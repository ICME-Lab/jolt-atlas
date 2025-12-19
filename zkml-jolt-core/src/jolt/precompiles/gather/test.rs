//! Common test utility for gather operations

use crate::jolt::{
    JoltProverPreprocessing, JoltSharedPreprocessing, JoltVerifierPreprocessing,
    bytecode::BytecodePreprocessing,
    dag::state_manager::StateManager,
    fp_lookups::FpLookupPreprocessing,
    pcs::{OpeningId, SumcheckId},
    precompiles::{PrecompilePreprocessing, PrecompileSNARK},
    sumcheck::{BatchedSumcheck, SumcheckInstance},
    trace::JoltONNXCycle,
    witness::VirtualPolynomial,
};
use ark_bn254::Fr;
use ark_std::One;
use itertools::Itertools;
use jolt_core::{
    field::JoltField,
    poly::{
        commitment::{commitment_scheme::CommitmentScheme, mock::MockCommitScheme},
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{BIG_ENDIAN, OpeningPoint},
    },
    transcripts::{Blake2bTranscript, Transcript},
    utils::thread::unsafe_allocate_zero_vec,
};
use onnx_tracer::{ProgramIO, tensor::Tensor};
use rand::{RngCore, SeedableRng, rngs::StdRng};
use std::collections::BTreeMap;

pub type TestInstances = (
    (
        Vec<Box<dyn SumcheckInstance<Fr>>>,
        Vec<Vec<i64>>,
        Vec<Vec<i64>>,
        Vec<(usize, usize, usize)>, // dims
    ),
    Vec<Box<dyn SumcheckInstance<Fr>>>,
);

// Creates a random `Gather` instance, and outputs a tuple of (a, B, output)
pub fn random_gather(
    rng: &mut StdRng,
    // Number of dictionnary entries to recover
    num_lookups: usize,
    // Number of words in the dictionnary to gather from
    num_words: usize,
    // Number of dimensions per word of dictionnary
    word_dim: usize,
) -> (Vec<Fr>, Vec<Fr>, Vec<Fr>) {
    let read_addresses: Vec<Fr> = (0..num_lookups)
        .map(|_| Fr::from_u32(rng.next_u32() % num_words as u32))
        .collect();

    let dictionnary: Vec<Fr> = (0..num_words * word_dim).map(|_| Fr::random(rng)).collect();

    // Expected output of the gather node: a matrix where each row `i` corresponds to the row at index `a[i]` in B
    let output: Vec<Fr> = read_addresses
        .iter()
        .flat_map(|&index| {
            let index = index.to_u64().unwrap() as usize;
            dictionnary[index * word_dim..(index + 1) * word_dim].to_vec()
        })
        .collect();

    (read_addresses, dictionnary, output)
}

pub struct VirtualOpening {
    poly: VirtualPolynomial,
    sumcheck: SumcheckId,
    point: Vec<Fr>,
    claim: Fr,
}

// TODO(AntoineF4C5): Hope is to remove this and have cleaner implementation
impl VirtualOpening {
    pub fn new(poly: VirtualPolynomial, sumcheck: SumcheckId, point: Vec<Fr>, claim: Fr) -> Self {
        Self {
            poly,
            sumcheck,
            point,
            claim,
        }
    }

    pub fn insert_in_prover_sm<'a, PT, CS>(&self, sm: &mut StateManager<'a, Fr, PT, CS>)
    where
        PT: Transcript,
        CS: CommitmentScheme<Field = Fr>,
    {
        sm.get_prover_accumulator().borrow_mut().append_virtual(
            self.poly,
            self.sumcheck,
            self.point.clone().into(),
            self.claim,
        );
    }

    pub fn insert_in_verifier_sm<'a, PT, CS>(&self, sm: &mut StateManager<'a, Fr, PT, CS>)
    where
        PT: Transcript,
        CS: CommitmentScheme<Field = Fr>,
    {
        sm.get_verifier_accumulator()
            .borrow_mut()
            .openings_mut()
            .insert(
                OpeningId::Virtual(self.poly, self.sumcheck),
                (OpeningPoint::default(), self.claim.clone()),
            );

        sm.get_verifier_accumulator().borrow_mut().append_virtual(
            self.poly,
            self.sumcheck,
            self.point.clone().into(),
        );
    }
}

/// Generic test harness for gather operations
pub fn test_sumcheck_instances(
    instance_generator: impl Fn(StdRng, usize, (usize, usize, usize, usize)) -> TestInstances,
    max_dims: (usize, usize, usize, usize),
    seed: u64,
    num_instances: usize,
) {
    let bytecode_preprocessing = BytecodePreprocessing::default();
    let shared_preprocessing = JoltSharedPreprocessing {
        bytecode: bytecode_preprocessing,
        precompiles: PrecompilePreprocessing::empty(),
        fp_lookups: FpLookupPreprocessing::empty(),
    };

    let prover_preprocessing: JoltProverPreprocessing<Fr, MockCommitScheme<Fr>> =
        JoltProverPreprocessing {
            generators: (),
            shared: shared_preprocessing.clone(),
        };

    let verifier_preprocessing: JoltVerifierPreprocessing<Fr, MockCommitScheme<Fr>> =
        JoltVerifierPreprocessing {
            generators: (),
            shared: shared_preprocessing,
        };

    let program_io = ProgramIO {
        input: Tensor::new(None, &[]).unwrap(),
        output: Tensor::new(None, &[]).unwrap(),
        max_lookup_input: 0,
        min_lookup_input: 0,
    };

    let trace = vec![JoltONNXCycle::no_op(); 16];
    let mut prover_sm = StateManager::<'_, Fr, Blake2bTranscript, _>::new_prover(
        &prover_preprocessing,
        trace.clone(),
        program_io.clone(),
    );

    let rng = StdRng::seed_from_u64(seed);
    let ((mut prover_instances, a_instances, b_instances, b_dims), verifier_instances) =
        instance_generator(rng, num_instances, max_dims);

    let proof = PrecompileSNARK::prove_batched_sumchecks(&mut prover_instances, &prover_sm);
    let acc = prover_sm.get_prover_accumulator();

    for index in 0..num_instances {
        let (m, k, _n) = b_dims[index];
        let (r_a, a_claim) = acc.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::PrecompileA(index),
            SumcheckId::PrecompileExecution,
        );
        let mut one_hot_a: Vec<Fr> = unsafe_allocate_zero_vec(m * k);
        for (j, &i) in a_instances[index].iter().enumerate() {
            one_hot_a[j * k + i as usize] = Fr::one();
        }
        println!("a_one-hot: {one_hot_a:?}");
        assert_eq!(
            MultilinearPolynomial::from(one_hot_a.to_vec()).evaluate(&r_a.r),
            a_claim
        );
        let (r_b, b_claim) = acc.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::PrecompileB(index),
            SumcheckId::PrecompileExecution,
        );
        assert_eq!(
            MultilinearPolynomial::from(b_instances[index].to_vec()).evaluate(&r_b.r),
            b_claim
        )
    }

    // Verify proof
    let verifier_sm = StateManager::<'_, Fr, Blake2bTranscript, _>::new_verifier(
        &verifier_preprocessing,
        program_io,
        trace.len(),
        1 << 8,
        prover_sm.twist_sumcheck_switch_index,
    );

    let prover_state = prover_sm.prover_state.as_mut().unwrap();
    let openings = std::mem::take(&mut prover_state.accumulator.borrow_mut().openings);
    let opening_accumulator = verifier_sm.get_verifier_accumulator();

    for (key, (_, claim)) in openings.iter() {
        opening_accumulator
            .borrow_mut()
            .openings_mut()
            .insert(*key, (OpeningPoint::default(), *claim));
    }

    PrecompileSNARK::verify_batched_sumchecks(&proof, verifier_instances, &verifier_sm).unwrap();
}

pub struct TestInstance {
    a: Vec<Fr>,
    b: Vec<Fr>,
    c: Vec<Fr>,
}

impl TestInstance {
    pub fn new(a: &[Fr], b: &[Fr], c: &[Fr]) -> Self {
        Self {
            a: a.to_vec(),
            b: b.to_vec(),
            c: c.to_vec(),
        }
    }
    pub fn get(self) -> (Vec<Fr>, Vec<Fr>, Vec<Fr>) {
        (self.a, self.b, self.c)
    }
}

pub fn test_gather_sumcheck(
    instance_generator: impl Fn(
        &mut StdRng,
        &mut StateManager<'_, Fr, Blake2bTranscript, MockCommitScheme<Fr>>,
        &mut StateManager<'_, Fr, Blake2bTranscript, MockCommitScheme<Fr>>,
        (usize, usize, usize), // TODO(AntoineF4C5): Temporary, this information should be held in state_manager
        usize,                 // num_instances
    ) -> (
        TestInstance,
        Box<dyn SumcheckInstance<Fr>>,
        Box<dyn SumcheckInstance<Fr>>,
    ),
    (num_lookups, num_words, word_dim): (usize, usize, usize),
    seed: u64,
    num_instances: usize,
) -> (
    Vec<TestInstance>,
    BTreeMap<OpeningId, (OpeningPoint<BIG_ENDIAN, Fr>, Fr)>,
) {
    let bytecode_pp = BytecodePreprocessing::default();
    let shared_pp = JoltSharedPreprocessing {
        bytecode: bytecode_pp,
        precompiles: PrecompilePreprocessing::empty(),
        fp_lookups: Default::default(),
    };

    let prover_preprocessing: JoltProverPreprocessing<Fr, MockCommitScheme<Fr>> =
        JoltProverPreprocessing {
            generators: (),
            shared: shared_pp.clone(),
        };

    let verifier_preprocessing: JoltVerifierPreprocessing<Fr, MockCommitScheme<Fr>> =
        JoltVerifierPreprocessing {
            generators: (),
            shared: shared_pp,
        };
    let program_io = ProgramIO {
        input: Tensor::new(None, &[]).unwrap(),
        output: Tensor::new(None, &[]).unwrap(),
        min_lookup_input: 0,
        max_lookup_input: 0,
    };

    let trace = vec![JoltONNXCycle::no_op(); 32];

    let mut prover_sm = StateManager::<'_, Fr, Blake2bTranscript, _>::new_prover(
        &prover_preprocessing,
        trace.clone(),
        program_io.clone(),
    );

    let mut verifier_sm = StateManager::<'_, Fr, Blake2bTranscript, _>::new_verifier(
        &verifier_preprocessing,
        program_io,
        trace.len(),
        1 << 8,
        prover_sm.twist_sumcheck_switch_index,
    );

    let mut rng = StdRng::seed_from_u64(seed);

    let mut test_instances: Vec<TestInstance> = Vec::new();
    let mut prover_instances: Vec<Box<dyn SumcheckInstance<Fr>>> = Vec::new();
    let mut verifier_instances: Vec<Box<dyn SumcheckInstance<Fr>>> = Vec::new();
    for i in 0..num_instances {
        let (test_instance, prover_instance, verifier_instance) = instance_generator(
            &mut rng,
            &mut prover_sm,
            &mut verifier_sm,
            (num_lookups, num_words, word_dim),
            i,
        );

        test_instances.push(test_instance);
        prover_instances.push(prover_instance);
        verifier_instances.push(verifier_instance);
    }

    let (prover_sumcheck, verifier_sumcheck): (
        Vec<&mut dyn SumcheckInstance<Fr>>,
        Vec<&dyn SumcheckInstance<Fr>>,
    ) = prover_instances
        .iter_mut()
        .zip_eq(verifier_instances.iter())
        .map(|(psc, vsc)| {
            (
                &mut **psc as &mut dyn SumcheckInstance<Fr>,
                &**vsc as &dyn SumcheckInstance<Fr>,
            )
        })
        .collect::<Vec<(&mut dyn SumcheckInstance<Fr>, &dyn SumcheckInstance<Fr>)>>()
        .into_iter()
        .unzip();

    let (proof, _r_sumcheck) = BatchedSumcheck::prove(
        prover_sumcheck,
        Some(prover_sm.get_prover_accumulator()),
        &mut *prover_sm.get_transcript().borrow_mut(),
    );

    // Take claims
    let prover_acc = prover_sm.get_prover_accumulator();
    let prover_acc_borrow = prover_acc.borrow();
    let verifier_accumulator = verifier_sm.get_verifier_accumulator();
    let mut verifier_acc_borrow = verifier_accumulator.borrow_mut();

    for (key, (_, value)) in prover_acc_borrow.evaluation_openings().iter() {
        let empty_point = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![]);
        verifier_acc_borrow
            .openings_mut()
            .insert(*key, (empty_point, *value));
    }
    drop((prover_acc_borrow, verifier_acc_borrow));

    let res = BatchedSumcheck::verify(
        &proof,
        verifier_sumcheck,
        Some(verifier_sm.get_verifier_accumulator()),
        &mut *verifier_sm.get_transcript().borrow_mut(),
    );

    assert!(
        res.is_ok(),
        "Sumcheck verification failed with error: {:?}",
        res.err()
    );

    (
        test_instances,
        prover_sm
            .get_prover_accumulator()
            .borrow()
            .evaluation_openings()
            .clone(),
    )
}
