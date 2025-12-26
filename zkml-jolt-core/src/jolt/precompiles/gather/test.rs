//! Common test utility for gather operations

use crate::jolt::{
    JoltProverPreprocessing, JoltSharedPreprocessing, JoltVerifierPreprocessing,
    bytecode::BytecodePreprocessing,
    dag::state_manager::StateManager,
    pcs::{OpeningId, SumcheckId},
    precompiles::PrecompilePreprocessing,
    sumcheck::{BatchedSumcheck, SumcheckInstance},
    trace::JoltONNXCycle,
    witness::VirtualPolynomial,
};
use ark_bn254::Fr;
use itertools::Itertools;
use jolt_core::{
    field::JoltField,
    poly::{
        commitment::{commitment_scheme::CommitmentScheme, mock::MockCommitScheme},
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{BIG_ENDIAN, OpeningPoint},
    },
    transcripts::{Blake2bTranscript, Transcript},
    utils::math::Math,
};
use onnx_tracer::{ProgramIO, tensor::Tensor};
use rand::{Rng, RngCore, SeedableRng, rngs::StdRng};
use std::collections::BTreeMap;

// Creates a random `Gather` instance.
pub fn random_gather(
    rng: &mut StdRng,
    // Number of dictionnary entries to recover
    max_num_lookups: usize,
    // Number of words in the dictionnary to gather from
    max_num_words: usize,
    // Number of dimensions per word of dictionnary
    max_word_dim: usize,
) -> TestIO {
    let num_lookups = rng.gen_range(1..max_num_lookups);
    let num_words = rng.gen_range(1..max_num_words);
    let word_dim = rng.gen_range(1..max_word_dim);

    let read_addresses_vals: Vec<i32> = (0..num_lookups)
        .map(|_| rng.gen_range(0..num_words) as i32)
        .collect();

    let dictionnary_vals: Vec<i32> = (0..num_words * word_dim)
        .map(|_| rng.next_u32() as i32)
        .collect();

    // Expected output of the gather node: a matrix where each row `i` corresponds to the row at index `a[i]` in B
    let output_vals: Vec<i32> = read_addresses_vals
        .iter()
        .flat_map(|&index| {
            let index = index as usize;
            dictionnary_vals[index * word_dim..(index + 1) * word_dim].to_vec()
        })
        .collect();

    let mut ra_tensor = Tensor::new(Some(&read_addresses_vals), &[num_lookups]).unwrap();
    let mut dict_tensor = Tensor::new(Some(&dictionnary_vals), &[num_words, word_dim]).unwrap();
    let mut out_tensor = Tensor::new(Some(&output_vals), &[num_lookups, word_dim]).unwrap();

    ra_tensor.pad_next_power_of_two();
    dict_tensor.pad_next_power_of_two();
    out_tensor.pad_next_power_of_two();

    let read_addresses: Vec<Fr> = ra_tensor.iter().map(|&e| Fr::from_u32(e as u32)).collect();
    let dictionnary: Vec<Fr> = dict_tensor
        .iter()
        .map(|&e| Fr::from_u32(e as u32))
        .collect();
    let output: Vec<Fr> = out_tensor.iter().map(|&e| Fr::from_u32(e as u32)).collect();

    TestIO::new(
        read_addresses,
        dictionnary,
        output,
        (num_lookups, num_words, word_dim),
    )
}

pub struct TestInstance<F: JoltField> {
    prover_instance: Box<dyn SumcheckInstance<F>>,
    verifier_instance: Box<dyn SumcheckInstance<F>>,
    openings: Vec<VirtualOpening>,
    io: TestIO,
}

impl<F: JoltField> TestInstance<F> {
    pub fn new(
        pi: Box<dyn SumcheckInstance<F>>,
        vi: Box<dyn SumcheckInstance<F>>,
        openings: Vec<VirtualOpening>,
        io: TestIO,
    ) -> Self {
        Self {
            prover_instance: pi,
            verifier_instance: vi,
            openings,
            io,
        }
    }
}

pub struct TestIO {
    a: Vec<Fr>,
    b: Vec<Fr>,
    c: Vec<Fr>,
    // (num_lookups, num_words, word_dim) | unpadded
    dims: (usize, usize, usize),
}

impl TestIO {
    pub fn new(a: Vec<Fr>, b: Vec<Fr>, c: Vec<Fr>, dims: (usize, usize, usize)) -> Self {
        Self { a, b, c, dims }
    }

    pub fn values(&self) -> (&Vec<Fr>, &Vec<Fr>, &Vec<Fr>) {
        (&self.a, &self.b, &self.c)
    }

    pub fn dims(&self) -> (usize, usize, usize) {
        self.dims
    }
}

pub struct VirtualOpening {
    pub poly: VirtualPolynomial,
    pub sumcheck: SumcheckId,
    pub r: Vec<Fr>,
    pub claim: Fr,
}

impl VirtualOpening {
    pub fn new(poly: VirtualPolynomial, sumcheck: SumcheckId, r: Vec<Fr>, claim: Fr) -> Self {
        Self {
            poly,
            sumcheck,
            r,
            claim,
        }
    }

    pub fn initial_claim<PT, CS>(
        prover_sm: &mut StateManager<'_, Fr, PT, CS>,
        verifier_sm: &mut StateManager<'_, Fr, PT, CS>,
        poly_id: VirtualPolynomial,
        sumcheck: SumcheckId,
        poly: MultilinearPolynomial<Fr>,
        // Whether to use an existing challenge vector
        r: Option<Vec<Fr>>,
    ) -> Self
    where
        PT: Transcript,
        CS: CommitmentScheme<Field = Fr>,
    {
        let num_vars = poly.len().log_2();

        let r = if let Some(r) = r {
            assert_eq!(r.len(), num_vars);
            r
        } else {
            // If no challenge is fed, get a new one from the transcript
            let r: Vec<Fr> = prover_sm
                .get_transcript()
                .borrow_mut()
                .challenge_vector(num_vars);
            let _r: Vec<Fr> = verifier_sm
                .get_transcript()
                .borrow_mut()
                .challenge_vector(num_vars);
            assert_eq!(r, _r);
            r
        };

        let claim = poly.evaluate(&r);

        prover_sm
            .get_prover_accumulator()
            .borrow_mut()
            .openings
            .insert(
                OpeningId::Virtual(poly_id, sumcheck),
                (r.clone().into(), claim),
            );
        verifier_sm
            .get_verifier_accumulator()
            .borrow_mut()
            .openings_mut()
            .insert(
                OpeningId::Virtual(poly_id, sumcheck),
                (r.clone().into(), claim),
            );

        Self::new(poly_id, sumcheck, r, claim)
    }

    pub fn insert_in_prover_sm<'a, PT, CS>(&self, sm: &mut StateManager<'a, Fr, PT, CS>)
    where
        PT: Transcript,
        CS: CommitmentScheme<Field = Fr>,
    {
        sm.get_prover_accumulator().borrow_mut().append_virtual(
            self.poly,
            self.sumcheck,
            self.r.clone().into(),
            self.claim,
        );
    }

    pub fn insert_in_verifier_sm<'a, PT, CS>(&self, sm: &mut StateManager<'a, Fr, PT, CS>)
    where
        PT: Transcript,
        CS: CommitmentScheme<Field = Fr>,
    {
        sm.get_verifier_accumulator().borrow_mut().append_virtual(
            self.poly,
            self.sumcheck,
            self.r.clone().into(),
        );
    }
}

pub fn test_gather_sumcheck(
    instance_generator: impl Fn(
        &mut StdRng,
        &mut StateManager<'_, Fr, Blake2bTranscript, MockCommitScheme<Fr>>,
        &mut StateManager<'_, Fr, Blake2bTranscript, MockCommitScheme<Fr>>,
        (usize, usize, usize), // dims
        usize,                 // num_instances
    ) -> TestInstance<Fr>,
    (max_num_lookups, max_num_words, max_word_dim): (usize, usize, usize),
    seed: u64,
    num_instances: usize,
) -> (
    Vec<TestIO>,
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

    let mut prover_instances: Vec<Box<dyn SumcheckInstance<Fr>>> = Vec::new();
    let mut verifier_instances: Vec<Box<dyn SumcheckInstance<Fr>>> = Vec::new();
    let mut test_io_vec: Vec<TestIO> = Vec::new();
    let mut openings_acc = Vec::new();

    for i in 0..num_instances {
        let TestInstance {
            prover_instance,
            verifier_instance,
            openings,
            io,
        } = instance_generator(
            &mut rng,
            &mut prover_sm,
            &mut verifier_sm,
            (max_num_lookups, max_num_words, max_word_dim),
            i,
        );

        prover_instances.push(prover_instance);
        verifier_instances.push(verifier_instance);
        test_io_vec.push(io);
        openings_acc.extend(openings);
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

    openings_acc
        .iter()
        .for_each(|opening| opening.insert_in_prover_sm(&mut prover_sm));

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

    openings_acc
        .iter()
        .for_each(|opening| opening.insert_in_verifier_sm(&mut verifier_sm));

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
        test_io_vec,
        prover_sm
            .get_prover_accumulator()
            .borrow()
            .evaluation_openings()
            .clone(),
    )
}
