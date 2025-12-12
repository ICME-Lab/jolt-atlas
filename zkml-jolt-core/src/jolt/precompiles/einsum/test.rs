//! Common test utility for einsum operations

use crate::jolt::{
    JoltProverPreprocessing, JoltSharedPreprocessing, JoltVerifierPreprocessing,
    bytecode::BytecodePreprocessing,
    dag::state_manager::StateManager,
    pcs::SumcheckId,
    precompiles::{PrecompilePreprocessing, PrecompileSNARK},
    sumcheck::SumcheckInstance,
    trace::JoltONNXCycle,
    witness::VirtualPolynomial,
};
use ark_bn254::Fr;
use jolt_core::{
    poly::{
        commitment::mock::MockCommitScheme,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::OpeningPoint,
    },
    transcripts::Blake2bTranscript,
};
use onnx_tracer::{ProgramIO, tensor::Tensor};
use rand::{SeedableRng, rngs::StdRng};

pub type TestInstances = (
    (
        Vec<Box<dyn SumcheckInstance<Fr>>>,
        Vec<Vec<i64>>,
        Vec<Vec<i64>>,
    ),
    Vec<Box<dyn SumcheckInstance<Fr>>>,
);

/// Generic test harness for einsum operations
pub fn test_einsum_instances(
    instance_generator: impl Fn(StdRng, usize, (usize, usize, usize, usize)) -> TestInstances,
    max_dims: (usize, usize, usize, usize),
    seed: u64,
    num_instances: usize,
) {
    let bytecode_preprocessing = BytecodePreprocessing::default();
    let shared_preprocessing = JoltSharedPreprocessing {
        bytecode: bytecode_preprocessing,
        precompiles: PrecompilePreprocessing::empty(),
        fp_lookups: Default::default(),
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
        min_lookup_input: 0,
        max_lookup_input: 0,
    };

    let trace = vec![JoltONNXCycle::no_op(); 16];
    let mut prover_sm = StateManager::<'_, Fr, Blake2bTranscript, _>::new_prover(
        &prover_preprocessing,
        trace.clone(),
        program_io.clone(),
    );

    let rng = StdRng::seed_from_u64(seed);
    let ((mut prover_instances, a_instances, b_instances), verifier_instances) =
        instance_generator(rng, num_instances, max_dims);

    let proof = PrecompileSNARK::prove_batched_sumchecks(&mut prover_instances, &prover_sm);
    let acc = prover_sm.get_prover_accumulator();

    for index in 0..num_instances {
        let (r_a, a_claim) = acc.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::PrecompileA(index),
            SumcheckId::PrecompileExecution,
        );
        assert_eq!(
            MultilinearPolynomial::from(a_instances[index].to_vec()).evaluate(&r_a.r),
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
