/// Run with tracing:
/// ```bash
/// # Chrome Tracing JSON output (view in chrome://tracing)
/// cargo run --example microgpt -- --trace
///
/// # Terminal output with timing
/// cargo run --example microgpt -- --trace-terminal
/// ```
use atlas_onnx_tracer::{model::Model, tensor::Tensor};
use common::utils::logging::setup_tracing;
use jolt_atlas_core::onnx_proof::{
    AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing,
    Blake2bTranscript, Bn254, Fr, HyperKZG, ONNXProof,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn main() {
    let (_guard, _tracing_enabled) = setup_tracing("microGPT ONNX Proof");
    let working_dir = "atlas-onnx-tracer/models/microgpt/";
    let mut rng = StdRng::seed_from_u64(0x42);

    // Model hyperparameters (matching the microGPT Python script by @karpathy)
    // vocab_size=32, n_embd=16, n_head=4, n_layer=1, block_size=16
    let vocab_size: i32 = 32;
    let block_size = 16;

    // Generate random token IDs in [0, vocab_size)
    let input_data: Vec<i32> = (0..block_size)
        .map(|_| rng.gen_range(0..vocab_size))
        .collect();
    let input = Tensor::new(Some(&input_data), &[1, block_size]).unwrap();
    let model = Model::load(&format!("{working_dir}network.onnx"), &Default::default());

    let pp = AtlasSharedPreprocessing::preprocess(model);
    let prover_preprocessing = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(pp);

    let (proof, io, _debug_info) =
        ONNXProof::<Fr, Blake2bTranscript, HyperKZG<Bn254>>::prove(&prover_preprocessing, &[input]);

    let verifier_preprocessing =
        AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_preprocessing);

    proof.verify(&verifier_preprocessing, &io, None).unwrap();
}
