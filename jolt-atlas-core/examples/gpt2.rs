/// Run with tracing:
/// ```bash
/// # Chrome Tracing JSON output (view in chrome://tracing)
/// cargo run --release --package jolt-atlas-core --example gpt2  -- --trace
///
/// # Terminal output with timing
/// cargo run --release --package jolt-atlas-core --example gpt2  -- --trace-terminal
/// ```
use atlas_onnx_tracer::{
    model::{Model, RunArgs},
    tensor::Tensor,
};
use common::utils::logging::setup_tracing;
use jolt_atlas_core::onnx_proof::{
    AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing,
    Blake2bTranscript, Bn254, Fr, HyperKZG, ONNXProof,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn main() {
    let (_guard, _tracing_enabled) = setup_tracing("nanoGPT ONNX Proof");

    // Reduce sequence_length for faster tracing; increase as needed (max 1024).
    let seq_len: usize = 16;
    let run_args = RunArgs::new([
        ("batch_size", 1),
        ("sequence_length", seq_len),
        ("past_sequence_length", 0),
    ])
    .with_pre_rebase_nonlinear(true);
    let model = Model::load("atlas-onnx-tracer/models/gpt2/model.onnx", &run_args);
    println!("{}", model.pretty_print());
    println!("max num vars: {}", model.max_num_vars());

    let mut rng = StdRng::seed_from_u64(42);
    let vocab_size: i32 = 50257;

    // Input 0 → node 1: input_ids — random token IDs used as Gather indices
    let input_ids_data: Vec<i32> = (0..seq_len).map(|_| rng.gen_range(0..vocab_size)).collect();
    let input_ids = Tensor::new(Some(&input_ids_data), &[1, seq_len]).unwrap();

    // Input 1 → node 4: position_ids — sequential positions used as Gather indices
    let position_ids_data: Vec<i32> = (0..seq_len as i32).collect();
    let position_ids = Tensor::new(Some(&position_ids_data), &[1, seq_len]).unwrap();

    // Input 2 → node 57: attention_mask — all 1s (attend everywhere)
    // The model's Cast handler divides by scale to de-quantize, so we provide
    // the mask in quantized form: 1.0 in fixed-point = 1 << scale.
    let scale = run_args.scale;
    let attention_mask_data: Vec<i32> = vec![1 << scale; seq_len];
    let attention_mask = Tensor::new(Some(&attention_mask_data), &[1, seq_len]).unwrap();

    tracing::info!("Loaded model and generated input data");
    let pp = AtlasSharedPreprocessing::preprocess(model);
    let prover_preprocessing = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(pp);

    let timing = std::time::Instant::now();
    let (proof, io, _debug_info) = ONNXProof::<Fr, Blake2bTranscript, HyperKZG<Bn254>>::prove(
        &prover_preprocessing,
        &[input_ids, position_ids, attention_mask],
    );
    println!("Proof generation took {:.2?}", timing.elapsed());

    let verifier_preprocessing =
        AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_preprocessing);

    proof.verify(&verifier_preprocessing, &io, None).unwrap();
    println!("Proof verified successfully!");
}
