/// Run with tracing:
/// ```bash
/// # Chrome Tracing JSON output (view in chrome://tracing)
/// cargo run --release --package jolt-atlas-core --example bge  -- --trace
///
/// # Terminal output with timing
/// cargo run --release --package jolt-atlas-core --example bge  -- --trace-terminal
/// ```
///
/// Requires BGE ONNX model download first:
/// ```bash
/// python scripts/download_bge_small_en_v1_5.py
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
    let (_guard, _tracing_enabled) = setup_tracing("BGE ONNX Proof");

    // Reduce sequence_length for faster tracing; increase as needed.
    let seq_len: usize = 16;
    let run_args = RunArgs::new([
        ("batch_size", 1),
        ("sequence_length", seq_len),
        ("past_sequence_length", 0),
    ])
    .with_pre_rebase_nonlinear(true);
    let model = Model::load(
        "atlas-onnx-tracer/models/bge-small-en-v1.5/network.onnx",
        &run_args,
    );
    println!("{}", model.pretty_print());
    println!("max num vars: {}", model.max_num_vars());

    let mut rng = StdRng::seed_from_u64(43);
    let vocab_size: i32 = 30522;

    // Input 0: input_ids — random token IDs used as Gather indices
    let input_ids_data: Vec<i32> = (0..seq_len).map(|_| rng.gen_range(0..vocab_size)).collect();
    let input_ids = Tensor::new(Some(&input_ids_data), &[1, seq_len]).unwrap();

    // Input 1: token_type_ids — segment IDs in {0, 1}. Use all zeros for single-segment input.
    let token_type_ids_data: Vec<i32> = vec![0; seq_len];
    let token_type_ids = Tensor::new(Some(&token_type_ids_data), &[1, seq_len]).unwrap();

    // Input 2: attention_mask — all 1s (attend everywhere), in BERT-style binary form.
    let attention_mask_data: Vec<i32> = vec![1; seq_len];
    let attention_mask = Tensor::new(Some(&attention_mask_data), &[1, seq_len]).unwrap();

    tracing::info!("Loaded model and generated input data");
    let pp = AtlasSharedPreprocessing::preprocess(model);
    let prover_preprocessing = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(pp);

    let timing = std::time::Instant::now();
    let (proof, io, _debug_info) = ONNXProof::<Fr, Blake2bTranscript, HyperKZG<Bn254>>::prove(
        &prover_preprocessing,
        &[input_ids, token_type_ids, attention_mask],
    );
    println!("Proof generation took {:.2?}", timing.elapsed());

    let verifier_preprocessing =
        AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_preprocessing);

    proof.verify(&verifier_preprocessing, &io, None).unwrap();
    println!("Proof verified successfully!");
}
