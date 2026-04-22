/// Run with tracing:
/// ```bash
/// # Chrome Tracing JSON output (view in chrome://tracing)
/// cargo run --release --package jolt-atlas-core --example qwen -- --trace
///
/// # Terminal output with timing
/// cargo run --release --package jolt-atlas-core --example qwen -- --trace-terminal
///
/// # Override sequence length (default: 16)
/// cargo run --release --package jolt-atlas-core --example qwen -- --seq-len 32
///
/// # Reuse cached shared preprocessing (builds and saves it on first use)
/// cargo run --release --package jolt-atlas-core --example qwen -- --use-cache
///
/// # Disable pre-rebase nonlinear decomposition
/// cargo run --release --package jolt-atlas-core --example qwen -- --no-pre-rebase-nonlinear
/// ```
///
/// Requires the Qwen ONNX model to be present first.
use atlas_onnx_tracer::{
    model::{Model, RunArgs},
    tensor::Tensor,
};
use bincode::config::standard;
use common::utils::logging::setup_tracing;
use jolt_atlas_core::onnx_proof::{
    AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing,
    Blake2bTranscript, Bn254, Fr, HyperKZG, ONNXProof,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{env, fs, path::Path};

const MODEL_PATH: &str = "atlas-onnx-tracer/models/qwen/network.onnx";
const SHARED_PP_CACHE_PATH: &str = "atlas-onnx-tracer/models/qwen/shared_preprocessing.bin";

fn parse_seq_len_arg(args: &[String]) -> usize {
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        if arg == "--seq-len" {
            let value = args.next().expect("--seq-len requires a value");
            return value
                .parse::<usize>()
                .expect("--seq-len must be a positive integer");
        }
    }
    16
}

fn load_or_build_shared_preprocessing(
    run_args: &RunArgs,
    use_cache: bool,
) -> AtlasSharedPreprocessing {
    if use_cache && Path::new(SHARED_PP_CACHE_PATH).exists() {
        let bytes = fs::read(SHARED_PP_CACHE_PATH).expect("failed to read shared preprocessing");
        let (shared, _): (AtlasSharedPreprocessing, usize) =
            bincode::serde::decode_from_slice(&bytes, standard())
                .expect("failed to decode shared preprocessing");
        return shared;
    }

    let model = Model::load(MODEL_PATH, run_args);
    println!("{}", model.pretty_print());
    println!("max num vars: {}", model.max_num_vars());

    let shared = AtlasSharedPreprocessing::preprocess(model);

    if use_cache {
        let bytes = bincode::serde::encode_to_vec(&shared, standard())
            .expect("failed to encode shared preprocessing");
        fs::write(SHARED_PP_CACHE_PATH, bytes).expect("failed to write shared preprocessing");
        println!("saved shared preprocessing cache to {SHARED_PP_CACHE_PATH}");
    }

    shared
}

fn main() {
    let (_guard, _tracing_enabled) = setup_tracing("Qwen ONNX Proof");
    let args: Vec<String> = env::args().collect();
    let use_cache = args.iter().any(|arg| arg == "--use-cache");
    let pre_rebase_nonlinear = !args.iter().any(|arg| arg == "--no-pre-rebase-nonlinear");
    let seq_len = parse_seq_len_arg(&args);

    let run_args = RunArgs::new([
        ("batch_size", 1),
        ("sequence_length", seq_len),
        ("past_sequence_length", 0),
    ])
    .with_pre_rebase_nonlinear(pre_rebase_nonlinear);

    let mut rng = StdRng::seed_from_u64(44);
    let vocab_size: i32 = 151936;

    let input_ids_data: Vec<i32> = (0..seq_len).map(|_| rng.gen_range(0..vocab_size)).collect();
    let input_ids = Tensor::new(Some(&input_ids_data), &[1, seq_len]).unwrap();

    let token_type_ids_data: Vec<i32> = vec![0; seq_len];
    let token_type_ids = Tensor::new(Some(&token_type_ids_data), &[1, seq_len]).unwrap();

    let attention_mask_data: Vec<i32> = vec![1; seq_len];
    let attention_mask = Tensor::new(Some(&attention_mask_data), &[1, seq_len]).unwrap();

    tracing::info!("Loaded input data");
    let pp = load_or_build_shared_preprocessing(&run_args, use_cache);
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
