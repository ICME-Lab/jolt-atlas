/// Run with tracing:
/// ```bash
/// # Chrome Tracing JSON output (view in chrome://tracing)
/// cargo run --release --package jolt-atlas-core --example qwen -- --trace
///
/// # Terminal output with timing
/// cargo run --release --package jolt-atlas-core --example qwen -- --trace-terminal
///
/// # Override the input prompt
/// cargo run --release --package jolt-atlas-core --example qwen -- --input "Hello, world!"
///
/// # Over-provision the Dory SRS (wider columns => fewer tier-2 pairings per commit)
/// cargo run --release --package jolt-atlas-core --example qwen -- --srs-vars 34
///
/// # Reuse cached shared preprocessing (only valid for inputs that tokenize to the same
/// # sequence length as the cached run)
/// cargo run --release --package jolt-atlas-core --example qwen -- --use-cache
/// ```
///
/// Requires the Qwen ONNX model (`python scripts/download_qwen.py`) and
/// `common::consts::MODEL_SCALE == 14` (Qwen's export scale) — several proving-side lookup
/// tables are const-generic on `MODEL_SCALE` rather than runtime scale, so this can't be passed
/// as a flag; edit `common/src/consts/general.rs` and recompile if needed.
use atlas_onnx_tracer::{
    model::{Model, RunArgs},
    tensor::Tensor,
};
use bincode::config::standard;
use common::utils::logging::setup_tracing;
use jolt_atlas_core::onnx_proof::{
    AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing,
    Blake2bTranscript, DoryScheme, Fr, ONNXProof,
};
use std::{env, fs, path::Path};
use tokenizers::Tokenizer;

const MODEL_PATH: &str = "atlas-onnx-tracer/models/qwen/network.onnx";
const TOKENIZER_PATH: &str = "atlas-onnx-tracer/models/qwen/tokenizer.json";
const SHARED_PP_CACHE_PATH: &str = "atlas-onnx-tracer/models/qwen/shared_preprocessing.bin";
/// Same default as `qwen_quant_error_analysis`'s `setup(...)` call, so the two examples are
/// directly comparable out of the box.
const DEFAULT_PROMPT: &str = "The quick brown fox jumps over the lazy dog";

fn parse_srs_vars_arg(args: &[String]) -> Option<usize> {
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        if arg == "--srs-vars" {
            return Some(
                args.next()
                    .expect("--srs-vars requires a value")
                    .parse()
                    .expect("--srs-vars must be an integer"),
            );
        }
    }
    None
}

fn parse_input_arg(args: &[String]) -> String {
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        if arg == "--input" {
            return args.next().expect("--input requires a value").clone();
        }
    }
    DEFAULT_PROMPT.to_string()
}

/// Per-operator histogram of the statically derived saturating-clamp widths
/// (see `atlas_onnx_tracer::model::clamp_width`).
fn print_clamp_width_histogram(model: &Model) {
    use std::collections::BTreeMap;
    let mut hist: BTreeMap<(String, usize), usize> = BTreeMap::new();
    for node in model.graph.nodes.values() {
        if atlas_onnx_tracer::model::clamp_width::clamp_value_bound(node, &model.graph.nodes)
            .is_some()
        {
            let op = format!("{:?}", node.operator);
            let op = op
                .split(|c: char| !c.is_alphanumeric())
                .next()
                .unwrap()
                .to_string();
            *hist.entry((op, node.sat_clamp_bits)).or_default() += 1;
        }
    }
    println!("saturating-clamp widths (op, bits) -> nodes:");
    for ((op, bits), n) in hist {
        println!("  {op:<16} {bits:>2} bits  x{n}");
    }
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
    print_clamp_width_histogram(&model);

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
    assert_eq!(
        common::consts::MODEL_SCALE,
        14,
        "this example requires common::consts::MODEL_SCALE == 14 (Qwen's export scale); \
         currently compiled with MODEL_SCALE = {}. Edit common/src/consts/general.rs and \
         recompile.",
        common::consts::MODEL_SCALE
    );

    let (_guard, _tracing_enabled) = setup_tracing("Qwen ONNX Proof");
    let args: Vec<String> = env::args().collect();
    let use_cache = args.iter().any(|arg| arg == "--use-cache");
    let input_text = parse_input_arg(&args);

    let tokenizer = Tokenizer::from_file(TOKENIZER_PATH)
        .expect("failed to load tokenizer.json – run scripts/download_qwen.py first");
    let encoding = tokenizer
        .encode(input_text.as_str(), false)
        .expect("tokenization failed");
    let token_ids = encoding.get_ids().to_vec();
    let seq_len = token_ids.len();
    tracing::info!(input = %input_text, seq_len, "Tokenized input");

    let run_args = RunArgs::new([
        ("batch_size", 1),
        ("sequence_length", seq_len),
        ("past_sequence_length", 0),
    ]);
    let scale = run_args.scale;

    let input_ids_data: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();
    let input_ids = Tensor::new(Some(&input_ids_data), &[1, seq_len]).unwrap();

    // Qwen export input 1 is `position_ids`, fixed-point (i << scale) so RoPE's
    // Einsum(inv_freq, pos)/2^scale = inv_freq * i. Matches `qea::make_qwen_i32_inputs`,
    // validated against Tract in `qwen_quant_error_analysis`.
    let max_pos = (i32::MAX >> scale) as usize;
    assert!(
        seq_len.saturating_sub(1) <= max_pos,
        "seq_len={seq_len} too large for scale={scale} (max position index {max_pos})"
    );
    let position_ids_data: Vec<i32> = (0..seq_len as i32).map(|i| i << scale).collect();
    let position_ids = Tensor::new(Some(&position_ids_data), &[1, seq_len]).unwrap();
    // Input 2 is `attention_mask`; provide quantized 1.0s (attend everywhere).
    let attention_mask_data: Vec<i32> = vec![1 << scale; seq_len];
    let attention_mask = Tensor::new(Some(&attention_mask_data), &[1, seq_len]).unwrap();

    tracing::info!("Loaded input data");
    let pp = load_or_build_shared_preprocessing(&run_args, use_cache);
    let prover_preprocessing = match parse_srs_vars_arg(&args) {
        Some(n) => AtlasProverPreprocessing::<Fr, DoryScheme>::new_with_srs_num_vars(pp, n),
        None => AtlasProverPreprocessing::<Fr, DoryScheme>::new(pp),
    };

    let timing = std::time::Instant::now();
    let (proof, io, _debug_info) = ONNXProof::<Fr, Blake2bTranscript, DoryScheme>::prove(
        &prover_preprocessing,
        &[input_ids, position_ids, attention_mask],
    );
    println!("Proof generation took {:.2?}", timing.elapsed());
    println!(
        "Proof size: {:.2} MiB ({} committed polys, {} sumcheck proofs)",
        ark_serialize::CanonicalSerialize::serialized_size(&proof, ark_serialize::Compress::Yes)
            as f64
            / (1024.0 * 1024.0),
        proof.commitments.len(),
        proof.proofs.len(),
    );

    let verifier_preprocessing =
        AtlasVerifierPreprocessing::<Fr, DoryScheme>::from(&prover_preprocessing);

    proof.verify(&verifier_preprocessing, &io, None).unwrap();
    println!("Proof verified successfully!");
}
