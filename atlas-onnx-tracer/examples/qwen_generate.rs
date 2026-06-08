use atlas_onnx_tracer::{
    model::Model,
    utils::{
        qea::{init_logging, make_qwen_i32_inputs, make_run_args, tract_greedy_generate},
        quantize,
    },
};
use tokenizers::Tokenizer;
use tracing::info;

const ONNX_PATH: &str = "atlas-onnx-tracer/models/qwen/network.onnx";
const TOKENIZER_PATH: &str = "atlas-onnx-tracer/models/qwen/tokenizer.json";
const VOCAB_SIZE: usize = 151936;

const SCALE: i32 = 20;

const DEFAULT_PROMPT: &str = "The quick brown fox";
const N_TOKENS: usize = 20;

/// Qwen2-0.5B text generation demo.
///
/// Runs greedy generation in two modes and prints both outputs side-by-side:
///
/// **TRACT** (f32, ground truth): produces coherent English using Tract's native
/// f32 inference engine — this is what the model should output.
///
/// **QUANT** (i32, our engine): runs the same model through the i32 fixed-point
/// quantized path. `fused-ops` improvements over the un-fixed baseline:
///   - Cos/Sin RoPE fix (removed scale=8-specific `const_rem` that corrupted RoPE)
///   - Scale=16 (4× finer weight quantization than scale=14)
///   - i64 softmax (`s² = S²` in i64 enabling scale>15)
///   - Teleportation removed from Sigmoid/Tanh/Erf in fused-ops path
///
/// At scale=16 the quantized model produces valid English tokens; further
/// precision improvements are needed for fully coherent generation.
///
/// # Setup
///
/// ```sh
/// python scripts/download_qwen.py
/// ```
///
/// # Usage
///
/// ```sh
/// cargo run -r -p atlas-onnx-tracer --features fused-ops --example qwen_generate
/// ```
///
/// Override prompt:
/// ```sh
/// QWEN_PROMPT="Once upon a time" cargo run -r -p atlas-onnx-tracer --features fused-ops --example qwen_generate
/// ```
fn main() {
    init_logging();

    let prompt = std::env::var("QWEN_PROMPT").unwrap_or_else(|_| DEFAULT_PROMPT.to_string());
    let tokenizer = Tokenizer::from_file(TOKENIZER_PATH)
        .expect("failed to load tokenizer – run scripts/download_qwen.py first");

    let model_label = if ONNX_PATH.contains("gptq") {
        "Qwen2-0.5B (GPTQ)"
    } else {
        "Qwen2-0.5B"
    };
    info!("Model  : {model_label}");
    info!("Prompt : \"{prompt}\"");
    info!("Tokens : {N_TOKENS}");
    info!("");

    let encoding = tokenizer.encode(prompt.as_str(), false).expect("tokenize");
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();

    // ── TRACT (f32) generation ──────────────────────────────────────────────
    info!("TRACT (f32, ground truth):");
    let tract_ids = tract_greedy_generate(ONNX_PATH, &prompt_ids, N_TOKENS, VOCAB_SIZE);
    let tract_text = tokenizer
        .decode(&tract_ids, false)
        .unwrap_or_else(|_| "<decode error>".to_string());
    info!("  \"{tract_text}\"");
    info!("");

    // ── QUANT (i32) generation ──────────────────────────────────────────────
    info!("QUANT (i32, scale=2^{SCALE}, fused-ops):");
    let quant_ids = quant_greedy_generate(&prompt_ids, N_TOKENS);
    let quant_text = tokenizer
        .decode(&quant_ids, false)
        .unwrap_or_else(|_| "<decode error>".to_string());
    info!("  \"{quant_text}\"");
}

fn quant_greedy_generate(prompt_ids: &[u32], n_tokens: usize) -> Vec<u32> {
    let mut ids = prompt_ids.to_vec();
    let scale_mult = quantize::scale_to_multiplier(SCALE);
    for _ in 0..n_tokens {
        let seq_len = ids.len();
        let run_args = make_run_args(seq_len, SCALE);
        let model = Model::load(ONNX_PATH, &run_args);
        let inputs = make_qwen_i32_inputs(&ids, seq_len, SCALE);
        let outputs = model.forward(&inputs);
        let logits: &[i32] = outputs[0].data();
        let last_pos = (seq_len - 1) * VOCAB_SIZE;
        let next = logits[last_pos..last_pos + VOCAB_SIZE]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let ar = **a as f64 / scale_mult;
                let br = **b as f64 / scale_mult;
                ar.partial_cmp(&br).unwrap()
            })
            .map(|(id, _)| id as u32)
            .unwrap();
        ids.push(next);
    }
    ids
}
