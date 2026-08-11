// Shared across examples; each one only uses a subset, so dead_code would fire per-binary.
#[path = "utils/qea.rs"]
#[allow(dead_code)]
mod qea;

use atlas_onnx_tracer::{model::Model, utils::quantize};
use qea::{
    Ctx, init_logging, make_f64_inputs, make_qwen_i32_inputs, make_run_args, run_tract_f32,
    step1_quant_vs_tract, step2a_per_node_drift_cumulative, step2b_per_node_drift_isolated,
    step3_shadow_vs_tract, step4_weight_quant_effect, step5_greedy_generation,
};
use tokenizers::Tokenizer;
use tracing::{debug, info};

const ONNX_PATH: &str = "atlas-onnx-tracer/models/qwen/network.onnx";
const TOKENIZER_PATH: &str = "atlas-onnx-tracer/models/qwen/tokenizer.json";
const VOCAB_SIZE: usize = 151936;
const SCALE: i32 = 14;

/// Quantization error analysis for Qwen.
///
/// Compares four views of the same forward pass (see GLOSSARY in the output):
///
/// | Label     | Description                                          |
/// |-----------|------------------------------------------------------|
/// | TRACT     | f32 Tract ONNX reference (ground truth)              |
/// | QUANT     | Our i32 fixed-point engine (scale = 2^N)             |
/// | SHADOW    | f64 math, quantized weights (isolates rounding)      |
/// | TRUE-F64  | f64 math, original weights (isolates weight quant)   |
///
/// The output is organised into five numbered steps:
///
/// 1. **QUANT vs TRACT** — overall quantization error.
/// 2. **SHADOW vs QUANT per-node** — where rounding error accumulates.
/// 3. **SHADOW vs TRACT** — verifies the shadow is faithful.
/// 4. **TRUE-F64 vs TRACT / SHADOW vs TRUE-F64** — isolates weight-quant error.
/// 5. **TRACT greedy generation** — sanity check.
///
/// # Setup
///
/// ```sh
/// python3 -m venv .venv && source .venv/bin/activate
/// python scripts/download_qwen.py
/// ```
///
/// # Usage
///
/// Default (info-level output, tract logs silenced):
/// ```sh
/// cargo run -r -p atlas-onnx-tracer --example qwen_quant_error_analysis
/// ```
///
/// Show debug output (shapes, token details):
/// ```sh
/// RUST_LOG=debug cargo run -r -p atlas-onnx-tracer --example qwen_quant_error_analysis
/// ```
///
/// Show everything *including* tract internals:
/// ```sh
/// RUST_LOG=trace cargo run -r -p atlas-onnx-tracer --example qwen_quant_error_analysis
/// ```
fn main() {
    init_logging();
    let ctx = setup("The quick brown fox jumps over the lazy dog");

    print_glossary(&ctx);
    step1_quant_vs_tract(&ctx);
    let shadow = step2a_per_node_drift_cumulative(&ctx);
    step2b_per_node_drift_isolated(&ctx);
    step3_shadow_vs_tract(&ctx, &shadow);
    step4_weight_quant_effect(&ctx, &shadow);
    step5_greedy_generation(&ctx);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Model-specific setup
// ═══════════════════════════════════════════════════════════════════════════

fn setup(text: &str) -> Ctx {
    let tokenizer = Tokenizer::from_file(TOKENIZER_PATH)
        .expect("failed to load tokenizer.json – run scripts/download_qwen.py first");

    info!("Input text : \"{text}\"");
    let encoding = tokenizer.encode(text, false).expect("tokenization failed");
    let token_ids = encoding.get_ids().to_vec();
    let seq_len = token_ids.len();
    debug!(tokens = ?encoding.get_tokens(), "Tokens");
    debug!(?token_ids, "Token IDs");
    debug!(seq_len, "Sequence length");

    let run_args = make_run_args(seq_len, SCALE);
    let scale = run_args.scale;
    let scale_mult = quantize::scale_to_multiplier(scale);

    info!("Running TRACT (f32 reference) …");
    let f32_outputs = run_tract_f32(ONNX_PATH, &token_ids, seq_len, &run_args);
    debug!(shape = ?f32_outputs[0].dims(), "TRACT output shape");

    info!("Running QUANT (i32 fixed-point, scale=2^{scale}) …");
    let model = Model::load(ONNX_PATH, &run_args);
    let quant_inputs = make_qwen_i32_inputs(&token_ids, seq_len, scale);
    let i32_outputs = model.forward(&quant_inputs);
    debug!(shape = ?i32_outputs[0].dims(), "QUANT output shape");

    let last_pos_start = (seq_len - 1) * VOCAB_SIZE;
    let ref_logits =
        qea::last_position_logits_f32(f32_outputs[0].data(), last_pos_start, VOCAB_SIZE);
    let deq_logits = qea::last_position_logits_i32(
        i32_outputs[0].data(),
        last_pos_start,
        VOCAB_SIZE,
        scale_mult,
    );

    let shadow_inputs = make_f64_inputs(&token_ids, seq_len);

    Ctx {
        tokenizer,
        model,
        run_args,
        scale,
        onnx_path: ONNX_PATH,
        vocab_size: VOCAB_SIZE,
        token_ids,
        seq_len,
        last_pos_start,
        quant_inputs,
        shadow_inputs,
        ref_logits,
        deq_logits,
    }
}

fn print_glossary(ctx: &Ctx) {
    qea::print_section("GLOSSARY");
    info!("  This analysis compares four views of the same Qwen forward pass:");
    info!("");
    info!("    TRACT     ONNX model evaluated by Tract at f32 precision.");
    info!("              This is the ground-truth reference.");
    info!("");
    info!(
        "    QUANT     Our quantized integer engine (scale = 2^{}).",
        ctx.scale
    );
    info!("              All weights and activations are i32 fixed-point.");
    info!("");
    info!("    SHADOW    f64 simulation running in lockstep with QUANT.");
    info!("              Uses the SAME quantized constants, but f64 arithmetic.");
    info!("              Drift from QUANT = rounding error in integer math.");
    info!("");
    info!("    TRUE-F64  f64 simulation using ORIGINAL f32 weights (not quantized).");
    info!("              Drift from SHADOW = error caused by weight quantization.");
    info!("              Drift from TRACT  = negligible (f64 vs f32 precision).");
    info!("");
    info!("  All logit comparisons use the LAST token position (vocab = {VOCAB_SIZE}).");
}
