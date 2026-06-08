use atlas_onnx_tracer::{
    model::{Model, RunArgs},
    tensor::Tensor,
    utils::{
        qea::{
            self, ShadowResult, THIN, extract_shadow_logits, init_logging,
            last_position_logits_f32, last_position_logits_i32, make_f64_inputs,
            make_qwen_i32_inputs, make_run_args, print_comparison_metrics, print_step,
            run_tract_f32, top_k_entries, tract_greedy_generate,
        },
        quantize,
    },
};
use tokenizers::Tokenizer;
use tracing::{debug, info};

const SCALE: i32 = 20; // 2^20 = 1,048,576 multiplier for fixed-point representation (roughly 6 decimal digits of precision)
const ONNX_PATH: &str = "atlas-onnx-tracer/models/qwen/network.onnx";
const TOKENIZER_PATH: &str = "atlas-onnx-tracer/models/qwen/tokenizer.json";
const VOCAB_SIZE: usize = 151936;

/// Quantization error analysis for Qwen2-0.5B.
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
/// python atlas-onnx-tracer/.venv/bin/python scripts/download_qwen.py
/// ```
///
/// # Usage
///
/// Default (info-level output, tract logs silenced):
/// ```sh
/// cargo run -r -p atlas-onnx-tracer --features fused-ops --example quant_error_analysis_qwen
/// ```
///
/// Show debug output (shapes, token details):
/// ```sh
/// RUST_LOG=debug cargo run -r -p atlas-onnx-tracer --features fused-ops --example quant_error_analysis_qwen
/// ```
fn main() {
    init_logging();
    let ctx = setup("The quick brown fox jumps over the lazy dog");

    print_glossary(&ctx);
    step1_quant_vs_tract(&ctx);
    let shadow = step2_per_node_drift(&ctx);
    step3_shadow_vs_tract(&ctx, &shadow);
    step4_weight_quant_effect(&ctx, &shadow);
    step5_greedy_generation(&ctx);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Analysis pipeline
// ═══════════════════════════════════════════════════════════════════════════

struct Ctx {
    tokenizer: Tokenizer,
    model: Model,
    run_args: RunArgs,
    scale: i32,
    token_ids: Vec<u32>,
    seq_len: usize,
    last_pos_start: usize,
    quant_inputs: Vec<Tensor<i32>>,
    ref_logits: Vec<f64>,
    deq_logits: Vec<f64>,
}

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
    let ref_logits = last_position_logits_f32(f32_outputs[0].data(), last_pos_start, VOCAB_SIZE);
    let deq_logits = last_position_logits_i32(
        i32_outputs[0].data(),
        last_pos_start,
        VOCAB_SIZE,
        scale_mult,
    );

    Ctx {
        tokenizer,
        model,
        run_args,
        scale,
        token_ids,
        seq_len,
        last_pos_start,
        quant_inputs,
        ref_logits,
        deq_logits,
    }
}

fn print_glossary(ctx: &Ctx) {
    qea::print_section("GLOSSARY");
    info!("  This analysis compares four views of the same Qwen2-0.5B forward pass:");
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

fn step1_quant_vs_tract(ctx: &Ctx) {
    print_step(1, "QUANT vs TRACT — overall quantization error");
    info!("  Question: how much does our integer engine diverge from f32?");
    info!("");
    print_comparison_metrics(&ctx.ref_logits, &ctx.deq_logits, "  ");
    info!("");
    print_top_k_side_by_side(
        "TRACT",
        &ctx.ref_logits,
        "QUANT",
        &ctx.deq_logits,
        &ctx.tokenizer,
        5,
    );
}

fn step2_per_node_drift(ctx: &Ctx) -> ShadowResult {
    print_step(2, "SHADOW vs QUANT — per-node drift");
    info!("  Each row compares the dequantized i32 output against the f64 shadow");
    info!("  after every graph node. Shows where rounding error accumulates.");
    info!("");

    let shadow_inputs = make_f64_inputs(&ctx.token_ids, ctx.seq_len);
    let trace = ctx
        .model
        .trace_with_shadow(&ctx.quant_inputs, &shadow_inputs, ctx.scale);
    trace.print_report();

    info!("");
    info!("{THIN}");
    info!("  Aggregated by operator type:");
    info!("{THIN}");
    trace.print_op_class_summary();

    let logits = extract_shadow_logits(&trace, "Shadow", ctx.last_pos_start, VOCAB_SIZE);
    ShadowResult { trace, logits }
}

fn step3_shadow_vs_tract(ctx: &Ctx, sr: &ShadowResult) {
    print_step(3, "SHADOW vs TRACT — shadow faithfulness");
    info!("  Question: does the f64 shadow (with quantized weights) agree with");
    info!("  the f32 Tract reference? High agreement means the shadow is valid.");
    info!("");
    print_comparison_metrics(&sr.logits, &ctx.ref_logits, "  ");
    info!("");
    print_top_k_side_by_side(
        "SHADOW",
        &sr.logits,
        "TRACT",
        &ctx.ref_logits,
        &ctx.tokenizer,
        5,
    );
}

fn step4_weight_quant_effect(ctx: &Ctx, sr: &ShadowResult) {
    print_step(4, "TRUE-F64 vs TRACT — weight quantization effect");
    info!("  TRUE-F64 uses original f32 weights + f64 arithmetic.");
    info!("  Comparing TRUE-F64 to TRACT should show near-zero error (just f64↔f32).");
    info!("  Comparing SHADOW to TRUE-F64 isolates the error from quantizing weights.");
    info!("");

    let original_constants = ctx
        .model
        .load_original_f64_constants(ONNX_PATH, &ctx.run_args);
    debug!(
        count = original_constants.len(),
        "Loaded original f64 constants from Tract"
    );

    let inputs = make_f64_inputs(&ctx.token_ids, ctx.seq_len);
    let true_shadow = ctx
        .model
        .trace_with_true_f64_shadow(&inputs, &original_constants, ctx.scale);
    let true_logits =
        extract_shadow_logits(&true_shadow, "True shadow", ctx.last_pos_start, VOCAB_SIZE);

    info!("{THIN}");
    info!("  [4a] TRUE-F64 vs TRACT  (expected: near-zero error)");
    info!("{THIN}");
    print_comparison_metrics(&true_logits, &ctx.ref_logits, "  ");
    info!("");
    print_top_k_side_by_side(
        "TRUE-F64",
        &true_logits,
        "TRACT",
        &ctx.ref_logits,
        &ctx.tokenizer,
        5,
    );

    info!("");
    info!("{THIN}");
    info!("  [4b] SHADOW vs TRUE-F64  (difference = weight quantization error)");
    info!("{THIN}");
    print_comparison_metrics(&sr.logits, &true_logits, "  ");

    info!("");
    info!("{THIN}");
    info!("  [4c] Per-node SHADOW vs TRUE-F64 (worst 20 nodes by CosSim)");
    info!("{THIN}");
    let mut diffs: Vec<(usize, String, f64, f64)> = sr
        .trace
        .f64_outputs
        .iter()
        .filter_map(|(node_idx, shadow_out)| {
            let true_out = true_shadow.f64_outputs.get(node_idx)?;
            if shadow_out.len() != true_out.len() || shadow_out.is_empty() {
                return None;
            }
            let cos = atlas_onnx_tracer::utils::metrics::cosine_similarity(
                shadow_out.data(),
                true_out.data(),
            );
            let rel_mse =
                atlas_onnx_tracer::utils::metrics::relative_mse(true_out.data(), shadow_out.data());
            if cos.is_nan() || cos > 0.9999 {
                return None;
            }
            let op_name = sr
                .trace
                .node_metrics
                .iter()
                .find(|m| m.idx == *node_idx)
                .map(|m| m.op_name.clone())
                .unwrap_or_else(|| "?".to_string());
            Some((*node_idx, op_name, cos, rel_mse))
        })
        .collect();
    diffs.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    if diffs.is_empty() {
        info!("  No nodes with CosSim < 0.9999 between SHADOW and TRUE-F64.");
    } else {
        info!(
            "  {:<6} {:<20} {:<12} {:<12}",
            "Node", "Op", "CosSim", "RelMSE"
        );
        info!("  {}", "-".repeat(54));
        for (idx, op, cos, relmse) in diffs.iter().take(20) {
            info!("  {:<6} {:<20} {:<12.6} {:<12.6}", idx, op, cos, relmse);
        }
        let mut by_idx = diffs.clone();
        by_idx.sort_by_key(|x| x.0);
        info!("");
        info!("  First 5 diverging nodes (sorted by node index):");
        for (idx, op, cos, relmse) in by_idx.iter().take(5) {
            info!("  {:<6} {:<20} {:<12.6} {:<12.6}", idx, op, cos, relmse);
        }
        let mut by_op: std::collections::BTreeMap<String, (usize, f64)> =
            std::collections::BTreeMap::new();
        for (_, op, cos, _) in &diffs {
            let e = by_op.entry(op.clone()).or_insert((0, 1.0f64));
            e.0 += 1;
            e.1 = e.1.min(*cos);
        }
        info!("");
        info!("  Count by op type (nodes with CosSim < 0.9999):");
        for (op, (cnt, worst)) in &by_op {
            info!("  {:<20} count={:<4} worst_cos={:.6}", op, cnt, worst);
        }
        info!("");
        info!("  Total diverging nodes: {}", diffs.len());
    }
}

fn step5_greedy_generation(ctx: &Ctx) {
    print_step(5, "TRACT greedy generation (sanity check)");
    info!("  Greedy-decoding 10 tokens with the f32 Tract model.");
    info!("");
    let generated = tract_greedy_generate(ONNX_PATH, &ctx.token_ids, 10, VOCAB_SIZE);
    let text = ctx
        .tokenizer
        .decode(&generated, false)
        .unwrap_or_else(|_| "<decode error>".to_string());
    info!("  \"{text}\"");
}

// ── Printing helpers ──────────────────────────────────────────────────────────

fn print_top_k_side_by_side(
    label_a: &str,
    logits_a: &[f64],
    label_b: &str,
    logits_b: &[f64],
    tokenizer: &Tokenizer,
    k: usize,
) {
    let top_a = top_k_entries(logits_a, k);
    let top_b = top_k_entries(logits_b, k);

    let decode = |id: usize| -> String {
        tokenizer
            .decode(&[id as u32], false)
            .unwrap_or_else(|_| "<unk>".to_string())
            .replace('\n', "\\n")
            .replace('\r', "\\r")
    };

    info!(
        "  {:<30}  |  {:<30}",
        format!("Top-{k} {label_a}"),
        format!("Top-{k} {label_b}"),
    );
    info!("  {:<30}  |  {:<30}", "-".repeat(30), "-".repeat(30));
    for i in 0..k {
        let (id_a, logit_a) = top_a[i];
        let (id_b, logit_b) = top_b[i];
        info!(
            "  {:<2}. {:>8.4}  {:<16}  |  {:<2}. {:>8.4}  {:<16}",
            i + 1,
            logit_a,
            format!("\"{}\"", decode(id_a)),
            i + 1,
            logit_b,
            format!("\"{}\"", decode(id_b)),
        );
    }
}
