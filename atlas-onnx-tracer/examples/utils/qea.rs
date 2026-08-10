/// Shared helpers used across quantization-analysis and generation examples.
///
/// Factored out here to avoid copying boilerplate between
/// `quant_error_analysis`, `quant_error_analysis_qwen`, and `qwen_generate`.
use atlas_onnx_tracer::{
    model::{Model, RunArgs, shadow_trace::ShadowTrace},
    tensor::Tensor,
    utils::metrics,
};
use tokenizers::Tokenizer;
use tracing::{debug, info};
use tracing_subscriber::EnvFilter;

/// Heavy section separator line.
pub const SEP: &str = "═══════════════════════════════════════════════════════════";
/// Thin sub-section separator line.
pub const THIN: &str = "───────────────────────────────────────────────────────────";

/// Per-node shadow trace together with extracted last-position logits.
pub struct ShadowResult {
    /// The full per-node shadow trace.
    pub trace: ShadowTrace,
    /// Last-position logits extracted from the final shadow node.
    pub logits: Vec<f64>,
}

/// Shared context produced by `setup` and threaded through every analysis step.
pub struct Ctx {
    /// Tokenizer for the prompt text and for decoding predicted tokens.
    pub tokenizer: Tokenizer,
    /// The loaded, quantized model.
    pub model: Model,
    /// The `RunArgs` the model was loaded with.
    pub run_args: RunArgs,
    /// The quantization scale (also `run_args.scale`).
    pub scale: i32,
    /// Path to the model's `network.onnx`.
    pub onnx_path: &'static str,
    /// Output vocabulary size.
    pub vocab_size: usize,
    /// Encoded prompt token ids.
    pub token_ids: Vec<u32>,
    /// Number of prompt tokens.
    pub seq_len: usize,
    /// Flat-array offset of the last token position's logits.
    pub last_pos_start: usize,
    /// Quantized i32 model inputs.
    pub quant_inputs: Vec<Tensor<i32>>,
    /// f64 shadow input tensors (unquantized real values), shared by every step that runs an f64
    /// trace, so it's built once instead of per-step.
    pub shadow_inputs: Vec<Tensor<f64>>,
    /// Last-position logits from the f32 Tract reference.
    pub ref_logits: Vec<f64>,
    /// Last-position logits from the dequantized i32 QUANT engine.
    pub deq_logits: Vec<f64>,
}

// ── Logging ──────────────────────────────────────────────────────────────────

/// Initialise `tracing-subscriber` silencing tract crate noise by default.
/// Override with `RUST_LOG`.
pub fn init_logging() {
    let tract_suppressions = [
        "tract_core=warn",
        "tract_data=warn",
        "tract_hir=warn",
        "tract_linalg=warn",
        "tract_nnef=warn",
        "tract_onnx=warn",
        "tract_onnx_opl=warn",
        "tract_extra=warn",
        "tract_pulse=warn",
        "tract_pulse_opl=warn",
    ];
    let base = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());
    let combined = format!("{},{}", base, tract_suppressions.join(","));
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(combined))
        .with_target(false)
        .without_time()
        .init();
}

// ── Section / step banners ────────────────────────────────────────────────────

/// Print a titled section banner.
pub fn print_section(title: &str) {
    info!("\n");
    info!("{SEP}");
    info!("  {title}");
    info!("{SEP}");
}

/// Print a step header (`[id/5] title`). `id` is usually a step number ("3"), but sub-steps of
/// the same logical step use a letter suffix ("2a", "2b") rather than incrementing the count.
pub fn print_step(id: &str, title: &str) {
    info!("\n");
    info!("{SEP}");
    info!("  [{id}/5] {title}");
    info!("{SEP}");
}

// ── Logit metrics ─────────────────────────────────────────────────────────────

/// Print a standard suite of comparison metrics between two logit vectors.
pub fn print_comparison_metrics(a: &[f64], b: &[f64], pad: &str) {
    info!(
        "{pad}Cosine similarity   : {:.6}",
        metrics::cosine_similarity(a, b)
    );
    info!("{pad}MSE                 : {:.6}", metrics::mse(a, b));
    info!("{pad}RMSE                : {:.6}", metrics::rmse(a, b));
    info!(
        "{pad}Max absolute error  : {:.6}",
        metrics::max_abs_error(a, b)
    );
    info!(
        "{pad}Mean absolute error : {:.6}",
        metrics::mean_abs_error(a, b)
    );
    info!(
        "{pad}Relative MSE        : {:.6}",
        metrics::relative_mse(a, b)
    );
    info!(
        "{pad}KL divergence       : {:.6}",
        metrics::kl_divergence_from_logits(a, b)
    );
    info!(
        "{pad}Top-1 agreement     : {:.2}%",
        metrics::top_k_agreement(a, b, 1) * 100.0
    );
    info!(
        "{pad}Top-5 agreement     : {:.2}%",
        metrics::top_k_agreement(a, b, 5) * 100.0
    );
    info!(
        "{pad}Top-10 agreement    : {:.2}%",
        metrics::top_k_agreement(a, b, 10) * 100.0
    );
    info!(
        "{pad}Spearman rank corr  : {:.6}",
        metrics::spearman_rank_correlation(a, b)
    );
    info!(
        "{pad}Pearson correlation : {:.6}",
        metrics::pearson_correlation(a, b)
    );
}

/// Return `(token_id, logit)` pairs sorted descending by logit.
pub fn top_k_entries(logits: &[f64], k: usize) -> Vec<(usize, f64)> {
    let mut indexed: Vec<(usize, f64)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    indexed
}

// ── Logit extraction ──────────────────────────────────────────────────────────

/// Extract last-position logits from an f32 output tensor, cast to f64.
pub fn last_position_logits_f32(data: &[f32], start: usize, vocab_size: usize) -> Vec<f64> {
    data[start..start + vocab_size]
        .iter()
        .map(|&v| v as f64)
        .collect()
}

/// Extract last-position logits from an i32 output tensor, dequantized to f64.
pub fn last_position_logits_i32(
    data: &[i32],
    start: usize,
    vocab_size: usize,
    scale_mult: f64,
) -> Vec<f64> {
    data[start..start + vocab_size]
        .iter()
        .map(|&v| v as f64 / scale_mult)
        .collect()
}

/// Extract the final-node f64 logits from a shadow trace.
pub fn extract_shadow_logits(
    shadow: &ShadowTrace,
    label: &str,
    last_pos_start: usize,
    vocab_size: usize,
) -> Vec<f64> {
    let &node_idx = shadow.f64_outputs.keys().next_back().unwrap();
    let tensor = &shadow.f64_outputs[&node_idx];
    let data = tensor.data();
    debug!(
        label,
        node_idx,
        shape = ?tensor.dims(),
        numel = data.len(),
        "Shadow output"
    );
    data[last_pos_start..last_pos_start + vocab_size].to_vec()
}

// ── Input construction ────────────────────────────────────────────────────────

/// Build the basic `RunArgs` (batch=1, no padding) with the given scale.
pub fn make_run_args(seq_len: usize, scale: i32) -> RunArgs {
    RunArgs::new([
        ("batch_size", 1),
        ("sequence_length", seq_len),
        ("past_sequence_length", 0),
    ])
    .set_scale(scale)
    .with_padding(false)
}

/// Prepare f64 shadow input tensors: `[input_ids, position_ids, attention_mask]`.
pub fn make_f64_inputs(token_ids: &[u32], seq_len: usize) -> Vec<Tensor<f64>> {
    let ids: Vec<f64> = token_ids.iter().map(|&id| id as f64).collect();
    let pos: Vec<f64> = (0..seq_len).map(|i| i as f64).collect();
    let mask: Vec<f64> = vec![1.0; seq_len];
    vec![
        Tensor::new(Some(&ids), &[1, seq_len]).unwrap(),
        Tensor::new(Some(&pos), &[1, seq_len]).unwrap(),
        Tensor::new(Some(&mask), &[1, seq_len]).unwrap(),
    ]
}

/// Prepare GPT-2-style quantized i32 inputs: `[input_ids, position_ids, attention_mask]`.
///
/// Position IDs are raw (unscaled) integers — GPT-2 uses them only as a Gather index into a
/// learned positional embedding table, not arithmetically (unlike Qwen's RoPE).
pub fn make_gpt2_i32_inputs(token_ids: &[u32], seq_len: usize, scale: i32) -> Vec<Tensor<i32>> {
    let ids: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();
    let pos: Vec<i32> = (0..seq_len as i32).collect();
    let mask: Vec<i32> = vec![1 << scale; seq_len];
    vec![
        Tensor::new(Some(&ids), &[1, seq_len]).unwrap(),
        Tensor::new(Some(&pos), &[1, seq_len]).unwrap(),
        Tensor::new(Some(&mask), &[1, seq_len]).unwrap(),
    ]
}

/// Prepare Qwen-style quantized i32 inputs: `[input_ids, position_ids, attention_mask]`.
///
/// Position IDs are scaled (`i << scale`) to match RoPE Einsum expectations.
pub fn make_qwen_i32_inputs(token_ids: &[u32], seq_len: usize, scale: i32) -> Vec<Tensor<i32>> {
    let ids: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();
    // position_ids in fixed-point: i × 2^scale so Einsum(freq_q, pos_q)/2^scale = freq × i
    let max_pos = (i32::MAX >> scale) as usize;
    assert!(
        seq_len.saturating_sub(1) <= max_pos,
        "make_qwen_i32_inputs: seq_len={seq_len} too large for scale={scale} (max position index {max_pos})"
    );
    let pos: Vec<i32> = (0..seq_len as i32).map(|i| i << scale).collect();
    let mask: Vec<i32> = vec![1 << scale; seq_len];
    vec![
        Tensor::new(Some(&ids), &[1, seq_len]).unwrap(),
        Tensor::new(Some(&pos), &[1, seq_len]).unwrap(),
        Tensor::new(Some(&mask), &[1, seq_len]).unwrap(),
    ]
}

// ── Tract helpers ─────────────────────────────────────────────────────────────

/// Run the Tract f32 reference forward pass for `[input_ids, attention_mask, position_ids]`.
pub fn run_tract_f32(
    onnx_path: &str,
    token_ids: &[u32],
    seq_len: usize,
    run_args: &RunArgs,
) -> Vec<Tensor<f32>> {
    let f32_ids: Vec<f32> = token_ids.iter().map(|&id| id as f32).collect();
    let f32_mask: Vec<f32> = vec![1.0; seq_len];
    let f32_pos: Vec<f32> = (0..seq_len as i64).map(|i| i as f32).collect();
    Model::run_tract_forward(
        onnx_path,
        run_args,
        &[
            (
                "input_ids",
                Tensor::new(Some(&f32_ids), &[1, seq_len]).unwrap(),
            ),
            (
                "attention_mask",
                Tensor::new(Some(&f32_mask), &[1, seq_len]).unwrap(),
            ),
            (
                "position_ids",
                Tensor::new(Some(&f32_pos), &[1, seq_len]).unwrap(),
            ),
        ],
    )
}

/// Greedy-decode `n_tokens` using the Tract f32 model.
pub fn tract_greedy_generate(
    onnx_path: &str,
    prompt_ids: &[u32],
    n_tokens: usize,
    vocab_size: usize,
) -> Vec<u32> {
    let mut ids = prompt_ids.to_vec();
    for _ in 0..n_tokens {
        let len = ids.len();
        let run_args = RunArgs::new([
            ("batch_size", 1),
            ("sequence_length", len),
            ("past_sequence_length", 0),
        ])
        .with_padding(false);

        let f32_ids: Vec<f32> = ids.iter().map(|&id| id as f32).collect();
        let f32_mask: Vec<f32> = vec![1.0; len];
        let f32_pos: Vec<f32> = (0..len as i64).map(|i| i as f32).collect();

        let outs = Model::run_tract_forward(
            onnx_path,
            &run_args,
            &[
                ("input_ids", Tensor::new(Some(&f32_ids), &[1, len]).unwrap()),
                (
                    "attention_mask",
                    Tensor::new(Some(&f32_mask), &[1, len]).unwrap(),
                ),
                (
                    "position_ids",
                    Tensor::new(Some(&f32_pos), &[1, len]).unwrap(),
                ),
            ],
        );
        let logits = outs[0].data();
        let start = (len - 1) * vocab_size;
        let next = logits[start..start + vocab_size]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as u32;
        ids.push(next);
    }
    ids
}

// ── Printing helpers ────────────────────────────────────────────────────────

/// Print top-k predicted tokens from two logit vectors side by side.
pub fn print_top_k_side_by_side(
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

// ── Analysis pipeline steps ───────────────────────────────────────────────────
//
// Shared by every quant-error-analysis example: identical regardless of model, since they only
// touch `Ctx` fields (including `ctx.onnx_path`/`ctx.vocab_size`), never a model-specific
// constant.

/// [1/5] Compare QUANT vs TRACT end-to-end.
pub fn step1_quant_vs_tract(ctx: &Ctx) {
    print_step("1", "QUANT vs TRACT — overall quantization error");
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

/// [2a/5] Run the f64 SHADOW alongside QUANT and report per-node drift, cumulative: each node's
/// QUANT inputs are QUANT's own (possibly already-drifted) previous outputs, so an early node's
/// error propagates and inflates every downstream node's apparent error too — see step 2b for the
/// isolated alternative.
pub fn step2a_per_node_drift_cumulative(ctx: &Ctx) -> ShadowResult {
    print_step("2a", "SHADOW vs QUANT — per-node drift (cumulative)");
    info!("  Each row compares the dequantized i32 output against the f64 shadow");
    info!("  after every graph node. Shows where rounding error accumulates.");
    info!("");

    let trace = ctx
        .model
        .trace_with_shadow(&ctx.quant_inputs, &ctx.shadow_inputs, ctx.scale);
    trace.print_report();

    info!("");
    info!("{THIN}");
    info!("  Aggregated by operator type:");
    info!("{THIN}");
    trace.print_op_class_summary();

    let logits = extract_shadow_logits(&trace, "Shadow", ctx.last_pos_start, ctx.vocab_size);
    ShadowResult { trace, logits }
}

/// [2b/5] Like step 2a, but re-quantizes each node's inputs from the SHADOW's own (ideal)
/// outputs instead of chaining QUANT's own history — isolates each node's own rounding error
/// from upstream propagated drift.
pub fn step2b_per_node_drift_isolated(ctx: &Ctx) {
    print_step(
        "2b",
        "SHADOW vs QUANT — per-node drift (isolated, no error propagation)",
    );
    info!("  Each node's QUANT inputs are re-quantized from SHADOW's own outputs, not chained");
    info!("  from QUANT's previous nodes — isolates each node's own rounding error.");
    info!("");

    let trace =
        ctx.model
            .trace_with_shadow_isolated(&ctx.quant_inputs, &ctx.shadow_inputs, ctx.scale);
    trace.print_report();

    info!("");
    info!("{THIN}");
    info!("  Aggregated by operator type:");
    info!("{THIN}");
    trace.print_op_class_summary();
}

/// [3/5] Compare SHADOW vs TRACT to verify the shadow is faithful.
pub fn step3_shadow_vs_tract(ctx: &Ctx, sr: &ShadowResult) {
    print_step("3", "SHADOW vs TRACT — shadow faithfulness");
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

/// [4/5] Run TRUE-F64 (original weights) and compare to isolate
/// weight-quantization error vs arithmetic rounding error.
pub fn step4_weight_quant_effect(ctx: &Ctx, sr: &ShadowResult) {
    print_step("4", "TRUE-F64 vs TRACT — weight quantization effect");
    info!("  TRUE-F64 uses original f32 weights + f64 arithmetic.");
    info!("  Comparing TRUE-F64 to TRACT should show near-zero error (just f64↔f32).");
    info!("  Comparing SHADOW to TRUE-F64 isolates the error from quantizing weights.");
    info!("");

    let original_constants = ctx
        .model
        .load_original_f64_constants(ctx.onnx_path, &ctx.run_args);
    debug!(
        count = original_constants.len(),
        "Loaded original f64 constants from Tract"
    );

    let true_shadow =
        ctx.model
            .trace_with_true_f64_shadow(&ctx.shadow_inputs, &original_constants, ctx.scale);
    let true_logits = extract_shadow_logits(
        &true_shadow,
        "True shadow",
        ctx.last_pos_start,
        ctx.vocab_size,
    );

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
}

/// [5/5] Greedy-decode 10 tokens with the Tract f32 model as a sanity check.
pub fn step5_greedy_generation(ctx: &Ctx) {
    print_step("5", "TRACT greedy generation (sanity check)");
    info!("  Greedy-decoding 10 tokens with the f32 Tract model.");
    info!("");
    let generated = tract_greedy_generate(ctx.onnx_path, &ctx.token_ids, 10, ctx.vocab_size);
    let text = ctx
        .tokenizer
        .decode(&generated, false)
        .unwrap_or_else(|_| "<decode error>".to_string());
    info!("  \"{text}\"");
}
