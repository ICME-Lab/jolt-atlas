use atlas_onnx_tracer::{
    model::{Model, RunArgs, shadow_trace::ShadowTrace},
    tensor::Tensor,
    utils::{metrics, quantize},
};
use tokenizers::Tokenizer;
use tracing::{debug, info};
use tracing_subscriber::EnvFilter;

const ONNX_PATH: &str = "atlas-onnx-tracer/models/gpt2/network.onnx";
const TOKENIZER_PATH: &str = "atlas-onnx-tracer/models/gpt2/tokenizer.json";
const VOCAB_SIZE: usize = 50257;
const SEP: &str = "═══════════════════════════════════════════════════════════";
const THIN: &str = "───────────────────────────────────────────────────────────";
#[cfg(feature = "fused-ops")]
const SCALE: i32 = 12;
#[cfg(not(feature = "fused-ops"))]
const SCALE: i32 = 7;

/// Quantization error analysis for GPT-2.
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
/// pip install 'optimum[exporters]' 'optimum[onnxruntime]'
/// python -m optimum.exporters.onnx --model gpt2 atlas-onnx-tracer/models/gpt2
/// ```
///
/// # Usage
///
/// Default (info-level output, tract logs silenced):
/// ```sh
/// cargo run -r -p atlas-onnx-tracer --features fused-ops --example quant_error_analysis
/// ```
///
/// Show debug output (shapes, token details):
/// ```sh
/// RUST_LOG=debug cargo run -r -p atlas-onnx-tracer --features fused-ops --example quant_error_analysis
/// ```
///
/// Show everything *including* tract internals:
/// ```sh
/// RUST_LOG=trace cargo run -r -p atlas-onnx-tracer --features fused-ops --example quant_error_analysis
/// ```
fn main() {
    init_logging();
    let ctx = setup("The white man worked as a");

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

/// Shared context produced by [`setup`] and threaded through every analysis step.
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

/// Per-node shadow trace together with the extracted last-position logits.
#[allow(dead_code)]
struct ShadowResult {
    trace: ShadowTrace,
    logits: Vec<f64>,
}

/// Load tokenizer, encode prompt, run Tract f32 + quantized i32 forward passes,
/// and extract last-position logits from both.
fn setup(text: &str) -> Ctx {
    let tokenizer = Tokenizer::from_file(TOKENIZER_PATH)
        .expect("failed to load tokenizer.json – see doc-comment for setup");

    info!("Input text : \"{text}\"");
    let encoding = tokenizer.encode(text, false).expect("tokenization failed");
    let token_ids = encoding.get_ids().to_vec();
    let seq_len = token_ids.len();
    debug!(tokens = ?encoding.get_tokens(), "Tokens");
    debug!(?token_ids, "Token IDs");
    debug!(seq_len, "Sequence length");

    let run_args = make_run_args(seq_len);
    let scale = run_args.scale;
    let scale_mult = quantize::scale_to_multiplier(scale);

    info!("Running TRACT (f32 reference) …");
    let f32_outputs = run_tract_f32(&token_ids, seq_len, &run_args);
    debug!(shape = ?f32_outputs[0].dims(), "TRACT output shape");

    info!("Running QUANT (i32 fixed-point, scale=2^{scale}) …");
    let model = Model::load(ONNX_PATH, &run_args);
    let quant_inputs = make_i32_inputs(&token_ids, seq_len, scale);
    let i32_outputs = model.forward(&quant_inputs);
    debug!(shape = ?i32_outputs[0].dims(), "QUANT output shape");

    let last_pos_start = (seq_len - 1) * VOCAB_SIZE;
    let ref_logits = last_position_logits_f32(f32_outputs[0].data(), last_pos_start);
    let deq_logits = last_position_logits_i32(i32_outputs[0].data(), last_pos_start, scale_mult);

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

/// Print the glossary / legend explaining each model.
fn print_glossary(ctx: &Ctx) {
    print_section("GLOSSARY");
    info!("  This analysis compares four views of the same GPT-2 forward pass:");
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

/// [1/5] Compare QUANT vs TRACT end-to-end.
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

/// [2/5] Run the f64 SHADOW alongside QUANT and report per-node drift.
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

    let logits = extract_shadow_logits(&trace, "Shadow", ctx.last_pos_start);
    ShadowResult { trace, logits }
}

/// [3/5] Compare SHADOW vs TRACT to verify the shadow is faithful.
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

/// [4/5] Run TRUE-F64 (original weights) and compare to isolate
/// weight-quantization error vs arithmetic rounding error.
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
    let true_logits = extract_shadow_logits(&true_shadow, "True shadow", ctx.last_pos_start);

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
fn step5_greedy_generation(ctx: &Ctx) {
    print_step(5, "TRACT greedy generation (sanity check)");
    info!("  Greedy-decoding 10 tokens with the f32 Tract model.");
    info!("");
    let generated = greedy_generate(&ctx.token_ids, 10);
    let text = ctx
        .tokenizer
        .decode(&generated, false)
        .unwrap_or_else(|_| "<decode error>".to_string());
    info!("  \"{text}\"");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Helper functions
// ═══════════════════════════════════════════════════════════════════════════

/// Build the standard `RunArgs` with scale=12 and no padding.
fn make_run_args(seq_len: usize) -> RunArgs {
    let mut args = RunArgs::new([
        ("batch_size", 1),
        ("sequence_length", seq_len),
        ("past_sequence_length", 0),
    ])
    .set_scale(SCALE)
    .with_padding(false);
    #[cfg(not(feature = "fused-ops"))]
    {
        args = args.with_pre_rebase_nonlinear(true);
    }
    args
}

/// Prepare quantized i32 input tensors: `[input_ids, position_ids, attention_mask]`.
fn make_i32_inputs(token_ids: &[u32], seq_len: usize, scale: i32) -> Vec<Tensor<i32>> {
    let ids: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();
    let pos: Vec<i32> = (0..seq_len as i32).collect();
    let mask: Vec<i32> = vec![1 << scale; seq_len];
    vec![
        Tensor::new(Some(&ids), &[1, seq_len]).unwrap(),
        Tensor::new(Some(&pos), &[1, seq_len]).unwrap(),
        Tensor::new(Some(&mask), &[1, seq_len]).unwrap(),
    ]
}

/// Prepare f64 shadow input tensors: `[input_ids, position_ids, attention_mask]`.
fn make_f64_inputs(token_ids: &[u32], seq_len: usize) -> Vec<Tensor<f64>> {
    let ids: Vec<f64> = token_ids.iter().map(|&id| id as f64).collect();
    let pos: Vec<f64> = (0..seq_len).map(|i| i as f64).collect();
    let mask: Vec<f64> = vec![1.0; seq_len];
    vec![
        Tensor::new(Some(&ids), &[1, seq_len]).unwrap(),
        Tensor::new(Some(&pos), &[1, seq_len]).unwrap(),
        Tensor::new(Some(&mask), &[1, seq_len]).unwrap(),
    ]
}

/// Run the Tract f32 reference forward pass.
fn run_tract_f32(token_ids: &[u32], seq_len: usize, run_args: &RunArgs) -> Vec<Tensor<f32>> {
    let f32_ids: Vec<f32> = token_ids.iter().map(|&id| id as f32).collect();
    let f32_mask: Vec<f32> = vec![1.0; seq_len];
    let f32_pos: Vec<f32> = (0..seq_len as i64).map(|i| i as f32).collect();
    Model::run_tract_forward(
        ONNX_PATH,
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

/// Extract last-position logits from f32 output, cast to f64.
fn last_position_logits_f32(data: &[f32], start: usize) -> Vec<f64> {
    data[start..start + VOCAB_SIZE]
        .iter()
        .map(|&v| v as f64)
        .collect()
}

/// Extract last-position logits from i32 output, dequantized to f64.
fn last_position_logits_i32(data: &[i32], start: usize, scale_mult: f64) -> Vec<f64> {
    data[start..start + VOCAB_SIZE]
        .iter()
        .map(|&v| v as f64 / scale_mult)
        .collect()
}

/// Extract the final-node f64 logits from a shadow trace.
fn extract_shadow_logits(shadow: &ShadowTrace, label: &str, last_pos_start: usize) -> Vec<f64> {
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
    data[last_pos_start..last_pos_start + VOCAB_SIZE].to_vec()
}

/// Greedy-decode `n_tokens` using the Tract f32 model.
fn greedy_generate(prompt_ids: &[u32], n_tokens: usize) -> Vec<u32> {
    let mut ids: Vec<u32> = prompt_ids.to_vec();
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
            ONNX_PATH,
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
        let start = (len - 1) * VOCAB_SIZE;
        let next = logits[start..start + VOCAB_SIZE]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as u32;
        ids.push(next);
    }
    ids
}

// ── Printing helpers ─────────────────────────────────────────────────────

/// Initialise `tracing-subscriber` with an `EnvFilter` that silences tract
/// crate noise by default.  Override with `RUST_LOG`.
///
/// Default filter: `info,tract_core=warn,tract_hir=warn,tract_onnx=warn,tract_linalg=warn`
fn init_logging() {
    // Always suppress tract crate noise, then layer on RUST_LOG (or default to info).
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
    let filter = EnvFilter::new(combined);

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .without_time()
        .init();
}

/// Print a section header banner.
fn print_section(title: &str) {
    info!("\n");
    info!("{SEP}");
    info!("  {title}");
    info!("{SEP}");
}

/// Print a numbered step header.
fn print_step(n: usize, title: &str) {
    info!("\n");
    info!("{SEP}");
    info!("  [{n}/5] {title}");
    info!("{SEP}");
}

/// Print the comparison metrics between two logit vectors.
fn print_comparison_metrics(a: &[f64], b: &[f64], pad: &str) {
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

/// Print top-k predicted tokens from *two* logit vectors side by side.
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
    };

    info!(
        "  {:<30}  |  {:<30}",
        format!("Top-{k} {label_a}"),
        format!("Top-{k} {label_b}"),
    );
    info!("  {:<30}  |  {:<30}", "-".repeat(30), "-".repeat(30),);
    for i in 0..k {
        let (id_a, logit_a) = top_a[i];
        let (id_b, logit_b) = top_b[i];
        let word_a = decode(id_a);
        let word_b = decode(id_b);
        info!(
            "  {:<2}. {:>8.4}  {:<16}  |  {:<2}. {:>8.4}  {:<16}",
            i + 1,
            logit_a,
            format!("\"{word_a}\""),
            i + 1,
            logit_b,
            format!("\"{word_b}\""),
        );
    }
}

/// Return vec of (token_id, logit) sorted descending by logit.
fn top_k_entries(logits: &[f64], k: usize) -> Vec<(usize, f64)> {
    let mut indexed: Vec<(usize, f64)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    indexed
}
