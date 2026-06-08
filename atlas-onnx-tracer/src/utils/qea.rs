/// Shared helpers used across quantization-analysis and generation examples.
///
/// Factored out here to avoid copying boilerplate between
/// `quant_error_analysis`, `quant_error_analysis_qwen`, and `qwen_generate`.
use crate::{
    model::{Model, RunArgs, shadow_trace::ShadowTrace},
    tensor::Tensor,
    utils::metrics,
};
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

/// Print a numbered step header (`[n/N] title`).
pub fn print_step(n: usize, title: &str) {
    info!("\n");
    info!("{SEP}");
    info!("  [{n}/5] {title}");
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
    let mut indexed: Vec<(usize, f64)> =
        logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
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

/// Prepare Qwen-style quantized i32 inputs: `[input_ids, position_ids, attention_mask]`.
///
/// Position IDs are scaled (`i << scale`) to match RoPE Einsum expectations.
pub fn make_qwen_i32_inputs(token_ids: &[u32], seq_len: usize, scale: i32) -> Vec<Tensor<i32>> {
    let ids: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();
    // position_ids in fixed-point: i × 2^scale so Einsum(freq_q, pos_q)/2^scale = freq × i
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
