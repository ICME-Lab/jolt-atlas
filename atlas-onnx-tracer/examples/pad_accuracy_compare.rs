//! TEMPORARY analysis example: quantify the accuracy cost of power-of-two padding.
//!
//! Runs the same GPT-2 forward pass three ways and compares last-position logits
//! against the Tract f32 reference:
//!   1. QUANT unpadded  (i32 fixed-point, original dims)
//!   2. QUANT padded    (i32 fixed-point, dims padded to next power of two)
//!   3. SHADOW padded   (f64 arithmetic on the padded graph — isolates semantic
//!      padding error from integer rounding error)
//!
//! ```sh
//! cargo run -r -p atlas-onnx-tracer --example pad_accuracy_compare
//! ```

use atlas_onnx_tracer::{
    model::{Model, RunArgs},
    tensor::Tensor,
    utils::{metrics, quantize},
};
use tokenizers::Tokenizer;
use tracing::info;
use tracing_subscriber::EnvFilter;

const ONNX_PATH: &str = "atlas-onnx-tracer/models/gpt2/network.onnx";
const TOKENIZER_PATH: &str = "atlas-onnx-tracer/models/gpt2/tokenizer.json";
const VOCAB_SIZE: usize = 50257;
const SCALE: i32 = 12;
const SEP: &str = "═══════════════════════════════════════════════════════════";

fn main() {
    init_logging();
    let text = "The white man worked as a";
    let tokenizer = Tokenizer::from_file(TOKENIZER_PATH).expect("failed to load tokenizer");
    let encoding = tokenizer.encode(text, false).expect("tokenization failed");
    let token_ids = encoding.get_ids().to_vec();
    let seq_len = token_ids.len();
    info!("Input: \"{text}\"  ({seq_len} tokens; padded seq = {})", seq_len.next_power_of_two());

    let scale_mult = quantize::scale_to_multiplier(SCALE);

    // ── Tract f32 reference ────────────────────────────────────────────────
    let run_args_ref = run_args(seq_len, false);
    info!("Running TRACT f32 reference …");
    let f32_ids: Vec<f32> = token_ids.iter().map(|&id| id as f32).collect();
    let f32_mask: Vec<f32> = vec![1.0; seq_len];
    let f32_pos: Vec<f32> = (0..seq_len as i64).map(|i| i as f32).collect();
    let ref_out = Model::run_tract_forward(
        ONNX_PATH,
        &run_args_ref,
        &[
            ("input_ids", Tensor::new(Some(&f32_ids), &[1, seq_len]).unwrap()),
            ("attention_mask", Tensor::new(Some(&f32_mask), &[1, seq_len]).unwrap()),
            ("position_ids", Tensor::new(Some(&f32_pos), &[1, seq_len]).unwrap()),
        ],
    );
    let start = (seq_len - 1) * VOCAB_SIZE;
    let ref_logits: Vec<f64> = ref_out[0].data()[start..start + VOCAB_SIZE]
        .iter()
        .map(|&v| v as f64)
        .collect();

    // ── QUANT unpadded ─────────────────────────────────────────────────────
    info!("Running QUANT unpadded …");
    let model_u = Model::load(ONNX_PATH, &run_args(seq_len, false));
    let out_u = model_u.forward(&i32_inputs(&token_ids, seq_len));
    let logits_u: Vec<f64> = out_u[0].data()[start..start + VOCAB_SIZE]
        .iter()
        .map(|&v| v as f64 / scale_mult)
        .collect();

    // ── QUANT padded ───────────────────────────────────────────────────────
    info!("Running QUANT padded …");
    let model_p = Model::load(ONNX_PATH, &run_args(seq_len, true));
    let out_p = model_p.forward(&i32_inputs(&token_ids, seq_len));
    info!("padded model output dims (after crop): {:?}", out_p[0].dims());
    assert_eq!(
        out_p[0].dims(),
        &[1, seq_len, VOCAB_SIZE],
        "forward() should crop padded outputs back to original dims"
    );
    let logits_p: Vec<f64> = out_p[0].data()[start..start + VOCAB_SIZE]
        .iter()
        .map(|&v| v as f64 / scale_mult)
        .collect();

    // ── Report ─────────────────────────────────────────────────────────────
    report("QUANT unpadded vs TRACT", &ref_logits, &logits_u, &tokenizer);
    report("QUANT padded   vs TRACT", &ref_logits, &logits_p, &tokenizer);

    // ── SHADOW padded (f64 arithmetic, padded graph semantics) ────────────
    info!("Running SHADOW (f64) on the padded graph …");
    let f64_ids: Vec<f64> = token_ids.iter().map(|&id| id as f64).collect();
    let f64_pos: Vec<f64> = (0..seq_len).map(|i| i as f64).collect();
    let f64_mask: Vec<f64> = vec![1.0; seq_len];
    let shadow_inputs = vec![
        Tensor::new(Some(&f64_ids), &[1, seq_len]).unwrap(),
        Tensor::new(Some(&f64_pos), &[1, seq_len]).unwrap(),
        Tensor::new(Some(&f64_mask), &[1, seq_len]).unwrap(),
    ];
    let trace = model_p.trace_with_shadow(&i32_inputs(&token_ids, seq_len), &shadow_inputs, SCALE);
    let &last_idx = trace.f64_outputs.keys().next_back().unwrap();
    let shadow_tensor = &trace.f64_outputs[&last_idx];
    let sdims = shadow_tensor.dims().to_vec();
    info!("shadow final node dims (padded): {:?}", sdims);
    let vocab_p = *sdims.last().unwrap();
    let sstart = (seq_len - 1) * vocab_p;
    let shadow_logits: Vec<f64> = shadow_tensor.data()[sstart..sstart + VOCAB_SIZE].to_vec();
    report(
        "SHADOW padded (f64) vs TRACT — error surviving f64 = graph-semantic error",
        &ref_logits,
        &shadow_logits,
        &tokenizer,
    );
}

fn run_args(seq_len: usize, pad: bool) -> RunArgs {
    RunArgs::new([
        ("batch_size", 1),
        ("sequence_length", seq_len),
        ("past_sequence_length", 0),
    ])
    .set_scale(SCALE)
    .with_padding(pad)
}

fn i32_inputs(token_ids: &[u32], seq_len: usize) -> Vec<Tensor<i32>> {
    let ids: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();
    let pos: Vec<i32> = (0..seq_len as i32).collect();
    let mask: Vec<i32> = vec![1 << SCALE; seq_len];
    vec![
        Tensor::new(Some(&ids), &[1, seq_len]).unwrap(),
        Tensor::new(Some(&pos), &[1, seq_len]).unwrap(),
        Tensor::new(Some(&mask), &[1, seq_len]).unwrap(),
    ]
}

fn report(title: &str, reference: &[f64], candidate: &[f64], tokenizer: &Tokenizer) {
    info!("");
    info!("{SEP}");
    info!("  {title}");
    info!("{SEP}");
    info!("  Cosine similarity   : {:.6}", metrics::cosine_similarity(reference, candidate));
    info!("  RMSE                : {:.6}", metrics::rmse(reference, candidate));
    info!("  Max absolute error  : {:.6}", metrics::max_abs_error(reference, candidate));
    info!("  KL divergence       : {:.6}", metrics::kl_divergence_from_logits(reference, candidate));
    info!("  Top-1 agreement     : {:.2}%", metrics::top_k_agreement(reference, candidate, 1) * 100.0);
    info!("  Top-5 agreement     : {:.2}%", metrics::top_k_agreement(reference, candidate, 5) * 100.0);
    info!("  Top-10 agreement    : {:.2}%", metrics::top_k_agreement(reference, candidate, 10) * 100.0);
    info!("  Spearman rank corr  : {:.6}", metrics::spearman_rank_correlation(reference, candidate));

    let decode = |id: usize| tokenizer.decode(&[id as u32], false).unwrap_or_else(|_| "<unk>".into());
    let top_ref = top_k(reference, 5);
    let top_cand = top_k(candidate, 5);
    info!("  {:<34} | {:<34}", "Top-5 TRACT", "Top-5 candidate");
    for i in 0..5 {
        let (ia, la) = top_ref[i];
        let (ib, lb) = top_cand[i];
        info!(
            "  {:>8.4}  {:<22} | {:>8.4}  {:<22}",
            la,
            format!("\"{}\"", decode(ia)),
            lb,
            format!("\"{}\"", decode(ib)),
        );
    }
}

fn top_k(logits: &[f64], k: usize) -> Vec<(usize, f64)> {
    let mut indexed: Vec<(usize, f64)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    indexed
}

fn init_logging() {
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
    let filter = EnvFilter::new(format!("{},{}", base, tract_suppressions.join(",")));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .without_time()
        .init();
}
