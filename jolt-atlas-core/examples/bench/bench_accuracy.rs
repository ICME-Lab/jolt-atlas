//! `tab:accuracy`: per model — cosine similarity, top-5 agreement, and perplexity (float vs
//! quantized) on the model's logits over a WikiText-2 sample. See
//! `../../paper-jolt-atlas2/BENCHMARKS.md` benchmark 7.
//!
//! BGE isn't built here: its ONNX export panics the tracer at forward-execution time (a
//! pre-existing bug, also hit by the stock `bge.rs` example).
//!
//! Methodology: all three metrics are computed at every interior position of a
//! `SAMPLE_TOKENS`-token WikiText-2 (raw, test split) prefix, not a single prompt continuation.
//! Top-5 agreement is float/quantized top-5 *token* overlap; perplexity is exp(mean NLL of the
//! true next token), computed once per side.
//!
//! `cargo run --release -p jolt-atlas-core --example bench_accuracy -- --model <gpt2|qwen>`
//! (Qwen needs `MODEL_SCALE = 14`, GPT-2 needs 12). Requires the WikiText-2 sample
//! (`scripts/download_wikitext.py`) and, for GPT-2 and Qwen, their tokenizers.
//! Writes `bench/results/tab_accuracy.json`.

mod harness;
mod models;
mod report;

use atlas_onnx_tracer::{
    model::{Model, RunArgs},
    tensor::Tensor,
    utils::{metrics, quantize},
};
use models::LoadedModel;
use report::{upsert_row, BenchModel, Meta, Row};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const SAMPLE_TOKENS: usize = 64;

const WIKITEXT_PATH: &str = "jolt-atlas-core/examples/bench/data/wikitext-2-test.txt";
const GPT2_TOKENIZER_PATH: &str = "atlas-onnx-tracer/models/gpt2/tokenizer.json";
const QWEN_TOKENIZER_PATH: &str = "atlas-onnx-tracer/models/qwen/tokenizer.json";

#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct AccuracyTableMeta {
    table: String,
    dataset: String,
    sample_tokens: usize,
}

impl Meta for AccuracyTableMeta {}

#[derive(Serialize, Deserialize)]
struct AccuracyRow {
    model: BenchModel,
    seq_len: usize,
    cosine_similarity: f64,
    top5_agreement: f64,
    perplexity_float: f64,
    perplexity_quantized: f64,
}

impl Row for AccuracyRow {
    fn name(&self) -> String {
        self.model.to_string()
    }
}

fn parse_model() -> BenchModel {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .map(|s| s.as_str())
    };
    let model = get("--model")
        .and_then(BenchModel::parse)
        .unwrap_or_else(|| panic!("--model <gpt2|qwen> is required (got {:?})", get("--model")));
    assert_ne!(
        model,
        BenchModel::Bge,
        "BGE isn't built here — its ONNX export currently panics the tracer at forward-execution \
         time (see this file's doc comment). Use --model gpt2 or --model qwen."
    );
    model
}

fn wikitext_sample() -> String {
    std::fs::read_to_string(WIKITEXT_PATH).unwrap_or_else(|e| {
        panic!("failed to read {WIKITEXT_PATH}: {e} — run scripts/download_wikitext.py first")
    })
}

/// Up to `n` token ids of `text` under the tokenizer at `tokenizer_path`.
fn tokenize_sample(tokenizer_path: &str, text: &str, n: usize) -> Vec<i32> {
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
        .unwrap_or_else(|e| panic!("failed to load tokenizer {tokenizer_path}: {e}"));
    let encoding = tokenizer.encode(text, false).expect("tokenization failed");
    let ids: Vec<i32> = encoding
        .get_ids()
        .iter()
        .take(n)
        .map(|&id| id as i32)
        .collect();
    assert!(
        ids.len() >= 2,
        "need at least 2 tokens (1 interior position) to measure top-5 agreement/perplexity, \
         got {} from {tokenizer_path}",
        ids.len()
    );
    ids
}

fn main() {
    let model = parse_model();
    let text = wikitext_sample();

    let (label, onnx_path, tokenizer_path) = match model {
        BenchModel::Gpt2 => (
            "gpt2",
            "atlas-onnx-tracer/models/gpt2/network.onnx",
            GPT2_TOKENIZER_PATH,
        ),
        BenchModel::Qwen => (
            "qwen",
            "atlas-onnx-tracer/models/qwen/network.onnx",
            QWEN_TOKENIZER_PATH,
        ),
        BenchModel::Bge => unreachable!("rejected in parse_model"),
    };
    let input_ids_data = tokenize_sample(tokenizer_path, &text, SAMPLE_TOKENS);
    let row = run_logits_model(label, onnx_path, input_ids_data);

    println!(
        "[{model}] seq_len={} cosine_similarity={:.6} top5_agreement={:.4} \
         perplexity(float)={:.4} perplexity(quantized)={:.4}",
        row.seq_len,
        row.cosine_similarity,
        row.top5_agreement,
        row.perplexity_float,
        row.perplexity_quantized,
    );

    let out_path = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/bench/results"
    ))
    .join("tab_accuracy.json");
    let meta = AccuracyTableMeta {
        table: "tab:accuracy".to_string(),
        dataset: "wikitext-2-raw-v1 (test split)".to_string(),
        sample_tokens: SAMPLE_TOKENS,
    };
    upsert_row(&out_path, meta, row);
}

/// GPT-2/Qwen: causal LM, compare logits at every interior position.
fn run_logits_model(label: &'static str, onnx_path: &str, input_ids_data: Vec<i32>) -> AccuracyRow {
    let seq_len = input_ids_data.len();
    let loaded = match label {
        "gpt2" => models::load_gpt2(input_ids_data.clone()),
        "qwen" => models::load_qwen(input_ids_data.clone()),
        _ => unreachable!(),
    };
    let LoadedModel {
        shared,
        inputs,
        scale,
        ..
    } = loaded;
    let vocab_size = match label {
        "gpt2" => models::GPT2_VOCAB_SIZE as usize,
        "qwen" => models::QWEN_VOCAB_SIZE as usize,
        _ => unreachable!(),
    };

    // ── Quantized ────────────────────────────────────────────────────────
    let scale_mult = quantize::scale_to_multiplier(scale as i32);
    let quant_out = shared.model().forward(&inputs);
    let quant_logits: Vec<f64> = quant_out[0]
        .data()
        .iter()
        .map(|&v| v as f64 / scale_mult)
        .collect();

    // ── Float reference (Tract) ─────────────────────────────────────────
    let run_args_ref = RunArgs::new([
        ("batch_size", 1),
        ("sequence_length", seq_len),
        ("past_sequence_length", 0),
    ]);
    let f32_ids: Vec<f32> = input_ids_data.iter().map(|&id| id as f32).collect();
    let f32_mask: Vec<f32> = vec![1.0; seq_len];
    let f32_pos: Vec<f32> = (0..seq_len as i64).map(|i| i as f32).collect();
    let ref_out = Model::run_tract_forward(
        onnx_path,
        &run_args_ref,
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
    );
    let ref_logits: Vec<f64> = ref_out[0].data().iter().map(|&v| v as f64).collect();

    assert_eq!(ref_logits.len(), seq_len * vocab_size);
    assert_eq!(quant_logits.len(), seq_len * vocab_size);

    let cosine_similarity = metrics::cosine_similarity(&ref_logits, &quant_logits);

    // position i predicts token[i+1]
    let mut top5_sum = 0.0;
    let mut nll_float_sum = 0.0;
    let mut nll_quant_sum = 0.0;
    let positions = seq_len - 1;
    for i in 0..positions {
        let r = &ref_logits[i * vocab_size..(i + 1) * vocab_size];
        let q = &quant_logits[i * vocab_size..(i + 1) * vocab_size];
        top5_sum += metrics::top_k_agreement(r, q, 5);
        let next_tok = input_ids_data[i + 1] as usize;
        nll_float_sum += neg_log_likelihood(r, next_tok);
        nll_quant_sum += neg_log_likelihood(q, next_tok);
    }

    AccuracyRow {
        model: BenchModel::parse(label).unwrap(),
        seq_len,
        cosine_similarity,
        top5_agreement: top5_sum / positions as f64,
        perplexity_float: (nll_float_sum / positions as f64).exp(),
        perplexity_quantized: (nll_quant_sum / positions as f64).exp(),
    }
}

/// -log(softmax(logits)[token]), the per-position term perplexity averages over.
fn neg_log_likelihood(logits: &[f64], token: usize) -> f64 {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = logits.iter().map(|&x| (x - max).exp()).sum();
    let log_prob = (logits[token] - max) - sum_exp.ln();
    -log_prob
}
