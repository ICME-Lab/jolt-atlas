//! `tab:models`: per-model params, seq len, scale `s`, `max_num_vars()`, node count. See
//! `../../paper-jolt-atlas2/BENCHMARKS.md`'s "Also needed: model metadata" section.
//!
//! Structural only (`Model::load`, no `forward`/proving) — this is also why BGE is included even
//! though `tab:accuracy` excludes it: BGE's tracer panic only happens during forward execution.
//!
//! `params` (sum of `Constant` node sizes) reads well above nominal HF param counts — GPT-2:
//! 286M here vs. 125M nominal — because committed tensors are padded to power-of-two dims and
//! GPT-2's tied `wte`/`lm_head` embedding is stored as two separate nodes. `node_count` is this
//! crate's own decomposed graph node count, not the ONNX/HF layer count. Both need a caveat next
//! to the table in the paper so they aren't read as directly comparable to published figures.
//!
//! `cargo run --release -p jolt-atlas-core --example bench_models -- --model <gpt2|bge|qwen>`
//! (Qwen needs `MODEL_SCALE = 14`, gpt2/bge need 12). Writes `bench/results/tab_models.json`.

mod harness;
mod models;
mod report;

use atlas_onnx_tracer::ops::Operator;
use report::{upsert_row, BenchModel, Meta, Row};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const DEFAULT_SEQ_LEN: usize = 64;

#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct ModelsTableMeta {
    table: String,
    seq_len: usize,
}

impl Meta for ModelsTableMeta {}

#[derive(Serialize, Deserialize)]
struct ModelsRow {
    model: BenchModel,
    scale: usize,
    params: usize,
    node_count: usize,
    max_num_vars: usize,
}

impl Row for ModelsRow {
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
    get("--model")
        .and_then(BenchModel::parse)
        .unwrap_or_else(|| {
            panic!(
                "--model <gpt2|bge|qwen> is required (got {:?})",
                get("--model")
            )
        })
}

fn parse_seq_len() -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == "--seq-len")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.parse().expect("--seq-len must be a positive integer"))
        .unwrap_or(DEFAULT_SEQ_LEN)
}

fn main() {
    let model = parse_model();
    let seq_len = parse_seq_len();

    let loaded = match model {
        BenchModel::Gpt2 => models::load_gpt2(models::random_input_ids(
            seq_len,
            models::GPT2_VOCAB_SIZE,
            42,
        )),
        BenchModel::Bge => models::load_bge(models::random_input_ids(
            seq_len,
            models::BGE_VOCAB_SIZE,
            43,
        )),
        BenchModel::Qwen => models::load_qwen(models::random_input_ids(
            seq_len,
            models::QWEN_VOCAB_SIZE,
            44,
        )),
    };

    let params: usize = loaded
        .shared
        .model()
        .nodes()
        .values()
        .map(|node| match &node.operator {
            Operator::Constant(c) => c.0.len(),
            _ => 0,
        })
        .sum();
    let node_count = loaded.shared.model().nodes().len();
    let max_num_vars = loaded.shared.model().max_num_vars();

    let row = ModelsRow {
        model,
        scale: loaded.scale,
        params,
        node_count,
        max_num_vars,
    };

    println!(
        "[{model}] scale={} params={} node_count={} max_num_vars={}",
        row.scale, row.params, row.node_count, row.max_num_vars,
    );

    let out_path = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/bench/results"
    ))
    .join("tab_models.json");
    let meta = ModelsTableMeta {
        table: "tab:models".to_string(),
        seq_len,
    };
    upsert_row(&out_path, meta, row);
}
