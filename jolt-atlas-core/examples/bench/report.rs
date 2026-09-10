//! Shared engine for paper-table JSON output: one file per `(table, model)`, written under
//! `examples/bench/results/`. See `../../../../paper-jolt-atlas2/BENCHMARKS.md`.
//!
//! Each `bench_*.rs` binary defines its own `*TableMeta` (whatever's constant across every row)
//! and `*Row` type, implementing [`Meta`]/[`Row`] here. [`upsert_row`] builds a table's file up
//! incrementally: each invocation measures one row and inserts or replaces it (by [`Row::name`])
//! rather than overwriting the whole file. It asserts the incoming meta matches what's already
//! on disk, so stale table-level fields (e.g. a different `--seq-len` without clearing the file)
//! are a hard error, not a quiet data-integrity bug.
#![allow(dead_code)]

use super::harness::RepeatedTiming;
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, path::Path, time::Duration};

/// Named `BenchPcs`, this is only a report label.
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum BenchPcs {
    HyperKzg,
    Dory,
}

impl BenchPcs {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "hyperkzg" => Some(BenchPcs::HyperKzg),
            "dory" => Some(BenchPcs::Dory),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            BenchPcs::HyperKzg => "hyperkzg",
            BenchPcs::Dory => "dory",
        }
    }
}

impl std::fmt::Display for BenchPcs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Named `BenchModel`.
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum BenchModel {
    Gpt2,
    Bge,
    Qwen,
}

impl BenchModel {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "gpt2" => Some(BenchModel::Gpt2),
            "bge" => Some(BenchModel::Bge),
            "qwen" => Some(BenchModel::Qwen),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            BenchModel::Gpt2 => "gpt2",
            BenchModel::Bge => "bge",
            BenchModel::Qwen => "qwen",
        }
    }
}

impl std::fmt::Display for BenchModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Associates a `CommitmentScheme` type with its [`BenchPcs`] label, so PCS-generic bench code
/// can derive the label from `PCS::BENCH_PCS` instead of threading it separately.
pub trait PcsLabel {
    const BENCH_PCS: BenchPcs;
}

impl PcsLabel for jolt_atlas_core::onnx_proof::HyperKZG<jolt_atlas_core::onnx_proof::Bn254> {
    const BENCH_PCS: BenchPcs = BenchPcs::HyperKzg;
}

impl PcsLabel for jolt_atlas_core::onnx_proof::DoryScheme {
    const BENCH_PCS: BenchPcs = BenchPcs::Dory;
}

#[derive(Serialize, Deserialize)]
pub struct TimingStats {
    pub min_s: f64,
    pub median_s: f64,
    pub mean_s: f64,
    pub max_s: f64,
    pub samples_s: Vec<f64>,
}

impl From<&RepeatedTiming> for TimingStats {
    fn from(t: &RepeatedTiming) -> Self {
        TimingStats {
            min_s: t.min().as_secs_f64(),
            median_s: t.median().as_secs_f64(),
            mean_s: t.mean().as_secs_f64(),
            max_s: t.max().as_secs_f64(),
            samples_s: t.samples.iter().map(Duration::as_secs_f64).collect(),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct PhasePercent {
    pub mean_s: f64,
    pub percent_of_prove: f64,
}

impl PhasePercent {
    pub fn new(mean_s: f64, prove_total_mean_s: f64) -> Self {
        PhasePercent {
            mean_s,
            percent_of_prove: 100.0 * mean_s / prove_total_mean_s,
        }
    }
}

/// One row of `tab:committed-polys` Table A (also reused for Table B, `tab:clamp-contributions`).
/// Dense and one-hot polys have different cost drivers (padded size vs. nonzero entry count), so
/// one-hot gets a size column and dense just a count.
#[derive(Serialize, Deserialize)]
pub struct CommittedPolyRow {
    pub category: String,
    pub operators: Vec<String>,
    pub nodes: usize,
    pub dense_polys: usize,
    pub one_hot_polys: usize,
    pub one_hot_entries: usize,
}

/// A table's meta: whatever's constant across every row in a `(table, model)` file. Needs an
/// explicit (always-empty) `impl Meta for MyTableMeta {}` so a missing derive errors at the
/// definition site, not deep inside [`upsert_row`].
pub trait Meta: Serialize + for<'de> Deserialize<'de> + PartialEq + Debug {}

/// `name` is the row's matching key within its file (e.g. "dory", "seq64") — derived from
/// columns the row already carries, never written to disk itself.
pub trait Row: Serialize + for<'de> Deserialize<'de> {
    fn name(&self) -> String;
}

/// The on-disk shape of a table file: its meta plus accumulated rows.
#[derive(Serialize, Deserialize)]
struct TableFile<M, R> {
    #[serde(flatten)]
    meta: M,
    rows: Vec<R>,
}

/// Insert or replace one row (by [`Row::name`]) in a table's JSON file, creating it if needed.
/// If the file exists, `meta` must equal what's already on disk (see module docs).
pub fn upsert_row<M: Meta, R: Row>(path: &Path, meta: M, row: R) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("failed to create bench results dir");
    }

    let mut rows: Vec<R> = Vec::new();
    if let Ok(existing_json) = std::fs::read_to_string(path) {
        let existing: TableFile<M, R> = serde_json::from_str(&existing_json)
            .expect("existing table JSON has a different shape than this bench produces");
        assert_eq!(
            existing.meta,
            meta,
            "{}'s table-level metadata changed since it was last written (existing: {:?}, now: \
             {:?}) — delete the file first if this is intentional (e.g. you changed --seq-len)",
            path.display(),
            existing.meta,
            meta,
        );
        rows = existing.rows;
    }

    let row_name = row.name();
    match rows.iter().position(|r| r.name() == row_name) {
        Some(i) => rows[i] = row,
        None => rows.push(row),
    }

    let table = TableFile { meta, rows };
    let json = serde_json::to_string_pretty(&table).expect("failed to serialize table JSON");
    std::fs::write(path, json).expect("failed to write bench JSON");
    println!("wrote {} (row {row_name:?})", path.display());
}
