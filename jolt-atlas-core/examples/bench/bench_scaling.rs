//! `tab:scaling`: GPT-2, Dory (fixed, not swept) — largest committed poly / prove / verify /
//! peak memory / prefill throughput, swept over sequence length. See
//! `../../paper-jolt-atlas2/BENCHMARKS.md` benchmark 6.
//!
//! One `--seq-len` per invocation (peak memory is whole-process). Report seq 16/32/64/128, +256
//! if it fits — 64 is the load-bearing row (DeepProve's shortest reported config); an OOM at 256
//! is itself the answer (the system's honest ceiling), not a bug to chase.
//!
//! `cargo run --release -p jolt-atlas-core --example bench_scaling -- --seq-len <n>` (requires
//! `MODEL_SCALE == 12`). Writes `bench/results/tab_scaling_gpt2.json`, one row per `--seq-len`.

mod harness;
mod models;
mod report;

use ark_serialize::{CanonicalSerialize, Compress};
use common::utils::logging::setup_tracing;
use jolt_atlas_core::onnx_proof::{
    AtlasProverPreprocessing, AtlasVerifierPreprocessing, Blake2bTranscript, DoryScheme, Fr,
    ONNXProof,
};
use models::LoadedModel;
use report::{upsert_row, BenchModel, BenchPcs, Meta, Row, TimingStats};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const WARMUP_ITERS: usize = 1;
const MEASURED_ITERS: usize = 3;
const DEFAULT_SEQ_LEN: usize = 16;

#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct ScalingTableMeta {
    table: String,
    model: BenchModel,
    pcs: BenchPcs,
    scale: usize,
    warmup_iters: usize,
    measured_iters: usize,
}

impl Meta for ScalingTableMeta {}

#[derive(Serialize, Deserialize)]
struct ScalingRow {
    seq_len: usize,
    max_num_vars: usize,
    prove: TimingStats,
    verify: TimingStats,
    peak_rss_bytes: Option<u64>,
    proof_size_bytes: usize,
    /// Prefill throughput (one forward pass), not autoregressive generation throughput.
    throughput_tok_per_min: f64,
}

impl Row for ScalingRow {
    fn name(&self) -> String {
        format!("seq{}", self.seq_len)
    }
}

fn parse_seq_len() -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == "--seq-len")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.parse().expect("--seq-len must be a number"))
        .unwrap_or(DEFAULT_SEQ_LEN)
}

fn main() {
    let (_guard, _tracing_enabled) = setup_tracing("tab:scaling bench");
    let seq_len = parse_seq_len();

    let loaded = models::load_gpt2(models::random_input_ids(
        seq_len,
        models::GPT2_VOCAB_SIZE,
        42,
    ));
    let (scale, max_num_vars) = (loaded.scale, loaded.max_num_vars);
    println!("[gpt2/dory] seq_len={seq_len} scale={scale} max_num_vars={max_num_vars}");

    let LoadedModel { shared, inputs, .. } = loaded;
    let prover_pp = AtlasProverPreprocessing::<Fr, DoryScheme>::new(shared);
    let verifier_pp = AtlasVerifierPreprocessing::<Fr, DoryScheme>::from(&prover_pp);
    joltworks::utils::thread::wait_for_background_drops();

    // See `bench_e2e.rs::run_e2e` for why background drops are settled after every iteration.
    let mut last_proof_io = None;
    let prove = harness::time_repeated_with_settle(
        WARMUP_ITERS,
        MEASURED_ITERS,
        || {
            let (proof, io, _debug) =
                ONNXProof::<Fr, Blake2bTranscript, DoryScheme>::prove(&prover_pp, &inputs);
            last_proof_io = Some((proof, io));
        },
        joltworks::utils::thread::wait_for_background_drops,
    );
    let (proof, io) = last_proof_io.expect("at least one measured iteration");
    let proof_size_bytes = proof.serialized_size(Compress::Yes);

    let verify = harness::time_repeated_with_settle(
        WARMUP_ITERS,
        MEASURED_ITERS,
        || {
            proof.verify(&verifier_pp, &io, None).unwrap();
        },
        joltworks::utils::thread::wait_for_background_drops,
    );

    let peak_rss_bytes = harness::peak_rss_bytes();
    let throughput_tok_per_min = seq_len as f64 * 60.0 / prove.mean().as_secs_f64();

    println!(
        "[gpt2/dory] seq_len={seq_len} prove(mean)={:.2?} verify(mean)={:.2?} \
         size={proof_size_bytes}B peak_rss={peak_rss_bytes:?} throughput={throughput_tok_per_min:.1}tok/min",
        prove.mean(),
        verify.mean(),
    );

    let row = ScalingRow {
        seq_len,
        max_num_vars,
        prove: TimingStats::from(&prove),
        verify: TimingStats::from(&verify),
        peak_rss_bytes,
        proof_size_bytes,
        throughput_tok_per_min,
    };

    let out_path = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/bench/results"
    ))
    .join("tab_scaling_gpt2.json");
    let meta = ScalingTableMeta {
        table: "tab:scaling".to_string(),
        model: BenchModel::Gpt2,
        pcs: BenchPcs::Dory,
        scale,
        warmup_iters: WARMUP_ITERS,
        measured_iters: MEASURED_ITERS,
    };
    upsert_row(&out_path, meta, row);
}
