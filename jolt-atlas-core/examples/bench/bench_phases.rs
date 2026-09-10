//! `tab:phases`: GPT-2, both PCS — witness commitment / IOP proving / batched clamp proof /
//! reduction opening / PCS evaluation proof, as absolute time and % of prove.
//! See `../../paper-jolt-atlas2/BENCHMARKS.md` benchmark 2.
//!
//! Separate binary from `bench_e2e.rs`: needs its own `tracing_subscriber` layer
//! (`bench/phase_layer.rs`) with no `fmt` layer, which conflicts with `bench_e2e.rs`'s own
//! global subscriber install.
//!
//! `cargo run --release -p jolt-atlas-core --example bench_phases -- --pcs <hyperkzg|dory>`
//! (requires `MODEL_SCALE == 12`). Writes `bench/results/tab_phases_gpt2.json`, one row per PCS.

mod harness;
mod models;
mod phase_layer;
mod report;

use jolt_atlas_core::onnx_proof::{
    AtlasProverPreprocessing, AtlasVerifierPreprocessing, Blake2bTranscript, Bn254, DoryScheme, Fr,
    HyperKZG, ONNXProof,
};
use joltworks::poly::commitment::commitment_scheme::CommitmentScheme;
use phase_layer::StageTimingLayer;
use report::{upsert_row, BenchModel, BenchPcs, Meta, PhasePercent, Row};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::PathBuf,
    time::{Duration, Instant},
};
use tracing_subscriber::prelude::*;

const WARMUP_ITERS: usize = 1;
const MEASURED_ITERS: usize = 3;
const DEFAULT_SEQ_LEN: usize = 16;

#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct PhaseTableMeta {
    table: String,
    model: BenchModel,
    seq_len: usize,
    scale: usize,
    warmup_iters: usize,
    measured_iters: usize,
}

impl Meta for PhaseTableMeta {}

#[derive(Serialize, Deserialize)]
struct PhaseRow {
    pcs: BenchPcs,
    prove_total_mean_s: f64,
    witness_commitment: PhasePercent,
    iop_proving: PhasePercent,
    batched_clamp_proof: PhasePercent,
    opening_reduction: PhasePercent,
    pcs_eval_proof: PhasePercent,
}

impl Row for PhaseRow {
    fn name(&self) -> String {
        self.pcs.to_string()
    }
}

const WITNESS_COMMITMENT: &str = "ONNXProof::commit_witness_polynomials";
const IOP_PROVING: &str = "ONNXProof::iop_nodes";
const BATCHED_CLAMP_PROOF: &str = "deferred_lookups::prove_all";
const OPENING_REDUCTION: &str = "ONNXProof::opening_reduction";
const PCS_EVAL_PROOF: &str = "ONNXProof::pcs_prove";

fn parse_args() -> (BenchPcs, usize) {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .map(|s| s.as_str())
    };
    let pcs = get("--pcs")
        .and_then(BenchPcs::parse)
        .unwrap_or_else(|| panic!("--pcs <hyperkzg|dory> is required (got {:?})", get("--pcs")));
    let seq_len = get("--seq-len")
        .map(|s| s.parse().expect("--seq-len must be a number"))
        .unwrap_or(DEFAULT_SEQ_LEN);
    (pcs, seq_len)
}

fn main() {
    let layer = StageTimingLayer::default();
    tracing_subscriber::registry().with(layer.clone()).init();

    let (pcs, seq_len) = parse_args();
    let input_ids = models::random_input_ids(seq_len, models::GPT2_VOCAB_SIZE, 42);
    let loaded = models::load_gpt2(input_ids);
    let scale = loaded.scale;
    println!(
        "[gpt2] seq_len={} scale={} max_num_vars={}",
        loaded.seq_len, loaded.scale, loaded.max_num_vars
    );

    let row = match pcs {
        BenchPcs::HyperKzg => run_phases::<HyperKZG<Bn254>>(loaded, &layer),
        BenchPcs::Dory => run_phases::<DoryScheme>(loaded, &layer),
    };

    let out_path = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/bench/results"
    ))
    .join("tab_phases_gpt2.json");
    let meta = PhaseTableMeta {
        table: "tab:phases".to_string(),
        model: BenchModel::Gpt2,
        seq_len,
        scale,
        warmup_iters: WARMUP_ITERS,
        measured_iters: MEASURED_ITERS,
    };
    upsert_row(&out_path, meta, row);
}

fn run_phases<PCS: CommitmentScheme<Field = Fr> + report::PcsLabel>(
    loaded: models::LoadedModel,
    layer: &StageTimingLayer,
) -> PhaseRow {
    let pcs = PCS::BENCH_PCS;
    let models::LoadedModel { shared, inputs, .. } = loaded;

    let prover_pp = AtlasProverPreprocessing::<Fr, PCS>::new(shared);
    joltworks::utils::thread::wait_for_background_drops();

    let mut prove_total_samples = Vec::with_capacity(MEASURED_ITERS);
    let mut phase_samples: HashMap<&'static str, Vec<Duration>> = HashMap::new();
    let mut last_proof_io = None;

    for i in 0..(WARMUP_ITERS + MEASURED_ITERS) {
        let start = Instant::now();
        let (proof, io, _debug) =
            ONNXProof::<Fr, Blake2bTranscript, PCS>::prove(&prover_pp, &inputs);
        let elapsed = start.elapsed();
        let durations = layer.take();
        last_proof_io = Some((proof, io));
        joltworks::utils::thread::wait_for_background_drops();

        if i >= WARMUP_ITERS {
            prove_total_samples.push(elapsed);
            for name in [
                WITNESS_COMMITMENT,
                IOP_PROVING,
                BATCHED_CLAMP_PROOF,
                OPENING_REDUCTION,
                PCS_EVAL_PROOF,
            ] {
                phase_samples
                    .entry(name)
                    .or_default()
                    .push(durations.get(name).copied().unwrap_or_default());
            }
        }
    }

    // Correctness gate only — not timed, not part of the table.
    let (proof, io) = last_proof_io.expect("at least one measured iteration");
    let verifier_pp = AtlasVerifierPreprocessing::<Fr, PCS>::from(&prover_pp);
    proof
        .verify(&verifier_pp, &io, None)
        .expect("proof produced during phase timing failed to verify");

    let prove_total_mean_s = mean_secs(&prove_total_samples);
    let phase_mean = |name: &str| mean_secs(&phase_samples[name]);

    println!(
        "[gpt2/{pcs}] prove(mean)={prove_total_mean_s:.2}s witness_commit={:.2}s \
         iop={:.2}s clamp={:.2}s opening_reduction={:.2}s pcs_prove={:.2}s",
        phase_mean(WITNESS_COMMITMENT),
        phase_mean(IOP_PROVING),
        phase_mean(BATCHED_CLAMP_PROOF),
        phase_mean(OPENING_REDUCTION),
        phase_mean(PCS_EVAL_PROOF),
    );

    PhaseRow {
        pcs,
        prove_total_mean_s,
        witness_commitment: PhasePercent::new(phase_mean(WITNESS_COMMITMENT), prove_total_mean_s),
        iop_proving: PhasePercent::new(phase_mean(IOP_PROVING), prove_total_mean_s),
        batched_clamp_proof: PhasePercent::new(phase_mean(BATCHED_CLAMP_PROOF), prove_total_mean_s),
        opening_reduction: PhasePercent::new(phase_mean(OPENING_REDUCTION), prove_total_mean_s),
        pcs_eval_proof: PhasePercent::new(phase_mean(PCS_EVAL_PROOF), prove_total_mean_s),
    }
}

fn mean_secs(samples: &[Duration]) -> f64 {
    samples.iter().sum::<Duration>().as_secs_f64() / samples.len() as f64
}
