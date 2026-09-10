//! `tab:clamp-ablation`: GPT-2, one PCS, full vs. saturating-clamp-disabled — prove time /
//! committed one-hot / proof size. See `../../paper-jolt-atlas2/BENCHMARKS.md` benchmark 4.
//!
//! Disabling clamp is a **compile-time** switch (`clamp-ablation` feature), not a runtime flag —
//! each PCS needs two invocations:
//!
//! ```bash
//! cargo run --release -p jolt-atlas-core --example bench_clamp_ablation -- --pcs dory
//! cargo run --release --features clamp-ablation -p jolt-atlas-core \
//!     --example bench_clamp_ablation -- --pcs dory   # UNSOUND: clamp buckets never proven
//! ```
//!
//! `pcs` lives in the table meta, not the row — running with a different `--pcs` than what's
//! already on disk is a hard error (delete the file first to switch). Requires `MODEL_SCALE ==
//! 12`. Writes `bench/results/tab_clamp_ablation_gpt2.json`, one row per `full|disabled`.

mod harness;
mod models;
mod report;

use ark_serialize::{CanonicalSerialize, Compress};
use common::utils::logging::setup_tracing;
use jolt_atlas_core::onnx_proof::{
    AtlasProverPreprocessing, Blake2bTranscript, Bn254, DoryScheme, Fr, HyperKZG, ONNXProof,
};
use joltworks::poly::{
    commitment::commitment_scheme::CommitmentScheme, multilinear_polynomial::MultilinearPolynomial,
};
use models::LoadedModel;
use report::{upsert_row, BenchModel, BenchPcs, Meta, PcsLabel, Row, TimingStats};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const WARMUP_ITERS: usize = 1;
const MEASURED_ITERS: usize = 3;
const SEQ_LEN: usize = 16;

#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct ClampAblationTableMeta {
    table: String,
    model: BenchModel,
    pcs: BenchPcs,
    seq_len: usize,
    scale: usize,
    warmup_iters: usize,
    measured_iters: usize,
}

impl Meta for ClampAblationTableMeta {}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
#[serde(rename_all = "lowercase")]
enum Config {
    Full,
    Disabled,
}

impl Config {
    fn as_str(self) -> &'static str {
        match self {
            Config::Full => "full",
            Config::Disabled => "disabled",
        }
    }
}

impl std::fmt::Display for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Serialize, Deserialize)]
struct ClampAblationRow {
    config: Config,
    prove: TimingStats,
    committed_polys: usize,
    committed_one_hot_entries: usize,
    proof_size_bytes: usize,
}

impl Row for ClampAblationRow {
    fn name(&self) -> String {
        self.config.to_string()
    }
}

const CONFIG: Config = if cfg!(feature = "clamp-ablation") {
    Config::Disabled
} else {
    Config::Full
};

fn parse_pcs() -> BenchPcs {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .map(|s| s.as_str())
    };
    get("--pcs")
        .and_then(BenchPcs::parse)
        .unwrap_or_else(|| panic!("--pcs <hyperkzg|dory> is required (got {:?})", get("--pcs")))
}

fn main() {
    let (_guard, _tracing_enabled) = setup_tracing("tab:clamp-ablation bench");
    let pcs = parse_pcs();
    let loaded = models::load_gpt2(models::random_input_ids(
        SEQ_LEN,
        models::GPT2_VOCAB_SIZE,
        42,
    ));
    let (seq_len, scale) = (loaded.seq_len, loaded.scale);
    println!(
        "[gpt2/{pcs}/{CONFIG}] seq_len={seq_len} scale={scale} max_num_vars={}",
        loaded.max_num_vars
    );

    let row = match pcs {
        BenchPcs::HyperKzg => run::<HyperKZG<Bn254>>(loaded),
        BenchPcs::Dory => run::<DoryScheme>(loaded),
    };

    let out_path = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/bench/results"
    ))
    .join("tab_clamp_ablation_gpt2.json");
    let meta = ClampAblationTableMeta {
        table: "tab:clamp-ablation".to_string(),
        model: BenchModel::Gpt2,
        pcs,
        seq_len,
        scale,
        warmup_iters: WARMUP_ITERS,
        measured_iters: MEASURED_ITERS,
    };
    upsert_row(&out_path, meta, row);
}

fn run<PCS: CommitmentScheme<Field = Fr> + PcsLabel>(loaded: LoadedModel) -> ClampAblationRow {
    let pcs = PCS::BENCH_PCS;
    let LoadedModel { shared, inputs, .. } = loaded;
    let prover_pp = AtlasProverPreprocessing::<Fr, PCS>::new(shared);
    joltworks::utils::thread::wait_for_background_drops();

    // `disabled` never generates clamp buckets (see `global_clamp::bucket_witnesses`).
    let trace = prover_pp.model().trace(&inputs);
    let poly_map: std::collections::BTreeMap<common::CommittedPoly, MultilinearPolynomial<Fr>> =
        ONNXProof::<Fr, Blake2bTranscript, PCS>::polynomial_map(prover_pp.model(), &trace);
    let committed_polys = poly_map.len();
    let committed_one_hot_entries: usize = poly_map
        .values()
        .map(|mle| match mle {
            MultilinearPolynomial::OneHot(one_hot) => one_hot
                .nonzero_indices
                .iter()
                .filter(|x| x.is_some())
                .count(),
            _ => 0,
        })
        .sum();

    let mut last_proof = None;
    let prove = harness::time_repeated_with_settle(
        WARMUP_ITERS,
        MEASURED_ITERS,
        || {
            let (proof, _io, _debug) =
                ONNXProof::<Fr, Blake2bTranscript, PCS>::prove(&prover_pp, &inputs);
            last_proof = Some(proof);
        },
        joltworks::utils::thread::wait_for_background_drops,
    );
    let proof = last_proof.expect("at least one measured iteration");
    let proof_size_bytes = proof.serialized_size(Compress::Yes);

    println!(
        "[gpt2/{pcs}/{CONFIG}] prove(mean)={:.2?} committed_polys={committed_polys} \
         one_hot_entries={committed_one_hot_entries} proof_size={proof_size_bytes}B",
        prove.mean(),
    );

    ClampAblationRow {
        config: CONFIG,
        prove: TimingStats::from(&prove),
        committed_polys,
        committed_one_hot_entries,
        proof_size_bytes,
    }
}
