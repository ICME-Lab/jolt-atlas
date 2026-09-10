//! `tab:e2e`: per model x PCS — setup / prove / verify / total / peak memory / proof size.
//! See `../../paper-jolt-atlas2/BENCHMARKS.md` benchmark 1.
//!
//! One PCS per invocation (peak memory is whole-process). One model per invocation too:
//! `MODEL_SCALE` is compile-time (GPT-2 12, Qwen 14).
//!
//! `cargo run --release -p jolt-atlas-core --example bench_e2e -- --model <gpt2|bge|qwen> --pcs
//! <hyperkzg|dory>`. Writes `bench/results/tab_e2e_<model>.json`, one row per PCS.

mod harness;
mod models;
mod report;

use ark_serialize::{CanonicalSerialize, Compress};
use common::utils::logging::setup_tracing;
use jolt_atlas_core::onnx_proof::{
    AtlasProverPreprocessing, AtlasVerifierPreprocessing, Blake2bTranscript, Bn254, DoryScheme, Fr,
    HyperKZG, ONNXProof,
};
use joltworks::poly::commitment::commitment_scheme::CommitmentScheme;
use models::LoadedModel;
use report::{upsert_row, BenchModel, BenchPcs, Meta, Row, TimingStats};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const WARMUP_ITERS: usize = 1;
const MEASURED_ITERS: usize = 3;

#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct E2eTableMeta {
    table: String,
    model: BenchModel,
    seq_len: usize,
    scale: usize,
    warmup_iters: usize,
    measured_iters: usize,
}

impl Meta for E2eTableMeta {}

#[derive(Serialize, Deserialize)]
struct E2eRow {
    pcs: BenchPcs,
    setup_s: f64,
    prove: TimingStats,
    verify: TimingStats,
    /// setup + mean(prove) + mean(verify), the paper's "total" column.
    total_s: f64,
    peak_rss_bytes: Option<u64>,
    proof_size_bytes: usize,
}

impl Row for E2eRow {
    fn name(&self) -> String {
        self.pcs.to_string()
    }
}

/// What `--seq-len`/`--input` resolved to — the two flags are mutually exclusive.
pub enum InputArg {
    Default,
    SeqLen(usize),
    Input(String),
}

const DEFAULT_SEQ_LEN: usize = 16;
const DEFAULT_QWEN_PROMPT: &str = "The quick brown fox jumps over the lazy dog";

fn gpt2_input_ids(input_arg: &InputArg) -> Vec<i32> {
    match input_arg {
        InputArg::Default => models::random_input_ids(DEFAULT_SEQ_LEN, models::GPT2_VOCAB_SIZE, 42),
        InputArg::SeqLen(seq_len) => {
            models::random_input_ids(*seq_len, models::GPT2_VOCAB_SIZE, 42)
        }
        InputArg::Input(text) => models::gpt2_token_ids(text),
    }
}

/// No tokenizer for bge — `--input` errors.
fn bge_input_ids(input_arg: &InputArg) -> Vec<i32> {
    let seq_len = match input_arg {
        InputArg::Default => DEFAULT_SEQ_LEN,
        InputArg::SeqLen(n) => *n,
        InputArg::Input(_) => panic!(
            "--input has no effect on bge: its input is randomly generated token ids, not \
             embedded text — use --seq-len instead"
        ),
    };
    models::random_input_ids(seq_len, models::BGE_VOCAB_SIZE, 43)
}

fn qwen_input_ids(input_arg: &InputArg) -> Vec<i32> {
    match input_arg {
        InputArg::Default => models::qwen_token_ids(DEFAULT_QWEN_PROMPT),
        InputArg::SeqLen(seq_len) => {
            models::random_input_ids(*seq_len, models::QWEN_VOCAB_SIZE, 44)
        }
        InputArg::Input(text) => models::qwen_token_ids(text),
    }
}

struct Args {
    model: BenchModel,
    pcs: BenchPcs,
    input_arg: InputArg,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .map(|s| s.as_str())
    };

    let model = get("--model")
        .and_then(BenchModel::parse)
        .unwrap_or_else(|| {
            panic!(
                "--model <gpt2|bge|qwen> is required (got {:?})",
                get("--model")
            )
        });
    let pcs = get("--pcs")
        .and_then(BenchPcs::parse)
        .unwrap_or_else(|| panic!("--pcs <hyperkzg|dory> is required (got {:?})", get("--pcs")));
    let seq_len = get("--seq-len").map(|s| s.parse().expect("--seq-len must be a number"));
    let input = get("--input").map(str::to_string);
    let input_arg = match (seq_len, input) {
        (Some(_), Some(_)) => panic!("--seq-len and --input are mutually exclusive"),
        (Some(n), None) => InputArg::SeqLen(n),
        (None, Some(s)) => InputArg::Input(s),
        (None, None) => InputArg::Default,
    };

    Args {
        model,
        pcs,
        input_arg,
    }
}

fn main() {
    let (_guard, _tracing_enabled) = setup_tracing("tab:e2e bench");
    let args = parse_args();

    let loaded = match args.model {
        BenchModel::Gpt2 => models::load_gpt2(gpt2_input_ids(&args.input_arg)),
        BenchModel::Bge => models::load_bge(bge_input_ids(&args.input_arg)),
        BenchModel::Qwen => models::load_qwen(qwen_input_ids(&args.input_arg)),
    };

    let model_label = loaded.label;
    let (seq_len, scale, max_num_vars) = (loaded.seq_len, loaded.scale, loaded.max_num_vars);
    println!(
        "[{model_label}/{}] seq_len={seq_len} scale={scale} max_num_vars={max_num_vars}",
        args.pcs
    );

    let row = match args.pcs {
        BenchPcs::HyperKzg => run_e2e::<HyperKZG<Bn254>>(loaded),
        BenchPcs::Dory => run_e2e::<DoryScheme>(loaded),
    };

    let out_path = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/bench/results"
    ))
    .join(format!("tab_e2e_{model_label}.json"));
    let meta = E2eTableMeta {
        table: "tab:e2e".to_string(),
        model: args.model,
        seq_len,
        scale,
        warmup_iters: WARMUP_ITERS,
        measured_iters: MEASURED_ITERS,
    };
    upsert_row(&out_path, meta, row);
}

fn run_e2e<PCS: CommitmentScheme<Field = Fr> + report::PcsLabel>(loaded: LoadedModel) -> E2eRow {
    let pcs = PCS::BENCH_PCS;
    let LoadedModel {
        label,
        shared,
        inputs,
        ..
    } = loaded;

    let setup_start = std::time::Instant::now();
    let prover_pp = AtlasProverPreprocessing::<Fr, PCS>::new(shared);
    let setup_s = setup_start.elapsed().as_secs_f64();
    let verifier_pp = AtlasVerifierPreprocessing::<Fr, PCS>::from(&prover_pp);
    joltworks::utils::thread::wait_for_background_drops();

    // Keep the last proof for `verify` below. Settles background drops every iteration — without
    // waiting, iteration N+1's allocations can start before N's frees land, stacking peak memory
    // until the process OOMs.
    let mut last_proof_io = None;
    let prove = harness::time_repeated_with_settle(
        WARMUP_ITERS,
        MEASURED_ITERS,
        || {
            let (proof, io, _debug) =
                ONNXProof::<Fr, Blake2bTranscript, PCS>::prove(&prover_pp, &inputs);
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
    let total_s = setup_s + prove.mean().as_secs_f64() + verify.mean().as_secs_f64();

    println!(
        "[{label}] setup={setup_s:.2}s prove(mean)={:.2?} verify(mean)={:.2?} \
         size={proof_size_bytes}B peak_rss={peak_rss_bytes:?}",
        prove.mean(),
        verify.mean(),
    );

    E2eRow {
        pcs,
        setup_s,
        prove: TimingStats::from(&prove),
        verify: TimingStats::from(&verify),
        total_s,
        peak_rss_bytes,
        proof_size_bytes,
    }
}
