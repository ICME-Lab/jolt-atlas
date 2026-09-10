//! `tab:zk-overhead`: GPT-2, HyperKZG — the same input proven both ways, comparing prove /
//! verify / proof size. See `../../paper-jolt-atlas2/BENCHMARKS.md` benchmark 5.
//!
//! `cargo run --release --features zk -p jolt-atlas-core --example bench_zk_overhead` (requires
//! `MODEL_SCALE == 12`). Writes `bench/results/tab_zk_overhead_gpt2.json`: "zk"/"non-zk"/
//! "overhead" rows.

mod harness;
mod models;
mod report;

#[cfg(not(feature = "zk"))]
fn main() {
    eprintln!("This example requires the `zk` feature. Re-run with:");
    eprintln!(
        "  cargo run --release --features zk --package jolt-atlas-core --example bench_zk_overhead"
    );
    std::process::exit(1);
}

#[cfg(feature = "zk")]
const WARMUP_ITERS: usize = 1;
#[cfg(feature = "zk")]
const MEASURED_ITERS: usize = 3;
#[cfg(feature = "zk")]
const SEQ_LEN: usize = 16;

#[cfg(feature = "zk")]
#[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug, Clone)]
struct ZkOverheadTableMeta {
    table: String,
    model: report::BenchModel,
    pcs: report::BenchPcs,
    seq_len: usize,
    scale: usize,
    warmup_iters: usize,
    measured_iters: usize,
}

#[cfg(feature = "zk")]
impl report::Meta for ZkOverheadTableMeta {}

#[cfg(feature = "zk")]
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, PartialEq, Debug)]
#[serde(rename_all = "lowercase")]
enum Config {
    Zk,
    NonZk,
    Overhead,
}

#[cfg(feature = "zk")]
impl Config {
    fn as_str(self) -> &'static str {
        match self {
            Config::Zk => "zk",
            Config::NonZk => "non-zk",
            Config::Overhead => "overhead",
        }
    }
}

#[cfg(feature = "zk")]
impl std::fmt::Display for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Either the config's own measurement, or (on the `Overhead` row) the zk/non-zk ratio for that
/// same column — keeps the ratio next to what it's a ratio of. `#[serde(untagged)]` so it's just
/// the measurement's own JSON shape on disk, never a `{"Measured": ...}` wrapper.
#[cfg(feature = "zk")]
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
enum Column<T> {
    Measured(T),
    Overhead(f64),
}

/// `threads` discloses that zk runs on a throttled pool while non-zk runs at full parallelism
/// (see `main`) — prove/verify times aren't apples-to-apples on thread count, so each row states
/// its own.
#[cfg(feature = "zk")]
#[derive(serde::Serialize, serde::Deserialize)]
struct ZkOverheadRow {
    config: Config,
    threads: Option<usize>,
    prove: Column<report::TimingStats>,
    verify: Column<report::TimingStats>,
    proof_size_bytes: Column<usize>,
}

#[cfg(feature = "zk")]
impl report::Row for ZkOverheadRow {
    fn name(&self) -> String {
        self.config.to_string()
    }
}

#[cfg(feature = "zk")]
fn main() {
    // Local (not global) bounded rayon pool for the ZK calls only: a bigger stack (ZK's deep MLE
    // work overflows the default 2MB), and a lower thread count (the patched arkworks MSM builds
    // a nested ThreadPoolBuilder per MSM chunk, which can exhaust the OS's pthread limit at full
    // parallelism). A global change would also throttle non-zk, which we want at full speed.
    // `ZK_BENCH_THREADS` overrides the default (2) without a code edit — see
    // wiki/jolt-atlas/book/src/underway/zk-prove-overhead.md. Each row reports its own thread
    // count since this makes prove/verify not apples-to-apples between configs.
    let zk_num_threads: usize = std::env::var("ZK_BENCH_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let nonzk_num_threads = rayon::current_num_threads();
    let zk_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(zk_num_threads)
        .stack_size(32 * 1024 * 1024)
        .build()
        .expect("failed to build ZK rayon pool");
    eprintln!("[bench] nonzk_threads={nonzk_num_threads} zk_threads={zk_num_threads}");

    use ark_serialize::{CanonicalSerialize, Compress};
    use common::utils::logging::setup_tracing;
    use jolt_atlas_core::onnx_proof::{
        AtlasProverPreprocessing, AtlasVerifierPreprocessing, Blake2bTranscript, Bn254, Fr,
        HyperKZG, ONNXProof,
    };
    use models::LoadedModel;
    use report::{upsert_row, TimingStats};
    use std::path::PathBuf;

    let (_guard, _tracing_enabled) = setup_tracing("tab:zk-overhead bench");

    let LoadedModel {
        shared,
        inputs,
        seq_len,
        scale,
        max_num_vars,
        ..
    } = models::load_gpt2(models::random_input_ids(
        SEQ_LEN,
        models::GPT2_VOCAB_SIZE,
        42,
    ));
    println!("[gpt2] seq_len={seq_len} scale={scale} max_num_vars={max_num_vars}");

    let prover_pp = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(shared);
    let verifier_pp = AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_pp);
    joltworks::utils::thread::wait_for_background_drops();

    // ── Non-ZK path ──────────────────────────────────────────────────────
    let mut last_nonzk = None;
    let nonzk_prove = harness::time_repeated_with_settle(
        WARMUP_ITERS,
        MEASURED_ITERS,
        || {
            let (proof, io, _dbg) =
                ONNXProof::<Fr, Blake2bTranscript, HyperKZG<Bn254>>::prove(&prover_pp, &inputs);
            last_nonzk = Some((proof, io));
        },
        joltworks::utils::thread::wait_for_background_drops,
    );
    let (nonzk_proof, nonzk_io) = last_nonzk.expect("at least one measured iteration");
    let nonzk_proof_size_bytes = nonzk_proof.serialized_size(Compress::Yes);

    let nonzk_verify = harness::time_repeated_with_settle(
        WARMUP_ITERS,
        MEASURED_ITERS,
        || {
            nonzk_proof.verify(&verifier_pp, &nonzk_io, None).unwrap();
        },
        joltworks::utils::thread::wait_for_background_drops,
    );

    // ── ZK path (BlindFold) ──────────────────────────────────────────────
    let gens = joltworks::poly::commitment::pedersen::PedersenGenerators::<
        joltworks::curve::Bn254Curve,
    >::deterministic(4096);

    let mut last_zk = None;
    let zk_prove = harness::time_repeated_with_settle(
        WARMUP_ITERS,
        MEASURED_ITERS,
        || {
            let (bundle, zk_io) = zk_pool
                .install(|| jolt_atlas_core::onnx_proof::zk::prove_zk(&prover_pp, &inputs, &gens));
            last_zk = Some((bundle, zk_io));
        },
        joltworks::utils::thread::wait_for_background_drops,
    );
    let (zk_bundle, zk_io) = last_zk.expect("at least one measured iteration");
    let zk_proof_size_bytes = zk_bundle_size(&zk_bundle);

    let zk_verify = harness::time_repeated_with_settle(
        WARMUP_ITERS,
        MEASURED_ITERS,
        || {
            zk_pool
                .install(|| {
                    jolt_atlas_core::onnx_proof::zk::verify_zk(
                        &zk_bundle,
                        &verifier_pp,
                        &zk_io,
                        &gens,
                    )
                })
                .expect("ZK verification should succeed");
        },
        joltworks::utils::thread::wait_for_background_drops,
    );

    let prove_overhead = zk_prove.mean().as_secs_f64() / nonzk_prove.mean().as_secs_f64();
    let verify_overhead = zk_verify.mean().as_secs_f64() / nonzk_verify.mean().as_secs_f64();
    let proof_size_overhead = zk_proof_size_bytes as f64 / nonzk_proof_size_bytes as f64;

    println!(
        "[gpt2] prove:  non-zk(mean)={:.2?}  zk(mean)={:.2?}  ratio={prove_overhead:.2}x",
        nonzk_prove.mean(),
        zk_prove.mean(),
    );
    println!(
        "[gpt2] verify: non-zk(mean)={:.2?}  zk(mean)={:.2?}  ratio={verify_overhead:.2}x",
        nonzk_verify.mean(),
        zk_verify.mean(),
    );
    println!(
        "[gpt2] size:   non-zk={nonzk_proof_size_bytes}B  zk={zk_proof_size_bytes}B  \
         ratio={proof_size_overhead:.2}x",
    );

    let out_path = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/bench/results"
    ))
    .join("tab_zk_overhead_gpt2.json");
    let meta = ZkOverheadTableMeta {
        table: "tab:zk-overhead".to_string(),
        model: report::BenchModel::Gpt2,
        pcs: report::BenchPcs::HyperKzg,
        seq_len,
        scale,
        warmup_iters: WARMUP_ITERS,
        measured_iters: MEASURED_ITERS,
    };

    upsert_row(
        &out_path,
        meta.clone(),
        ZkOverheadRow {
            config: Config::NonZk,
            threads: Some(nonzk_num_threads),
            prove: Column::Measured(TimingStats::from(&nonzk_prove)),
            verify: Column::Measured(TimingStats::from(&nonzk_verify)),
            proof_size_bytes: Column::Measured(nonzk_proof_size_bytes),
        },
    );
    upsert_row(
        &out_path,
        meta.clone(),
        ZkOverheadRow {
            config: Config::Zk,
            threads: Some(zk_num_threads),
            prove: Column::Measured(TimingStats::from(&zk_prove)),
            verify: Column::Measured(TimingStats::from(&zk_verify)),
            proof_size_bytes: Column::Measured(zk_proof_size_bytes),
        },
    );
    upsert_row(
        &out_path,
        meta,
        ZkOverheadRow {
            config: Config::Overhead,
            threads: None,
            prove: Column::Overhead(prove_overhead),
            verify: Column::Overhead(verify_overhead),
            proof_size_bytes: Column::Overhead(proof_size_overhead),
        },
    );
}

/// Wire-protocol size only; skips fields the verifier reconstructs from the transcript/R1CS
/// layout (`stage_configs`, `baked`, `zk_sumcheck_num_instances`).
#[cfg(feature = "zk")]
fn zk_bundle_size(b: &jolt_atlas_core::onnx_proof::zk::ZkProofBundle) -> usize {
    use ark_serialize::CanonicalSerialize;

    let mut n = 0;
    n += b.blindfold_proof.compressed_size();

    // BlindFoldVerifierInput: three Vec<C::G1>.
    let bvi = &b.blindfold_verifier_input;
    n += bvi.round_commitments.compressed_size();
    n += bvi.output_claims_row_commitments.compressed_size();
    n += bvi.eval_commitments.compressed_size();

    // BTreeMap<usize, EvalReductionProof<F>> — sum keys + values.
    for (k, v) in &b.eval_reduction_proofs {
        n += k.compressed_size();
        n += v.compressed_size();
    }
    for (k, v) in &b.eval_reduction_h_commitments {
        n += k.compressed_size();
        n += v.compressed_size();
    }

    n += b.commitments.compressed_size();
    n += b.output_claim.compressed_size();

    // Vec<(usize, ZkSumcheckProof)> — fields are pub, inline-size.
    for (idx, zk) in &b.zk_sumcheck_proofs {
        n += idx.compressed_size();
        n += zk.round_commitments.compressed_size();
        n += zk.poly_degrees.compressed_size();
        n += zk.output_claims_commitments.compressed_size();
    }

    for (k, v) in &b.inter_stage_commitments {
        n += k.compressed_size();
        n += v.compressed_size();
    }
    for (k, v) in &b.auxiliary_claims {
        n += k.compressed_size();
        n += v.compressed_size();
    }
    n
}
