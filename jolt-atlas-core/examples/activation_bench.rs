//! Benchmark: proving time for Erf/Sigmoid/Tanh over a large input tensor.
//!
//! Run with:
//! ```bash
//! cargo run --release --package jolt-atlas-core --example activation_bench
//! ```
//! Add `-- --trace-terminal` for a per-span timing breakdown.

use atlas_onnx_tracer::{
    model::{test::ModelBuilder, Model},
    tensor::Tensor,
};
use common::utils::logging::setup_tracing;
use jolt_atlas_core::onnx_proof::{
    AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing,
    Blake2bTranscript, Bn254, Fr, HyperKZG, ONNXProof,
};
use rand::{rngs::StdRng, SeedableRng};

const SCALE: u32 = 12;
const T: usize = 1 << 16;
/// Number of prove/verify repetitions per op (beyond the discarded warmup run).
const REPS: usize = 5;

fn erf_model() -> Model {
    let mut b = ModelBuilder::with_scale(SCALE);
    let i = b.input(vec![T]);
    let res = b.erf(i);
    b.mark_output(res);
    b.build()
}

fn erf_small_table_model() -> Model {
    let mut b = ModelBuilder::with_scale(SCALE);
    let i = b.input(vec![T]);
    let res = b.erf_small_table(i);
    b.mark_output(res);
    b.build()
}

fn sigmoid_model() -> Model {
    let mut b = ModelBuilder::with_scale(SCALE);
    let i = b.input(vec![T]);
    let res = b.sigmoid(i);
    b.mark_output(res);
    b.build()
}

fn tanh_model() -> Model {
    let mut b = ModelBuilder::with_scale(SCALE);
    let i = b.input(vec![T]);
    let res = b.tanh(i);
    b.mark_output(res);
    b.build()
}

/// Preprocess `model` once, then prove+verify it against `REPS` freshly-sampled
/// inputs (plus one discarded warmup run) to avoid attributing warmup/first-call
/// overhead to the measured timings. Prints min/avg prove and verify time.
fn bench_one(name: &str, model: Model, rng: &mut StdRng) {
    let pp = AtlasSharedPreprocessing::preprocess(model);
    let prover_pp = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(pp);
    let verifier_pp = AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_pp);

    let mut run = || {
        // `ModelBuilder::erf`/`sigmoid` hardcode tau=2 regardless of scale (unlike
        // `tanh`, which scales tau with `neural_teleport_tau`), so the valid raw
        // input bound stays [-65536, 65534] (quotient range ±2^15 times tau=2) at
        // any scale. Stay safely inside that. At scale=12 this is well within the
        // [-8, 8) real-value range (fixed-point [-32768, 32768)), so this doesn't
        // exercise saturation for the baseline the way scale=8 did -- an existing
        // ModelBuilder quirk, not something this benchmark works around.
        let input = Tensor::<i32>::random_range(rng, &[T], -60000..60000);

        let t = std::time::Instant::now();
        let (proof, io, debug_info) =
            ONNXProof::<Fr, Blake2bTranscript, HyperKZG<Bn254>>::prove(&prover_pp, &[input]);
        let prove_time = t.elapsed();

        let t = std::time::Instant::now();
        proof.verify(&verifier_pp, &io, debug_info).unwrap();
        let verify_time = t.elapsed();

        (prove_time, verify_time)
    };

    // Discarded warmup run.
    run();

    let mut prove_times = Vec::with_capacity(REPS);
    let mut verify_times = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let (p, v) = run();
        prove_times.push(p);
        verify_times.push(v);
    }

    let prove_min = prove_times.iter().min().unwrap();
    let prove_avg = prove_times.iter().sum::<std::time::Duration>() / REPS as u32;
    let verify_min = verify_times.iter().min().unwrap();
    let verify_avg = verify_times.iter().sum::<std::time::Duration>() / REPS as u32;

    println!(
        "{name:<10} prove: min {prove_min:>10.2?}  avg {prove_avg:>10.2?}   verify: min {verify_min:>10.2?}  avg {verify_avg:>10.2?}"
    );
}

fn main() {
    let (_guard, _tracing_enabled) = setup_tracing("activation bench");
    let mut rng = StdRng::seed_from_u64(0x420);

    println!("=== baseline (neural_teleport, log_table=16), scale={SCALE} ===");
    bench_one("erf", erf_model(), &mut rng);
    bench_one("sigmoid", sigmoid_model(), &mut rng);
    bench_one("tanh", tanh_model(), &mut rng);

    println!("\n=== clamped small table (log_table=12), scale={SCALE} ===");
    bench_one("erf", erf_small_table_model(), &mut rng);
}
