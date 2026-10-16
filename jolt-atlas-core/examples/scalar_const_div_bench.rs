//! Prove/verify timing benchmark for a standalone ScalarConstDiv node.
//!
//! ```bash
//! cargo run --release --package jolt-atlas-core --example scalar_const_div_bench
//! # Also prints mean-of-NUM_RUNS timings for ONNXProof::{commit_witness_polynomials,
//! # iop,prove_reduced_openings} (accumulated in-process from their tracing spans):
//! cargo run --release --package jolt-atlas-core --example scalar_const_div_bench -- --trace-terminal
//! ```
use atlas_onnx_tracer::{model::test::ModelBuilder, tensor::Tensor};
use common::consts::MODEL_SCALE;
use jolt_atlas_core::onnx_proof::{
    AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing,
    Blake2bTranscript, Bn254, Fr, HyperKZG, ONNXProof,
};
use rand::{rngs::StdRng, SeedableRng};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tracing_subscriber::{layer::SubscriberExt, registry::LookupSpan, util::SubscriberInitExt};

const INPUT_LENS: &[usize] = &[1 << 8, 1 << 16];
const DIVISOR: i32 = 128;
const NUM_RUNS: usize = 5;

/// Top-level `ONNXProof::prove` stages to accumulate mean timings for.
const STAGE_SPANS: &[&str] = &[
    "ONNXProof::commit_witness_polynomials",
    "ONNXProof::iop",
    "ONNXProof::prove_reduced_openings",
];

fn scalar_const_div_model(input_len: usize) -> atlas_onnx_tracer::model::Model {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![input_len]);
    let res = b.scalar_const_div(i, DIVISOR);
    b.mark_output(res);
    b.build()
}

fn run_once(
    prover_preprocessing: &AtlasProverPreprocessing<Fr, HyperKZG<Bn254>>,
    verifier_preprocessing: &AtlasVerifierPreprocessing<Fr, HyperKZG<Bn254>>,
    input: &Tensor<i32>,
) -> (Duration, Duration) {
    let prove_start = Instant::now();
    let (proof, io, debug_info) = ONNXProof::<Fr, Blake2bTranscript, HyperKZG<Bn254>>::prove(
        prover_preprocessing,
        &[input.clone()],
    );
    let prove_time = prove_start.elapsed();

    let verify_start = Instant::now();
    proof
        .verify(verifier_preprocessing, &io, debug_info)
        .unwrap();
    let verify_time = verify_start.elapsed();

    (prove_time, verify_time)
}

fn mean(durations: &[Duration]) -> Duration {
    durations.iter().sum::<Duration>() / durations.len() as u32
}

/// Timestamp stashed in a span's extensions on entry, read back on close.
struct SpanStart(Instant);

/// Per-stage-name accumulated durations, shared between the tracing layer (writer)
/// and `main` (reader, once per input length).
#[derive(Default, Clone)]
struct StageTimings(Arc<Mutex<HashMap<&'static str, Vec<Duration>>>>);

impl StageTimings {
    /// Drains and returns each stage's mean duration, in `STAGE_SPANS` order.
    fn take_means(&self) -> Vec<(&'static str, Duration)> {
        let mut recorded = self.0.lock().unwrap();
        STAGE_SPANS
            .iter()
            .filter_map(|&name| {
                let durations = recorded.remove(name)?;
                (!durations.is_empty()).then(|| (name, mean(&durations)))
            })
            .collect()
    }
}

/// Records each `STAGE_SPANS` span's wall-clock duration into `StageTimings`.
struct StageTimingLayer {
    timings: StageTimings,
}

impl<S> tracing_subscriber::Layer<S> for StageTimingLayer
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_enter(&self, id: &tracing::span::Id, ctx: tracing_subscriber::layer::Context<'_, S>) {
        let Some(span) = ctx.span(id) else { return };
        if STAGE_SPANS.contains(&span.name()) {
            span.extensions_mut().insert(SpanStart(Instant::now()));
        }
    }

    fn on_close(&self, id: tracing::span::Id, ctx: tracing_subscriber::layer::Context<'_, S>) {
        let Some(span) = ctx.span(&id) else { return };
        let name = span.name();
        if !STAGE_SPANS.contains(&name) {
            return;
        }
        let Some(elapsed) = span
            .extensions()
            .get::<SpanStart>()
            .map(|start| start.0.elapsed())
        else {
            return;
        };
        self.timings
            .0
            .lock()
            .unwrap()
            .entry(name)
            .or_default()
            .push(elapsed);
    }
}

/// Registers only [`StageTimingLayer`] — no `fmt` layer, so nothing prints to the
/// terminal except this bench's own `println!`s.
fn setup_stage_timing_tracing(timings: StageTimings) {
    tracing_subscriber::registry()
        .with(StageTimingLayer { timings })
        .init();
}

fn main() {
    let mut rng = StdRng::seed_from_u64(0x5CD);
    let trace_terminal = std::env::args().any(|a| a == "--trace-terminal");
    let stage_timings = StageTimings::default();
    if trace_terminal {
        setup_stage_timing_tracing(stage_timings.clone());
    }

    println!("MODEL_SCALE = {MODEL_SCALE}, divisor = {DIVISOR}");
    println!("runs per input length = {NUM_RUNS} (+1 warmup, discarded)");

    for &input_len in INPUT_LENS {
        let input = Tensor::<i32>::random_small(&mut rng, &[input_len]);

        let model = scalar_const_div_model(input_len);
        let pp = AtlasSharedPreprocessing::preprocess(model);
        let prover_preprocessing = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(pp);
        let verifier_preprocessing =
            AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_preprocessing);

        // Warmup run, discarded (page faults, allocator warmup, etc.); also discards
        // its span timings so they don't skew the per-stage means below.
        run_once(&prover_preprocessing, &verifier_preprocessing, &input);
        stage_timings.take_means();

        let mut prove_times = Vec::with_capacity(NUM_RUNS);
        let mut verify_times = Vec::with_capacity(NUM_RUNS);
        for _ in 0..NUM_RUNS {
            let (prove_time, verify_time) =
                run_once(&prover_preprocessing, &verifier_preprocessing, &input);
            prove_times.push(prove_time);
            verify_times.push(verify_time);
        }

        println!("--- input length = {input_len} ---");
        println!("prove times  = {prove_times:?}");
        println!("verify times = {verify_times:?}");
        println!("mean prove time  = {:?}", mean(&prove_times));
        println!("mean verify time = {:?}", mean(&verify_times));
        if trace_terminal {
            for (name, stage_mean) in stage_timings.take_means() {
                println!("mean {name} = {stage_mean:?}");
            }
        }
    }
}
