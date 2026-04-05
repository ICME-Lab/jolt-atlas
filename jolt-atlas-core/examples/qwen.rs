/// Run with tracing:
/// ```bash
/// # Chrome Tracing JSON output (view in chrome://tracing)
/// cargo run --release --package jolt-atlas-core --example qwen -- --trace
///
/// # Terminal output with timing
/// cargo run --release --package jolt-atlas-core --example qwen -- --trace-terminal
///
/// # Reuse cached shared preprocessing (builds and saves it on first use)
/// cargo run --release --package jolt-atlas-core --example qwen -- --use-cache
/// ```
///
/// Requires the Qwen ONNX model to be present first.
use atlas_onnx_tracer::{
    model::{Model, RunArgs},
    tensor::Tensor,
};
use bincode::config::standard;
use jolt_atlas_core::onnx_proof::ops::{
    concat::{reset_concat_selector_build_metrics, snapshot_concat_selector_build_metrics},
    reshape::{reset_reshape_selector_build_metrics, snapshot_reshape_selector_build_metrics},
    slice::{reset_slice_selector_build_metrics, snapshot_slice_selector_build_metrics},
};
use jolt_atlas_core::onnx_proof::{
    snapshot_prove_exclusive_metrics, AtlasProverPreprocessing, AtlasSharedPreprocessing,
    AtlasVerifierPreprocessing, Blake2bTranscript, Bn254, Fr, HyperKZG, ONNXProof,
};
use joltworks::subprotocols::sumcheck::{
    reset_batched_sumcheck_prove_metrics, snapshot_batched_sumcheck_prove_metrics,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{
    collections::HashMap,
    env, fs,
    path::Path,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tracing::{info, info_span, span::Attributes, Id, Subscriber};
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::{
    fmt::format::FmtSpan,
    layer::{Context, SubscriberExt},
    registry::LookupSpan,
    util::SubscriberInitExt,
    EnvFilter, Layer,
};

const MODEL_PATH: &str = "atlas-onnx-tracer/models/qwen/network.onnx";
const SHARED_PP_CACHE_PATH: &str = "atlas-onnx-tracer/models/qwen/shared_preprocessing.bin";

#[derive(Clone, Debug)]
struct SpanState {
    name: &'static str,
    started_at: Instant,
}

#[derive(Clone, Debug, Default)]
struct SpanAggregate {
    total: Duration,
    max: Duration,
    count: u64,
}

#[derive(Clone, Debug, Default)]
struct SpanMetricsCollector {
    totals: Arc<Mutex<HashMap<String, SpanAggregate>>>,
}

impl SpanMetricsCollector {
    fn record(&self, name: &str, elapsed: Duration) {
        let mut totals = self.totals.lock().expect("span metrics mutex poisoned");
        let entry = totals.entry(name.to_string()).or_default();
        entry.total += elapsed;
        entry.max = entry.max.max(elapsed);
        entry.count += 1;
    }

    fn sorted_rows(&self) -> Vec<(String, SpanAggregate)> {
        let totals = self.totals.lock().expect("span metrics mutex poisoned");
        let mut rows: Vec<_> = totals.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        rows.sort_by(|a, b| b.1.total.cmp(&a.1.total));
        rows
    }

    fn print_top(&self, title: &str, limit: usize, filter: impl Fn(&str) -> bool) {
        println!("{title}:");
        for (name, agg) in self
            .sorted_rows()
            .into_iter()
            .filter(|(name, _)| filter(name))
            .take(limit)
        {
            println!(
                "  {name}: total={:.2?}, count={}, max={:.2?}",
                agg.total, agg.count, agg.max
            );
        }
    }
}

#[derive(Clone, Debug)]
struct SpanMetricsLayer {
    collector: SpanMetricsCollector,
}

impl SpanMetricsLayer {
    fn new(collector: SpanMetricsCollector) -> Self {
        Self { collector }
    }
}

impl<S> Layer<S> for SpanMetricsLayer
where
    S: Subscriber + for<'span> LookupSpan<'span>,
{
    fn on_new_span(&self, _attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            let mut extensions = span.extensions_mut();
            extensions.insert(SpanState {
                name: span.metadata().name(),
                started_at: Instant::now(),
            });
        }
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(&id) {
            let extensions = span.extensions();
            if let Some(state) = extensions.get::<SpanState>() {
                self.collector
                    .record(state.name, state.started_at.elapsed());
            }
        }
    }
}

struct QwenTracingGuard {
    chrome_guard: Option<tracing_chrome::FlushGuard>,
    span_metrics: SpanMetricsCollector,
}

fn setup_qwen_tracing(title: &str) -> QwenTracingGuard {
    let args: Vec<String> = std::env::args().collect();
    let enable_trace_json = args.contains(&"--trace".to_string());
    let enable_trace_terminal = args.contains(&"--trace-terminal".to_string());
    let span_metrics = SpanMetricsCollector::default();
    let metrics_layer = SpanMetricsLayer::new(span_metrics.clone());

    if enable_trace_json {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry()
            .with(metrics_layer)
            .with(chrome_layer)
            .init();

        println!("=== {title} (Chrome Tracing enabled) ===");
        println!("Trace output: trace-<timestamp>.json (viewable in chrome://tracing)\n");
        QwenTracingGuard {
            chrome_guard: Some(guard),
            span_metrics,
        }
    } else if enable_trace_terminal {
        let fmt_layer = tracing_subscriber::fmt::layer().with_span_events(FmtSpan::CLOSE);
        let env_filter =
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        tracing_subscriber::registry()
            .with(metrics_layer)
            .with(env_filter)
            .with(fmt_layer)
            .init();

        println!("=== {title} (Terminal Tracing enabled) ===");
        println!("Log level: Set via RUST_LOG environment variable (default: info)\n");
        QwenTracingGuard {
            chrome_guard: None,
            span_metrics,
        }
    } else {
        tracing_subscriber::registry().with(metrics_layer).init();
        println!("=== {title} ===");
        println!("(Run with --trace for JSON output or --trace-terminal for terminal output)\n");
        QwenTracingGuard {
            chrome_guard: None,
            span_metrics,
        }
    }
}

struct SharedPreprocessingLoad {
    shared: AtlasSharedPreprocessing,
    cache_hit: bool,
    cache_read: Duration,
    trace: Duration,
    shared_preprocess: Duration,
}

#[derive(Clone, Debug, Default)]
struct QwenMetrics {
    cache_hit: bool,
    cache_read: Duration,
    trace: Duration,
    shared_preprocess: Duration,
    prover_preprocess: Duration,
    prove: Duration,
    verify: Duration,
    total: Duration,
}

impl QwenMetrics {
    fn print_summary(&self, span_metrics: &SpanMetricsCollector) {
        println!("Qwen metrics:");
        println!("  cache_hit: {}", self.cache_hit);
        if self.cache_hit {
            println!("  cache_read: {:.2?}", self.cache_read);
        } else {
            println!("  trace: {:.2?}", self.trace);
            println!("  shared_preprocess: {:.2?}", self.shared_preprocess);
        }
        println!("  prover_preprocess: {:.2?}", self.prover_preprocess);
        println!("  prove: {:.2?}", self.prove);
        println!("  verify: {:.2?}", self.verify);
        println!("  total: {:.2?}", self.total);

        let prove_exclusive = snapshot_prove_exclusive_metrics();
        println!("Prove top-level phases:");
        let top_level_total: Duration = prove_exclusive
            .top_level
            .iter()
            .map(|phase| phase.duration)
            .sum();
        for phase in &prove_exclusive.top_level {
            println!("  {}: {:.2?}", phase.name, phase.duration);
        }
        println!("  summed_top_level: {top_level_total:.2?}");

        println!("Reduced openings phases:");
        let reduced_total: Duration = prove_exclusive
            .reduced_openings
            .iter()
            .map(|phase| phase.duration)
            .sum();
        for phase in &prove_exclusive.reduced_openings {
            println!("  {}: {:.2?}", phase.name, phase.duration);
        }
        println!("  summed_reduced_openings: {reduced_total:.2?}");

        let mut op_agg: HashMap<String, (u64, Duration)> = HashMap::new();
        for node in &prove_exclusive.iop_nodes {
            let entry = op_agg.entry(node.op_name.clone()).or_default();
            entry.0 += 1;
            entry.1 += node.duration;
        }
        let mut op_phase_agg: HashMap<(String, String), (u64, Duration)> = HashMap::new();
        for phase in &prove_exclusive.iop_node_phases {
            let entry = op_phase_agg
                .entry((phase.op_name.clone(), phase.phase_name.clone()))
                .or_default();
            entry.0 += 1;
            entry.1 += phase.duration;
        }
        let mut op_rows: Vec<_> = op_agg.into_iter().collect();
        op_rows.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));
        let mut op_phase_rows: Vec<_> = op_phase_agg.into_iter().collect();
        op_phase_rows.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));
        let iop_total: Duration = prove_exclusive
            .iop_nodes
            .iter()
            .map(|node| node.duration)
            .sum();
        let mut node_rows = prove_exclusive.iop_nodes.clone();
        node_rows.sort_by(|a, b| b.duration.cmp(&a.duration));
        println!("IOP top nodes:");
        for node in node_rows.iter().take(25) {
            println!(
                "  node {} {}: {:.2?}",
                node.idx, node.op_name, node.duration
            );
        }
        println!("IOP per-op totals:");
        for (op_name, (count, duration)) in op_rows.iter().take(25) {
            println!("  {op_name}: total={duration:.2?}, count={count}");
        }
        println!("  summed_iop_nodes: {iop_total:.2?}");
        println!("IOP per-op phase totals:");
        for ((op_name, phase_name), (count, duration)) in op_phase_rows.iter().take(40) {
            println!("  {op_name}/{phase_name}: total={duration:.2?}, count={count}");
        }
        println!("Constant node phases:");
        for ((op_name, phase_name), (count, duration)) in op_phase_rows
            .iter()
            .filter(|((op_name, _), _)| op_name == "Constant")
        {
            println!("  {op_name}/{phase_name}: total={duration:.2?}, count={count}");
        }

        span_metrics.print_top("Top spans", 25, |_| true);
        span_metrics.print_top("Top op spans", 25, |name| {
            name.contains("::prove")
                || name.contains("::verify")
                || name.contains("Prover::initialize")
                || name.contains("Verifier::new")
                || name.contains("SumcheckProver::compute_message")
        });
        span_metrics.print_top("Top compute_message spans", 40, |name| {
            name.contains("compute_message")
        });
        span_metrics.print_top("Top eval reduction spans", 30, |name| {
            name.contains("EvalReduction") || name.contains("NodeEvalReduction")
        });

        let reshape = snapshot_reshape_selector_build_metrics();
        println!("Reshape selector build metrics:");
        println!("  calls: {}", reshape.calls);
        println!("  total: {:.2?}", reshape.total);
        println!(
            "  linear_to_coord(input): {:.2?}",
            reshape.linear_to_coord_input
        );
        println!(
            "  linear_to_coord(output): {:.2?}",
            reshape.linear_to_coord_output
        );
        println!(
            "  coord_to_linear(input): {:.2?}",
            reshape.coord_to_linear_input
        );
        println!(
            "  coord_to_linear(output): {:.2?}",
            reshape.coord_to_linear_output
        );
        println!("  output_bits: {:.2?}", reshape.output_bits);
        println!("  eq_evals: {:.2?}", reshape.eq_evals);
        println!("  eq_lookup: {:.2?}", reshape.eq_lookup);

        let concat = snapshot_concat_selector_build_metrics();
        println!("Concat selector build metrics:");
        println!("  calls: {}", concat.calls);
        println!("  total: {:.2?}", concat.total);
        println!(
            "  linear_to_coord(input): {:.2?}",
            concat.linear_to_coord_input
        );
        println!(
            "  coord_to_linear(input): {:.2?}",
            concat.coord_to_linear_input
        );
        println!(
            "  coord_to_linear(output): {:.2?}",
            concat.coord_to_linear_output
        );
        println!("  output_bits: {:.2?}", concat.output_bits);
        println!("  eq_evals: {:.2?}", concat.eq_evals);
        println!("  eq_lookup: {:.2?}", concat.eq_lookup);

        let slice = snapshot_slice_selector_build_metrics();
        println!("Slice selector build metrics:");
        println!("  calls: {}", slice.calls);
        println!("  total: {:.2?}", slice.total);
        println!(
            "  linear_to_coord(output): {:.2?}",
            slice.linear_to_coord_output
        );
        println!(
            "  coord_to_linear(input): {:.2?}",
            slice.coord_to_linear_input
        );
        println!(
            "  coord_to_linear(output): {:.2?}",
            slice.coord_to_linear_output
        );
        println!("  output_bits: {:.2?}", slice.output_bits);
        println!("  eq_evals: {:.2?}", slice.eq_evals);
        println!("  eq_lookup: {:.2?}", slice.eq_lookup);

        let batched_sumcheck = snapshot_batched_sumcheck_prove_metrics();
        println!("BatchedSumcheck::prove metrics:");
        println!("  calls: {}", batched_sumcheck.calls);
        println!("  instances_total: {}", batched_sumcheck.instances_total);
        println!("  max_instances: {}", batched_sumcheck.max_instances);
        println!("  rounds_total: {}", batched_sumcheck.rounds_total);
        println!("  max_rounds: {}", batched_sumcheck.max_rounds);
        println!("  total: {:.2?}", batched_sumcheck.total);
        println!(
            "  append_input_claims: {:.2?}",
            batched_sumcheck.append_input_claims
        );
        println!(
            "  batching_coeffs: {:.2?}",
            batched_sumcheck.batching_coeffs
        );
        println!(
            "  initialize_individual_claims: {:.2?}",
            batched_sumcheck.initialize_individual_claims
        );
        println!(
            "  compute_messages: {:.2?}",
            batched_sumcheck.compute_messages
        );
        println!(
            "  combine_univariate_polys: {:.2?}",
            batched_sumcheck.combine_univariate_polys
        );
        println!("  compress: {:.2?}", batched_sumcheck.compress);
        println!(
            "  transcript_and_challenge: {:.2?}",
            batched_sumcheck.transcript_and_challenge
        );
        println!(
            "  update_individual_claims: {:.2?}",
            batched_sumcheck.update_individual_claims
        );
        println!(
            "  ingest_challenges: {:.2?}",
            batched_sumcheck.ingest_challenges
        );
        println!(
            "  finalize_instances: {:.2?}",
            batched_sumcheck.finalize_instances
        );
        println!("  cache_openings: {:.2?}", batched_sumcheck.cache_openings);
    }
}

fn load_or_build_shared_preprocessing(
    run_args: &RunArgs,
    use_cache: bool,
) -> SharedPreprocessingLoad {
    if use_cache && Path::new(SHARED_PP_CACHE_PATH).exists() {
        let cache_span = info_span!("qwen.cache_read", path = SHARED_PP_CACHE_PATH);
        let _entered = cache_span.enter();
        let timing = Instant::now();
        let bytes = fs::read(SHARED_PP_CACHE_PATH).expect("failed to read shared preprocessing");
        let (shared, _): (AtlasSharedPreprocessing, usize) =
            bincode::serde::decode_from_slice(&bytes, standard())
                .expect("failed to decode shared preprocessing");
        let cache_read = timing.elapsed();
        info!(?cache_read, "Loaded shared preprocessing from cache");
        return SharedPreprocessingLoad {
            shared,
            cache_hit: true,
            cache_read,
            trace: Duration::ZERO,
            shared_preprocess: Duration::ZERO,
        };
    }

    let trace_span = info_span!("qwen.trace", model_path = MODEL_PATH);
    let _entered = trace_span.enter();
    let trace_timing = Instant::now();
    let model = Model::load(MODEL_PATH, run_args);
    println!("{}", model.pretty_print());
    println!("max num vars: {}", model.max_num_vars());
    let trace = trace_timing.elapsed();
    drop(_entered);

    let preprocess_span = info_span!("qwen.shared_preprocess");
    let _entered = preprocess_span.enter();
    let preprocess_timing = Instant::now();
    let shared = AtlasSharedPreprocessing::preprocess(model);
    let shared_preprocess = preprocess_timing.elapsed();
    info!(?trace, ?shared_preprocess, "Built shared preprocessing");

    if use_cache {
        let bytes = bincode::serde::encode_to_vec(&shared, standard())
            .expect("failed to encode shared preprocessing");
        fs::write(SHARED_PP_CACHE_PATH, bytes).expect("failed to write shared preprocessing");
        println!("saved shared preprocessing cache to {SHARED_PP_CACHE_PATH}");
    }

    SharedPreprocessingLoad {
        shared,
        cache_hit: false,
        cache_read: Duration::ZERO,
        trace,
        shared_preprocess,
    }
}

fn main() {
    let guard = setup_qwen_tracing("Qwen ONNX Proof");
    let use_cache = env::args().any(|arg| arg == "--use-cache");
    let total_timing = Instant::now();
    reset_reshape_selector_build_metrics();
    reset_concat_selector_build_metrics();
    reset_slice_selector_build_metrics();
    reset_batched_sumcheck_prove_metrics();

    let seq_len: usize = 16;
    let run_args = RunArgs::new([
        ("batch_size", 1),
        ("sequence_length", seq_len),
        ("past_sequence_length", 0),
    ])
    .with_pre_rebase_nonlinear(true);

    let mut rng = StdRng::seed_from_u64(44);
    let vocab_size: i32 = 151936;

    let input_ids_data: Vec<i32> = (0..seq_len).map(|_| rng.gen_range(0..vocab_size)).collect();
    let input_ids = Tensor::new(Some(&input_ids_data), &[1, seq_len]).unwrap();

    let token_type_ids_data: Vec<i32> = vec![0; seq_len];
    let token_type_ids = Tensor::new(Some(&token_type_ids_data), &[1, seq_len]).unwrap();

    let attention_mask_data: Vec<i32> = vec![1; seq_len];
    let attention_mask = Tensor::new(Some(&attention_mask_data), &[1, seq_len]).unwrap();

    info!("Loaded input data");
    let shared_load = load_or_build_shared_preprocessing(&run_args, use_cache);

    let prover_preprocess_span = info_span!("qwen.prover_preprocess");
    let _entered = prover_preprocess_span.enter();
    let prover_preprocess_timing = Instant::now();
    let prover_preprocessing =
        AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(shared_load.shared);
    let prover_preprocess = prover_preprocess_timing.elapsed();
    info!(?prover_preprocess, "Built prover preprocessing");

    let prove_span = info_span!("qwen.prove");
    let _entered = prove_span.enter();
    let timing = Instant::now();
    let (proof, io, _debug_info) = ONNXProof::<Fr, Blake2bTranscript, HyperKZG<Bn254>>::prove(
        &prover_preprocessing,
        &[input_ids, token_type_ids, attention_mask],
    );
    let prove = timing.elapsed();
    println!("Proof generation took {prove:.2?}");
    info!(?prove, "Finished proof generation");

    let verifier_preprocess_span = info_span!("qwen.verifier_preprocess");
    let _entered = verifier_preprocess_span.enter();
    let verifier_preprocessing =
        AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_preprocessing);

    let verify_span = info_span!("qwen.verify");
    let _entered = verify_span.enter();
    let verify_timing = Instant::now();
    proof.verify(&verifier_preprocessing, &io, None).unwrap();
    let verify = verify_timing.elapsed();
    println!("Proof verified successfully!");
    info!(?verify, "Finished verification");

    let metrics = QwenMetrics {
        cache_hit: shared_load.cache_hit,
        cache_read: shared_load.cache_read,
        trace: shared_load.trace,
        shared_preprocess: shared_load.shared_preprocess,
        prover_preprocess,
        prove,
        verify,
        total: total_timing.elapsed(),
    };
    metrics.print_summary(&guard.span_metrics);
    drop(guard.chrome_guard);
}
