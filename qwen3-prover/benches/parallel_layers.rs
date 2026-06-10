use std::{
    env,
    hint::black_box,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use ark_bn254::Bn254;
use joltworks::{
    poly::commitment::{
        commitment_scheme::CommitmentScheme,
        hyperkzg::{HyperKZG, HyperKZGProverKey},
    },
    transcripts::{Blake2bTranscript, Transcript},
};
use qwen3_prover::{
    commitment::{CommitLayerParams, commit_layer_hidden_openings},
    layer::{LayerOutput, prove_layer},
};
use qwen3_tracer::{TraceLayerInput, layer_input_from_trace_dir};
use rayon::prelude::*;

const LAYERS: usize = 28;

struct LayerBenchResult {
    layer: usize,
    elapsed: Duration,
    proved: bool,
}

fn main() {
    type Pcs = HyperKZG<Bn254>;

    let rayon_threads = env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok());
    let mut pool = rayon::ThreadPoolBuilder::new().stack_size(64 * 1024 * 1024);
    if let Some(threads) = rayon_threads {
        pool = pool.num_threads(threads);
    }
    pool.build_global()
        .expect("rayon global thread pool initializes once");

    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("qwen3-prover has a workspace parent");
    let trace_dir = env::var_os("QWEN3_TRACE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace.join("traces/qwen3-0.6b/fox_eos_full_awy"));
    let q8_cache = env::var_os("QWEN3_Q8_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace.join("models/qwen3-0.6b/model.q8.bin"));

    let layer_indices = bench_layers();
    let load_start = Instant::now();
    let layers = layer_indices
        .iter()
        .copied()
        .map(|layer| {
            layer_input_from_trace_dir(&trace_dir, &q8_cache, layer)
                .unwrap_or_else(|error| panic!("layer {layer} trace conversion failed: {error}"))
        })
        .collect::<Vec<_>>();
    let load_elapsed = load_start.elapsed();

    let setup_start = Instant::now();
    let setup = Pcs::setup_prover(20);
    let setup_elapsed = setup_start.elapsed();

    let commit_params = CommitLayerParams {
        pcs_domain_size: 1 << 20,
    };

    let prove_start = Instant::now();
    let mut results = layers
        .into_par_iter()
        .zip(layer_indices.into_par_iter())
        .map(|(traced, layer)| {
            let layer_start = Instant::now();
            let mut transcript = Blake2bTranscript::default();
            let output = prove_bench_layer(traced, commit_params, &setup, &mut transcript);
            if let Some(output) = output.as_ref() {
                black_box(output.commitments.hidden_out.commitments.len());
            }
            LayerBenchResult {
                layer,
                elapsed: layer_start.elapsed(),
                proved: output.is_some(),
            }
        })
        .collect::<Vec<_>>();
    let prove_elapsed = prove_start.elapsed();
    results.sort_by_key(|result| result.layer);

    println!("qwen3 parallel layer prove bench");
    println!("layers: {}", results.len());
    println!("rayon threads: {}", rayon::current_num_threads());
    println!("trace_load: {}", format_duration(load_elapsed));
    println!("pcs_setup:  {}", format_duration(setup_elapsed));
    println!("prove_layers: {}", format_duration(prove_elapsed));
    println!();
    println!("per-layer wall time");
    println!("layer  status  elapsed");
    println!("-----  ------  -------");
    for result in results {
        println!(
            "{:>5}  {:>6}  {}",
            result.layer,
            if result.proved { "ok" } else { "failed" },
            format_duration(result.elapsed)
        );
    }
}

fn prove_bench_layer<T>(
    traced: TraceLayerInput,
    commit_params: CommitLayerParams,
    setup: &HyperKZGProverKey<Bn254>,
    transcript: &mut T,
) -> Option<LayerOutput>
where
    T: Transcript,
{
    let hidden_commitments =
        commit_layer_hidden_openings(&traced.input.opening_witnesses, commit_params, setup).ok()?;
    prove_layer(
        traced.input,
        hidden_commitments,
        commit_params,
        setup,
        transcript,
    )
}

fn bench_layers() -> Vec<usize> {
    if let Ok(value) = env::var("QWEN3_BENCH_LAYERS") {
        return value
            .split(',')
            .map(|layer| {
                layer
                    .trim()
                    .parse::<usize>()
                    .expect("QWEN3_BENCH_LAYERS contains layer indices")
            })
            .collect();
    }
    (0..LAYERS).collect()
}

fn format_duration(duration: Duration) -> String {
    format!("{:.3}s", duration.as_secs_f64())
}
