use std::{
    error::Error,
    path::PathBuf,
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
use qwen3_common::{TraceLayerRawInput, layer_raw_input_from_trace_dir};
use qwen3_prover::{
    commitment::{CommitLayerParams, commit_layer_hidden_openings},
    layer::{LayerOutput, prove_layer},
    layer_input::layer_prover_input,
};
use rayon::prelude::*;

const LAYERS: usize = 28;

struct Args {
    trace: PathBuf,
    q8_cache: PathBuf,
    layers: Vec<usize>,
    jobs: Option<usize>,
}

fn main() -> Result<(), Box<dyn Error>> {
    type Pcs = HyperKZG<Bn254>;

    let args = Args::parse()?;
    if let Some(jobs) = args.jobs {
        rayon::ThreadPoolBuilder::new()
            .num_threads(jobs)
            .stack_size(64 * 1024 * 1024)
            .build_global()?;
    }
    let load_start = Instant::now();
    let layers = args
        .layers
        .iter()
        .copied()
        .map(|layer| {
            layer_raw_input_from_trace_dir(&args.trace, &args.q8_cache, layer)
                .map(|input| (layer, input))
                .map_err(|error| format!("layer {layer} trace conversion failed: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
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
        .map(|(layer, traced)| {
            let layer_start = Instant::now();
            let mut transcript = Blake2bTranscript::default();
            let output = prove_trace_layer(traced, commit_params, &setup, &mut transcript);
            (layer, output.is_some(), layer_start.elapsed())
        })
        .collect::<Vec<_>>();
    results.sort_by_key(|(layer, _, _)| *layer);
    let prove_elapsed = prove_start.elapsed();

    println!("qwen3 trace prove");
    println!("trace: {}", args.trace.display());
    println!("q8_cache: {}", args.q8_cache.display());
    println!("layers: {}", args.layers.len());
    println!("rayon threads: {}", rayon::current_num_threads());
    println!("trace_load: {}", format_duration(load_elapsed));
    println!("pcs_setup: {}", format_duration(setup_elapsed));
    println!("prove_layers: {}", format_duration(prove_elapsed));
    println!();
    println!("layer  status  elapsed");
    println!("-----  ------  -------");
    for (layer, proved, elapsed) in results {
        println!(
            "{:>5}  {:>6}  {}",
            layer,
            if proved { "ok" } else { "failed" },
            format_duration(elapsed)
        );
    }

    Ok(())
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .ok_or("qwen3-prover has no workspace parent")?
            .to_path_buf();
        let mut trace = None;
        let mut q8_cache = workspace.join("models/qwen3-0.6b/model.q8.bin");
        let mut layers = None;
        let mut jobs = None;

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--trace" => {
                    trace = Some(PathBuf::from(args.next().ok_or("--trace requires a path")?));
                }
                "--q8-cache" => {
                    q8_cache = PathBuf::from(args.next().ok_or("--q8-cache requires a path")?);
                }
                "--layer" | "--layers" => {
                    layers = Some(parse_layers(
                        &args.next().ok_or("--layer requires a value")?,
                    )?);
                }
                "--jobs" => {
                    let value = args.next().ok_or("--jobs requires a value")?.parse()?;
                    if value == 0 {
                        return Err("--jobs must be positive".into());
                    }
                    jobs = Some(value);
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => return Err(format!("unknown option {other:?}").into()),
            }
        }

        let trace = trace.ok_or("--trace is required")?;
        let layers = layers.ok_or("--layer is required")?;
        if layers.is_empty() {
            return Err("--layer must contain at least one layer".into());
        }
        for &layer in &layers {
            if layer >= LAYERS {
                return Err(format!("layer {layer} is out of range 0..{LAYERS}").into());
            }
        }

        Ok(Self {
            trace,
            q8_cache,
            layers,
            jobs,
        })
    }
}

fn parse_layers(value: &str) -> Result<Vec<usize>, Box<dyn Error>> {
    if value == "all" {
        return Ok((0..LAYERS).collect());
    }
    value
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .map_err(|error| format!("invalid layer {part:?}: {error}").into())
        })
        .collect()
}

fn print_help() {
    println!(
        "qwen3-prove-trace\n\
         \n\
         Prove one or more Qwen3 layers from a generated trace.\n\
         \n\
         Options:\n\
           --trace PATH          trace directory produced by qwen3_generate --trace\n\
           --layer N            layer index to prove, or comma-separated indices\n\
           --layers LIST        alias for --layer; use all for every layer\n\
           --jobs N             number of Rayon worker threads; default Rayon chooses\n\
           --q8-cache PATH      model.q8.bin path; default models/qwen3-0.6b/model.q8.bin\n"
    );
}

fn prove_trace_layer<T>(
    traced: TraceLayerRawInput,
    commit_params: CommitLayerParams,
    setup: &HyperKZGProverKey<Bn254>,
    transcript: &mut T,
) -> Option<LayerOutput>
where
    T: Transcript,
{
    let input = layer_prover_input(traced.shape, traced.weights, traced.raw_witness)?;
    let hidden_commitments =
        commit_layer_hidden_openings(&input.opening_witnesses, commit_params, setup).ok()?;
    prove_layer(input, hidden_commitments, commit_params, setup, transcript)
}

fn format_duration(duration: Duration) -> String {
    format!("{:.3}s", duration.as_secs_f64())
}
