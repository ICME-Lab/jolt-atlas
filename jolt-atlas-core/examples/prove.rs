//! Prove and verify any model in the zoo.
//!
//! ```bash
//! # Checked-in stand-ins (no download needed)
//! cargo run --release -p jolt-atlas-core --example prove -- --model models/qwen2-tiny
//! cargo run --release -p jolt-atlas-core --example prove -- --model models/gpt2-tiny --pcs dory
//!
//! # Real models (scripts/download_*.py first)
//! cargo run --release -p jolt-atlas-core --example prove -- --model models/gpt2 --pcs dory
//! cargo run --release -p jolt-atlas-core --example prove -- --model models/qwen --pcs dory \
//!     --input "The quick brown fox jumps over the lazy dog" --use-cache
//! ```
//!
//! Options:
//! * `--model <dir>`          directory holding `network.onnx` (required)
//! * `--pcs hyperkzg|dory`    commitment scheme (default `hyperkzg`)
//! * `--seq-len N`            sequence length binding (default 16; ignored with `--input`)
//! * `--input "<text>"`       tokenize with `<dir>/tokenizer.json`; sequence length = token count
//! * `--srs-vars N`           over-provision the SRS (Dory: wider columns, fewer tier-2 pairings)
//! * `--use-cache`            reuse `<dir>/shared_preprocessing.seq<N>.bin` when present
//! * `--stats`                print the model, clamp-width histogram and packed buckets
//! * `--trace` / `--trace-terminal`  Chrome-trace JSON / terminal timings
//!
//! Inputs other than the token ids are synthesized from the importer's input
//! descriptors (`atlas_onnx_import::fixtures`): masks all ones, positions
//! `0..n` (scaled when the model expects fixed point), token types zero.
use std::{collections::HashMap, path::PathBuf, time::Instant};

use atlas_onnx_import::fixtures::{inputs_with_ids, standard_bindings, synth_inputs, GOLDEN_SEED};
use atlas_onnx_import::{ImportedModel, InputDescriptor, InputKind};
use atlas_trace::{Model, Tensor};
use bincode::config::standard;
use common::utils::logging::setup_tracing;
use jolt_atlas_core::loader::{load_imported, RunArgs};
use jolt_atlas_core::node_view::ModelView;
use jolt_atlas_core::onnx_proof::{
    proof_serialization::serialize_proof, AtlasProverPreprocessing, AtlasSharedPreprocessing,
    AtlasVerifierPreprocessing, Blake2bTranscript, Bn254, DoryScheme, Fr, HyperKZG, ONNXProof,
};
use joltworks::poly::commitment::commitment_scheme::CommitmentScheme;

struct Args {
    model_dir: PathBuf,
    dory: bool,
    seq_len: usize,
    input: Option<String>,
    srs_vars: Option<usize>,
    use_cache: bool,
    stats: bool,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let value = |flag: &str| -> Option<String> {
        argv.iter().position(|a| a == flag).map(|i| {
            argv.get(i + 1)
                .unwrap_or_else(|| panic!("{flag} needs a value"))
                .clone()
        })
    };
    let model_dir = PathBuf::from(value("--model").unwrap_or_else(|| {
        eprintln!(
            "usage: prove --model <dir> [--pcs hyperkzg|dory] [--seq-len N] [--input TEXT] \
             [--srs-vars N] [--use-cache] [--stats] [--trace|--trace-terminal]"
        );
        std::process::exit(2);
    }));
    let dory = match value("--pcs").as_deref() {
        None | Some("hyperkzg") => false,
        Some("dory") => true,
        Some(other) => panic!("unknown --pcs {other:?} (hyperkzg|dory)"),
    };
    Args {
        model_dir,
        dory,
        seq_len: value("--seq-len").map_or(16, |v| v.parse().expect("--seq-len N")),
        input: value("--input"),
        srs_vars: value("--srs-vars").map(|v| v.parse().expect("--srs-vars N")),
        use_cache: argv.iter().any(|a| a == "--use-cache"),
        stats: argv.iter().any(|a| a == "--stats"),
    }
}

/// Token ids from `<dir>/tokenizer.json`.
fn tokenize(model_dir: &std::path::Path, text: &str) -> Vec<i32> {
    let path = model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&path)
        .unwrap_or_else(|e| panic!("--input needs {}: {e}", path.display()));
    let encoding = tokenizer.encode(text, false).expect("tokenization failed");
    encoding.get_ids().iter().map(|&id| id as i32).collect()
}

/// Per-operator histogram of the statically derived saturating-clamp widths,
/// the packed lookup buckets, and the largest nodes.
fn print_stats(model: &Model) {
    use jolt_atlas_core::onnx_proof::global_clamp::{clamp_buckets, remainder_buckets};
    println!("{}", model.pretty_print());
    println!("max num vars: {}", model.max_num_vars());
    let op_name = |idx: usize| {
        let op = format!("{:?}", model.view(idx).operator);
        op.split(|c: char| !c.is_alphanumeric())
            .next()
            .unwrap()
            .to_string()
    };
    let mut hist: std::collections::BTreeMap<(String, usize), usize> = Default::default();
    for node in &model.views() {
        if jolt_atlas_core::node_view::clamp_value_bound(model, node).is_some() {
            *hist
                .entry((op_name(node.idx), node.sat_clamp_bits))
                .or_default() += 1;
        }
    }
    println!("saturating-clamp widths (op, bits) -> nodes:");
    for ((op, bits), n) in hist {
        println!("  {op:<16} {bits:>2} bits  x{n}");
    }
    println!("packed buckets (kind idx width log_t nodes | largest node):");
    for b in clamp_buckets(model)
        .iter()
        .chain(remainder_buckets(model).iter())
    {
        let big = b.nodes.iter().max_by_key(|n| n.log_t).unwrap();
        println!(
            "  {:?} #{:<3} w={:<2} log_t={:<2} nodes={:<4} | node {} {} log_t={}",
            b.kind,
            b.idx,
            b.width,
            b.log_t,
            b.nodes.len(),
            big.idx,
            op_name(big.idx),
            big.log_t
        );
    }
    let mut sizes: Vec<(usize, usize)> = model
        .views()
        .iter()
        .map(|n| (n.pow2_padded_num_output_elements(), n.idx))
        .collect();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    println!("largest nodes (padded elements, idx, op):");
    for (t, idx) in sizes.iter().take(12) {
        println!(
            "  2^{:<2} node {:<5} {}",
            t.trailing_zeros(),
            idx,
            op_name(*idx)
        );
    }
}

/// On-disk form of the input descriptors, next to the preprocessing cache.
#[derive(serde::Serialize, serde::Deserialize)]
struct CachedDescriptor {
    name: String,
    scaled: bool,
    dims: Vec<usize>,
}

/// Import (or load from cache) and return the shared preprocessing plus the
/// input descriptors needed to synthesize inputs. The cache is keyed by the
/// sequence length; a stale or format-incompatible cache is a miss.
fn load(args: &Args, run_args: &RunArgs) -> (AtlasSharedPreprocessing, Vec<InputDescriptor>) {
    let seq_len = run_args.variables["sequence_length"];
    let cache = args
        .model_dir
        .join(format!("shared_preprocessing.seq{seq_len}.bin"));
    let cache_inputs = args
        .model_dir
        .join(format!("shared_preprocessing.seq{seq_len}.inputs.json"));
    if args.use_cache && cache.exists() && cache_inputs.exists() {
        let bytes = std::fs::read(&cache).expect("read cache");
        let descriptors: Vec<CachedDescriptor> =
            serde_json::from_slice(&std::fs::read(&cache_inputs).expect("read cache"))
                .expect("cached descriptors");
        match bincode::serde::decode_from_slice::<AtlasSharedPreprocessing, _>(&bytes, standard()) {
            Ok((shared, _)) => {
                println!("loaded shared preprocessing from {}", cache.display());
                let descriptors = descriptors
                    .into_iter()
                    .map(|d| InputDescriptor {
                        name: d.name,
                        kind: if d.scaled {
                            InputKind::Scaled
                        } else {
                            InputKind::RawInt
                        },
                        dims: d.dims,
                    })
                    .collect();
                return (shared, descriptors);
            }
            Err(e) => eprintln!("stale shared-preprocessing cache ({e}); rebuilding"),
        }
    }

    let t = Instant::now();
    let imported = load_imported(
        args.model_dir.join("network.onnx").to_str().unwrap(),
        run_args,
    );
    println!(
        "imported {} atlas nodes in {:.2?}",
        imported.model.graph().nodes.len(),
        t.elapsed()
    );
    if args.stats {
        print_stats(&imported.model);
    }
    let ImportedModel {
        model,
        input_descriptors,
        ..
    } = imported;
    let shared = AtlasSharedPreprocessing::preprocess(model);
    if args.use_cache {
        std::fs::write(
            &cache,
            bincode::serde::encode_to_vec(&shared, standard()).expect("encode cache"),
        )
        .expect("write cache");
        let descriptors: Vec<CachedDescriptor> = input_descriptors
            .iter()
            .map(|d| CachedDescriptor {
                name: d.name.clone(),
                scaled: d.kind == InputKind::Scaled,
                dims: d.dims.clone(),
            })
            .collect();
        std::fs::write(
            &cache_inputs,
            serde_json::to_vec_pretty(&descriptors).expect("encode descriptors"),
        )
        .expect("write cache");
        println!("saved shared preprocessing cache to {}", cache.display());
    }
    (shared, input_descriptors)
}

/// Prove + verify with the chosen PCS, reporting timings and proof size.
fn run<PCS: CommitmentScheme<Field = Fr>>(
    label: &str,
    shared: AtlasSharedPreprocessing,
    inputs: &[Tensor<i32>],
    srs_vars: Option<usize>,
) {
    let t = Instant::now();
    let prover_pp = match srs_vars {
        Some(n) => AtlasProverPreprocessing::<Fr, PCS>::new_with_srs_num_vars(shared, n),
        None => AtlasProverPreprocessing::<Fr, PCS>::new(shared),
    };
    println!("[{label}] prover setup: {:.2?}", t.elapsed());

    let t = Instant::now();
    let (proof, io, _) = ONNXProof::<Fr, Blake2bTranscript, PCS>::prove(&prover_pp, inputs);
    println!("[{label}] prove: {:.2?}", t.elapsed());
    let bytes = serialize_proof(&proof).expect("proof serializes");
    println!(
        "[{label}] proof size: {:.2} MiB ({} commitments, {} sumcheck proofs)",
        bytes.len() as f64 / (1024.0 * 1024.0),
        proof.commitments.len(),
        proof.proofs.len()
    );

    let verifier_pp = AtlasVerifierPreprocessing::<Fr, PCS>::from(&prover_pp);
    let t = Instant::now();
    proof
        .verify(&verifier_pp, &io, None)
        .expect("verification failed");
    println!("[{label}] verify: {:.2?}", t.elapsed());
    println!("Proof verified successfully!");
}

fn main() {
    let args = parse_args();
    let (_guard, _) = setup_tracing("jolt-atlas prove");

    let ids = args
        .input
        .as_deref()
        .map(|text| tokenize(&args.model_dir, text));
    let seq_len = ids.as_ref().map_or(args.seq_len, Vec::len);
    let bindings: HashMap<String, usize> = standard_bindings(seq_len);
    let run_args = RunArgs::new(bindings);
    println!(
        "model {} | sequence_length {seq_len} | scale 2^{}",
        args.model_dir.display(),
        run_args.scale
    );

    let (shared, descriptors) = load(&args, &run_args);
    // `synth_inputs` only needs the model for the gather dictionary bound;
    // rebuild a minimal `ImportedModel` view around the preprocessed model.
    let imported = ImportedModel {
        model: shared.model().clone(),
        input_descriptors: descriptors,
        pass_reports: vec![],
    };
    let inputs = match &ids {
        Some(ids) => inputs_with_ids(&imported, ids, GOLDEN_SEED),
        None => synth_inputs(&imported, GOLDEN_SEED),
    };
    for (d, t) in imported.input_descriptors.iter().zip(&inputs) {
        println!("  input {:<16} {:?} {:?}", d.name, d.kind, t.dims());
    }

    if args.dory {
        run::<DoryScheme>("dory", shared, &inputs, args.srs_vars);
    } else {
        run::<HyperKZG<Bn254>>("hyperkzg", shared, &inputs, args.srs_vars);
    }
}
