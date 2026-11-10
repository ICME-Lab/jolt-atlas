//! Prove and verify any model in the zoo.
//!
//! ```bash
//! # Checked-in stand-ins (no download needed)
//! cargo run --release -p jolt-atlas-core --example prove -- --model models/qwen2-tiny
//! cargo run --release -p jolt-atlas-core --example prove -- --model models/gpt2-tiny
//!
//! # Real models (scripts/download_*.py first)
//! cargo run --release -p jolt-atlas-core --example prove -- --model models/gpt2
//! cargo run --release -p jolt-atlas-core --example prove -- --model models/qwen \
//!     --input "The quick brown fox jumps over the lazy dog" --use-cache
//! ```
//!
//! Options:
//! * `--model <dir>`          directory holding `network.onnx` (required)
//! * `--seq-len N`            sequence length binding (default 16; ignored with `--input`)
//! * `--input "<text>"`       tokenize with `<dir>/tokenizer.json`; sequence length = token count
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
    proof_serialization::serialize_proof, prove, verify, AkitaScheme, AtlasProverPreprocessing,
    AtlasSharedPreprocessing, AtlasVerifierPreprocessing, Blake2bProver, Blake2bVerifier, Fp128,
};

struct Args {
    model_dir: PathBuf,
    seq_len: usize,
    input: Option<String>,
    use_cache: bool,
    stats: bool,
    census: bool,
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
            "usage: prove --model <dir> [--seq-len N] [--input TEXT] [--use-cache] [--stats] \
             [--census] [--trace|--trace-terminal]"
        );
        std::process::exit(2);
    }));
    Args {
        model_dir,
        seq_len: value("--seq-len").map_or(16, |v| v.parse().expect("--seq-len N")),
        input: value("--input"),
        use_cache: argv.iter().any(|a| a == "--use-cache"),
        stats: argv.iter().any(|a| a == "--stats"),
        census: argv.iter().any(|a| a == "--census"),
    }
}

/// Short operator name (`Einsum`, `Add`, ...).
fn op_name(model: &Model, idx: usize) -> String {
    let op = format!("{:?}", model.view(idx).operator);
    op.split(|c: char| !c.is_alphanumeric())
        .next()
        .unwrap()
        .to_string()
}

/// Signed two's-complement width that holds every value of `q`.
fn needed_bits(q: &[i64]) -> usize {
    let max = q.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
    (u64::BITS - max.leading_zeros()) as usize + 1
}

/// Committed-digit census and value-magnitude audit from one trace, without
/// proving (Phase 2 of `docs/measurement-plan.md`).
fn census(model: &Model, inputs: &[Tensor<i32>]) {
    use jolt_atlas_core::onnx_proof::clamp_lookups::{clamp_intermediate, is_scalar};
    use jolt_atlas_core::onnx_proof::ops::NodeCommittedPolynomials;
    use jolt_atlas_core::onnx_proof::trace_view::TraceView;
    use std::collections::BTreeMap;

    let t = Instant::now();
    let trace = model.trace(inputs).expect("model execution failed");
    println!("trace execution: {:.2?}", t.elapsed());
    let views = model.views();
    let consumers: BTreeMap<usize, Vec<usize>> = views.iter().fold(BTreeMap::new(), |mut m, n| {
        for &i in &n.inputs {
            m.entry(i).or_default().push(n.idx);
        }
        m
    });
    let is_const = |idx: usize| {
        let n = model.view(idx);
        match &n.operator {
            atlas_trace::Op::Constant(_) => true,
            atlas_trace::Op::Broadcast(_) => {
                matches!(
                    model.view(n.inputs[0]).operator,
                    atlas_trace::Op::Constant(_)
                )
            }
            _ => false,
        }
    };

    // 1. Committed digit columns by producing op, split by polynomial family.
    //    One digit column = one one-hot polynomial of `padded` elements.
    #[derive(Default)]
    struct Fam {
        polys: usize,
        cells: usize,
    }
    let mut by_op: BTreeMap<(String, String), Fam> = BTreeMap::new();
    let mut total_cells = 0usize;
    for n in &views {
        let padded = n.pow2_padded_num_output_elements();
        for p in NodeCommittedPolynomials::get_committed_polynomials(n).expect("shape") {
            let name = format!("{p:?}");
            let fam = name.split('(').next().unwrap().to_string();
            let e = by_op.entry((op_name(model, n.idx), fam)).or_default();
            e.polys += 1;
            e.cells += padded;
            total_cells += padded;
        }
    }
    println!("committed digit columns by op and family (polys, cells=polys·padded, share):");
    let mut rows: Vec<_> = by_op.iter().collect();
    rows.sort_by_key(|(_, f)| std::cmp::Reverse(f.cells));
    for ((op, fam), f) in rows {
        println!(
            "  {op:<16} {fam:<30} polys {:>5}  cells 2^{:<5.1} {:>5.1}%",
            f.polys,
            (f.cells as f64).log2(),
            100.0 * f.cells as f64 / total_cells as f64
        );
    }
    println!(
        "  total cells 2^{:.1} ({} K=16 one-hot digits; {} key bits)",
        (total_cells as f64).log2(),
        total_cells,
        total_cells * 4
    );

    // 2. Clamp audit: static width vs the width the traced quotient needs, and
    //    the saturation count; rows by width for the row check.
    println!(
        "clamp audit per node (op, static w, needed w, saturated/elems, row width): nodes with \
         needed > 32 or saturation only"
    );
    let widths = [16usize, 20, 24, 28, 32, 36, 40, 48];
    let thresholds = [80usize, 112, 176, 1024, 4096];
    let mut digits_today = 0usize;
    let mut digits_observed = 0usize; // width = needed, rounded up to a digit
    let mut overflow_nodes = [0usize; 8];
    let mut overflow_elems = [0usize; 8];
    let mut digits_at = [0usize; 8]; // nodes under w' at w', others at today's width
    let mut needed_hist: BTreeMap<(String, usize), usize> = BTreeMap::new();
    let mut rows_by_thr = [0usize; 5];
    let mut digits_by_thr = [0usize; 5];
    let mut saturated_total = 0usize;
    let mut elems_total = 0usize;
    for n in &views {
        let clamped = jolt_atlas_core::node_view::clamp_value_bound(model, n)
            .expect("shape")
            .is_some()
            && !is_scalar(n);
        if !clamped {
            continue;
        }
        let q = clamp_intermediate(n, &trace);
        let q = q.data();
        let live = n.num_output_elements();
        let padded = n.pow2_padded_num_output_elements();
        let needed = needed_bits(&q[..live.min(q.len())]);
        let saturated = q
            .iter()
            .filter(|&&v| v > i32::MAX as i64 || v < i32::MIN as i64)
            .count();
        saturated_total += saturated;
        elems_total += live;
        let row_w = *n.output_dims.last().unwrap_or(&1);
        let today = n.sat_clamp_bits / 4;
        digits_today += today * padded;
        digits_observed += needed.div_ceil(4) * padded;
        for (k, &w) in widths.iter().enumerate() {
            let bound = 1i64 << (w - 1);
            let over = q.iter().filter(|&&v| v >= bound || v < -bound).count();
            if over > 0 {
                overflow_nodes[k] += 1;
                overflow_elems[k] += over;
                digits_at[k] += today * padded;
            } else {
                digits_at[k] += (w / 4) * padded;
            }
        }
        for (k, &thr) in thresholds.iter().enumerate() {
            if row_w >= thr {
                rows_by_thr[k] += padded / row_w.max(1);
                digits_by_thr[k] += today * padded;
            }
        }
        *needed_hist
            .entry((op_name(model, n.idx), needed))
            .or_default() += 1;
        if needed > 32 || saturated > 0 {
            println!(
                "  node {:<5} {:<14} static {:>2} needed {:>2} sat {}/{} row {}",
                n.idx,
                op_name(model, n.idx),
                n.sat_clamp_bits,
                needed,
                saturated,
                live,
                row_w
            );
        }
    }
    println!("needed width histogram (op, bits) -> nodes:");
    for ((op, bits), c) in &needed_hist {
        println!("  {op:<16} {bits:>2} bits  x{c}");
    }
    println!(
        "saturated elements: {saturated_total} of {elems_total}; clamp digit cells today 2^{:.1}, \
         at observed widths 2^{:.1} ({:.2}x)",
        (digits_today as f64).log2(),
        (digits_observed as f64).log2(),
        digits_today as f64 / digits_observed.max(1) as f64
    );
    println!(
        "per-op clamp width w' -> nodes/elements that exceed it, clamp digit cells (x vs today):"
    );
    for (k, &w) in widths.iter().enumerate() {
        println!(
            "  w'={w:>2}: {:>4} nodes, {:>9} elems over; cells 2^{:.1} ({:.2}x)",
            overflow_nodes[k],
            overflow_elems[k],
            (digits_at[k] as f64).log2(),
            digits_today as f64 / digits_at[k].max(1) as f64
        );
    }
    println!("clamp digit cells on rows of width >= threshold (rows, share of clamp cells):");
    for (k, &thr) in thresholds.iter().enumerate() {
        println!(
            "  >= {thr:>4}: rows {:>8}  cells {:>5.1}%",
            rows_by_thr[k],
            100.0 * digits_by_thr[k] as f64 / digits_today.max(1) as f64
        );
    }

    // 3. Softmax: spread of `max - x` over live lanes.
    println!("softmax rows: spread of (max - x) over live lanes:");
    let s = model.scale().log2() as usize;
    for n in &views {
        let Some(w) = trace.softmax_witness(n.idx) else {
            continue;
        };
        let rows = w.x.len() / w.last_dim;
        let mut hist = [0usize; 40];
        let mut max_d = 0i64;
        for k in 0..rows {
            for j in 0..w.active_len {
                let d = w.max_k[k] as i64 - w.x[k * w.last_dim + j] as i64;
                max_d = max_d.max(d);
                hist[(64 - d.max(0).leading_zeros()) as usize] += 1;
            }
        }
        let lanes = rows * w.active_len;
        let over: usize = hist[s + 4..].iter().sum();
        println!(
            "  node {:<5} rows {:>5} live {:>4} max(max-x) 2^{:.1}  lanes with max-x >= 2^{} : {} ({:.2}%)",
            n.idx,
            rows,
            w.active_len,
            (max_d.max(1) as f64).log2(),
            s + 3,
            over,
            100.0 * over as f64 / lanes.max(1) as f64
        );
    }

    // 4. Fusion candidates and shape-op alignment.
    let mut biased = 0usize;
    let mut biased_cells = 0usize;
    let mut act_after_einsum = 0usize;
    for n in &views {
        if let atlas_trace::Op::Einsum(_) = n.operator {
            for &c in consumers.get(&n.idx).map(Vec::as_slice).unwrap_or(&[]) {
                let cn = model.view(c);
                if matches!(cn.operator, atlas_trace::Op::Add(_))
                    && cn.inputs.iter().any(|&i| i != n.idx && is_const(i))
                {
                    biased += 1;
                    biased_cells += n.pow2_padded_num_output_elements();
                }
                if NodeCommittedPolynomials::get_committed_polynomials(&cn)
                    .expect("shape")
                    .iter()
                    .any(|p| format!("{p:?}").starts_with("NodeOutputRaD"))
                {
                    act_after_einsum += 1;
                }
            }
        }
    }
    println!(
        "einsum -> constant Add (bias): {biased} nodes, 2^{:.1} output cells; activation reading an einsum directly: {act_after_einsum}",
        (biased_cells.max(1) as f64).log2()
    );
    for n in &views {
        match &n.operator {
            atlas_trace::Op::Slice(sl) => println!(
                "  Slice node {} axis {} {}..{} of {:?}",
                n.idx,
                sl.axis,
                sl.start,
                sl.end,
                model.view(n.inputs[0]).output_dims
            ),
            atlas_trace::Op::Concat(c) => println!(
                "  Concat node {} axis {} inputs {:?} -> {:?}",
                n.idx, c.axis, n.inputs, n.output_dims
            ),
            _ => {}
        }
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
    println!(
        "max num vars: {}",
        model
            .max_num_vars()
            .expect("the model's nodes have the shapes their ops prove")
    );
    let op_name = |idx: usize| {
        let op = format!("{:?}", model.view(idx).operator);
        op.split(|c: char| !c.is_alphanumeric())
            .next()
            .unwrap()
            .to_string()
    };
    let mut hist: std::collections::BTreeMap<(String, usize), usize> = Default::default();
    for node in &model.views() {
        if jolt_atlas_core::node_view::clamp_value_bound(model, node)
            .expect("the model's nodes have the shapes their ops prove")
            .is_some()
        {
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
        .expect("the model's nodes have the shapes their ops prove")
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

/// Prove + verify, reporting timings and proof size.
fn run(shared: AtlasSharedPreprocessing, inputs: &[Tensor<i32>]) {
    let t = Instant::now();
    let prover_pp = AtlasProverPreprocessing::<Fp128, AkitaScheme>::new(shared);
    println!("prover setup: {:.2?}", t.elapsed());

    let t = Instant::now();
    let (proof, io, _) = prove::<Fp128, Blake2bProver, AkitaScheme>(&prover_pp, inputs);
    println!("prove: {:.2?}", t.elapsed());
    let bytes = serialize_proof(&proof).expect("proof serializes");
    println!(
        "proof size: {:.2} MiB ({} commitments, {} sumcheck proofs)",
        bytes.len() as f64 / (1024.0 * 1024.0),
        proof.commitments().len(),
        proof.proofs().len()
    );

    let verifier_pp = AtlasVerifierPreprocessing::<Fp128, AkitaScheme>::from(&prover_pp);
    let t = Instant::now();
    verify::<Fp128, Blake2bVerifier, AkitaScheme>(&proof, &verifier_pp, &io, None)
        .expect("verification failed");
    println!("verify: {:.2?}", t.elapsed());
    println!("Proof verified successfully!");
}

fn main() {
    let args = parse_args();
    let (_guard, _) = setup_tracing("jolt-atlas prove");

    let ids = args.input.as_deref().map(|text| {
        // The models need a power-of-two sequence length; keep the longest
        // power-of-two prefix of the text.
        let mut ids = tokenize(&args.model_dir, text);
        let keep = 1usize << ids.len().ilog2();
        ids.truncate(keep);
        ids
    });
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

    if args.census {
        census(shared.model(), &inputs);
        return;
    }
    run(shared, &inputs);
}
