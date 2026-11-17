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
//! * `--stats`                print the model, the packed range buckets and the largest nodes
//! * `--trace` / `--trace-terminal`  Chrome-trace JSON / terminal timings
//!
//! Inputs other than the token ids are synthesized from the importer's input
//! descriptors (`atlas_onnx_import::fixtures`): masks all ones, positions
//! `0..n` (scaled when the model expects fixed point), token types zero.
use std::{collections::HashMap, path::PathBuf, time::Instant};

use atlas_algebra::poly::multilinear::MultilinearPolynomial;
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
fn needed_bits(q: &[i32]) -> usize {
    let max = q.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
    (u32::BITS - max.leading_zeros()) as usize + 1
}

/// Committed-digit census and value-magnitude audit from one trace, without
/// proving (Phase 2 of `docs/measurement-plan.md`).
fn census(model: &Model, inputs: &[Tensor<i32>]) {
    use common::CommittedPoly;
    use jolt_atlas_core::onnx_proof::schedule::Schedule;
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

    // 1. Committed columns, as the prover commits them: the packed bucket polynomials (dense
    //    digit columns of the range claims) and the per-node polynomials. A row is one cycle;
    //    a one-hot row has K = 16 cells, a dense digit row one.
    #[derive(Default)]
    struct Fam {
        polys: usize,
        empty: usize,
        rows: usize,
        nonzeros: usize,
    }
    let count = |fam: &mut Fam, poly: &MultilinearPolynomial<Fp128>| {
        fam.polys += 1;
        match poly {
            MultilinearPolynomial::OneHot(oh) => {
                let nz = oh.nonzero_indices.iter().filter(|k| k.is_some()).count();
                fam.rows += oh.nonzero_indices.len();
                fam.nonzeros += nz;
                if nz == 0 {
                    fam.empty += 1;
                }
            }
            other => fam.rows += other.len(),
        }
    };
    let mut by_fam: BTreeMap<String, Fam> = BTreeMap::new();
    for (p, poly) in
        jolt_atlas_core::onnx_proof::range_bucket::BucketWitnesses::build(model, &trace)
            .committed_polys::<Fp128>(model)
    {
        let name = format!("{p:?}");
        let fam = name.split('(').next().unwrap().to_string();
        count(by_fam.entry(fam).or_default(), &poly);
    }
    let mut by_op: BTreeMap<(String, String), usize> = BTreeMap::new();
    let mut node_rows = 0usize;
    // The polynomials the schedule opens besides the buckets' digits: each names its node
    // first.
    let committed = Schedule::<Fp128>::compile(model)
        .expect("the model has a schedule")
        .committed();
    for p in &committed {
        if matches!(p, CommittedPoly::GlobalRemainderDigit(..)) {
            continue;
        }
        let name = format!("{p:?}");
        let (fam, args) = name.split_once('(').unwrap();
        let node: usize = args
            .split(|c: char| !c.is_ascii_digit())
            .next()
            .unwrap()
            .parse()
            .expect("a node polynomial names its node first");
        let padded = views[node].pow2_padded_num_output_elements();
        *by_op
            .entry((op_name(model, node), fam.to_string()))
            .or_default() += padded;
        node_rows += padded;
    }
    let bucket_rows: usize = by_fam.values().map(|f| f.rows).sum();
    let total_rows = bucket_rows + node_rows;
    println!("committed columns (polys, empty polys, rows=polys*cycles, nonzeros, share of rows):");
    for (fam, f) in &by_fam {
        println!(
            "  bucket {fam:<24} polys {:>4} empty {:>4} rows 2^{:<5.1} nonzeros 2^{:<5.1} {:>5.1}%",
            f.polys,
            f.empty,
            (f.rows.max(1) as f64).log2(),
            (f.nonzeros.max(1) as f64).log2(),
            100.0 * f.rows as f64 / total_rows as f64
        );
    }
    let mut rows: Vec<_> = by_op.iter().collect();
    rows.sort_by_key(|(_, r)| std::cmp::Reverse(**r));
    for ((op, fam), r) in rows {
        println!(
            "  node   {op:<10} {fam:<26} rows 2^{:<5.1} {:>5.1}%",
            (*r as f64).log2(),
            100.0 * *r as f64 / total_rows as f64
        );
    }
    println!(
        "  total rows 2^{:.1} = {} committed digit positions ({} cells at K=16; {} key bits)",
        (total_rows as f64).log2(),
        total_rows,
        total_rows * 16,
        total_rows * 4
    );

    // 2. Output audit: the width each range-checked output needs against the 32 bits its
    //    range claim commits (`out + 2^31`, 16 base-4 digits per element).
    let mut needed_hist: BTreeMap<(String, usize), usize> = BTreeMap::new();
    let mut digits_committed = 0usize;
    let mut digits_needed = 0usize;
    let mut elems_total = 0usize;
    for n in &views {
        if !jolt_atlas_core::onnx_proof::range_bucket::checks_its_output(n) {
            continue;
        }
        let out = trace.node_outputs[n.idx].data();
        let live = n.num_output_elements();
        let padded = n.pow2_padded_num_output_elements();
        let needed = needed_bits(&out[..live.min(out.len())]);
        elems_total += live;
        digits_committed += 16 * padded;
        digits_needed += needed.div_ceil(2) * padded;
        *needed_hist
            .entry((op_name(model, n.idx), needed))
            .or_default() += 1;
    }
    println!("range-checked outputs, needed width histogram (op, bits) -> nodes:");
    for ((op, bits), c) in &needed_hist {
        println!("  {op:<16} {bits:>2} bits  x{c}");
    }
    println!(
        "range-checked elements: {elems_total}; output digit rows 2^{:.1} (16 per element), at \
         the needed widths 2^{:.1} ({:.2}x)",
        (digits_committed as f64).log2(),
        (digits_needed as f64).log2(),
        digits_committed as f64 / digits_needed.max(1) as f64
    );

    // 3. Softmax: spread of the lookup address `c - x` over live lanes.
    println!("softmax rows: spread of (c - x) over live lanes:");
    let s = model.scale().log2() as usize;
    for n in &views {
        let Some(w) = trace.softmax_witness(n.idx) else {
            continue;
        };
        let rows = w.address.len() / w.last_dim;
        let mut hist = [0usize; 40];
        let mut max_d = 0i64;
        for k in 0..rows {
            for j in 0..w.active_len {
                let d = w.address[k * w.last_dim + j] as i64;
                max_d = max_d.max(d);
                hist[(64 - d.max(0).leading_zeros()) as usize] += 1;
            }
        }
        let lanes = rows * w.active_len;
        let over: usize = hist[s + 4..].iter().sum();
        println!(
            "  node {:<5} rows {:>5} live {:>4} max(c-x) 2^{:.1}  lanes with c-x >= 2^{} : {} ({:.2}%)",
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
                if committed.contains(&CommittedPoly::NodeOutputRaD(cn.idx, 0)) {
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

/// The model's nodes, the packed range buckets, and the largest nodes.
fn print_stats(model: &Model) {
    use jolt_atlas_core::onnx_proof::range_bucket::range_buckets;
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
    println!("range buckets (idx width log_t members | largest member):");
    for b in &range_buckets(model).expect("the model's range claims have bounds the buckets check")
    {
        let big = b.nodes.iter().max_by_key(|n| n.log_t).unwrap();
        println!(
            "  #{:<3} w={:<2} log_t={:<2} members={:<4} | node {} {} log_t={}",
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
