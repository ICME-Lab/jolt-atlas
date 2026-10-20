//! Print the lowered op inventory (op histogram + einsum equations) of a
//! model — the conformance view of what statement would be proven.
//!
//! ```sh
//! cargo run -p atlas-onnx-import --example dump_lowered_ops -- \
//!     atlas-onnx-tracer/models/gpt2/network.onnx batch_size=1 sequence_length=16 past_sequence_length=0
//! ```

use std::collections::BTreeMap;
use std::path::Path;

use atlas_onnx_import::LoadOptions;
use atlas_trace::Op;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1) else {
        eprintln!("usage: dump_lowered_ops <model.onnx> [key=value ...]");
        std::process::exit(1);
    };
    let mut opts = LoadOptions::default();
    for arg in &args[2..] {
        if let Some((k, v)) = arg.split_once('=') {
            opts.bindings
                .insert(k.to_string(), v.parse().expect("numeric binding"));
        }
    }

    let imported = match atlas_onnx_import::load(Path::new(path), &opts) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("import failed: {e}");
            std::process::exit(1);
        }
    };

    println!("== pass log ==");
    for r in &imported.pass_reports {
        if r.rewrites > 0 {
            println!("  {:<24} {:>6}", r.name, r.rewrites);
        }
    }

    println!("== inputs ==");
    for d in &imported.input_descriptors {
        println!("  {} {:?} {:?}", d.name, d.kind, d.dims);
    }

    let mut ops: BTreeMap<String, usize> = BTreeMap::new();
    let mut equations: BTreeMap<String, usize> = BTreeMap::new();
    for node in &imported.model.graph().nodes {
        let key = match &node.op {
            Op::Einsum(e) => {
                *equations.entry(e.equation.to_string()).or_default() += 1;
                "Einsum".to_string()
            }
            other => {
                // The display form may carry parameters; keep the base name.
                let s = other.to_string();
                s.split(['(', '{', '[']).next().unwrap_or(&s).to_string()
            }
        };
        *ops.entry(key).or_default() += 1;
    }

    println!(
        "== lowered ops ({} nodes) ==",
        imported.model.graph().nodes.len()
    );
    for (op, count) in &ops {
        println!("  {op:<24} x{count}");
    }
    if !equations.is_empty() {
        println!("== einsum equations ==");
        for (eq, count) in &equations {
            println!("  {eq:<24} x{count}");
        }
    }
}
