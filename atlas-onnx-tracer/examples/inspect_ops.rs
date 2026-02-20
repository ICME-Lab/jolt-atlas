use atlas_onnx_tracer::{
    model::{Model, RunArgs},
    node::handlers::HANDLERS,
};
use std::collections::{BTreeMap, BTreeSet};

/// Inspects an ONNX model and reports which ops it requires vs. which ops
/// atlas-onnx-tracer currently supports.
///
/// # Usage
///
/// ```sh
/// # Simple model (no dynamic dimensions):
/// cargo run --example inspect_ops -- atlas-onnx-tracer/models/perceptron/network.onnx
///
/// # Model with dynamic dimensions (pass key=value pairs):
/// cargo run --example inspect_ops -- atlas-onnx-tracer/models/gpt2/model.onnx batch_size=1 sequence_length=16
/// ```
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.onnx> [key=value ...]", args[0]);
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  {} models/perceptron/network.onnx", args[0]);
        eprintln!(
            "  {} models/gpt2/model.onnx batch_size=1 sequence_length=16",
            args[0]
        );
        std::process::exit(1);
    }

    let model_path = &args[1];

    // Parse optional key=value variable pairs from remaining args
    let mut run_args = RunArgs::default();
    for arg in &args[2..] {
        let parts: Vec<&str> = arg.splitn(2, '=').collect();
        if parts.len() == 2 {
            let key = parts[0].to_string();
            let value: usize = parts[1]
                .parse()
                .unwrap_or_else(|_| panic!("Invalid value for variable '{key}': {}", parts[1]));
            run_args.variables.insert(key, value);
        } else {
            eprintln!("Warning: ignoring argument '{arg}' (expected key=value)");
        }
    }

    // Load the model at the Tract level only (no atlas parsing)
    let (graph, _symbols) = Model::load_onnx_using_tract(model_path, &run_args);

    // Collect all op names used by the model, with counts
    let mut op_counts: BTreeMap<String, usize> = BTreeMap::new();
    for node in &graph.nodes {
        let name = node.op().name().to_string();
        *op_counts.entry(name).or_insert(0) += 1;
    }

    // Classify ops as supported or unsupported
    let supported_handlers: BTreeSet<&str> = HANDLERS.keys().copied().collect();

    let mut supported: BTreeMap<&str, usize> = BTreeMap::new();
    let mut unsupported: BTreeMap<&str, usize> = BTreeMap::new();

    for (op_name, count) in &op_counts {
        if supported_handlers.contains(op_name.as_str()) {
            supported.insert(op_name, *count);
        } else {
            unsupported.insert(op_name, *count);
        }
    }

    // Print results
    let total_nodes = graph.nodes.len();
    let total_unique_ops = op_counts.len();

    println!("Model: {model_path}");
    println!("Total nodes: {total_nodes}");
    println!("Unique ops:  {total_unique_ops}");
    println!();

    println!(
        "Supported ops ({}/{} unique):",
        supported.len(),
        total_unique_ops
    );
    for (op, count) in &supported {
        println!("  {op:<30} x{count}");
    }

    println!();
    println!(
        "Unsupported ops ({}/{} unique):",
        unsupported.len(),
        total_unique_ops
    );
    if unsupported.is_empty() {
        println!("  (none — all ops are supported!)");
    } else {
        for (op, count) in &unsupported {
            println!("  {op:<30} x{count}");
        }
    }

    println!();
    println!("All registered handlers ({}):", supported_handlers.len());
    for handler in &supported_handlers {
        let status = if op_counts.contains_key(*handler) {
            "used"
        } else {
            "    "
        };
        println!("  [{status}] {handler}");
    }
}
