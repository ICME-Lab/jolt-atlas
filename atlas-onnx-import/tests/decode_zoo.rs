//! Gate 1: every model in the repo zoo decodes and builds an importer graph
//! (bindings resolved, initializers attached, topological order derivable).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use atlas_onnx_import::{graph, proto};

fn zoo_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../atlas-onnx-tracer/models")
}

fn standard_bindings() -> HashMap<String, usize> {
    HashMap::from([
        ("batch_size".to_string(), 1),
        ("sequence_length".to_string(), 16),
        ("past_sequence_length".to_string(), 0),
        // A few models use other names.
        ("seq_len".to_string(), 16),
        ("N".to_string(), 1),
    ])
}

fn decode_one(dir: &Path) -> Result<(usize, usize), String> {
    let path = dir.join("network.onnx");
    let model = proto::load_model(&path).map_err(|e| e.to_string())?;
    let g = graph::from_proto(&model, dir, &standard_bindings()).map_err(|e| e.to_string())?;
    let order = g.topo_order().map_err(|e| e.to_string())?;
    Ok((order.len(), g.edges.len()))
}

/// Small models run in default CI.
#[test]
fn decode_small_models() {
    for name in [
        "perceptron",
        "mlp_square",
        "tanh",
        "erf",
        "gather",
        "slice",
        "reshape",
        "broadcast",
        "moveaxis",
        "rsqrt",
        "sum_2d_axis1",
        "test_add_sub_mul",
        "layernorm_head",
        "self_attention_layer",
        "multihead_attention",
        "transformer",
        "nanoGPT",
    ] {
        let dir = zoo_dir().join(name);
        if !dir.exists() {
            continue;
        }
        match decode_one(&dir) {
            Ok((nodes, edges)) => println!("{name}: {nodes} nodes, {edges} edges"),
            Err(e) => panic!("{name}: {e}"),
        }
    }
}

/// The full zoo, including the multi-hundred-MB models; run with
/// `cargo test -p atlas-onnx-import --test decode_zoo -- --ignored --nocapture`.
#[test]
#[ignore]
fn decode_full_zoo() {
    let mut failures = vec![];
    for entry in std::fs::read_dir(zoo_dir()).expect("zoo dir") {
        let dir = entry.expect("entry").path();
        if !dir.join("network.onnx").exists() {
            continue;
        }
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        match decode_one(&dir) {
            Ok((nodes, edges)) => println!("OK   {name}: {nodes} nodes, {edges} edges"),
            Err(e) => {
                println!("FAIL {name}: {e}");
                failures.push(name);
            }
        }
    }
    assert!(failures.is_empty(), "failed: {failures:?}");
}

/// Print the union of op types across the zoo (run with --ignored).
#[test]
#[ignore]
fn op_surface() {
    let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
    for entry in std::fs::read_dir(zoo_dir()).expect("zoo dir") {
        let dir = entry.expect("entry").path();
        if !dir.join("network.onnx").exists() {
            continue;
        }
        let model = proto::load_model(&dir.join("network.onnx")).unwrap();
        for node in &model.graph.as_ref().unwrap().node {
            *counts.entry(node.op_type.clone()).or_default() += 1;
        }
    }
    for (op, n) in counts {
        println!("{op}: {n}");
    }
}

/// Gate 2: every zoo model normalizes and fully shape-infers/folds.
#[test]
#[ignore]
fn analyze_full_zoo() {
    use atlas_onnx_import::{analyze, normalize};
    let mut failures = vec![];
    for entry in std::fs::read_dir(zoo_dir()).expect("zoo dir") {
        let dir = entry.expect("entry").path();
        if !dir.join("network.onnx").exists() {
            continue;
        }
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        let model = proto::load_model(&dir.join("network.onnx")).unwrap();
        let opset = model
            .opset_import
            .iter()
            .find(|o| o.domain.is_empty() || o.domain == "ai.onnx")
            .map(|o| o.version)
            .unwrap_or(13);
        let result = (|| -> Result<usize, String> {
            let mut g = graph::from_proto(&model, &dir, &standard_bindings())
                .map_err(|e| e.to_string())?;
            normalize::normalize(&mut g, opset).map_err(|e| e.to_string())?;
            analyze::analyze(&mut g).map_err(|e| e.to_string())?;
            analyze::dce(&mut g);
            // Every remaining edge referenced by a live node must have info.
            for (_, n) in g.live_nodes() {
                for &e in n.inputs.iter().chain(&n.outputs) {
                    if g.info(e).is_none() {
                        return Err(format!("edge {} has no shape", g.edge(e).name));
                    }
                }
            }
            Ok(g.live_nodes().count())
        })();
        match result {
            Ok(n) => println!("OK   {name}: {n} live nodes after analysis"),
            Err(e) => {
                println!("FAIL {name}: {e}");
                failures.push(name);
            }
        }
    }
    assert!(failures.is_empty(), "failed: {failures:?}");
}

/// Gate 3: end-to-end import (decode → passes → lower → compile).
///
/// Two fixtures are EXPECTED to be rejected: their padded execution cannot
/// be made coherent (`slice` extracts a non-power-of-two extent;
/// `concat_transformer_block` concatenates extents whose paddings do not
/// add up). The old crate accepted both and silently produced trace tensors
/// whose dims disagree with the declared padded dims — the same latent bug
/// class as its qwen repeat_kv scramble. The new pipeline fails loudly.
#[test]
#[ignore]
fn import_full_zoo() {
    const EXPECTED_REJECTED: &[&str] = &["slice", "concat_transformer_block"];
    let mut failures = vec![];
    for entry in std::fs::read_dir(zoo_dir()).expect("zoo dir") {
        let dir = entry.expect("entry").path();
        if !dir.join("network.onnx").exists() {
            continue;
        }
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        let opts = atlas_onnx_import::LoadOptions {
            bindings: standard_bindings(),
            ..Default::default()
        };
        let expected_reject = EXPECTED_REJECTED.contains(&name.as_str());
        match atlas_onnx_import::load(&dir.join("network.onnx"), &opts) {
            Ok(m) if expected_reject => {
                println!("UNEXPECTED OK {name}: {} nodes — update the test", m.model.graph().nodes.len());
                failures.push(name);
            }
            Ok(m) => {
                let n = m.model.graph().nodes.len();
                println!("OK   {name}: {n} atlas nodes");
            }
            Err(e) if expected_reject => {
                println!("REJ  {name} (expected): {e}");
            }
            Err(e) => {
                println!("FAIL {name}: {e}");
                failures.push(name);
            }
        }
    }
    assert!(failures.is_empty(), "failed: {failures:?}");
}

/// Debug: print the producer chain feeding gpt2's QKV split.
#[test]
#[ignore]
fn debug_gpt2_split_chain() {
    use atlas_onnx_import::{analyze, normalize, passes};
    let dir = zoo_dir().join("gpt2");
    let model = proto::load_model(&dir.join("network.onnx")).unwrap();
    let opset = model.opset_import.iter().find(|o| o.domain.is_empty()).map(|o| o.version).unwrap_or(13);
    let mut g = graph::from_proto(&model, &dir, &standard_bindings()).unwrap();
    normalize::normalize(&mut g, opset).unwrap();
    analyze::analyze(&mut g).unwrap();
    analyze::dce(&mut g);
    passes::run_all(&mut g).unwrap();
    // Find a non-pow2 atlas.Slice and walk its producers.
    for (_, n) in g.live_nodes() {
        if n.op == "atlas.Slice" {
            let (s, e) = (n.attr_int("start").unwrap(), n.attr_int("end").unwrap());
            if !((e - s) as usize).is_power_of_two() {
                println!("slice {} {}..{} axis {:?}", n.name, s, e, n.attr_int("axis"));
                let mut edge = n.inputs[0];
                for _ in 0..6 {
                    match g.producer(edge) {
                        Some((_, p)) => {
                            println!("  <- {} ({}) inputs {:?}", p.name, p.op,
                                p.inputs.iter().map(|&i| {
                                    let e = g.edge(i);
                                    format!("{}{}", e.name, if e.konst.is_some() {"[const]"} else {""})
                                }).collect::<Vec<_>>());
                            edge = p.inputs[0];
                        }
                        None => { println!("  <- (input/const edge {})", g.edge(edge).name); break; }
                    }
                }
                break;
            }
        }
    }
}

/// Debug: print the consumer chains of int inputs for a model.
#[test]
#[ignore]
fn debug_mask_chain() {
    use atlas_onnx_import::{analyze, normalize};
    let model_name = std::env::var("MODEL").unwrap_or_else(|_| "gpt2".into());
    let dir = zoo_dir().join(&model_name);
    let model = proto::load_model(&dir.join("network.onnx")).unwrap();
    let opset = model.opset_import.iter().find(|o| o.domain.is_empty()).map(|o| o.version).unwrap_or(13);
    let mut g = graph::from_proto(&model, &dir, &standard_bindings()).unwrap();
    normalize::normalize(&mut g, opset).unwrap();
    analyze::analyze(&mut g).unwrap();
    analyze::dce(&mut g);
    for &input in g.inputs.clone().iter() {
        let name = g.edge(input).name.clone();
        if !name.contains("mask") && !name.contains("position") { continue; }
        println!("== {model_name} input {name}");
        // BFS consumers to depth 5.
        let mut frontier = vec![(input, 0usize)];
        while let Some((e, depth)) = frontier.pop() {
            if depth > 4 { continue; }
            for (_, n) in g.live_nodes() {
                if n.inputs.contains(&e) {
                    println!("{:indent$}-> {} ({})", "", n.name, n.op, indent = depth * 2);
                    for &o in &n.outputs { frontier.push((o, depth + 1)); }
                }
            }
        }
    }
}
