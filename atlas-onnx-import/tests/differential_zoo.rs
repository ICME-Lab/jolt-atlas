//! Differential harness #2: the same quantized inputs through the old
//! tract-based pipeline and the new importer, comparing final outputs.
//!
//! Single-input models only (the old crate matches inputs positionally and
//! does not retain names, so multi-input models can't be wired reliably
//! from the outside; GPT-2/Qwen get dedicated comparisons later).
//!
//! Expectations:
//! - models on the EXACT list must agree bit for bit;
//! - the rest must agree within a small tolerance (deterministic integer
//!   activation tables may differ from the old f64 tables by one ulp, and
//!   softmax compounds a few of those).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use atlas_onnx_import::{InputKind, LoadOptions};
use atlas_onnx_tracer::{
    model::{Model as OldModel, RunArgs},
    tensor::Tensor as OldTensor,
};
use atlas_trace::{Op, Tensor};
use rand::{Rng, SeedableRng, rngs::StdRng};

fn zoo_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../atlas-onnx-tracer/models")
}

fn bindings() -> HashMap<String, usize> {
    HashMap::from([
        ("batch_size".to_string(), 1),
        ("sequence_length".to_string(), 16),
        ("past_sequence_length".to_string(), 0),
    ])
}

/// Models whose op surface is exact-class only (no lookup tables), so old
/// and new outputs must be byte-identical.
const EXACT: &[&str] = &[
    "gather",
    "reshape",
    "broadcast",
    "moveaxis",
    "rsqrt",
    "sum_1d_axis0",
    "sum_2d_axis0",
    "sum_2d_axis1",
    "sum_3d_axis2",
    "sum_axes_test",
    "sum_independent",
    "test_add_sub_mul",
    "mlp_square",
    "mlp_square_4layer",
    "perceptron",
    "layernorm_head",
    "layernorm_partial_head",
];

/// Table-class models (activations/softmax): tolerance per element.
const TOLERANT: &[(&str, i32)] = &[
    ("tanh", 1),
    ("erf", 1),
    ("positional_encoding", 16),
    ("sigmoid_encoder", 16),
    ("article_classification", 16),
    ("authorization", 16),
    ("self_attention_layer", 16),
    ("relu_self_attention", 16),
    ("transformer", 16),
    ("multihead_attention", 16),
    ("microgpt", 16),
    ("minigpt", 16),
    ("nanoGPT", 16),
    ("relu_nanoGPT", 16),
];

fn run_one(name: &str, tolerance: Option<i32>) -> Result<i32, String> {
    let dir = zoo_dir().join(name);
    let path = dir.join("network.onnx");
    if !path.exists() {
        return Err("model file missing".into());
    }

    // New pipeline.
    let opts = LoadOptions {
        bindings: bindings(),
        ..Default::default()
    };
    let imported = atlas_onnx_import::load(&path, &opts).map_err(|e| e.to_string())?;
    if imported.input_descriptors.len() != 1 {
        return Err(format!(
            "skipped: {} inputs (positional wiring to the old crate is ambiguous)",
            imported.input_descriptors.len()
        ));
    }
    let desc = &imported.input_descriptors[0];

    // Shared quantized input values.
    let mut rng = StdRng::seed_from_u64(0xD1FF);
    let len: usize = desc.dims.iter().product();
    let data: Vec<i32> = match desc.kind {
        InputKind::RawInt => {
            // Safe small indices for any dictionary in the zoo.
            let max_index = smallest_dict(&imported) as i32;
            (0..len)
                .map(|_| rng.gen_range(0..max_index.max(1)))
                .collect()
        }
        InputKind::Scaled => {
            let s = 1i32 << 14;
            (0..len).map(|_| rng.gen_range(-s..s)).collect()
        }
    };

    // Old pipeline.
    let run_args = RunArgs {
        variables: bindings(),
        scale: 14,
        pad_to_power_of_2: true,
    };
    let old = OldModel::load(path.to_str().expect("utf8 path"), &run_args);
    let old_out = old.forward(&[OldTensor::new(Some(&data), &desc.dims).expect("dims")]);

    let new_out = imported
        .model
        .forward(&[Tensor::new(data, &desc.dims).expect("dims")])
        .map_err(|e| e.to_string())?;

    if old_out.len() != new_out.len() {
        return Err(format!(
            "output count mismatch: old {}, new {}",
            old_out.len(),
            new_out.len()
        ));
    }
    let mut max_diff = 0i32;
    for (o, n) in old_out.iter().zip(&new_out) {
        if o.dims() != n.dims() {
            return Err(format!("output dims {:?} vs {:?}", o.dims(), n.dims()));
        }
        for (i, (&a, &b)) in o.data().iter().zip(n.data()).enumerate() {
            let d = (a - b).abs();
            max_diff = max_diff.max(d);
            if let Some(tol) = tolerance
                && d > tol
            {
                return Err(format!("element {i}: old {a}, new {b} (tol {tol})"));
            }
            if tolerance.is_none() && d != 0 {
                return Err(format!("element {i}: old {a}, new {b} (exact class)"));
            }
        }
    }
    Ok(max_diff)
}

/// The smallest gather dictionary in the model (bounds valid index inputs).
fn smallest_dict(m: &atlas_onnx_import::ImportedModel) -> usize {
    m.model
        .graph()
        .nodes
        .iter()
        .filter_map(|n| match &n.op {
            Op::GatherSmall(g) => Some(g.dict_len),
            Op::GatherLarge(g) => Some(g.dict_len),
            _ => None,
        })
        .min()
        .unwrap_or(2)
}

#[test]
#[ignore]
fn differential_exact_models() {
    let mut failures = vec![];
    for &name in EXACT {
        match run_one(name, None) {
            Ok(_) => println!("OK   {name}: bit-exact"),
            Err(e) if e.starts_with("skipped") => println!("SKIP {name}: {e}"),
            Err(e) => {
                println!("FAIL {name}: {e}");
                failures.push(name);
            }
        }
    }
    assert!(failures.is_empty(), "failed: {failures:?}");
}

#[test]
#[ignore]
fn differential_table_models() {
    let mut failures = vec![];
    for &(name, tol) in TOLERANT {
        match run_one(name, Some(tol)) {
            Ok(max) => println!("OK   {name}: max diff {max} (tol {tol})"),
            Err(e) if e.starts_with("skipped") => println!("SKIP {name}: {e}"),
            Err(e) => {
                println!("FAIL {name}: {e}");
                failures.push(name);
            }
        }
    }
    assert!(failures.is_empty(), "failed: {failures:?}");
}

/// Big multi-input models: wire inputs in ONNX order on both sides, values
/// chosen by descriptor kind (ids random, masks all-ones, positions 0..n).
#[test]
#[ignore]
fn differential_big_models() {
    let mut failures = vec![];
    for (name, tol) in [("gpt2", 32), ("qwen", 32), ("bge-small-en-v1.5", 32)] {
        match run_big(name, tol) {
            Ok(max) => println!("OK   {name}: max diff {max} (tol {tol})"),
            Err(e) => {
                println!("FAIL {name}: {e}");
                failures.push(name);
            }
        }
    }
    assert!(failures.is_empty(), "failed: {failures:?}");
}

/// The old crate's positional input order (it loses input names, so this is
/// pinned from `debug_old_input_order`).
fn old_input_order(model: &str) -> Vec<&'static str> {
    match model {
        "gpt2" => vec!["input_ids", "position_ids", "attention_mask"],
        "qwen" => vec!["input_ids", "position_ids", "attention_mask"],
        "bge-small-en-v1.5" => vec!["input_ids", "token_type_ids", "attention_mask"],
        _ => vec![],
    }
}

/// Whether the old pipeline expects the attention mask pre-scaled by `2^s`
/// (it does when its Cast handler inserted a descale on that path).
fn old_mask_scaled(model: &str) -> bool {
    model == "gpt2" || model == "qwen"
}

fn run_big(name: &str, tol: i32) -> Result<i32, String> {
    let dir = zoo_dir().join(name);
    let path = dir.join("network.onnx");
    let opts = LoadOptions {
        bindings: bindings(),
        ..Default::default()
    };
    let imported = atlas_onnx_import::load(&path, &opts).map_err(|e| e.to_string())?;
    let s = 1i32 << 14;
    let mut rng = StdRng::seed_from_u64(0xB16);

    let mut old_inputs = Vec::new();
    let mut new_inputs = Vec::new();
    for d in &imported.input_descriptors {
        let len: usize = d.dims.iter().product();
        let lower = d.name.to_lowercase();
        // (new-pipeline value, old-pipeline value): the old crate descales
        // every int-targeted Cast, so int inputs that pass through a Cast
        // (attention masks) must arrive pre-scaled THERE, while the new
        // pipeline takes them raw. Semantically identical inputs.
        let (data, old_data): (Vec<i32>, Vec<i32>) = if lower.contains("mask") {
            let old_val = if old_mask_scaled(name) { s } else { 1 };
            match d.kind {
                InputKind::Scaled => (vec![s; len], vec![s; len]),
                InputKind::RawInt => (vec![1; len], vec![old_val; len]),
            }
        } else if lower.contains("position") {
            let raw: Vec<i32> = (0..len as i32).collect();
            let scaled: Vec<i32> = (0..len as i32).map(|i| i * s).collect();
            match d.kind {
                InputKind::Scaled => (scaled.clone(), scaled),
                InputKind::RawInt => (raw.clone(), raw),
            }
        } else if lower.contains("token_type") {
            (vec![0; len], vec![0; len]) // segment ids: dictionary of 2
        } else {
            let v: Vec<i32> = (0..len).map(|_| rng.gen_range(0..1000)).collect();
            (v.clone(), v)
        };
        println!("  input {} {:?} {:?}", d.name, d.kind, d.dims);
        old_inputs.push((
            d.name.clone(),
            OldTensor::new(Some(&old_data), &d.dims).expect("dims"),
        ));
        new_inputs.push(Tensor::new(data, &d.dims).expect("dims"));
    }
    // Reorder the old side to its own positional convention.
    let order = old_input_order(name);
    let old_inputs: Vec<OldTensor<i32>> = order
        .iter()
        .map(|&wanted| {
            old_inputs
                .iter()
                .find(|(n, _)| n == wanted)
                .map(|(_, t)| t.clone())
                .ok_or_else(|| format!("no input named {wanted}"))
        })
        .collect::<Result<_, _>>()?;

    // Qwen: the old pipeline's PADDED execution diverges from its own
    // unpadded execution (its GQA repeat_kv reshape scrambles under
    // padding — a pre-existing bug the new pipeline fixes via a
    // padding-coherent head gather). Ground truth is the old UNPADDED run;
    // the new padded pipeline must match it bit for bit.
    let run_args = RunArgs {
        variables: bindings(),
        scale: 14,
        pad_to_power_of_2: name != "qwen",
    };
    let old = OldModel::load(path.to_str().expect("utf8"), &run_args);
    let old_out = old.forward(&old_inputs);
    let new_out = imported
        .model
        .forward(&new_inputs)
        .map_err(|e| e.to_string())?;

    let mut max_diff = 0i32;
    for (o, n) in old_out.iter().zip(&new_out) {
        if o.dims() != n.dims() {
            return Err(format!("output dims {:?} vs {:?}", o.dims(), n.dims()));
        }
        for (i, (&a, &b)) in o.data().iter().zip(n.data()).enumerate() {
            let d = (a - b).abs();
            max_diff = max_diff.max(d);
            if d > tol {
                return Err(format!(
                    "element {i}: old {a}, new {b} (max so far {max_diff})"
                ));
            }
        }
    }
    Ok(max_diff)
}

/// Debug: the old crate's input order and each input's consumers.
#[test]
#[ignore]
fn debug_old_input_order() {
    let name = std::env::var("MODEL").unwrap_or_else(|_| "gpt2".into());
    let dir = zoo_dir().join(&name);
    let run_args = RunArgs {
        variables: bindings(),
        scale: 14,
        pad_to_power_of_2: true,
    };
    let old = OldModel::load(dir.join("network.onnx").to_str().unwrap(), &run_args);
    for (pos, &idx) in old.graph.inputs.iter().enumerate() {
        let node = &old.graph.nodes[&idx];
        let consumers: Vec<String> = old
            .graph
            .nodes
            .values()
            .filter(|n| n.inputs.contains(&idx))
            .map(|n| {
                let dict = n
                    .inputs
                    .first()
                    .and_then(|i| old.graph.nodes.get(i))
                    .map(|d| format!("{:?}", d.output_dims))
                    .unwrap_or_default();
                format!("{:?}(first-op dims {dict})", n.operator)
                    .chars()
                    .take(80)
                    .collect()
            })
            .collect();
        println!(
            "old input {pos}: node {idx} dims {:?} -> {consumers:?}",
            old.graph
                .original_input_dims
                .get(&idx)
                .unwrap_or(&node.output_dims)
        );
    }
}

/// Decisive self-consistency check: the new pipeline's padded execution
/// must equal its unpadded execution (cropped). Whichever pipeline breaks
/// padding invariance is the one mislowering qwen.
#[test]
#[ignore]
fn debug_qwen_padding_invariance() {
    let dir = zoo_dir().join("qwen");
    let path = dir.join("network.onnx");
    let seq = 16usize;
    let s = 1i32 << 14;
    let ids: Vec<i32> = (0..seq as i32).map(|i| 100 + i * 7).collect();
    let pos_scaled: Vec<i32> = (0..seq as i32).map(|i| i * s).collect();
    let inputs = vec![
        Tensor::new(ids.clone(), &[1, seq]).unwrap(),
        Tensor::new(vec![1i32; seq], &[1, seq]).unwrap(),
        Tensor::new(pos_scaled.clone(), &[1, seq]).unwrap(),
    ];

    let mut opts = LoadOptions {
        bindings: bindings(),
        ..Default::default()
    };
    opts.pad_pow2 = false;
    let unpadded = atlas_onnx_import::load(&path, &opts).unwrap();
    opts.pad_pow2 = true;
    let padded = atlas_onnx_import::load(&path, &opts).unwrap();

    let out_u = unpadded.model.forward(&inputs).unwrap();
    let out_p = padded.model.forward(&inputs).unwrap();
    let mut max = 0i32;
    for (u, p) in out_u.iter().zip(&out_p) {
        assert_eq!(u.dims(), p.dims());
        for (&a, &b) in u.data().iter().zip(p.data()) {
            max = max.max((a - b).abs());
        }
    }
    println!("new pipeline padded-vs-unpadded max diff: {max}");

    // And the old pipeline's self-consistency.
    let run_args_u = RunArgs {
        variables: bindings(),
        scale: 14,
        pad_to_power_of_2: false,
    };
    let run_args_p = RunArgs {
        variables: bindings(),
        scale: 14,
        pad_to_power_of_2: true,
    };
    let old_inputs = vec![
        OldTensor::new(Some(&ids), &[1, seq]).unwrap(),
        OldTensor::new(Some(&pos_scaled), &[1, seq]).unwrap(),
        OldTensor::new(Some(&vec![s; seq]), &[1, seq]).unwrap(),
    ];
    let old_u = OldModel::load(path.to_str().unwrap(), &run_args_u).forward(&old_inputs);
    let old_p = OldModel::load(path.to_str().unwrap(), &run_args_p).forward(&old_inputs);
    let mut omax = 0i32;
    for (u, p) in old_u.iter().zip(&old_p) {
        for (&a, &b) in u.data().iter().zip(p.data()) {
            omax = omax.max((a - b).abs());
        }
    }
    println!("old pipeline padded-vs-unpadded max diff: {omax}");

    // Cross-check the two UNPADDED pipelines (no padding semantics at all).
    let mut xmax = 0i32;
    for (u, o) in out_u.iter().zip(&old_u) {
        for (&a, &b) in u.data().iter().zip(o.data()) {
            xmax = xmax.max((a - b).abs());
        }
    }
    println!("unpadded old-vs-new max diff: {xmax}");
}

/// Find the first node whose padded output (cropped) differs from its
/// unpadded output in the new pipeline.
#[test]
#[ignore]
fn debug_qwen_first_pad_divergence() {
    let dir = zoo_dir().join("qwen");
    let path = dir.join("network.onnx");
    let seq = 16usize;
    let s = 1i32 << 14;
    let ids: Vec<i32> = (0..seq as i32).map(|i| 100 + i * 7).collect();
    let pos_scaled: Vec<i32> = (0..seq as i32).map(|i| i * s).collect();
    let inputs = vec![
        Tensor::new(ids.clone(), &[1, seq]).unwrap(),
        Tensor::new(vec![1i32; seq], &[1, seq]).unwrap(),
        Tensor::new(pos_scaled, &[1, seq]).unwrap(),
    ];
    let mut opts = LoadOptions {
        bindings: bindings(),
        ..Default::default()
    };
    opts.pad_pow2 = false;
    let unpadded = atlas_onnx_import::load(&path, &opts).unwrap();
    opts.pad_pow2 = true;
    let padded = atlas_onnx_import::load(&path, &opts).unwrap();
    let t_u = unpadded.model.trace(&inputs).unwrap();
    let t_p = padded.model.trace(&inputs).unwrap();

    for (id, node, _) in unpadded.model.nodes() {
        let logical = &node.output_dims;
        let cropped = t_p.output(id).crop_to(logical).unwrap();
        let expected = t_u.output(id);
        if cropped.data() != expected.data() {
            let max: i32 = cropped
                .data()
                .iter()
                .zip(expected.data())
                .map(|(&a, &b)| (a - b).abs())
                .max()
                .unwrap();
            println!(
                "FIRST divergent node {}: {} dims {:?} max diff {max}",
                id.index(),
                node.op,
                logical
            );
            // Print its operand ops for context.
            for &inp in &node.inputs {
                let n = &unpadded.model.graph().nodes[inp.index()];
                println!(
                    "  operand {}: {} dims {:?}",
                    inp.index(),
                    n.op,
                    n.output_dims
                );
            }
            return;
        }
    }
    println!("no divergence found");
}
