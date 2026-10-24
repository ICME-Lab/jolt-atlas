//! Fuzz the full import pipeline on arbitrary bytes: protobuf decode →
//! graph build → normalize → analyze (shapes + folding) → DCE → input
//! classification → rewrite passes → lower. Mirrors `load_graph`, which is
//! path-based and so can't be fuzzed directly.
//!
//! `ImportError` returns are fine at every stage; panics, overflow aborts,
//! OOMs, and timeouts are findings.

#![no_main]

use std::collections::HashMap;
use std::path::Path;

use atlas_onnx_import::{analyze, graph, inputs, lower, normalize, passes, proto};
use atlas_trace::Scale;
use libfuzzer_sys::fuzz_target;
use prost::Message;

fuzz_target!(|data: &[u8]| {
    let Ok(model) = proto::ModelProto::decode(data) else {
        return;
    };

    let opset = model
        .opset_import
        .iter()
        .find(|o| o.domain.is_empty() || o.domain == "ai.onnx")
        .map(|o| o.version)
        .unwrap_or(13);

    // Bind the symbolic dims real models use so symbolic-shaped inputs get
    // past resolution instead of erroring out immediately.
    let bindings: HashMap<String, usize> =
        [("batch_size".to_string(), 1), ("sequence_length".to_string(), 8)].into();

    // A directory that doesn't exist: tensors using the external-data
    // convention fail with a clean Io error instead of reading local files.
    let model_dir = Path::new("/nonexistent-atlas-fuzz-model-dir");

    let Ok(mut g) = graph::from_proto(&model, model_dir, &bindings) else {
        return;
    };
    if normalize::normalize(&mut g, opset).is_err() {
        return;
    }
    if analyze::analyze(&mut g).is_err() {
        return;
    }
    analyze::dce(&mut g);
    let graph_inputs = g.inputs.clone();
    let Ok(descriptors) = inputs::classify_inputs(&g, &graph_inputs) else {
        return;
    };
    if passes::run_all(&mut g, Scale::new(14).expect("in range")).is_err() {
        return;
    }
    let scale = Scale::new(14).expect("in range");
    let _ = lower::lower(&g, scale, &descriptors);
});
