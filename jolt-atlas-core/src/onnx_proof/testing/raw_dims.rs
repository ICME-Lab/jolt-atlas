//! Compiling with padding keeps the raw dims: each model is compiled without
//! padding (whose dims are the shape-inference result) and its graph
//! recompiled with padding, and the two must expose the same dims.

use atlas_onnx_import::fixtures::standard_bindings;
use atlas_trace::{CompileOptions, Model};

use crate::loader::{load_imported, RunArgs};

/// The same graph compiled with `pad_pow2 = true`.
fn padded_twin(unpadded: &Model) -> Model {
    Model::compile(
        unpadded.graph().clone(),
        CompileOptions {
            scale: unpadded.scale(),
            pad_pow2: true,
            version: unpadded.version(),
        },
    )
    .expect("padded compile")
}

fn assert_raw_dims(name: &str, unpadded: &Model) {
    let padded = padded_twin(unpadded);
    for (i, &id) in unpadded.inputs().iter().enumerate() {
        assert_eq!(
            padded.raw_input_dims()[i],
            unpadded[id].raw_or_padded_outdims(),
            "{name}: input {i} dims"
        );
    }
    for (i, &id) in unpadded.outputs().iter().enumerate() {
        assert_eq!(
            padded.raw_output_dims()[i],
            unpadded[id].raw_or_padded_outdims(),
            "{name}: output {i} dims"
        );
    }
}

#[test]
fn zoo_models_keep_raw_dims() {
    for name in ["op_gauntlet", "gpt2-tiny", "qwen2-tiny", "bge-tiny"] {
        let unpadded = load_imported(
            &format!("../models/{name}/network.onnx"),
            &RunArgs::new(standard_bindings(16)).with_padding(false),
        )
        .model;
        assert_raw_dims(name, &unpadded);
    }
}
