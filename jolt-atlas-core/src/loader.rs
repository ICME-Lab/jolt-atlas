//! ONNX model loading for examples, tests, and benches: a thin panicking
//! convenience over [`atlas_onnx_import::load`].

use serde::{Deserialize, Serialize};

/// Runtime arguments for loading a model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunArgs {
    /// Symbolic dim bindings (`batch_size`, `sequence_length`, …).
    pub variables: std::collections::HashMap<String, usize>,
    /// Global fixed-point scale (log2).
    pub scale: i32,
    /// Pad all dims to powers of two.
    pub pad_to_power_of_2: bool,
}

impl Default for RunArgs {
    fn default() -> Self {
        let mut variables = std::collections::HashMap::new();
        variables.insert("batch_size".to_string(), 1);
        RunArgs {
            variables,
            scale: common::consts::MODEL_SCALE as i32,
            pad_to_power_of_2: true,
        }
    }
}

impl RunArgs {
    /// RunArgs with the given symbolic dim bindings.
    pub fn new<K: Into<String>>(bindings: impl IntoIterator<Item = (K, usize)>) -> Self {
        let mut args = RunArgs::default();
        args.variables
            .extend(bindings.into_iter().map(|(k, v)| (k.into(), v)));
        args
    }

    /// Toggle power-of-two padding.
    pub fn with_padding(mut self, pad: bool) -> Self {
        self.pad_to_power_of_2 = pad;
        self
    }

    /// Set the fixed-point scale (log2).
    pub fn set_scale(mut self, scale: i32) -> Self {
        self.scale = scale;
        self
    }
}

/// Load and compile an ONNX model, panicking on failure (examples and tests;
/// use [`atlas_onnx_import::load`] for typed errors).
pub fn load_model(path: &str, run_args: &RunArgs) -> atlas_trace::Model {
    let imported = atlas_onnx_import::load(
        std::path::Path::new(path),
        &atlas_onnx_import::LoadOptions {
            bindings: run_args.variables.clone(),
            scale: atlas_trace::Scale::new(run_args.scale as u8).expect("scale in range"),
            pad_pow2: run_args.pad_to_power_of_2,
        },
    )
    .unwrap_or_else(|e| panic!("load_model({path}): {e}"));
    imported.model
}
