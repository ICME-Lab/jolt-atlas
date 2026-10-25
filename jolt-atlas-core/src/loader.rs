//! ONNX model loading for examples, tests, and benches: a thin panicking
//! convenience over [`atlas_onnx_import::load`].

use serde::{Deserialize, Serialize};

/// Load and compile an ONNX model, panicking on failure (examples and tests;
/// use [`atlas_onnx_import::load`] for typed errors).
pub fn load_model(path: &str, run_args: &RunArgs) -> atlas_trace::Model {
    load_imported(path, run_args).model
}

/// Like [`load_model`], but keeps the input descriptors and the pass log
/// (needed to synthesize inputs and to assert which importer rewrites fired).
pub fn load_imported(path: &str, run_args: &RunArgs) -> atlas_onnx_import::ImportedModel {
    atlas_onnx_import::load(
        std::path::Path::new(path),
        &atlas_onnx_import::LoadOptions {
            bindings: run_args.variables.clone(),
            scale: atlas_trace::Scale::new(run_args.scale).expect("scale in range"),
            pad_pow2: run_args.pad_to_power_of_2,
        },
    )
    .unwrap_or_else(|e| panic!("load_model({path}): {e}"))
}

/// Runtime arguments for loading a model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunArgs {
    /// Symbolic dim bindings (`batch_size`, `sequence_length`, …).
    pub variables: std::collections::HashMap<String, usize>,
    /// Global fixed-point scale (log2).
    pub scale: u8,
    /// Pad all dims to powers of two.
    pub pad_to_power_of_2: bool,
}

impl Default for RunArgs {
    fn default() -> Self {
        let mut variables = std::collections::HashMap::new();
        variables.insert("batch_size".to_string(), 1);
        RunArgs {
            variables,
            scale: common::consts::MODEL_SCALE as u8,
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
    pub fn set_scale(mut self, scale: u8) -> Self {
        self.scale = scale;
        self
    }
}
