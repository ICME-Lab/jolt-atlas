//! # atlas-onnx-import
//!
//! Tract-free ONNX frontend for `atlas-trace`: protobuf decode (prost,
//! vendored proto, checked-in codegen), static shape inference with
//! user-supplied dim bindings, whitelist constant folding, a fixed pipeline
//! of hand-written rewrite passes, weight quantization, and lowering to the
//! `atlas-trace` IR.

#![warn(missing_docs)]

pub mod analyze;
pub mod error;
#[cfg(any(test, feature = "test-util"))]
pub mod fixtures;
pub mod fold_fns;
pub mod graph;
pub mod inputs;
pub mod lower;
pub mod normalize;
pub mod passes;
pub mod proto;
pub mod quantize;
pub mod shape_fns;

use std::{collections::HashMap, path::Path};

use atlas_trace::{CompileOptions, Model, Scale};

/// Fallback ONNX opset version used when the model doesn't declare one.
const DEFAULT_ONNX_OPSET: i64 = 13;

pub use error::ImportError;
pub use inputs::{InputDescriptor, InputKind};
pub use lower::Lowered;
pub use passes::PassReport;

/// Options for [`load`].
#[derive(Clone, Debug)]
pub struct LoadOptions {
    /// Values for symbolic input dims (`batch_size`, `sequence_length`, …).
    pub bindings: HashMap<String, usize>,
    /// The global fixed-point scale.
    pub scale: Scale,
    /// Pad all dims to powers of two (what the prover needs).
    pub pad_pow2: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions {
            bindings: HashMap::new(),
            scale: Scale::new(14).expect("in range"),
            pad_pow2: true,
        }
    }
}

/// An imported, compiled model plus its input descriptors and the pass log.
pub struct ImportedModel {
    /// The compiled atlas-trace model, ready to trace.
    pub model: Model,
    /// How to quantize each input, in input order.
    pub input_descriptors: Vec<InputDescriptor>,
    /// Rewrites applied per pass (diagnostics).
    pub pass_reports: Vec<PassReport>,
}

/// Import an ONNX file end to end: decode → normalize → analyze (shapes +
/// constant folding) → rewrite passes → lower → compile.
pub fn load(path: &Path, opts: &LoadOptions) -> Result<ImportedModel, ImportError> {
    let lowered = load_graph(path, opts)?;
    let model = Model::compile(
        lowered.graph,
        CompileOptions {
            scale: opts.scale,
            pad_pow2: opts.pad_pow2,
            version: atlas_trace::SemVersion::V1,
        },
    )?;
    Ok(ImportedModel {
        model,
        input_descriptors: lowered.input_descriptors,
        pass_reports: lowered.pass_reports,
    })
}

/// A lowered (uncompiled) graph plus diagnostics; useful for golden tests.
pub struct LoweredGraph {
    /// The atlas-trace graph.
    pub graph: atlas_trace::Graph,
    /// Input descriptors.
    pub input_descriptors: Vec<InputDescriptor>,
    /// Pass log.
    pub pass_reports: Vec<PassReport>,
}

/// Import an ONNX file up to (but not including) compilation.
pub fn load_graph(path: &Path, opts: &LoadOptions) -> Result<LoweredGraph, ImportError> {
    let model_dir = path.parent().unwrap_or(Path::new("."));
    let proto_model = proto::load_model(path)?;
    let opset = proto_model
        .opset_import
        .iter()
        .find(|o| o.domain.is_empty() || o.domain == "ai.onnx")
        .map(|o| o.version)
        .unwrap_or(DEFAULT_ONNX_OPSET);

    let mut g = graph::from_proto(&proto_model, model_dir, &opts.bindings)?;
    normalize::normalize(&mut g, opset)?;
    analyze::analyze(&mut g)?;
    analyze::dce(&mut g);
    // Classify inputs while the ONNX surface (Cast/Gather uses) is intact.
    let descriptors = inputs::classify_inputs(&g, &g.inputs.clone())?;
    let pass_reports = passes::run_all(&mut g, opts.scale)?;
    let lowered = lower::lower(&g, opts.scale, &descriptors)?;
    Ok(LoweredGraph {
        graph: lowered.graph,
        input_descriptors: lowered.input_descriptors,
        pass_reports,
    })
}
