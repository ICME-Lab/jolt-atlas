//! Per-input quantization descriptors.
//!
//! The importer classifies each graph input so callers no longer need to
//! know the scaling convention by heart (the old examples multiplied some
//! inputs by the scale factor by hand).

use atlas_trace::{Scale, Tensor};

use crate::{
    error::ImportError,
    graph::{DType, EdgeId, IGraph},
};

/// How an input's values map into the fixed-point domain.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum InputKind {
    /// Integer values used as indices or selectors; passed through as-is.
    RawInt,
    /// Real-valued data; multiplied by `2^scale` when quantized.
    Scaled,
}

/// One graph input's name, kind, and expected dims.
#[derive(Clone, Debug)]
pub struct InputDescriptor {
    /// The ONNX input name.
    pub name: String,
    /// How to quantize values for it.
    pub kind: InputKind,
    /// Expected dims (unpadded).
    pub dims: Vec<usize>,
}

impl InputDescriptor {
    /// Quantize a slice of real-valued or integer data for this input.
    pub fn quantize_f64(&self, values: &[f64], scale: Scale) -> Result<Tensor<i32>, ImportError> {
        let data: Vec<i32> = match self.kind {
            InputKind::RawInt => values.iter().map(|&v| v as i32).collect(),
            InputKind::Scaled => values
                .iter()
                .map(|&v| (v * scale.denom_i64() as f64).round() as i32)
                .collect(),
        };
        Tensor::new(data, &self.dims).map_err(|e| ImportError::Malformed(e.to_string()))
    }
}

/// Classify every graph input by inspecting its uses (before the rewrite
/// passes run, while the ONNX surface is still visible):
///
/// - float inputs are [`InputKind::Scaled`];
/// - integer/bool inputs used as `Gather` indices are [`InputKind::RawInt`];
/// - integer inputs that flow into the float domain through `Cast` are
///   [`InputKind::Scaled`] (the surviving Cast lowers to identity, so the
///   caller supplies already-scaled values through the descriptor);
/// - an input used both ways is an error;
/// - otherwise integer inputs default to [`InputKind::RawInt`].
pub fn classify_inputs(g: &IGraph, inputs: &[EdgeId]) -> Result<Vec<InputDescriptor>, ImportError> {
    inputs
        .iter()
        .map(|&edge| {
            let info = g.info(edge).ok_or_else(|| {
                ImportError::Malformed(format!("input {} has no shape", g.edge(edge).name))
            })?;
            let name = g.edge(edge).name.clone();
            let kind = match info.dtype {
                DType::F32 => InputKind::Scaled,
                DType::I64 | DType::I32 | DType::Bool => classify_int_input(g, edge, &name)?,
            };
            Ok(InputDescriptor {
                name,
                kind,
                dims: info.dims.clone(),
            })
        })
        .collect()
}

fn classify_int_input(g: &IGraph, edge: EdgeId, name: &str) -> Result<InputKind, ImportError> {
    let mut used_as_index = false;
    let mut used_as_float = false;
    // Walk transitively through structural (value-preserving) ops: masks
    // typically pass through Unsqueeze/Reshape before their Cast.
    let mut frontier = vec![edge];
    let mut seen = std::collections::HashSet::new();
    while let Some(e) = frontier.pop() {
        if !seen.insert(e) {
            continue;
        }
        for (_, node) in g.live_nodes() {
            for (pos, &input) in node.inputs.iter().enumerate() {
                if input != e {
                    continue;
                }
                // This walk must cover (at least) every op the cast-policy
                // lineage check (`passes::scaled_lineage`) accepts, or an
                // input could be classified RawInt while its Cast still
                // lowers to identity — silently off by 2^scale.
                match node.op.as_str() {
                    "Gather" if pos == 1 => used_as_index = true,
                    "Gather" => frontier.extend(node.outputs.iter().copied()),
                    "Cast"
                        if g.info(node.outputs[0])
                            .is_some_and(|i| i.dtype == DType::F32) =>
                    {
                        used_as_float = true;
                    }
                    "Unsqueeze" | "Squeeze" | "Reshape" | "Flatten" | "Identity" | "Slice"
                    | "Expand" | "Tile" | "Transpose" | "Concat" | "Split" => {
                        frontier.extend(node.outputs.iter().copied());
                    }
                    _ => {}
                }
            }
        }
    }
    match (used_as_index, used_as_float) {
        (true, true) => Err(ImportError::Malformed(format!(
            "input {name} is used both as indices and as real-valued data; \
             split it into two inputs"
        ))),
        (false, true) => Ok(InputKind::Scaled),
        _ => Ok(InputKind::RawInt),
    }
}
