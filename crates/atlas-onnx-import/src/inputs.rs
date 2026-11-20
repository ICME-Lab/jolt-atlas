//! Per-input quantization descriptors.
//!
//! The importer classifies each graph input so callers no longer need to
//! know the scaling convention by heart.

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
    // Consumer adjacency, built once: each per-input walk below then visits
    // only the consumers of its frontier edges, instead of rescanning every
    // live node per edge (quadratic on transformer-scale graphs).
    let mut consumers: std::collections::HashMap<EdgeId, Vec<usize>> =
        std::collections::HashMap::new();
    for (idx, node) in g.live_nodes() {
        for &input in &node.inputs {
            let entry = consumers.entry(input).or_default();
            if entry.last() != Some(&idx) {
                entry.push(idx);
            }
        }
    }
    inputs
        .iter()
        .map(|&edge| {
            let info = g.info(edge).ok_or_else(|| {
                ImportError::Malformed(format!("input {} has no shape", g.edge(edge).name))
            })?;
            let name = g.edge(edge).name.clone();
            let kind = match info.dtype {
                DType::F32 => InputKind::Scaled,
                DType::I64 | DType::I32 | DType::Bool => {
                    classify_int_input(g, &consumers, edge, &name)?
                }
            };
            Ok(InputDescriptor {
                name,
                kind,
                dims: info.dims.clone(),
            })
        })
        .collect()
}

fn classify_int_input(
    g: &IGraph,
    consumers: &std::collections::HashMap<EdgeId, Vec<usize>>,
    edge: EdgeId,
    name: &str,
) -> Result<InputKind, ImportError> {
    let mut used_as_index = false;
    let mut used_as_float = false;
    // Walk transitively through structural (value-preserving) ops: masks
    // typically pass through Unsqueeze/Reshape before their Cast. The op set
    // is `passes::VALUE_PRESERVING_OPS`, shared with the cast-policy lineage
    // check (`passes::scaled_lineage`): the walks must cover the same ops, or
    // an input could be classified RawInt while its Cast still lowers to
    // identity — silently off by 2^scale.
    //
    // Each frontier entry carries whether the input's VALUE is still intact
    // (`direct`). Float use counts only on direct paths (mirroring
    // `scaled_lineage`); index use counts through integer arithmetic too,
    // since an arithmetically derived Gather index still means the caller
    // must supply raw integers.
    let mut frontier = vec![(edge, true)];
    let mut seen = std::collections::HashSet::new();
    while let Some((e, direct)) = frontier.pop() {
        if !seen.insert((e, direct)) {
            continue;
        }
        for &node_idx in consumers.get(&e).into_iter().flatten() {
            let Some(node) = g.nodes[node_idx].as_ref() else {
                continue;
            };
            let out_is_int = node.outputs.first().is_some_and(|&o| {
                g.info(o)
                    .is_some_and(|i| matches!(i.dtype, DType::I64 | DType::I32 | DType::Bool))
            });
            for (pos, &input) in node.inputs.iter().enumerate() {
                if input != e {
                    continue;
                }
                match node.op.as_str() {
                    "Gather" if pos == 1 => used_as_index = true,
                    "Cast"
                        if direct
                            && g.info(node.outputs[0])
                                .is_some_and(|i| i.dtype == DType::F32) =>
                    {
                        used_as_float = true;
                    }
                    // Int→int Cast: cast_policy later rewrites it to
                    // Identity, so this walk must see through it too.
                    "Cast" if out_is_int => {
                        frontier.extend(node.outputs.iter().map(|&o| (o, direct)));
                    }
                    op if crate::passes::VALUE_PRESERVING_OPS.contains(&op) => {
                        frontier.extend(node.outputs.iter().map(|&o| (o, direct)));
                    }
                    // Integer arithmetic: the value is no longer the input's,
                    // but a downstream index use still pins the raw domain.
                    "Add" | "Sub" | "Mul" | "Neg" | "Where" | "And" | "Equal" | "Less"
                    | "LessOrEqual" | "Greater"
                        if out_is_int =>
                    {
                        frontier.extend(node.outputs.iter().map(|&o| (o, false)));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Attr, ConstData, ConstTensor, INode, TensorInfo};

    fn int_info(dims: &[usize], dtype: DType) -> TensorInfo {
        TensorInfo {
            dtype,
            dims: dims.to_vec(),
        }
    }

    fn fresh(g: &mut IGraph, name: &str, dtype: DType) -> EdgeId {
        g.fresh_edge(name, int_info(&[4], dtype))
    }

    /// An input consumed as float data (direct Cast) AND as a Gather index
    /// derived through integer arithmetic is a domain conflict: classifying
    /// it Scaled would feed 2^s-scaled values into the index path.
    #[test]
    fn mixed_float_and_arithmetic_index_use_is_rejected() {
        let mut g = IGraph::default();
        let x = fresh(&mut g, "x", DType::I64);
        g.inputs.push(x);

        let as_float = fresh(&mut g, "as_float", DType::F32);
        g.push_node(INode {
            name: "to_float".into(),
            op: "Cast".into(),
            attrs: Default::default(),
            inputs: vec![x],
            outputs: vec![as_float],
        });

        let two = fresh(&mut g, "two", DType::I64);
        g.edge_mut(two).konst = Some(ConstTensor {
            dtype: DType::I64,
            dims: vec![1],
            data: ConstData::I64(vec![2]),
        });
        let scaled_idx = fresh(&mut g, "idx", DType::I64);
        g.push_node(INode {
            name: "mul".into(),
            op: "Mul".into(),
            attrs: Default::default(),
            inputs: vec![x, two],
            outputs: vec![scaled_idx],
        });
        let dict = fresh(&mut g, "dict", DType::F32);
        let gathered = fresh(&mut g, "gathered", DType::F32);
        g.push_node(INode {
            name: "gather".into(),
            op: "Gather".into(),
            attrs: [("axis".to_string(), Attr::Int(0))].into(),
            inputs: vec![dict, scaled_idx],
            outputs: vec![gathered],
        });

        let err = classify_inputs(&g, &[x]).unwrap_err();
        assert!(
            err.to_string().contains("both as indices"),
            "expected domain-conflict rejection, got: {err}"
        );
    }

    /// cast_policy sees through int→int Casts (it rewrites them to
    /// Identity), so classification must too, or the input descriptor says
    /// RawInt while the graph treats the values as scaled.
    #[test]
    fn widening_cast_chain_classifies_scaled() {
        let mut g = IGraph::default();
        let x = fresh(&mut g, "x", DType::I32);
        g.inputs.push(x);
        let wide = fresh(&mut g, "wide", DType::I64);
        g.push_node(INode {
            name: "widen".into(),
            op: "Cast".into(),
            attrs: Default::default(),
            inputs: vec![x],
            outputs: vec![wide],
        });
        let f = fresh(&mut g, "f", DType::F32);
        g.push_node(INode {
            name: "to_float".into(),
            op: "Cast".into(),
            attrs: Default::default(),
            inputs: vec![wide],
            outputs: vec![f],
        });

        let desc = classify_inputs(&g, &[x]).unwrap();
        assert_eq!(desc[0].kind, InputKind::Scaled);
    }

    /// A float use reached only through integer arithmetic is NOT a scaled
    /// use (cast_policy's scaled_lineage would reject that cast anyway):
    /// the input stays RawInt.
    #[test]
    fn arithmetic_then_cast_does_not_classify_scaled() {
        let mut g = IGraph::default();
        let x = fresh(&mut g, "x", DType::I64);
        g.inputs.push(x);
        let two = fresh(&mut g, "two", DType::I64);
        g.edge_mut(two).konst = Some(ConstTensor {
            dtype: DType::I64,
            dims: vec![1],
            data: ConstData::I64(vec![2]),
        });
        let doubled = fresh(&mut g, "doubled", DType::I64);
        g.push_node(INode {
            name: "mul".into(),
            op: "Mul".into(),
            attrs: Default::default(),
            inputs: vec![x, two],
            outputs: vec![doubled],
        });
        let f = fresh(&mut g, "f", DType::F32);
        g.push_node(INode {
            name: "to_float".into(),
            op: "Cast".into(),
            attrs: Default::default(),
            inputs: vec![doubled],
            outputs: vec![f],
        });

        let desc = classify_inputs(&g, &[x]).unwrap();
        assert_eq!(desc[0].kind, InputKind::RawInt);
    }
}
