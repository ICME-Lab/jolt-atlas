//! Opset normalization: one internal form per operator, whatever opset
//! (10–18 in the zoo) the exporter targeted.
//!
//! After this pass:
//! - `Constant` nodes are gone (payload attached to their output edge);
//! - `Squeeze`/`Unsqueeze`/`ReduceSum`/`ReduceMean` carry an `axes` ints
//!   attribute (never an input);
//! - `Slice` carries `starts`/`ends`/`axes`/`steps` ints attributes;
//! - `Split` carries `axis` and (if explicit) `split` attributes;
//! - `Softmax` / `Gather` / `Concat` / `Flatten` / `Gemm` /
//!   `LayerNormalization` have their defaulted attributes materialized.

use crate::{
    error::ImportError,
    graph::{Attr, ConstData, ConstTensor, DType, IGraph, TensorInfo},
};

/// Move a constant operand into an ints attribute, erroring when runtime.
fn input_to_ints_attr(
    g: &mut IGraph,
    node_idx: usize,
    input_pos: usize,
    attr: &'static str,
) -> Result<(), ImportError> {
    let node = g.nodes[node_idx].as_ref().expect("live");
    let Some(&edge) = node.inputs.get(input_pos) else {
        return Ok(()); // optional input absent
    };
    let (name, op) = (node.name.clone(), node.op.clone());
    let Some(k) = g.konst(edge) else {
        return Err(ImportError::NonConstant {
            node: name,
            op_type: op,
            input: attr.into(),
        });
    };
    let values = k
        .as_i64s()
        .ok_or_else(|| ImportError::UnsupportedAttribute {
            node: name.clone(),
            op_type: op.clone(),
            reason: format!("{attr} operand must be integer"),
        })?
        .to_vec();
    let node = g.nodes[node_idx].as_mut().expect("live");
    node.attrs.insert(attr.into(), Attr::Ints(values));
    node.inputs.truncate(input_pos.min(node.inputs.len()));
    Ok(())
}

/// Convert the const operands of shape-parameter ops into attributes.
///
/// Called from the analysis pass per node, in topological order — at that
/// point every foldable upstream value has already been folded, so a
/// shape-subgraph-computed `ends` operand is constant here even though it
/// was not at normalize time.
pub fn resolve_const_operands(g: &mut IGraph, idx: usize) -> Result<(), ImportError> {
    let Some(node) = g.nodes[idx].as_ref() else {
        return Ok(());
    };
    match node.op.as_str() {
        "Squeeze" | "Unsqueeze" if node.inputs.len() > 1 => {
            input_to_ints_attr(g, idx, 1, "axes")?;
        }
        "ReduceSum" | "ReduceMean" => {
            if node.inputs.len() > 1 {
                input_to_ints_attr(g, idx, 1, "axes")?;
            }
            // Resolve the ONNX empty-axes convention here, so every
            // downstream consumer of the `axes` attribute sees one form:
            // empty/absent axes mean identity when noop_with_empty_axes=1
            // (the node becomes Identity) and reduce-all when 0 (the attr
            // is dropped; a missing attr is reduce-all everywhere).
            let node = g.nodes[idx].as_mut().expect("live");
            let axes_empty = match node.attrs.get("axes") {
                Some(Attr::Ints(v)) => v.is_empty(),
                None => true,
                _ => false,
            };
            if axes_empty {
                if node.attr_int("noop_with_empty_axes").unwrap_or(0) != 0 {
                    node.op = "Identity".into();
                    node.attrs.clear();
                } else {
                    node.attrs.remove("axes");
                }
            }
        }
        "Slice" => {
            let n = g.nodes[idx].as_ref().expect("live").inputs.len();
            if n > 1 {
                if n > 4 {
                    input_to_ints_attr(g, idx, 4, "steps")?;
                }
                if g.nodes[idx].as_ref().expect("live").inputs.len() > 3 {
                    input_to_ints_attr(g, idx, 3, "axes")?;
                }
                input_to_ints_attr(g, idx, 2, "ends")?;
                input_to_ints_attr(g, idx, 1, "starts")?;
            }
        }
        "Split" if g.nodes[idx].as_ref().expect("live").inputs.len() > 1 => {
            input_to_ints_attr(g, idx, 1, "split")?;
        }
        _ => {}
    }
    Ok(())
}

/// Normalize the graph in place. `opset` is the ai.onnx opset version.
pub fn normalize(g: &mut IGraph, opset: i64) -> Result<(), ImportError> {
    for idx in 0..g.nodes.len() {
        let Some(node) = g.nodes[idx].as_ref() else {
            continue;
        };
        let op = node.op.clone();
        match op.as_str() {
            "Constant" => {
                let node = g.nodes[idx].as_ref().expect("live");
                let name = node.name.clone();
                let out = *node.outputs.first().ok_or_else(|| {
                    ImportError::Malformed(format!("Constant {name} has no output"))
                })?;
                let tensor = match (
                    node.attrs.get("value").cloned(),
                    node.attr_float("value_float"),
                    node.attr_int("value_int"),
                    node.attrs.get("value_ints").cloned(),
                ) {
                    (Some(Attr::Tensor(t)), ..) => t,
                    (_, Some(f), ..) => ConstTensor {
                        dtype: DType::F32,
                        dims: vec![],
                        data: ConstData::F32(vec![f]),
                    },
                    (_, _, Some(i), _) => ConstTensor {
                        dtype: DType::I64,
                        dims: vec![],
                        data: ConstData::I64(vec![i]),
                    },
                    (_, _, _, Some(Attr::Ints(v))) => ConstTensor {
                        dims: vec![v.len()],
                        dtype: DType::I64,
                        data: ConstData::I64(v),
                    },
                    _ => {
                        return Err(ImportError::UnsupportedAttribute {
                            node: name,
                            op_type: "Constant".into(),
                            reason: "unsupported Constant payload form".into(),
                        });
                    }
                };
                g.edge_mut(out).info = Some(TensorInfo {
                    dtype: tensor.dtype,
                    dims: tensor.dims.clone(),
                });
                g.edge_mut(out).konst = Some(tensor);
                g.remove_node(idx);
            }
            // Squeeze/Unsqueeze/ReduceSum/ReduceMean/Slice/Split carry
            // shape parameters as inputs in newer opsets; those inputs may
            // be computed by shape subgraphs, so the conversion to
            // attributes happens in topological order during analysis
            // (`resolve_const_operands`). Only defaults are set here.
            "ReduceSum" | "ReduceMean" => set_default_int(g, idx, "keepdims", 1),
            "Split" => set_default_int(g, idx, "axis", 0),
            "Softmax" => {
                let default = if opset >= 13 { -1 } else { 1 };
                set_default_int(g, idx, "axis", default);
            }
            "Gather" => set_default_int(g, idx, "axis", 0),
            "Flatten" => set_default_int(g, idx, "axis", 1),
            "Concat" => {
                let node = g.nodes[idx].as_ref().expect("live");
                if node.attr_int("axis").is_none() {
                    return Err(ImportError::UnsupportedAttribute {
                        node: node.name.clone(),
                        op_type: op,
                        reason: "Concat requires an axis".into(),
                    });
                }
            }
            "Gemm" => {
                set_default_int(g, idx, "transA", 0);
                set_default_int(g, idx, "transB", 0);
                set_default_float(g, idx, "alpha", 1.0);
                set_default_float(g, idx, "beta", 1.0);
            }
            "LayerNormalization" => {
                set_default_int(g, idx, "axis", -1);
                set_default_float(g, idx, "epsilon", 1e-5);
            }
            "Cast" => {
                let node = g.nodes[idx].as_ref().expect("live");
                if node.attr_int("to").is_none() {
                    return Err(ImportError::UnsupportedAttribute {
                        node: node.name.clone(),
                        op_type: op,
                        reason: "Cast requires `to`".into(),
                    });
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn set_default_int(g: &mut IGraph, idx: usize, name: &str, value: i64) {
    let node = g.nodes[idx].as_mut().expect("live");
    node.attrs
        .entry(name.to_string())
        .or_insert(Attr::Int(value));
}

fn set_default_float(g: &mut IGraph, idx: usize, name: &str, value: f32) {
    let node = g.nodes[idx].as_mut().expect("live");
    node.attrs
        .entry(name.to_string())
        .or_insert(Attr::Float(value));
}
