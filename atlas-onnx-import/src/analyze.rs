//! The single forward analysis pass: interleaved shape inference and
//! whitelist constant folding over the topological order.
//!
//! Interleaving is required — shape-construction subgraphs (`Shape` →
//! `Gather` → `Concat` → `Reshape`) need folded values to resolve shapes,
//! and folding needs inferred shapes (`Shape` reads its input's dims).

use crate::{
    error::ImportError,
    fold_fns,
    graph::{IGraph, TensorInfo, checked_element_count},
    shape_fns,
};

/// Every shape that enters the graph passes this gate: with each tensor's
/// element count capped, no downstream partial product of dims can overflow.
fn check_dims(node: &str, op: &str, dims: &[usize]) -> Result<(), ImportError> {
    if checked_element_count(dims).is_none() {
        return Err(ImportError::Shape {
            node: node.into(),
            op_type: op.into(),
            reason: format!("shape {dims:?} exceeds the element cap"),
        });
    }
    Ok(())
}

/// Analyze the graph: after this, every edge has a [`TensorInfo`] and every
/// import-time-constant edge carries its payload. Nodes whose outputs all
/// folded are removed.
pub fn analyze(g: &mut IGraph) -> Result<(), ImportError> {
    for idx in g.topo_order()? {
        // Shape parameters supplied as (now folded) inputs become attrs.
        crate::normalize::resolve_const_operands(g, idx)?;

        let node = g.nodes[idx].as_ref().expect("live");
        let name = node.name.clone();
        let op = node.op.clone();

        // Try folding when the whitelist covers the op and the needed
        // operands are constant (`Shape` needs only dims; the rest need
        // every operand).
        let inputs_const = node.inputs.iter().all(|&e| g.konst(e).is_some());
        let can_fold = fold_fns::foldable(&op) && (op == "Shape" || inputs_const);
        if can_fold {
            let node = g.nodes[idx].as_ref().expect("live");
            match fold_fns::fold(node, g) {
                Ok(Some(results)) => {
                    let outputs = node.outputs.clone();
                    if results.len() != outputs.len() {
                        return Err(ImportError::Malformed(format!(
                            "{name}: fold produced {} outputs, node has {}",
                            results.len(),
                            outputs.len()
                        )));
                    }
                    for (edge, tensor) in outputs.into_iter().zip(results) {
                        check_dims(&name, &op, &tensor.dims)?;
                        g.edge_mut(edge).info = Some(TensorInfo {
                            dtype: tensor.dtype,
                            dims: tensor.dims.clone(),
                        });
                        g.edge_mut(edge).konst = Some(tensor);
                    }
                    g.remove_node(idx);
                    continue;
                }
                Ok(None) => {}
                Err(reason) => {
                    return Err(ImportError::Shape {
                        node: name,
                        op_type: op,
                        reason: format!("constant fold failed: {reason}"),
                    });
                }
            }
        }

        // Shape inference.
        let node = g.nodes[idx].as_ref().expect("live");
        let infos = shape_fns::infer(node, g).map_err(|reason| ImportError::Shape {
            node: name.clone(),
            op_type: op.clone(),
            reason,
        })?;
        let outputs = node.outputs.clone();
        if infos.len() != outputs.len() {
            return Err(ImportError::Malformed(format!(
                "{name}: inferred {} outputs, node has {}",
                infos.len(),
                outputs.len()
            )));
        }
        for (edge, info) in outputs.into_iter().zip(infos) {
            check_dims(&name, &op, &info.dims)?;
            g.edge_mut(edge).info = Some(info);
        }
    }
    Ok(())
}

/// Drop nodes whose outputs are unused (classic DCE over the SSA graph).
pub fn dce(g: &mut IGraph) {
    loop {
        let dead: Vec<usize> = g
            .live_nodes()
            .filter(|(_, n)| n.outputs.iter().all(|&e| g.use_count(e) == 0))
            .map(|(i, _)| i)
            .collect();
        if dead.is_empty() {
            return;
        }
        for idx in dead {
            g.remove_node(idx);
        }
    }
}
