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

/// Analyze the graph: after this, every edge has a [`TensorInfo`] and every
/// import-time-constant edge carries its payload. Nodes whose outputs all
/// folded are removed.
pub fn analyze(g: &mut IGraph) -> Result<(), ImportError> {
    // Import-wide fold allowance: per-fold budgets scale with input payloads,
    // so synthesized constants must not be able to inflate their own
    // allowance across a chain of folds.
    let mut fold_remaining = fold_fns::FOLD_TOTAL_ELEMENTS;
    for idx in g.topo_order()? {
        // Shape parameters supplied as (now folded) inputs become attrs.
        crate::normalize::resolve_const_operands(g, idx)?;
        if try_fold(g, idx, &mut fold_remaining)? {
            continue;
        }
        infer_shapes(g, idx)?;
    }
    Ok(())
}

/// Constant-fold `idx` if the whitelist covers its op and the needed operands
/// are constant (`Shape` needs only its input's dims; the rest need every
/// operand). On success the outputs carry their payloads and the node is
/// removed; returns whether that happened.
fn try_fold(g: &mut IGraph, idx: usize, fold_remaining: &mut usize) -> Result<bool, ImportError> {
    let node = g.nodes[idx].as_ref().expect("live");
    let (name, op) = (node.name.clone(), node.op.clone());

    let inputs_const = node.inputs.iter().all(|&e| g.konst(e).is_some());
    if !(fold_fns::foldable(&op) && (op == "Shape" || inputs_const)) {
        return Ok(false);
    }
    let Some(results) = fold_fns::fold(node, g, *fold_remaining)
        .map_err(|reason| shape_err(&name, &op, format!("constant fold failed: {reason}")))?
    else {
        return Ok(false);
    };
    let produced: usize = results.iter().map(|t| t.len()).sum();
    *fold_remaining = fold_remaining.saturating_sub(produced);

    let outputs = node.outputs.clone();
    check_output_count(&name, "fold produced", results.len(), outputs.len())?;
    for (edge, tensor) in outputs.into_iter().zip(results) {
        check_dims(&name, &op, &tensor.dims)?;
        g.edge_mut(edge).info = Some(TensorInfo {
            dtype: tensor.dtype,
            dims: tensor.dims.clone(),
        });
        g.edge_mut(edge).konst = Some(tensor);
    }
    g.remove_node(idx);
    Ok(true)
}

/// Run shape inference for `idx` and record a [`TensorInfo`] on each output.
fn infer_shapes(g: &mut IGraph, idx: usize) -> Result<(), ImportError> {
    let node = g.nodes[idx].as_ref().expect("live");
    let (name, op) = (node.name.clone(), node.op.clone());

    let infos = shape_fns::infer(node, g).map_err(|reason| shape_err(&name, &op, reason))?;

    let outputs = node.outputs.clone();
    check_output_count(&name, "inferred", infos.len(), outputs.len())?;
    for (edge, info) in outputs.into_iter().zip(infos) {
        check_dims(&name, &op, &info.dims)?;
        g.edge_mut(edge).info = Some(info);
    }
    Ok(())
}

fn shape_err(node: &str, op: &str, reason: String) -> ImportError {
    ImportError::Shape {
        node: node.into(),
        op_type: op.into(),
        reason,
    }
}

fn check_output_count(name: &str, what: &str, got: usize, have: usize) -> Result<(), ImportError> {
    if got != have {
        return Err(ImportError::Malformed(format!(
            "{name}: {what} {got} outputs, node has {have}"
        )));
    }
    Ok(())
}

/// Remove operations that cannot affect any graph output.
///
/// An edge is considered used if a node consumes it or it is a graph output.
/// Start with nodes whose outputs have no uses, then remove them from a
/// worklist. Removing a node makes its input edges one use less popular; if
/// that makes an input unused, its producer may also be dead and can be added
/// to the worklist. This propagates backward through dead chains.
///
/// Use counts are computed once and updated as nodes are removed. That avoids
/// rescanning the whole graph after each deletion, which would be quadratic
/// for long dead chains.
pub fn dce(g: &mut IGraph) {
    let mut counts = g.use_counts();
    let mut worklist: Vec<usize> = g
        .live_nodes()
        .filter(|(_, n)| n.outputs.iter().all(|&e| counts[e.0] == 0))
        .map(|(i, _)| i)
        .collect();

    while let Some(idx) = worklist.pop() {
        // Defensive: counts only decrease and a node is queued only once all
        // of its outputs hit zero, so a queued node is never revived and never
        // queued twice. Checked rather than assumed.
        let Some(node) = g.nodes[idx].as_ref() else {
            continue;
        };
        if !node.outputs.iter().all(|&e| counts[e.0] == 0) {
            continue;
        }

        // Clone the inputs first: remove_node mutates the graph.
        let inputs = node.inputs.clone();
        g.remove_node(idx);

        for &e in &inputs {
            counts[e.0] -= 1;

            // If the edge just became unused, its producer may be dead too.
            if counts[e.0] != 0 {
                continue;
            }
            let Some((producer_idx, producer)) = g.producer(e) else {
                continue; // Graph inputs and constants do not have producers.
            };
            let all_outputs_unused = producer.outputs.iter().all(|&output| counts[output.0] == 0);
            if all_outputs_unused {
                worklist.push(producer_idx);
            }
        }
    }
}

/// Every shape that enters the graph passes this gate: with each tensor's
/// element count capped, no downstream partial product of dims can overflow.
fn check_dims(node: &str, op: &str, dims: &[usize]) -> Result<(), ImportError> {
    if checked_element_count(dims).is_none() {
        return Err(shape_err(
            node,
            op,
            format!("shape {dims:?} exceeds the element cap"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::INode;

    #[test]
    fn dce_removes_dead_chains_and_keeps_live_nodes() {
        let mut g = IGraph::default();
        let x = g.edge_by_name("x");
        g.inputs.push(x);
        let e1 = g.edge_by_name("e1");
        let e2 = g.edge_by_name("e2");
        let live = g.edge_by_name("live");
        let mk = |name: &str, i, o| INode {
            name: name.into(),
            op: "Identity".into(),
            attrs: Default::default(),
            inputs: vec![i],
            outputs: vec![o],
        };
        g.push_node(mk("dead1", x, e1));
        g.push_node(mk("dead2", e1, e2));
        let keep = g.push_node(mk("keep", x, live));
        g.outputs.push(live);
        dce(&mut g);
        assert_eq!(g.live_nodes().count(), 1);
        assert!(g.nodes[keep].is_some());
    }
}
