//! Exact integer masks and selection lowered to existing native relations.
//! These helpers append a block and return its output tensor. The complete
//! graph still requires normal validation, authenticated registration and proof.
use super::native_graph::{NativeGraph, NativeGraphNode};
use crate::utils::errors::ProofVerifyError;
fn next(graph: &NativeGraph, inputs: &[usize]) -> Result<usize, ProofVerifyError> {
    let first = graph
        .num_inputs()
        .checked_add(graph.nodes.len())
        .ok_or_else(|| {
            ProofVerifyError::InvalidOpeningProof("Native graph index overflow".into())
        })?;
    if first.checked_add(5).is_none() || inputs.iter().any(|id| *id >= first) {
        return Err(ProofVerifyError::InvalidOpeningProof(
            "Integer block input must precede its output".into(),
        ));
    }
    Ok(first)
}
/// Boolean AND. Both original inputs must be zero or one.
pub fn append_and(
    graph: &mut NativeGraph,
    left: usize,
    right: usize,
) -> Result<usize, ProofVerifyError> {
    let first = next(graph, &[left, right])?;
    graph.nodes.extend([
        NativeGraphNode::lookup(left, vec![0, 2], 1),
        NativeGraphNode::lookup(right, vec![0, 1], 1),
        NativeGraphNode::mul(first, first + 1, 1),
    ]);
    Ok(first + 2)
}
/// Choose the true or false tensor using a Boolean mask. Every selected
/// signed32 value is unchanged, including both signed endpoints.
pub fn append_select(
    graph: &mut NativeGraph,
    mask: usize,
    when_true: usize,
    when_false: usize,
) -> Result<usize, ProofVerifyError> {
    let first = next(graph, &[mask, when_true, when_false])?;
    graph.nodes.extend([
        NativeGraphNode::lookup(mask, vec![0, 2], 1),
        NativeGraphNode::lookup(mask, vec![2, 0], 1),
        NativeGraphNode::mul(when_true, first, 1),
        NativeGraphNode::mul(when_false, first + 1, 1),
        NativeGraphNode::add(first + 2, first + 3),
    ]);
    Ok(first + 4)
}
/// Exact negation on the domain where the signed32 result exists.
/// The minimum signed integer is rejected, rather than assigned a clamp or
/// wraparound result. Model integration must retain this proved domain.
pub fn append_checked_neg(
    graph: &mut NativeGraph,
    input: usize,
) -> Result<usize, ProofVerifyError> {
    let first = next(graph, &[input])?;
    graph.nodes.extend([
        NativeGraphNode::sub(input, input),
        NativeGraphNode::sub(first, input),
        NativeGraphNode::add(input, first + 1),
        // For ordinary inputs the checked sum is zero. Negating MIN through
        // the clamp would give MAX, leaving -1, which this lookup rejects.
        NativeGraphNode::lookup(first + 2, vec![0, 0], 1),
    ]);
    Ok(first + 1)
}
/// Integer IsNan with an output shape equal to the input shape.
/// All integer values are finite. The original tensor still has signed ranges.
pub fn append_is_nan(graph: &mut NativeGraph, input: usize) -> Result<usize, ProofVerifyError> {
    let first = next(graph, &[input])?;
    graph.nodes.push(NativeGraphNode::sub(input, input));
    Ok(first)
}
