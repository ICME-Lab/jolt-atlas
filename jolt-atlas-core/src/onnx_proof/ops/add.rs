use crate::node_view::ComputationNode;
use crate::onnx_proof::trace_view::TraceView;
use atlas_algebra::field::JoltField;
use atlas_trace::Trace;
pub use jolt_atlas_verifier::onnx_proof::ops::add::*;

/// The claims of an [`accumulation`]: the two operands at the output point.
pub fn accumulation_claims<F: JoltField>(
    node: &ComputationNode,
    trace: &Trace,
    point: &[F],
) -> Vec<F> {
    node.inputs[..2]
        .iter()
        .map(|&operand| TraceView::value_at(trace, operand, point))
        .collect()
}
