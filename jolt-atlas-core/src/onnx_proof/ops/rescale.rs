use atlas_algebra::field::JoltField;
pub use jolt_atlas_verifier::onnx_proof::ops::rescale::*;

use crate::node_view::ComputationNode;
use crate::onnx_proof::trace_view::TraceView;
use atlas_trace::Trace;

/// The claim of a [`rescale`]: the wide operand at the output point.
pub fn rescale_claims<F: JoltField>(node: &ComputationNode, trace: &Trace, point: &[F]) -> Vec<F> {
    vec![TraceView::value_at(trace, node.inputs[0], point)]
}
