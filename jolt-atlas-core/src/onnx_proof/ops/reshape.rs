use atlas_algebra::field::JoltField;
use atlas_algebra::poly::multilinear::EvaluateMle;
pub use jolt_atlas_verifier::onnx_proof::ops::reshape::*;

use crate::node_view::ComputationNode;
use crate::onnx_proof::trace_view::TraceView;
use atlas_trace::Trace;

/// The claim of a [`flat_copy`]: the operand at the output point.
pub fn flat_copy_claims<F: JoltField>(
    node: &ComputationNode,
    trace: &Trace,
    point: &[F],
) -> Vec<F> {
    let operand = TraceView::operand_tensors(trace, node)[0];
    vec![operand.evaluate_mle(point)]
}
