use atlas_algebra::field::JoltField;
use atlas_algebra::poly::multilinear::EvaluateMle;
pub use jolt_atlas_verifier::onnx_proof::ops::slice::*;

use crate::onnx_proof::relation::OpRel;
use crate::onnx_proof::trace_view::TraceView;
use atlas_trace::Trace;

/// The claim of a [`subcube_select`]: the operand at its point for the output point.
pub fn subcube_select_claims<F: JoltField>(
    relation: &OpRel<F>,
    trace: &Trace,
    point: &[F],
) -> Vec<F> {
    let operand = TraceView::operand_tensors(trace, &relation.node)[0];
    let point = operand_point(relation, point)
        .expect("the incoming point is a point of the node's padded domain");
    vec![operand.evaluate_mle(&point)]
}
