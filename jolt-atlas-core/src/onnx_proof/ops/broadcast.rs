use atlas_algebra::field::JoltField;
pub use jolt_atlas_verifier::onnx_proof::ops::broadcast::*;

use crate::onnx_proof::relation::OpRel;
use crate::onnx_proof::trace_view::TraceView;
use atlas_algebra::poly::multilinear::EvaluateMle;
use atlas_trace::Trace;

/// The claim of a [`broadcast`]: the operand at the non-broadcast coordinates of the output
/// point.
pub fn broadcast_claims<F: JoltField>(relation: &OpRel<F>, trace: &Trace, point: &[F]) -> Vec<F> {
    let operand = TraceView::operand_tensors(trace, &relation.node)[0];
    let (r_input, _) =
        split(relation, point).expect("the incoming point is a point of the node's padded domain");
    vec![operand.evaluate_mle(&r_input)]
}
