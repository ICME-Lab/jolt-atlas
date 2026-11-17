use atlas_algebra::field::JoltField;
use atlas_algebra::poly::multilinear::EvaluateMle;
pub use jolt_atlas_verifier::onnx_proof::ops::concat::*;

use crate::onnx_proof::relation::OpRel;
use crate::onnx_proof::trace_view::TraceView;
use atlas_trace::Trace;

/// The claims of a [`subcube_tiling`]: each operand at its point for the output point.
pub fn subcube_tiling_claims<F: JoltField>(
    relation: &OpRel<F>,
    trace: &Trace,
    point: &[F],
) -> Vec<F> {
    let points = operand_points(relation, point)
        .expect("the incoming point is a point of the node's padded domain");
    TraceView::operand_tensors(trace, &relation.node)
        .into_iter()
        .zip(points)
        .map(|(operand, point)| operand.evaluate_mle(&point))
        .collect()
}
