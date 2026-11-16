use atlas_algebra::field::JoltField;
use atlas_algebra::poly::multilinear::EvaluateMle;
pub use jolt_atlas_verifier::onnx_proof::ops::concat::*;

use crate::onnx_proof::relation::{AtlasRelation, RelationSetup, TieClaims};
use crate::onnx_proof::trace_view::TraceView;
use atlas_sumcheck::opening_proof::OpeningAccumulator;
use atlas_trace::Trace;

impl<F: JoltField> TieClaims<F> for SubcubeTilingRelation {
    fn claims(
        &self,
        trace: &Trace,
        setup: &RelationSetup<F>,
        _accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F> {
        let points = self
            .operand_points(&setup.incoming_point)
            .expect("the incoming point is a point of the node's padded domain");
        TraceView::operand_tensors(trace, self.node())
            .into_iter()
            .zip(points)
            .map(|(operand, point)| operand.evaluate_mle(&point))
            .collect()
    }
}
