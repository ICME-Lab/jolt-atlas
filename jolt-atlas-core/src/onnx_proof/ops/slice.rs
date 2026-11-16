use atlas_algebra::field::JoltField;
use atlas_algebra::poly::multilinear::EvaluateMle;
pub use jolt_atlas_verifier::onnx_proof::ops::slice::*;

use crate::onnx_proof::relation::{AtlasRelation, RelationSetup, TieClaims};
use crate::onnx_proof::trace_view::TraceView;
use atlas_sumcheck::opening_proof::OpeningAccumulator;
use atlas_trace::Trace;

impl<F: JoltField> TieClaims<F> for SubcubeSelectRelation {
    fn claims(
        &self,
        trace: &Trace,
        setup: &RelationSetup<F>,
        _accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F> {
        let operand = TraceView::operand_tensors(trace, self.node())[0];
        let point = self
            .operand_point(&setup.incoming_point)
            .expect("the incoming point is a point of the node's padded domain");
        vec![operand.evaluate_mle(&point)]
    }
}
