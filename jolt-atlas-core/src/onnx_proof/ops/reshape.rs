use atlas_algebra::field::JoltField;
use atlas_algebra::poly::multilinear::EvaluateMle;
pub use jolt_atlas_verifier::onnx_proof::ops::reshape::*;

use crate::onnx_proof::relation::{AtlasRelation, RelationSetup, TieClaims};
use crate::onnx_proof::trace_view::TraceView;
use atlas_sumcheck::opening_proof::OpeningAccumulator;
use atlas_trace::Trace;

impl<F: JoltField> TieClaims<F> for FlatCopyRelation {
    fn claims(
        &self,
        trace: &Trace,
        setup: &RelationSetup<F>,
        _accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F> {
        let operand = TraceView::operand_tensors(trace, self.node())[0];
        vec![operand.evaluate_mle(&setup.incoming_point)]
    }
}
