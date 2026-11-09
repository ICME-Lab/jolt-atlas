use crate::onnx_proof::relation::{AtlasRelation, RelationSetup, TieClaims};
use crate::onnx_proof::trace_view::TraceView;
use atlas_algebra::{field::JoltField, poly::multilinear_polynomial::EvaluateMle};
use atlas_sumcheck::opening_proof::OpeningAccumulator;
use atlas_trace::{trace::LayerData, Trace};
pub use jolt_atlas_verifier::onnx_proof::ops::add::*;

impl<F: JoltField, S: AccumulationShape> TieClaims<F> for AccumulationRelation<S> {
    fn claims(
        &self,
        trace: &Trace,
        setup: &RelationSetup<F>,
        _accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F> {
        let LayerData { operands, .. } = TraceView::layer_data(trace, self.node());
        operands[..2]
            .iter()
            .map(|operand| operand.evaluate_mle(&setup.incoming_point))
            .collect()
    }
}
