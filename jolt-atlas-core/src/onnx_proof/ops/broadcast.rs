use atlas_algebra::field::JoltField;
use atlas_trace::op::Broadcast;
pub use jolt_atlas_verifier::onnx_proof::ops::broadcast::*;

crate::impl_plan_op_prover!(Broadcast);

use crate::onnx_proof::relation::{AtlasRelation, RelationSetup, TieClaims};
use crate::onnx_proof::trace_view::TraceView;
use atlas_algebra::poly::multilinear_polynomial::EvaluateMle;
use atlas_sumcheck::opening_proof::OpeningAccumulator;
use atlas_trace::Trace;

impl<F: JoltField> TieClaims<F> for BroadcastRelation {
    fn claims(
        &self,
        trace: &Trace,
        setup: &RelationSetup<F>,
        _accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F> {
        let operand = TraceView::operand_tensors(trace, self.node())[0];
        let (r_input, _) = self
            .split(&setup.incoming_point)
            .expect("the incoming point is a point of the node's padded domain");
        vec![operand.evaluate_mle(&r_input)]
    }
}
