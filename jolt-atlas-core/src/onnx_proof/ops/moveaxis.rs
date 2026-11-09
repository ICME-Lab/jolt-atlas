use atlas_algebra::field::JoltField;
use atlas_trace::op::MoveAxis;
pub use jolt_atlas_verifier::onnx_proof::ops::moveaxis::*;

crate::impl_plan_op_prover!(MoveAxis);

use crate::onnx_proof::relation::{AtlasRelation, RelationSetup, TieClaims};
use atlas_sumcheck::opening_proof::OpeningAccumulator;
use atlas_trace::Trace;

impl<F: JoltField> TieClaims<F> for AxisPermutationRelation {
    fn claims(
        &self,
        _trace: &Trace,
        _setup: &RelationSetup<F>,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F> {
        // The operand at the permuted point is the output at `r`.
        vec![
            accumulator
                .get_node_output_opening(self.node().idx)
                .expect("the prover holds every claim it reads")
                .1,
        ]
    }
}
