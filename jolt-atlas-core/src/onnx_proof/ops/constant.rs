use atlas_algebra::field::JoltField;
use atlas_trace::op::Constant;
pub use jolt_atlas_verifier::onnx_proof::ops::constant::*;

crate::impl_plan_op_prover!(Constant);

use crate::onnx_proof::relation::{RelationSetup, TieClaims};
use atlas_sumcheck::opening_proof::OpeningAccumulator;
use atlas_trace::op::OpSpec;
use atlas_trace::Trace;

impl<F: JoltField, S: OpSpec + 'static> TieClaims<F> for PublicRelation<S> {
    fn claims(
        &self,
        _trace: &Trace,
        _setup: &RelationSetup<F>,
        _accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F> {
        Vec::new()
    }
}
