use crate::node_view::ComputationNode;
use crate::onnx_proof::{
    ops::add::{accumulation_plan, AccumulationRelation},
    plan::{OpPlan, Plan},
};
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_trace::{op::Sub, trace::ModelExecutionIO, Model};

crate::impl_plan_op!(Sub, AccumulationRelation<Sub>);

/// Proof for saturating element-wise subtraction: `output = SatClamp(left - right)`.
///
/// Identical in structure to [`Add`](super::add) but with a subtraction tie:
/// the accumulation `acc = left - right` is widened to `i64` and opened
/// ([`VirtualPoly::ClampAcc`]), its clamp is proven exactly or in its packed clamp bucket, and
/// the operand tie ([`AccumulationRelation`]) checks `left(r) - right(r) == acc(r)`. See
/// [`Add`](super::add) for the full breakdown.
///
/// [`VirtualPoly::ClampAcc`]: common::VirtualPoly::ClampAcc
impl OpPlan for Sub {
    fn plan(
        &self,
        node: &ComputationNode,
        model: &Model,
        io: &ModelExecutionIO,
    ) -> Result<Plan, ProofVerifyError> {
        Ok(accumulation_plan::<Sub>(node, model, io))
    }
}
