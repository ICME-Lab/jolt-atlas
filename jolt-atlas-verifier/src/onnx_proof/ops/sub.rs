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
/// the accumulation `acc = left - right` is widened to `i64`, the clamp lookup
/// proves `output(r) = SatClamp(acc(r))` and ties `acc` ([`VirtualPoly::ClampAcc`])
/// to the one-hot read-address poly ([`VirtualPoly::ClampRa`]), the one-hot
/// checks discharge `ClampRa` against [`CommittedPoly::ClampRaD`], and the
/// operand tie ([`AccumulationRelation`]) checks `left(r) - right(r) == acc(r)`. See
/// [`Add`](super::add) for the full three-part breakdown.
///
/// [`VirtualPoly::ClampAcc`]: common::VirtualPoly::ClampAcc
/// [`VirtualPoly::ClampRa`]: common::VirtualPoly::ClampRa
/// [`CommittedPoly::ClampRaD`]: common::CommittedPoly::ClampRaD
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
