use crate::node_view::ComputationNode;
use crate::onnx_proof::{
    ops::add::{accumulation_plan, AccumulationRelation},
    plan::{OpPlan, Plan},
};
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_trace::{op::Sub, trace::ModelExecutionIO, Model};

crate::impl_plan_op!(Sub, AccumulationRelation<Sub>);

/// Proof for exact element-wise subtraction: `output = left - right`, where a difference
/// outside `i32` is an execution error.
///
/// Identical in structure to [`Add`](super::add) but with a subtraction tie: the operand tie
/// ([`AccumulationRelation`]) checks `left(r) - right(r) == output(r)` at the reduced output
/// point, and the output's range claim is in the packed range buckets. See
/// [`Add`](super::add) for the full breakdown.
impl OpPlan for Sub {
    fn plan(
        &self,
        node: &ComputationNode,
        _model: &Model,
        _io: &ModelExecutionIO,
    ) -> Result<Plan, ProofVerifyError> {
        Ok(accumulation_plan::<Sub>(node))
    }
}
