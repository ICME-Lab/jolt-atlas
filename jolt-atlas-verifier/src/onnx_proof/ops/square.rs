use crate::node_view::ComputationNode;
use crate::onnx_proof::{
    ops::mul::ProductRelation,
    plan::{fused_rescale_plan, OpPlan, OpRelation, Plan},
    ProofType,
};
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_trace::{op::Square, trace::ModelExecutionIO, Model};
use jolt_claims::SymbolicSumcheck;

/// Element-wise square as a product relation.
pub type SquareRelation = ProductRelation<Square>;

crate::impl_plan_op!(Square, SquareRelation);

impl OpPlan for Square {
    fn plan(
        &self,
        node: &ComputationNode,
        model: &Model,
        io: &ModelExecutionIO,
    ) -> Result<Plan, ProofVerifyError> {
        fused_rescale_plan(
            node,
            model,
            io,
            ProofType::RescaleArith,
            OpRelation::Square(SquareRelation::new(node.clone())),
        )
    }
}
