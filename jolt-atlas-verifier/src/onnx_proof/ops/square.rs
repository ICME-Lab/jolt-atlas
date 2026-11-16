use crate::node_view::ComputationNode;
use crate::onnx_proof::{
    ops::mul::ProductRelation,
    plan::{fused_rescale_plan, OpPlan, OpRelation, Plan},
    ProofType,
};
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_claims::SymbolicSumcheck;
use atlas_trace::{op::Square, trace::ModelExecutionIO, Model};

/// Element-wise square as a product relation.
pub type SquareRelation = ProductRelation<Square>;

crate::impl_plan_op!(Square, SquareRelation);

impl OpPlan for Square {
    fn plan(
        &self,
        node: &ComputationNode,
        _model: &Model,
        _io: &ModelExecutionIO,
    ) -> Result<Plan, ProofVerifyError> {
        Ok(fused_rescale_plan(
            node,
            ProofType::RescaleArith,
            OpRelation::Square(SquareRelation::new(node.clone())),
        ))
    }
}
