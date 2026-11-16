use crate::node_view::ComputationNode;
use crate::onnx_proof::{
    ops::mul::ProductRelation,
    plan::{fused_rescale_plan, OpPlan, OpRelation, Plan},
    ProofType,
};
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_claims::SymbolicSumcheck;
use atlas_trace::{op::Cube, trace::ModelExecutionIO, Model};

/// Element-wise cube as a product relation.
pub type CubeRelation = ProductRelation<Cube>;

crate::impl_plan_op!(Cube, CubeRelation);

impl OpPlan for Cube {
    fn plan(
        &self,
        node: &ComputationNode,
        _model: &Model,
        _io: &ModelExecutionIO,
    ) -> Result<Plan, ProofVerifyError> {
        Ok(fused_rescale_plan(
            node,
            ProofType::RescaleArith,
            OpRelation::Cube(CubeRelation::new(node.clone())),
        ))
    }
}
