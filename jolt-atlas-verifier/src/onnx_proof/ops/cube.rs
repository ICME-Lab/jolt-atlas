use crate::node_view::ComputationNode;
use crate::onnx_proof::{
    ops::mul::product,
    relation::OpRelKind,
    schedule::{fused_rescale_plan, OpPlan, OpRelation, ScheduleBuilder},
};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_trace::{op::Cube, Model};

/// Element-wise cube as a product ([`product`]).
impl OpPlan for Cube {
    fn plan<F: JoltField>(
        &self,
        node: &ComputationNode,
        _model: &Model,
        schedule: &mut ScheduleBuilder<F>,
    ) -> Result<(), ProofVerifyError> {
        fused_rescale_plan(
            node,
            OpRelation::Op(product(node, OpRelKind::Cube, &[0, 0, 0])),
            schedule,
        );
        Ok(())
    }
}
