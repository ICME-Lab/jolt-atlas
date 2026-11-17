use crate::node_view::ComputationNode;
use crate::onnx_proof::{
    ops::mul::product,
    relation::OpRelKind,
    schedule::{fused_rescale_plan, OpPlan, OpRelation, ScheduleBuilder},
};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_trace::{op::Square, Model};

/// Element-wise square as a product ([`product`]).
impl OpPlan for Square {
    fn plan<F: JoltField>(
        &self,
        node: &ComputationNode,
        _model: &Model,
        schedule: &mut ScheduleBuilder<F>,
    ) -> Result<(), ProofVerifyError> {
        fused_rescale_plan(
            node,
            OpRelation::Op(product(node, OpRelKind::Square, &[0, 0])),
            schedule,
        );
        Ok(())
    }
}
