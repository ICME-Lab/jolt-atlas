use crate::node_view::ComputationNode;
use crate::onnx_proof::{
    ops::mul::product,
    relation::OpRelKind,
    schedule::{fused_rescale_plan, OpPlan, OpRelation, ScheduleBuilder},
    ProofType,
};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_trace::{op::Cube, trace::ModelExecutionIO, Model};

/// Element-wise cube as a product ([`product`]).
impl OpPlan for Cube {
    fn plan<F: JoltField>(
        &self,
        node: &ComputationNode,
        _model: &Model,
        _io: &ModelExecutionIO,
        schedule: &mut ScheduleBuilder<F>,
    ) -> Result<(), ProofVerifyError> {
        fused_rescale_plan(
            node,
            ProofType::RescaleArith,
            OpRelation::Op(product(node, OpRelKind::Cube, &[0, 0, 0])),
            schedule,
        );
        Ok(())
    }
}
