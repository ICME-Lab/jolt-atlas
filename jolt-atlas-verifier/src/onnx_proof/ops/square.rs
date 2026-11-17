use crate::node_view::ComputationNode;
use crate::onnx_proof::{
    ops::mul::product,
    relation::OpRelKind,
    schedule::{fused_rescale_plan, OpPlan, OpRelation, ScheduleBuilder},
    ProofType,
};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_trace::{op::Square, trace::ModelExecutionIO, Model};

/// Element-wise square as a product ([`product`]).
impl OpPlan for Square {
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
            OpRelation::Op(product(node, OpRelKind::Square, &[0, 0])),
            schedule,
        );
        Ok(())
    }
}
