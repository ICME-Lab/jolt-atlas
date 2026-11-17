use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::neg::signed_copy;
use crate::onnx_proof::relation::OpRelKind;
use crate::onnx_proof::schedule::{OpPlan, ScheduleBuilder};
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_trace::op::Identity;

impl OpPlan for Identity {
    fn plan<F: atlas_algebra::field::JoltField>(
        &self,
        node: &ComputationNode,
        _model: &atlas_trace::Model,
        _io: &atlas_trace::trace::ModelExecutionIO,
        schedule: &mut ScheduleBuilder<F>,
    ) -> Result<(), ProofVerifyError> {
        schedule.reduce();
        schedule.tie(signed_copy(node, OpRelKind::Identity));
        Ok(())
    }
}
