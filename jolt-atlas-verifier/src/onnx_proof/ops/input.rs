use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::constant::public_input;
use crate::onnx_proof::schedule::{OpPlan, ScheduleBuilder};
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_trace::op::Input;

impl OpPlan for Input {
    fn plan<F: atlas_algebra::field::JoltField>(
        &self,
        node: &ComputationNode,
        model: &atlas_trace::Model,
        schedule: &mut ScheduleBuilder<F>,
    ) -> Result<(), ProofVerifyError> {
        schedule.reduce();
        schedule.tie(public_input(model, node)?);
        Ok(())
    }
}
