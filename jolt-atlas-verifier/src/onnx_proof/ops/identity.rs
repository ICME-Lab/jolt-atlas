use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::{neg::signed_copy, OpLayout};
use crate::onnx_proof::plan::{OpPlan, Plan, Step};
use crate::onnx_proof::relation::OpRelKind;
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_trace::op::Identity;

impl OpLayout for Identity {}

impl OpPlan for Identity {
    fn plan<F: atlas_algebra::field::JoltField>(
        &self,
        node: &ComputationNode,
        _model: &atlas_trace::Model,
        _io: &atlas_trace::trace::ModelExecutionIO,
    ) -> Result<Plan<F>, ProofVerifyError> {
        Ok(Plan(vec![
            Step::Reduce,
            Step::Tie {
                relation: signed_copy(node, OpRelKind::Identity),
            },
        ]))
    }
}
