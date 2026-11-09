use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::neg::SignedCopyRelation;
use crate::onnx_proof::plan::{OpPlan, OpTie, Plan, Step};
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_trace::op::Identity;

crate::impl_plan_op!(Identity, SignedCopyRelation<Identity>);

impl OpPlan for Identity {
    fn plan(
        &self,
        node: &ComputationNode,
        _model: &atlas_trace::Model,
        _io: &atlas_trace::trace::ModelExecutionIO,
    ) -> Result<Plan, ProofVerifyError> {
        Ok(Plan(vec![
            Step::Reduce,
            Step::Tie {
                relation: OpTie::Identity(
                    <SignedCopyRelation<Identity> as jolt_claims::SymbolicSumcheck>::new(
                        node.clone(),
                    ),
                ),
            },
        ]))
    }
}
