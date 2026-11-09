use crate::node_view::ComputationNode;
use crate::onnx_proof::plan::{OpPlan, OpRelation, Plan, Step};
use crate::onnx_proof::{ops::mul::ProductRelation, ProofType};
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_trace::op::And;

/// `And` on booleans is their product.
pub type AndRelation = ProductRelation<And>;

crate::impl_plan_op!(And, AndRelation);

impl OpPlan for And {
    fn plan(
        &self,
        node: &ComputationNode,
        _model: &atlas_trace::Model,
        _io: &atlas_trace::trace::ModelExecutionIO,
    ) -> Result<Plan, ProofVerifyError> {
        Ok(Plan(vec![
            Step::Reduce,
            Step::Relation {
                proof: ProofType::Execution,
                relation: OpRelation::And(<AndRelation as jolt_claims::SymbolicSumcheck>::new(
                    node.clone(),
                )),
            },
        ]))
    }
}
