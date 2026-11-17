use crate::node_view::ComputationNode;
use crate::onnx_proof::{
    ops::{mul::product, OpLayout},
    plan::{OpPlan, OpRelation, Plan, Step},
    relation::OpRelKind,
    ProofType,
};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_trace::op::And;

impl OpLayout for And {}

/// `And` on booleans is their product ([`product`]).
impl OpPlan for And {
    fn plan<F: JoltField>(
        &self,
        node: &ComputationNode,
        _model: &atlas_trace::Model,
        _io: &atlas_trace::trace::ModelExecutionIO,
    ) -> Result<Plan<F>, ProofVerifyError> {
        Ok(Plan(vec![
            Step::Reduce,
            Step::Relation {
                proof: ProofType::Execution,
                relation: OpRelation::Op(product(node, OpRelKind::And, &[0, 1])),
            },
        ]))
    }
}
