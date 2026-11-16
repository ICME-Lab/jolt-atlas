use crate::node_view::ComputationNode;
use crate::onnx_proof::{
    ops::sum::axis::SumAxisRelation,
    plan::{OpPlan, OpRelation, Plan, Step},
    ProofType,
};
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_trace::{op::Sum, trace::ModelExecutionIO, Model};

crate::impl_plan_op!(Sum, SumAxisRelation);

/// Axis-wise sum implementations for sumcheck protocol.
pub mod axis;

/// Exact axis-sum proof: `output[k] = Σ_axis operand[k, ·]`, where a sum outside `i32` is an
/// execution error. The proof is:
///
/// 1. **Evaluation reduction**: the node's output claims are reduced to one opening at a
///    point `r`.
/// 2. **Axis reduction** ([`SumAxisRelation`], `ProofType::SumReduction`): prove
///    `output(r) = Σ_axis operand`, reducing to the operand opening.
///
/// Unless the node is a model output, its output is shown to lie in `i32` by a range claim in
/// the packed range buckets after the node loop, as for [`Add`](super::add).
impl OpPlan for Sum {
    fn plan(
        &self,
        node: &ComputationNode,
        model: &Model,
        _io: &ModelExecutionIO,
    ) -> Result<Plan, ProofVerifyError> {
        Ok(Plan(vec![
            Step::Reduce,
            Step::Relation {
                proof: ProofType::SumReduction,
                relation: OpRelation::SumAxis(SumAxisRelation::for_node(model, node.clone())?),
            },
        ]))
    }
}
