use crate::node_view::ComputationNode;
use crate::onnx_proof::{
    ops::{sum::axis::axis_sum, OpLayout},
    schedule::{OpPlan, OpRelation, ScheduleBuilder},
    ProofType,
};
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_trace::{op::Sum, trace::ModelExecutionIO, Model};

impl OpLayout for Sum {}

/// Axis-wise sum implementations for sumcheck protocol.
pub mod axis;

/// Exact axis-sum proof: `output[k] = Σ_axis operand[k, ·]`, where a sum outside `i32` is an
/// execution error. The proof is:
///
/// 1. **Evaluation reduction**: the node's output claims are reduced to one opening at a
///    point `r`.
/// 2. **Axis reduction** ([`axis_sum`], `ProofType::SumReduction`): prove
///    `output(r) = Σ_axis operand`, reducing to the operand opening.
///
/// Unless the node is a model output, its output is shown to lie in `i32` by a range claim in
/// the packed range buckets after the node loop, as for [`Add`](super::add).
impl OpPlan for Sum {
    fn plan<F: atlas_algebra::field::JoltField>(
        &self,
        node: &ComputationNode,
        model: &Model,
        _io: &ModelExecutionIO,
        schedule: &mut ScheduleBuilder<F>,
    ) -> Result<(), ProofVerifyError> {
        schedule.reduce();
        schedule.relation(
            ProofType::SumReduction,
            OpRelation::Op(axis_sum(model, node)?),
        );
        Ok(())
    }
}
