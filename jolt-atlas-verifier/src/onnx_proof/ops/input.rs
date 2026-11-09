use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::constant::{PublicRelation, PublicShape};
use crate::onnx_proof::plan::{OpPlan, OpTie, Plan, Step};
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_trace::op::Input;
use jolt_claims::SymbolicSumcheck;
use std::sync::Arc;

crate::impl_plan_op!(Input, PublicRelation<Input>);

impl OpPlan for Input {
    fn plan(
        &self,
        node: &ComputationNode,
        _model: &atlas_trace::Model,
        io: &atlas_trace::trace::ModelExecutionIO,
    ) -> Result<Plan, ProofVerifyError> {
        // The public tensor the IO carries for this input node.
        let input = io
            .input_indices
            .iter()
            .position(|&idx| idx == node.idx)
            .and_then(|position| io.inputs.get(position))
            .ok_or_else(|| {
                ProofVerifyError::Malformed(format!(
                    "public IO: no tensor for input node {}",
                    node.idx
                ))
            })?;
        Ok(Plan(vec![
            Step::Reduce,
            Step::Tie {
                relation: OpTie::Input(PublicRelation::new(PublicShape::new(
                    node.clone(),
                    Arc::new(input.clone()),
                ))),
            },
        ]))
    }
}
