use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::{constant::public_value, OpLayout};
use crate::onnx_proof::plan::{OpPlan, Plan, Step};
use crate::onnx_proof::relation::OpRelKind;
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_trace::op::Input;
use std::sync::Arc;

impl OpLayout for Input {}

impl OpPlan for Input {
    fn plan<F: atlas_algebra::field::JoltField>(
        &self,
        node: &ComputationNode,
        _model: &atlas_trace::Model,
        io: &atlas_trace::trace::ModelExecutionIO,
    ) -> Result<Plan<F>, ProofVerifyError> {
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
                relation: public_value(node, OpRelKind::Input, Arc::new(input.clone())),
            },
        ]))
    }
}
