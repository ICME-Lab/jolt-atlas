use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::{OpLayout, VerifyOp};
use crate::onnx_proof::{
    ops::constant::{PublicRelation, PublicShape},
    relation::verify_tie,
    Verifier,
};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_trace::op::Input;
use atlas_transcript::Transcript;
use jolt_claims::SymbolicSumcheck;
use std::sync::Arc;

crate::bind_relation!(Input, PublicRelation<Input>);

impl OpLayout for Input {}

impl<F: JoltField, T: Transcript> VerifyOp<F, T> for Input {
    #[tracing::instrument(skip_all, name = "Input::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let position = verifier
            .io
            .input_indices
            .iter()
            .position(|&idx| idx == node.idx)
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let shape = PublicShape::new(node.clone(), Arc::new(verifier.io.inputs[position].clone()));
        verify_tie(
            &PublicRelation::<Input>::new(shape),
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )
    }
}
