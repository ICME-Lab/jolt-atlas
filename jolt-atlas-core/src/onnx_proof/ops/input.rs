use crate::node_view::ComputationNode;
use crate::{
    onnx_proof::{ops::OperatorProofTrait, ProofId, Prover, Verifier},
    utils::opening_access::AccOpeningAccessor,
};
use atlas_trace::op::Input;
use joltworks::{
    field::JoltField, poly::multilinear_polynomial::EvaluateMle,
    subprotocols::sumcheck::SumcheckInstanceProof, transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Input {
    #[tracing::instrument(skip_all, name = "Input::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        // Assert claim is already cached
        let _opening = AccOpeningAccessor::new(&prover.accumulator, node).get_reduced_opening();
        vec![]
    }

    #[tracing::instrument(skip_all, name = "Input::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        // Check input_claim == IO.evaluate_input(r_input)
        let (r_node_input, input_claim) =
            AccOpeningAccessor::new(&verifier.accumulator, node).get_reduced_opening();
        let input = &verifier.io.inputs[verifier
            .io
            .input_indices
            .iter()
            .position(|&idx| idx == node.idx)
            .unwrap()];
        let expected_claim = input.evaluate_mle(&r_node_input.r);
        if expected_claim != input_claim {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Input claim does not match expected claim".to_string(),
            ));
        }
        Ok(())
    }
}
