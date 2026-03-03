use crate::onnx_proof::{ops::OperatorProofTrait, ProofId, Prover, Verifier};
use onnx_tracer::{node::ComputationNode, ops::Input};
use joltworks::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::OpeningAccumulator,
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Input {
    #[tracing::instrument(skip_all, name = "Input::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        // Assert claim is already cached
        let _opening = prover.accumulator.get_node_output_opening(node.idx);
        vec![]
    }

    #[tracing::instrument(skip_all, name = "Input::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        // Check input_claim == IO.evaluate_input(r_input)
        let (r_node_input, input_claim) = verifier.accumulator.get_node_output_opening(node.idx);
        let input = verifier.io.inputs[verifier
            .io
            .input_indices
            .iter()
            .position(|&idx| idx == node.idx)
            .unwrap()]
        .clone();
        let expected_claim = MultilinearPolynomial::from(input).evaluate(&r_node_input.r);
        if expected_claim != input_claim {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Input claim does not match expected claim".to_string(),
            ));
        }
        Ok(())
    }
}
