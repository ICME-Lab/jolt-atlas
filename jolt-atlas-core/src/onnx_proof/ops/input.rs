use crate::onnx_proof::{ops::OperatorHandler, ProofId, Prover, Verifier};
use atlas_onnx_tracer::{node::ComputationNode, ops::Input};
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

impl<F: JoltField, T: Transcript> OperatorHandler<F, T> for Input {
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        use common::VirtualPolynomial;
        use joltworks::poly::opening_proof::SumcheckId;

        // Assert claim is already cached
        let node_poly = VirtualPolynomial::NodeOutput(node.idx);
        let opening = prover
            .accumulator
            .assert_virtual_polynomial_opening_exists(node_poly, SumcheckId::Execution);
        assert!(opening.is_some());
        vec![]
    }

    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        use common::VirtualPolynomial;
        use joltworks::poly::opening_proof::SumcheckId;

        // Check input_claim == IO.evaluate_input(r_input)
        let (r_node_input, input_claim) = verifier.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NodeOutput(node.idx),
            SumcheckId::Execution,
        );
        let expected_claim =
            MultilinearPolynomial::from(verifier.io.inputs[0].clone()).evaluate(&r_node_input.r);
        if expected_claim != input_claim {
            return Err(ProofVerifyError::InvalidOpeningProof);
        }
        Ok(())
    }
}
