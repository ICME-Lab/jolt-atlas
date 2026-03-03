use crate::onnx_proof::{ops::OperatorProofTrait, ProofId, Prover, Verifier};
use common::VirtualPolynomial;
use joltworks::{
    field::JoltField,
    poly::opening_proof::{OpeningAccumulator, SumcheckId},
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};
use onnx_tracer::{node::ComputationNode, ops::Identity};

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Identity {
    #[tracing::instrument(skip_all, name = "Identity::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let node_poly_opening = prover.accumulator.get_node_output_opening(node.idx);
        let (opening_point, claim) = node_poly_opening;
        prover.accumulator.append_virtual(
            &mut prover.transcript,
            VirtualPolynomial::NodeOutput(node.inputs[0]),
            SumcheckId::NodeExecution(node.idx),
            opening_point,
            claim,
        );
        vec![]
    }

    #[tracing::instrument(skip_all, name = "Identity::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let (opening_point, claim) = verifier.accumulator.get_node_output_opening(node.idx);
        verifier.accumulator.append_virtual(
            &mut verifier.transcript,
            VirtualPolynomial::NodeOutput(node.inputs[0]),
            SumcheckId::NodeExecution(node.idx),
            opening_point,
        );
        let (_, operand_claim) = verifier.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NodeOutput(node.inputs[0]),
            SumcheckId::NodeExecution(node.idx),
        );

        if operand_claim != claim {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Identity claim should match operand claim".to_string(),
            ));
        }
        Ok(())
    }
}
