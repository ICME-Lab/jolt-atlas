use crate::onnx_proof::{ops::OperatorProofTrait, ProofId, Prover, Verifier};
use onnx_tracer::{node::ComputationNode, ops::IsNan};
use common::VirtualPolynomial;
use joltworks::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{OpeningAccumulator, SumcheckId},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for IsNan {
    #[tracing::instrument(skip_all, name = "IsNan::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let (opening_point, _claim) = prover.accumulator.get_node_output_opening(node.idx);
        let operand = prover.trace.operand_tensors(node)[0];
        let operand_claim = MultilinearPolynomial::from(operand.clone()).evaluate(&opening_point.r);
        prover.accumulator.append_virtual(
            &mut prover.transcript,
            VirtualPolynomial::NodeOutput(node.inputs[0]),
            SumcheckId::NodeExecution(node.idx),
            opening_point,
            operand_claim,
        );
        vec![]
    }

    #[tracing::instrument(skip_all, name = "IsNan::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let (opening_point, claim) = verifier.accumulator.get_node_output_opening(node.idx);
        if claim != F::zero() {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "isNan claim should be zero".to_string(),
            ));
        }
        verifier.accumulator.append_virtual(
            &mut verifier.transcript,
            VirtualPolynomial::NodeOutput(node.inputs[0]),
            SumcheckId::NodeExecution(node.idx),
            opening_point,
        );
        Ok(())
    }
}
