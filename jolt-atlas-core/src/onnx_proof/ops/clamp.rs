use crate::onnx_proof::{ops::OperatorProofTrait, ProofId, Prover, Verifier};
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
use onnx_tracer::{node::ComputationNode, ops::Clamp};

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Clamp {
    #[tracing::instrument(skip_all, name = "Clamp::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        // TODO: Clamp
        // Currently this is just a dummy implementation that passes down an operand claim for the rest of the proof system
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

    #[tracing::instrument(skip_all, name = "Clamp::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let (opening_point, _claim) = verifier.accumulator.get_node_output_opening(node.idx);
        verifier.accumulator.append_virtual(
            &mut verifier.transcript,
            VirtualPolynomial::NodeOutput(node.inputs[0]),
            SumcheckId::NodeExecution(node.idx),
            opening_point,
        );
        Ok(())
    }
}
