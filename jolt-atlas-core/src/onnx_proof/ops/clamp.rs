use crate::onnx_proof::{ops::OperatorProofTrait, ProofId, Prover, Verifier};
use atlas_onnx_tracer::{node::ComputationNode, ops::Clamp};
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

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Clamp {
    #[tracing::instrument(skip_all, name = "Clamp::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        // TODO: Clamp
        // Currently this is just a dummy implementation that passes down an operand claim for the rest of the proof system
        let node_poly = VirtualPolynomial::NodeOutput(node.idx);
        let (opening_point, _claim) = prover
            .accumulator
            .get_virtual_polynomial_opening(node_poly, SumcheckId::Execution);
        let operand = prover.trace.operand_tensors(node)[0];
        let operand_claim = MultilinearPolynomial::from(operand.clone()).evaluate(&opening_point.r);
        prover.accumulator.append_virtual(
            &mut prover.transcript,
            VirtualPolynomial::NodeOutput(node.inputs[0]),
            SumcheckId::Execution,
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
        let node_poly = VirtualPolynomial::NodeOutput(node.idx);
        let (opening_point, _claim) = verifier
            .accumulator
            .get_virtual_polynomial_opening(node_poly, SumcheckId::Execution);
        verifier.accumulator.append_virtual(
            &mut verifier.transcript,
            VirtualPolynomial::NodeOutput(node.inputs[0]),
            SumcheckId::Execution,
            opening_point,
        );
        Ok(())
    }
}
