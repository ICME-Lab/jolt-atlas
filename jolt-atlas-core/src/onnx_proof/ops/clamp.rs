use crate::{
    onnx_proof::{ops::OperatorProofTrait, ProofId, Prover, Verifier},
    utils::opening_id_builder::{OpeningIdBuilder, OpeningTarget},
};
use atlas_onnx_tracer::{node::ComputationNode, ops::Clamp};
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
        let opening_id = node.build_opening_id(OpeningTarget::Input(0));
        prover.accumulator.append_virtual(
            &mut prover.transcript,
            opening_id,
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
        let opening_id = node.build_opening_id(OpeningTarget::Input(0));
        verifier
            .accumulator
            .append_virtual(&mut verifier.transcript, opening_id, opening_point);
        Ok(())
    }
}
