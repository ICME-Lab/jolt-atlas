use crate::{
    onnx_proof::{ops::OperatorProofTrait, ProofId, Prover, Verifier},
    utils::opening_access::{AccOpeningAccessor, Target},
};
use atlas_onnx_tracer::{node::ComputationNode, ops::Clamp};
use joltworks::{
    field::JoltField,
    poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
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
        let accessor = AccOpeningAccessor::new(&mut prover.accumulator, node);
        let (opening_point, _claim) = accessor.get_reduced_opening();
        let operand = prover.trace.operand_tensors(node)[0];
        let operand_claim = MultilinearPolynomial::from(operand.clone()).evaluate(&opening_point.r);
        let mut provider = accessor.into_provider(&mut prover.transcript, opening_point);
        provider.append_nodeio(Target::Input(0), operand_claim);
        vec![]
    }

    #[tracing::instrument(skip_all, name = "Clamp::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let accessor = AccOpeningAccessor::new(&mut verifier.accumulator, node);
        let (opening_point, _claim) = accessor.get_reduced_opening();
        let mut provider = accessor.into_provider(&mut verifier.transcript, opening_point);
        provider.append_nodeio(Target::Input(0));
        Ok(())
    }
}
