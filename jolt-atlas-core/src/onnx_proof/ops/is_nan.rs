use crate::{
    onnx_proof::{ops::OperatorProofTrait, ProofId, Prover, Verifier},
    utils::opening_access::{AccOpeningAccessor, Target},
};
use atlas_onnx_tracer::{node::ComputationNode, ops::IsNan};
use joltworks::{
    field::JoltField,
    poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
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
        let accessor = AccOpeningAccessor::new(&mut prover.accumulator, node);
        let (opening_point, _claim) = accessor.get_reduced_opening();
        let operand = prover.trace.operand_tensors(node)[0];
        let operand_claim = MultilinearPolynomial::from(operand.clone()).evaluate(&opening_point.r);
        let mut provider = accessor.to_provider(&mut prover.transcript, opening_point);
        provider.append_node_io(Target::Input(0), operand_claim);
        vec![]
    }

    #[tracing::instrument(skip_all, name = "IsNan::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let accessor = AccOpeningAccessor::new(&mut verifier.accumulator, node);
        let (opening_point, claim) = accessor.get_reduced_opening();
        if claim != F::zero() {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "isNan claim should be zero".to_string(),
            ));
        }
        let mut provider = accessor.to_provider(&mut verifier.transcript, opening_point);
        provider.append_node_io(Target::Input(0));
        Ok(())
    }
}
