use crate::compat::{node::ComputationNode, ops::Constant};
use crate::onnx_proof::{
    ops::{OperatorProofTrait, Prover, Verifier},
    ProofId,
};
use crate::utils::opening_access::AccOpeningAccessor;

use joltworks::{
    field::JoltField, poly::multilinear_polynomial::EvaluateMle,
    subprotocols::sumcheck::SumcheckInstanceProof, transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Constant {
    #[tracing::instrument(skip_all, name = "Constant::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let _opening = AccOpeningAccessor::new(&prover.accumulator, node).get_reduced_opening();
        vec![]
    }

    #[tracing::instrument(skip_all, name = "Constant::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let (r_node_const, const_claim) =
            AccOpeningAccessor::new(&verifier.accumulator, node).get_reduced_opening();
        let expected_claim = self.0.evaluate_mle(&r_node_const.r);
        if expected_claim != const_claim {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Const claim does not match expected claim".to_string(),
            ));
        }
        Ok(())
    }
}
