use crate::onnx_proof::{
    ops::{OperatorProofTrait, Prover, Verifier},
    ProofId,
};
use atlas_onnx_tracer::{node::ComputationNode, ops::Constant};

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

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Constant {
    #[tracing::instrument(skip_all, name = "Constant::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let _opening = prover.accumulator.get_node_output_opening(node.idx);
        vec![]
    }

    #[tracing::instrument(skip_all, name = "Constant::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let (r_node_const, const_claim) = verifier.accumulator.get_node_output_opening(node.idx);
        let constant_tensor = self.0.padded_next_power_of_two();
        let expected_claim = MultilinearPolynomial::from(constant_tensor).evaluate(&r_node_const.r);
        if expected_claim != const_claim {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Const claim does not match expected claim".to_string(),
            ));
        }
        Ok(())
    }
}
