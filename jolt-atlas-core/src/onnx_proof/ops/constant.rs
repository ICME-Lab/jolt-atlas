use crate::onnx_proof::{
    ops::{OperatorProofTrait, Prover, Verifier},
    ProofId,
};
use atlas_onnx_tracer::{node::ComputationNode, ops::Constant};
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

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Constant {
    #[tracing::instrument(skip_all, name = "Constant::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let node_poly = VirtualPolynomial::NodeOutput(node.idx);
        prover
            .accumulator
            .assert_virtual_polynomial_opening_exists(node_poly, SumcheckId::Execution);
        vec![]
    }

    #[tracing::instrument(skip_all, name = "Constant::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let (r_node_const, const_claim) = verifier.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NodeOutput(node.idx),
            SumcheckId::Execution,
        );
        let expected_claim = MultilinearPolynomial::from(self.0.clone()).evaluate(&r_node_const.r);
        if expected_claim != const_claim {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Const claim does not match expected claim".to_string(),
            ));
        }
        Ok(())
    }
}
