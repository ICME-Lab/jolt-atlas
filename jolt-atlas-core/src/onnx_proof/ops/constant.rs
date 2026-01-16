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
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        use common::VirtualPolynomial;
        use joltworks::poly::opening_proof::{OpeningPoint, SumcheckId};

        let node_poly = VirtualPolynomial::NodeOutput(node.idx);
        let opening = prover
            .accumulator
            .assert_virtual_polynomial_opening_exists(node_poly, SumcheckId::Execution);

        if opening.is_none() {
            // Handle un-needed Relu operand const
            assert!(self.0.len() == 1);
            prover.accumulator.append_virtual(
                &mut prover.transcript,
                node_poly,
                SumcheckId::Execution,
                OpeningPoint::new(vec![F::Challenge::default()]),
                F::zero(),
            );
        }
        vec![]
    }

    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        use common::VirtualPolynomial;
        use joltworks::poly::opening_proof::{OpeningPoint, SumcheckId};

        if self.0.len() == 1 {
            // Handle un-needed Relu operand const
            verifier.accumulator.append_virtual(
                &mut verifier.transcript,
                VirtualPolynomial::NodeOutput(node.idx),
                SumcheckId::Execution,
                OpeningPoint::new(vec![F::Challenge::default()]),
            );
            let (_, const_claim) = verifier.accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(node.idx),
                SumcheckId::Execution,
            );
            if F::zero() != const_claim {
                return Err(ProofVerifyError::InvalidOpeningProof);
            }
        } else {
            let (r_node_const, const_claim) = verifier.accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(node.idx),
                SumcheckId::Execution,
            );
            let expected_claim =
                MultilinearPolynomial::from(self.0.clone()).evaluate(&r_node_const.r);
            if expected_claim != const_claim {
                return Err(ProofVerifyError::InvalidOpeningProof);
            }
        }
        Ok(())
    }
}
