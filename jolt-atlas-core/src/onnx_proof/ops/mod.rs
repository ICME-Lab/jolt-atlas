// TODO: Refactor duplicate logic in operators that use a zero-check sum-check & refactor duplicate test logic as-well
pub mod add;
pub mod broadcast;
pub mod constant;
pub mod cube;
pub mod div;
pub mod einsum;
pub mod gather;
pub mod iff;
pub mod input;
pub mod moveaxis;
pub mod mul;
pub mod relu;
pub mod reshape;
pub mod rsqrt;
pub mod softmax_axes;
pub mod square;
pub mod sub;

use atlas_onnx_tracer::{node::ComputationNode, ops::Operator};
// Re-export handler types for convenient access

// Re-export operator param/prover/verifier types for the handler macro
pub use add::{AddParams, AddProver, AddVerifier};
use common::CommittedPolynomial;
pub use cube::{CubeParams, CubeProver, CubeVerifier};
pub use div::{DivParams, DivProver, DivVerifier};
pub use iff::{IffParams, IffProver, IffVerifier};
use joltworks::{
    field::JoltField, subprotocols::sumcheck::SumcheckInstanceProof, transcripts::Transcript,
    utils::errors::ProofVerifyError,
};
pub use mul::{MulParams, MulProver, MulVerifier};
pub use rsqrt::{RsqrtParams, RsqrtProver, RsqrtVerifier};
pub use square::{SquareParams, SquareProver, SquareVerifier};
pub use sub::{SubParams, SubProver, SubVerifier};

use crate::onnx_proof::{ProofId, Prover, Verifier};

/// Trait for handling operator proving and verification.
///
/// Each operator type implements this trait to encapsulate its specific
/// proving and verification logic, which is then dispatched from the main
/// proof module.
pub trait OperatorProofTrait<F: JoltField, T: Transcript> {
    /// Prove the operation for the given computation node.
    ///
    /// Returns a vector of (ProofId, Proof) pairs. Most operators return a single
    /// proof, but some (like ReLU) return multiple proofs.
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)>;

    /// Verify the operation for the given computation node.
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError>;

    /// Get the committed polynomials involved in this operator's proof.
    fn get_committed_polynomials(&self, _node: &ComputationNode) -> Vec<CommittedPolynomial> {
        vec![]
    }
}

pub struct NodeCommittedPolynomials;

// TODO: Refactor duplicate `OperatorProofTrait::<F, T>::get_committed_polynomials(...)` logic in each operator into this dispatch method
impl NodeCommittedPolynomials {
    pub fn get_committed_polynomials<F: JoltField, T: Transcript>(
        node: &ComputationNode,
    ) -> Vec<CommittedPolynomial> {
        match &node.operator {
            Operator::Add(inner) => {
                OperatorProofTrait::<F, T>::get_committed_polynomials(inner, node)
            }
            Operator::Broadcast(inner) => {
                OperatorProofTrait::<F, T>::get_committed_polynomials(inner, node)
            }
            Operator::Constant(inner) => {
                OperatorProofTrait::<F, T>::get_committed_polynomials(inner, node)
            }
            Operator::Cube(inner) => {
                OperatorProofTrait::<F, T>::get_committed_polynomials(inner, node)
            }
            Operator::Div(inner) => {
                OperatorProofTrait::<F, T>::get_committed_polynomials(inner, node)
            }
            Operator::Einsum(inner) => {
                OperatorProofTrait::<F, T>::get_committed_polynomials(inner, node)
            }
            Operator::Gather(inner) => {
                OperatorProofTrait::<F, T>::get_committed_polynomials(inner, node)
            }
            Operator::Iff(inner) => {
                OperatorProofTrait::<F, T>::get_committed_polynomials(inner, node)
            }
            Operator::Input(inner) => {
                OperatorProofTrait::<F, T>::get_committed_polynomials(inner, node)
            }
            Operator::MoveAxis(inner) => {
                OperatorProofTrait::<F, T>::get_committed_polynomials(inner, node)
            }
            Operator::Mul(inner) => {
                OperatorProofTrait::<F, T>::get_committed_polynomials(inner, node)
            }
            Operator::ReLU(inner) => {
                OperatorProofTrait::<F, T>::get_committed_polynomials(inner, node)
            }
            Operator::Reshape(inner) => {
                OperatorProofTrait::<F, T>::get_committed_polynomials(inner, node)
            }
            Operator::Rsqrt(inner) => {
                OperatorProofTrait::<F, T>::get_committed_polynomials(inner, node)
            }
            Operator::Square(inner) => {
                OperatorProofTrait::<F, T>::get_committed_polynomials(inner, node)
            }
            Operator::SoftmaxAxes(inner) => {
                OperatorProofTrait::<F, T>::get_committed_polynomials(inner, node)
            }
            Operator::Sub(inner) => {
                OperatorProofTrait::<F, T>::get_committed_polynomials(inner, node)
            }
            _ => {
                println!("Unhandled operator in graph: {node:#?}");
                vec![]
            }
        }
    }
}
pub struct OperatorProver;

impl OperatorProver {
    pub fn prove<F, T>(
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)>
    where
        F: JoltField,
        T: Transcript,
    {
        match &node.operator {
            Operator::Add(inner) => inner.prove(node, prover),
            Operator::Broadcast(inner) => inner.prove(node, prover),
            Operator::Constant(inner) => inner.prove(node, prover),
            Operator::Cube(inner) => inner.prove(node, prover),
            Operator::Div(inner) => inner.prove(node, prover),
            Operator::Einsum(inner) => inner.prove(node, prover),
            Operator::Gather(inner) => inner.prove(node, prover),
            Operator::Iff(inner) => inner.prove(node, prover),
            Operator::Input(inner) => inner.prove(node, prover),
            Operator::MoveAxis(inner) => inner.prove(node, prover),
            Operator::Mul(inner) => inner.prove(node, prover),
            Operator::ReLU(inner) => inner.prove(node, prover),
            Operator::Reshape(inner) => inner.prove(node, prover),
            Operator::Rsqrt(inner) => inner.prove(node, prover),
            Operator::Square(inner) => inner.prove(node, prover),
            Operator::SoftmaxAxes(inner) => inner.prove(node, prover),
            Operator::Sub(inner) => inner.prove(node, prover),
            _ => {
                println!("Unhandled operator in graph: {node:#?}");
                vec![]
            }
        }
    }
}

pub struct OperatorVerifier;

impl OperatorVerifier {
    pub fn verify<F, T>(
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError>
    where
        F: JoltField,
        T: Transcript,
    {
        match &node.operator {
            Operator::Add(inner) => inner.verify(node, verifier),
            Operator::Broadcast(inner) => inner.verify(node, verifier),
            Operator::Constant(inner) => inner.verify(node, verifier),
            Operator::Cube(inner) => inner.verify(node, verifier),
            Operator::Div(inner) => inner.verify(node, verifier),
            Operator::Einsum(inner) => inner.verify(node, verifier),
            Operator::Gather(inner) => inner.verify(node, verifier),
            Operator::Iff(inner) => inner.verify(node, verifier),
            Operator::Input(inner) => inner.verify(node, verifier),
            Operator::MoveAxis(inner) => inner.verify(node, verifier),
            Operator::Mul(inner) => inner.verify(node, verifier),
            Operator::Reshape(inner) => inner.verify(node, verifier),
            Operator::ReLU(inner) => inner.verify(node, verifier),
            Operator::Rsqrt(inner) => inner.verify(node, verifier),
            Operator::Square(inner) => inner.verify(node, verifier),
            Operator::SoftmaxAxes(inner) => inner.verify(node, verifier),
            Operator::Sub(inner) => inner.verify(node, verifier),
            _ => {
                tracing::warn!("Unhandled operator in graph: {node:#?}");
                Ok(())
            }
        }
    }
}

/// Macro to implement OperatorProofTrait for standard sumcheck-based operators.
///
/// These operators all follow the same pattern:
/// 1. Create params from computation node and accumulator
/// 2. Initialize prover with trace and params
/// 3. Run sumcheck and store proof
#[macro_export]
macro_rules! impl_standard_sumcheck_proof_api {
    ($inner_ty:ty, $params_ty:ident, $prover_ty:ident, $verifier_ty:ident) => {
        impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for $inner_ty {
            fn prove(
                &self,
                node: &ComputationNode,
                prover: &mut Prover<F, T>,
            ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
                use joltworks::subprotocols::sumcheck::Sumcheck;
                use $crate::onnx_proof::ops::{$params_ty, $prover_ty};

                let params = $params_ty::new(node.clone(), &prover.accumulator);
                let mut prover_sumcheck = $prover_ty::initialize(&prover.trace, params);
                let (proof, _) = Sumcheck::prove(
                    &mut prover_sumcheck,
                    &mut prover.accumulator,
                    &mut prover.transcript,
                );
                vec![(ProofId(node.idx, ProofType::Execution), proof)]
            }

            fn verify(
                &self,
                node: &ComputationNode,
                verifier: &mut Verifier<'_, F, T>,
            ) -> Result<(), ProofVerifyError> {
                use joltworks::subprotocols::sumcheck::Sumcheck;
                use $crate::onnx_proof::ops::$verifier_ty;

                let proof = verifier
                    .proofs
                    .get(&ProofId(node.idx, ProofType::Execution))
                    .ok_or(ProofVerifyError::MissingProof(node.idx))?;
                let verifier_sumcheck = $verifier_ty::new(node.clone(), &verifier.accumulator);
                Sumcheck::verify(
                    proof,
                    &verifier_sumcheck,
                    &mut verifier.accumulator,
                    &mut verifier.transcript,
                )?;
                Ok(())
            }
        }
    };
}

pub use impl_standard_sumcheck_proof_api;
