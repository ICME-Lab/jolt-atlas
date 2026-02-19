//! # Neural Network Operator Proofs Module
//!
//! This module contains proving and verification logic for all supported ONNX operations
//! in the zero-knowledge proof system. Each operator module implements the [`OperatorProofTrait`]
//! to provide:
//!
//! - **Proving**: Generate sumcheck proofs for the operation's correctness
//! - **Verification**: Verify sumcheck proofs from the prover
//! - **Committed Polynomials**: Specify which polynomials need to be committed
//!
//! ## Operator Categories
//!
//! **Arithmetic Operations**: [`add`], [`sub`], [`mul`], [`div`], [`square`], [`cube`], [`scalar_const_div`]
//!
//! **Activation Functions**: [`relu`], [`tanh`], [`softmax_axes`]
//!
//! **Tensor Operations**: [`einsum`], [`sum`], [`reshape`], [`broadcast`], [`moveaxis`], [`gather`]
//!
//! **Special Operations**: [`rsqrt`] (reciprocal square root), [`clamp`], [`iff`] (conditional select)
//!
//! **Logic Operations**: [`and`], [`is_nan`]
//!
//! **Infrastructure**: [`input`], [`constant`], [`identity`]
//!
//! ## Architecture
//!
//! Most operators follow a standard pattern using the [`impl_standard_sumcheck_proof_api`] macro,
//! implementing three key components:
//!
//! - `Params` struct: Configuration and challenge points for the sumcheck
//! - `Prover` struct: Generate sumcheck messages during proving
//! - `Verifier` struct: Verify sumcheck messages and compute expected claims

// TODO: Refactor duplicate logic in operators that use a zero-check sum-check & refactor duplicate test logic as-well
/// Element-wise addition operation.
pub mod add;
/// Logical AND operation.
pub mod and;
/// Broadcast tensors to target shapes.
pub mod broadcast;
/// Clamp values to a specified range.
pub mod clamp;
/// Constant tensor nodes.
pub mod constant;
/// Element-wise cube (x³) operation.
pub mod cube;
/// Element-wise division operation.
pub mod div;
/// Einstein summation for tensor contractions.
pub mod einsum;
/// Gather elements from input tensor using indices.
pub mod gather;
/// Identity operation (pass-through).
pub mod identity;
/// Conditional selection (if-then-else) operation.
pub mod iff;
/// Input tensor nodes.
pub mod input;
/// Check for NaN values.
pub mod is_nan;
/// Reorder tensor axes.
pub mod moveaxis;
/// Element-wise multiplication operation.
pub mod mul;
/// ReLU activation function.
pub mod relu;
/// Reshape tensor dimensions.
pub mod reshape;
/// Reciprocal square root operation.
pub mod rsqrt;
/// Division by a scalar constant.
pub mod scalar_const_div;
/// Softmax operation along specified axes.
pub mod softmax_axes;
/// Element-wise square (x²) operation.
pub mod square;
/// Element-wise subtraction operation.
pub mod sub;
/// Sum reduction along axes.
pub mod sum;
/// Tanh activation function.
pub mod tanh;

#[cfg(test)]
mod test;

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

/// Dispatches on all provable `Operator` variants, binding the inner value
/// to `$inner` and evaluating `$body`. Unhandled variants hit `$fallback`.
macro_rules! dispatch_operator {
    ($node:expr, |$inner:ident| $body:expr, _ => $fallback:expr) => {
        match &$node.operator {
            Operator::And($inner) => $body,
            Operator::Add($inner) => $body,
            Operator::Broadcast($inner) => $body,
            Operator::Constant($inner) => $body,
            Operator::Cube($inner) => $body,
            Operator::Clamp($inner) => $body,
            Operator::Div($inner) => $body,
            Operator::Einsum($inner) => $body,
            Operator::Gather($inner) => $body,
            Operator::Identity($inner) => $body,
            Operator::Iff($inner) => $body,
            Operator::Input($inner) => $body,
            Operator::IsNan($inner) => $body,
            Operator::MoveAxis($inner) => $body,
            Operator::Mul($inner) => $body,
            Operator::ReLU($inner) => $body,
            Operator::Reshape($inner) => $body,
            Operator::Rsqrt($inner) => $body,
            Operator::ScalarConstDiv($inner) => $body,
            Operator::SoftmaxAxes($inner) => $body,
            Operator::Square($inner) => $body,
            Operator::Sub($inner) => $body,
            Operator::Sum($inner) => $body,
            Operator::Tanh($inner) => $body,
            _ => $fallback,
        }
    };
}

/// Helper for retrieving committed polynomials for a computation node.
///
/// This struct dispatches to the appropriate operator's `get_committed_polynomials`
/// implementation based on the node's operator type.
pub struct NodeCommittedPolynomials;

impl NodeCommittedPolynomials {
    /// Get all committed polynomials required for proving the given computation node.
    ///
    /// Different operators require different committed polynomials (e.g., division needs
    /// quotient polynomials, range checks need address polynomials, etc.).
    ///
    /// # Arguments
    ///
    /// * `node` - The computation node to get committed polynomials for
    ///
    /// # Returns
    ///
    /// A vector of [`CommittedPolynomial`] instances that the prover must commit to
    pub fn get_committed_polynomials<F: JoltField, T: Transcript>(
        node: &ComputationNode,
    ) -> Vec<CommittedPolynomial> {
        dispatch_operator!(
            node,
            |inner| OperatorProofTrait::<F, T>::get_committed_polynomials(inner, node),
            _ => panic!("Unhandled operator in graph: {node:#?}")
        )
    }
}

/// Prover dispatcher for operator proofs.
///
/// This struct provides a single entry point for generating proofs for any computation node,
/// dispatching to the appropriate operator-specific proving logic.
pub struct OperatorProver;

impl OperatorProver {
    /// Generate a proof for the given computation node.
    ///
    /// Dispatches to the operator-specific proving implementation and returns
    /// one or more sumcheck proofs (most operators return a single proof).
    ///
    /// # Arguments
    ///
    /// * `node` - The computation node to prove
    /// * `prover` - The prover state containing trace, transcript, and accumulator
    ///
    /// # Returns
    ///
    /// A vector of (ProofId, SumcheckInstanceProof) pairs
    pub fn prove<F, T>(
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)>
    where
        F: JoltField,
        T: Transcript,
    {
        dispatch_operator!(
            node,
            |inner| inner.prove(node, prover),
            _ => panic!("Unhandled operator in graph: {node:#?}")
        )
    }
}

/// Verifier dispatcher for operator proofs.
///
/// This struct provides a single entry point for verifying proofs for any computation node,
/// dispatching to the appropriate operator-specific verification logic.
pub struct OperatorVerifier;

impl OperatorVerifier {
    /// Verify a proof for the given computation node.
    ///
    /// Dispatches to the operator-specific verification implementation to check
    /// the sumcheck proof and update the accumulator with opening claims.
    ///
    /// # Arguments
    ///
    /// * `node` - The computation node to verify
    /// * `verifier` - The verifier state containing proofs, transcript, and accumulator
    ///
    /// # Returns
    ///
    /// `Ok(())` if verification succeeds, or a [`ProofVerifyError`] if it fails
    pub fn verify<F, T>(
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError>
    where
        F: JoltField,
        T: Transcript,
    {
        dispatch_operator!(
            node,
            |inner| inner.verify(node, verifier),
            _ => Err(ProofVerifyError::MissingProof(node.idx))
        )
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
            #[tracing::instrument(skip(node, prover))]
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

            #[tracing::instrument(skip(node, verifier))]
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
