//! # Neural Network Op Proofs Module
//!
//! This module contains proving and verification logic for all supported ONNX operations
//! in the zero-knowledge proof system. Each operator implements three traits:
//!
//! - [`OpLayout`]: the model-only facts (reduction flow, committed polynomials)
//! - [`ProveOp`]: generate sumcheck proofs for the operation's correctness
//! - [`VerifyOp`]: verify sumcheck proofs from the prover
//!
//! ## Op Categories
//!
//! **Arithmetic Operations**: [`add`], [`sub`], [`mul`], [`div`], [`square`], [`cube`], [`scalar_const_div`]
//!
//! **Activation Functions**: [`relu`], [`tanh`], [`softmax_last_axis`]
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
//! Every operator's protocol is a relation (`onnx_proof::relation`): its claim algebra, from
//! which the verifier is derived. An operator whose verification is a plan
//! (`onnx_proof::plan`: ordered steps over relations, run by the verifier's interpreter and
//! walked by the prover) binds it with `impl_plan_op`; the fused-rescale operators use
//! [`impl_fused_rescale_proof_api`]; the rest set up their relations in hand-written `ProveOp` /
//! `VerifyOp` impls.

/// Shared plumbing for the clamped Erf/Sigmoid/Tanh variants.
pub mod activation_clamped;
/// Element-wise addition operation.
pub mod add;
/// Logical AND operation.
pub mod and;
/// Broadcast tensors to target shapes.
pub mod broadcast;
/// Clamp values to a specified range.
pub mod clamp;
/// Concatenate tensors along an axis.
pub mod concat;
/// Constant tensor nodes.
pub mod constant;
/// Cos trigonometric function.
pub mod cos;
/// Element-wise cube (x³) operation.
pub mod cube;
/// Element-wise division operation.
pub mod div;
/// Einstein summation for tensor contractions.
pub mod einsum;
/// Erf activation function.
pub mod erf;
/// Op-scoped evaluation reduction helpers.
pub mod eval_reduction;
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
/// Fused mean-of-squares reduction operation.
pub mod mean_of_squares;
/// Reorder tensor axes.
pub mod moveaxis;
/// Element-wise multiplication operation.
pub mod mul;
/// Element-wise negation operation.
pub mod neg;
/// ReLU activation function.
pub mod relu;
/// Reshape tensor dimensions.
pub mod reshape;
/// Reciprocal square root operation.
pub mod rsqrt;
/// Division by a scalar constant.
pub mod scalar_const_div;
/// Sigmoid activation function.
pub mod sigmoid;
/// Sin trigonometric function.
pub mod sin;
/// Slice tensor along one axis.
pub mod slice;
/// Softmax operation along the last axis (axis = -1).
pub mod softmax_last_axis;
/// Element-wise square (x²) operation.
pub mod square;
/// Element-wise subtraction operation.
pub mod sub;
/// Sum reduction along axes.
pub mod sum;
/// Tanh activation function.
pub mod tanh;

use atlas_trace::Op;

use crate::node_view::ComputationNode;
// Re-export handler types for convenient access

// Re-export operator param/prover/verifier types for the handler macro
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_sumcheck::sumcheck::SumcheckInstanceProof;
use atlas_transcript::Transcript;
use common::CommittedPoly;
use eval_reduction::NodeEvalReduction;

use crate::onnx_proof::{ProofId, Verifier};

/// Standard execution proofs emitted by an operator.
pub type ExecutionProofs<F> = Vec<(ProofId, SumcheckInstanceProof<F>)>;

/// Eval-reduction flow mode for an operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReductionFlow {
    /// Use `prove` and run eval-reduction at the beginning (default behavior).
    Default,
    /// Use custom `prove_with_reduction` to control reduction timing.
    ///
    /// Used for operators that insert an additional opening to their output during proving
    Custom,
}

/// The model-only facts about an operator's proof: which reduction flow it uses and which
/// polynomials it commits. Prover and verifier read the same answers.
pub trait OpLayout {
    /// Select the eval-reduction flow for this operator.
    ///
    /// `Default` uses `prove`/`verify` with reduction at the beginning.
    /// `Custom` uses `prove_with_reduction`/`verify_with_reduction`.
    fn reduction_flow(&self) -> ReductionFlow {
        ReductionFlow::Default
    }

    /// Get the committed polynomials involved in this operator's proof. `Err` if the node does
    /// not have the shape the operator proves.
    fn get_committed_polynomials(
        &self,
        _node: &ComputationNode,
    ) -> Result<Vec<CommittedPoly>, ProofVerifyError> {
        Ok(vec![])
    }
}

/// An operator's verification logic, dispatched from the main proof module.
pub trait VerifyOp<F: JoltField, T: Transcript>: OpLayout {
    /// Verify the operation for the given computation node. An operator on the default
    /// reduction flow implements it; the default body rejects.
    fn verify(
        &self,
        node: &ComputationNode,
        _verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        Err(ProofVerifyError::Malformed(format!(
            "model: node {} selected the default reduction flow but its operator implements no verify",
            node.idx
        )))
    }

    /// Verify the operation and its node eval-reduction proof.
    ///
    /// Default behavior verifies eval reduction before operator verification.
    /// Operators that need additional openings can override this method and place
    /// `NodeEvalReduction::verify` at the matching verifier step. An operator on the custom
    /// reduction flow implements it; the default body rejects.
    fn verify_with_reduction(
        &self,
        node: &ComputationNode,
        _verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        Err(ProofVerifyError::Malformed(format!(
            "model: node {} selected the custom reduction flow but its operator implements no verify_with_reduction",
            node.idx
        )))
    }
}

/// Dispatches on every `Op` variant, binding the inner value to `$inner`
/// and evaluating `$body`. The match is exhaustive on purpose: adding a
/// variant to `atlas_trace::Op` must fail to compile here until the proof
/// layer handles it.
macro_rules! dispatch_operator {
    ($node:expr, |$inner:ident| $body:expr) => {
        match &$node.operator {
            Op::And($inner) => $body,
            Op::Add($inner) => $body,
            Op::Broadcast($inner) => $body,
            Op::Constant($inner) => $body,
            Op::Cos($inner) => $body,
            Op::Cube($inner) => $body,
            Op::Clamp($inner) => $body,
            Op::Concat($inner) => $body,
            Op::Div($inner) => $body,
            Op::Erf($inner) => $body,
            Op::Einsum($inner) => $body,
            Op::GatherSmall($inner) => $body,
            Op::GatherLarge($inner) => $body,
            Op::Identity($inner) => $body,
            Op::Iff($inner) => $body,
            Op::Input($inner) => $body,
            Op::IsNan($inner) => $body,
            Op::MeanOfSquares($inner) => $body,
            Op::MoveAxis($inner) => $body,
            Op::Mul($inner) => $body,
            Op::Neg($inner) => $body,
            Op::ReLU($inner) => $body,
            Op::Reshape($inner) => $body,
            Op::Rsqrt($inner) => $body,
            Op::ScalarConstDiv($inner) => $body,
            Op::Sigmoid($inner) => $body,
            Op::Sin($inner) => $body,
            Op::Slice($inner) => $body,
            Op::SoftmaxLastAxis($inner) => $body,
            Op::Square($inner) => $body,
            Op::Sub($inner) => $body,
            Op::Sum($inner) => $body,
            Op::Tanh($inner) => $body,
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
    /// A vector of [`CommittedPoly`] instances that the prover must commit to; `Err` if the
    /// node does not have the shape its operator proves.
    pub fn get_committed_polynomials(
        node: &ComputationNode,
    ) -> Result<Vec<CommittedPoly>, ProofVerifyError> {
        dispatch_operator!(node, |inner| OpLayout::get_committed_polynomials(
            inner, node
        ))
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
        dispatch_operator!(node, |inner| Self::verify_and_reduce_node(
            inner, node, verifier
        ))
    }

    fn verify_and_reduce_node<F, T, O>(
        operator: &O,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError>
    where
        F: JoltField,
        T: Transcript,
        O: VerifyOp<F, T>,
    {
        match operator.reduction_flow() {
            ReductionFlow::Default => {
                NodeEvalReduction::verify(verifier, node)?;
                operator.verify(node, verifier)
            }
            ReductionFlow::Custom => operator.verify_with_reduction(node, verifier),
        }
    }
}

/// Compile-time op binding of a relation-dispatched op: the relation's
/// [`Spec`](crate::onnx_proof::relation::AtlasRelation::Spec) must be the op it is dispatched for.
#[macro_export]
macro_rules! bind_relation {
    ($inner_ty:ty, $relation:ty) => {
        const _: () = {
            fn bound<R: $crate::onnx_proof::relation::AtlasRelation<Spec = $inner_ty>>() {}
            let _ = bound::<$relation>;
        };
    };
}

/// Macro to implement [`ProveOp`]/[`VerifyOp`] for element-wise operators that
/// fuse a rescaling division + saturating clamp (`Mul`, `Square`, `Cube`).
///
/// Wraps the operator's own arithmetic relation (emitted under [`ProofType::RescaleArith`]) with the shared
/// [`fused_rebase`](crate::onnx_proof::fused_rebase) stages:
///
/// 1. `prove_pre` — append the remainder advice + discharge the clamp.
/// 2. the operator's arithmetic sumcheck (`acc(r) = rescaled·2^S + R`).
/// 3. `prove_remainder_rc` — range-check the remainder (non-scalar only).
///
/// When the operator is *not* fused (`scale == 0`, a raw building-block product),
/// only step (2) runs and its claim is the plain node-output opening.
#[macro_export]
macro_rules! impl_fused_rescale_proof_api {
    ($inner_ty:ty, relation $relation:ty) => {
        $crate::bind_relation!($inner_ty, $relation);
        $crate::impl_fused_rescale_proof_api!(
            @impl $inner_ty,
            |node: &ComputationNode, prover: &mut Prover<F, T>| {
                use $crate::onnx_proof::relation::{RelationParams};
                let params = RelationParams::<F, $relation>::new(
                    <$relation as jolt_claims::SymbolicSumcheck>::new(node.clone()),
                    &prover.accumulator,
                    &mut prover.transcript,
                );
                RelationProver::initialize(&prover.trace, params)
            },
            |node: &ComputationNode,
             _: &atlas_trace::Model,
             accumulator: &atlas_sumcheck::opening_proof::VerifierOpeningAccumulator<F>,
             transcript: &mut T| {
                $crate::onnx_proof::relation::RelationVerifier::<F, $relation>::new(
                    <$relation as jolt_claims::SymbolicSumcheck>::new(node.clone()),
                    accumulator,
                    transcript,
                )
            }
        );
    };
    (@impl $inner_ty:ty, $make_prover:expr, $make_verifier:expr) => {
        impl $crate::onnx_proof::ops::OpLayout for $inner_ty {
            fn get_committed_polynomials(
                &self,
                node: &ComputationNode,
            ) -> Result<Vec<common::CommittedPoly>, ProofVerifyError> {
                Ok($crate::onnx_proof::fused_rebase::committed_polys(node))
            }
        }

        impl<F: JoltField, T: Transcript> $crate::onnx_proof::ops::VerifyOp<F, T> for $inner_ty {
            #[tracing::instrument(skip(node, verifier))]
            fn verify(
                &self,
                node: &ComputationNode,
                verifier: &mut Verifier<'_, F, T>,
            ) -> Result<(), ProofVerifyError> {
                use atlas_sumcheck::relation::verify_relation;
                use $crate::onnx_proof::fused_rebase;

                let fused = fused_rebase::fuses_rebase(node);
                if fused {
                    fused_rebase::verify_pre(node, verifier)?;
                }

                let proof = verifier.proofs.take(ProofId(node.idx, ProofType::RescaleArith))?;
                let verifier_sumcheck =
                    ($make_verifier)(
                        node,
                        &verifier.preprocessing.model,
                        &verifier.accumulator,
                        &mut verifier.transcript,
                    )?;
                verify_relation(
                    proof,
                    &verifier_sumcheck,
                    &mut verifier.accumulator,
                    &mut verifier.transcript,
                )?;

                if fused {
                    fused_rebase::verify_post(node, verifier)?;
                }
                Ok(())
            }
        }
    };
}

pub use impl_fused_rescale_proof_api;
