//! # Neural Network Op Proofs Module
//!
//! The verification of every supported ONNX operation. Each operator implements two traits:
//!
//! - [`OpLayout`]: the model-only facts (the committed polynomials)
//! - [`OpPlan`]: its verification plan (`onnx_proof::plan`), run by the interpreter
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
//! which the verifier is derived. An operator's verification is its plan (`onnx_proof::plan`:
//! ordered steps over relations, run by the verifier's interpreter and walked by the prover);
//! an operator with no committed polynomials binds its relation with `impl_plan_op`.

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

use crate::node_view::ComputationNode;
use crate::onnx_proof::{plan::OpPlan, Verifier};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_sumcheck::sumcheck::SumcheckInstanceProof;
use atlas_transcript::Transcript;
use common::CommittedPoly;

use crate::onnx_proof::ProofId;

/// Standard execution proofs emitted by an operator.
pub type ExecutionProofs<F> = Vec<(ProofId, SumcheckInstanceProof<F>)>;

/// The model-only facts about an operator's proof: which polynomials it commits. Prover and
/// verifier read the same answer.
pub trait OpLayout {
    /// Get the committed polynomials involved in this operator's proof. `Err` if the node does
    /// not have the shape the operator proves.
    fn get_committed_polynomials(
        &self,
        _node: &ComputationNode,
    ) -> Result<Vec<CommittedPoly>, ProofVerifyError> {
        Ok(vec![])
    }
}

/// Dispatches on every `Op` variant, binding the inner value to `$inner`
/// and evaluating `$body`. The match is exhaustive on purpose: adding a
/// variant to `atlas_trace::Op` must fail to compile here until the proof
/// layer handles it.
#[macro_export]
macro_rules! dispatch_operator {
    ($node:expr, |$inner:ident| $body:expr) => {
        match &$node.operator {
            atlas_trace::Op::And($inner) => $body,
            atlas_trace::Op::Add($inner) => $body,
            atlas_trace::Op::Broadcast($inner) => $body,
            atlas_trace::Op::Constant($inner) => $body,
            atlas_trace::Op::Cos($inner) => $body,
            atlas_trace::Op::Cube($inner) => $body,
            atlas_trace::Op::Clamp($inner) => $body,
            atlas_trace::Op::Concat($inner) => $body,
            atlas_trace::Op::Div($inner) => $body,
            atlas_trace::Op::Erf($inner) => $body,
            atlas_trace::Op::Einsum($inner) => $body,
            atlas_trace::Op::GatherSmall($inner) => $body,
            atlas_trace::Op::GatherLarge($inner) => $body,
            atlas_trace::Op::Identity($inner) => $body,
            atlas_trace::Op::Iff($inner) => $body,
            atlas_trace::Op::Input($inner) => $body,
            atlas_trace::Op::IsNan($inner) => $body,
            atlas_trace::Op::MeanOfSquares($inner) => $body,
            atlas_trace::Op::MoveAxis($inner) => $body,
            atlas_trace::Op::Mul($inner) => $body,
            atlas_trace::Op::Neg($inner) => $body,
            atlas_trace::Op::ReLU($inner) => $body,
            atlas_trace::Op::Reshape($inner) => $body,
            atlas_trace::Op::Rsqrt($inner) => $body,
            atlas_trace::Op::ScalarConstDiv($inner) => $body,
            atlas_trace::Op::Sigmoid($inner) => $body,
            atlas_trace::Op::Sin($inner) => $body,
            atlas_trace::Op::Slice($inner) => $body,
            atlas_trace::Op::SoftmaxLastAxis($inner) => $body,
            atlas_trace::Op::Square($inner) => $body,
            atlas_trace::Op::Sub($inner) => $body,
            atlas_trace::Op::Sum($inner) => $body,
            atlas_trace::Op::Tanh($inner) => $body,
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

/// Verifier dispatcher for operator proofs: the node's operator's plan, run by the
/// interpreter.
pub struct OperatorVerifier;

impl OperatorVerifier {
    /// Verify `node`: build its operator's plan from the model and the public IO, then run it.
    pub fn verify<F, T>(
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError>
    where
        F: JoltField,
        T: Transcript,
    {
        let plan = dispatch_operator!(node, |inner| OpPlan::plan(
            inner,
            node,
            verifier.preprocessing.model(),
            verifier.io
        ))?;
        verifier.run_plan(node, plan)
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
