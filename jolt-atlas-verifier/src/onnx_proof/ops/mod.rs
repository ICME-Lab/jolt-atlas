//! # Neural Network Op Proofs Module
//!
//! The verification of every supported ONNX operation. Each operator's `atlas_trace` payload
//! type implements [`OpPlan`](crate::onnx_proof::schedule::OpPlan): its steps of the model's
//! schedule (`onnx_proof::schedule`), run by the interpreter. The polynomials a proof commits
//! are the ones those steps open
//! ([`Schedule::committed`](crate::onnx_proof::schedule::Schedule::committed)).
//!
//! ## Op Categories
//!
//! **Arithmetic Operations**: [`add`], [`sub`], [`neg`], [`mul`], [`square`], [`cube`],
//! [`scalar_const_div`], [`mean_of_squares`]
//!
//! **Activation Functions**: [`relu`], [`activation_clamped`], [`smooth_relu`], [`trig`],
//! [`softmax_last_axis`]
//!
//! **Tensor Operations**: [`einsum`], [`sum`], [`reshape`], [`broadcast`], [`moveaxis`],
//! [`slice`], [`concat`], [`gather`]
//!
//! **Special Operations**: [`rsqrt`] (reciprocal square root), [`iff`] (conditional select)
//!
//! **Logic Operations**: [`and`], [`is_nan`]
//!
//! **Infrastructure**: [`input`], [`constant`], [`identity`], [`eval_reduction`]
//!
//! ## Architecture
//!
//! Every operator's protocol is a relation (`onnx_proof::relation`): its claim algebra, from
//! which the verifier is derived. An operator's verification is its plan: the ordered steps
//! over relations it writes to the model's schedule (`onnx_proof::schedule`), run by the
//! verifier's interpreter and walked by the prover.
//! An op's relations are values ([`OpRel`](crate::onnx_proof::relation::OpRel)) built by
//! constructor functions in its module, each documenting its S ⊆ R argument; the constructors
//! are called from the op's plan, so a relation is built only for a node of its op. The trig
//! ops share [`trig`] (the period reduction, the downscale, the Sin and Cos plans); the small
//! tables and the prefix-suffix reads are `onnx_proof::lookup`.

/// The clamped activations Erf, Sigmoid and Tanh.
pub mod activation_clamped;
/// Element-wise addition operation.
pub mod add;
/// Logical AND operation.
pub mod and;
/// Broadcast tensors to target shapes.
pub mod broadcast;
/// Concatenate tensors along an axis.
pub mod concat;
/// Constant tensor nodes.
pub mod constant;
/// Element-wise cube (x³) operation.
pub mod cube;
/// Einstein summation for tensor contractions.
pub mod einsum;
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
/// Rescale: a wide value taken to the model's scale.
pub mod rescale;
/// Reshape tensor dimensions.
pub mod reshape;
/// Reciprocal square root operation.
pub mod rsqrt;
/// Division by a scalar constant.
pub mod scalar_const_div;
/// Slice tensor along one axis.
pub mod slice;
/// The smooth ReLUs (GELU, SiLU): ReLU plus a table-lookup correction.
pub mod smooth_relu;
/// Softmax operation along the last axis (axis = -1).
pub mod softmax_last_axis;
/// Element-wise square (x²) operation.
pub mod square;
/// Element-wise subtraction operation.
pub mod sub;
/// Sum reduction along axes.
pub mod sum;
/// Sin and Cos: the period reduction, the downscale and the table read.
pub mod trig;

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
            atlas_trace::Op::Concat($inner) => $body,
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
            atlas_trace::Op::Rescale($inner) => $body,
            atlas_trace::Op::SmoothRelu($inner) => $body,
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
