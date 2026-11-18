//! Shared fused-rescaling (exact division by `2^S`) sub-protocol.
//!
//! Several operators fuse an i64 accumulate and a rescaling division by `2^S` into a single
//! node: einsum (`Σ_k L·R`), `Mul` (`left·right`), `Square` (`x²`), and `Cube` (`x³`). They
//! all share the same proof skeleton, which lives here so each operator only has to supply
//! its own *arithmetic* sumcheck (`acc(r) = Σ … `):
//!
//! ```text
//! acc(r) = out(r)·2^S + R(r)          (the operator's arithmetic sumcheck claim)
//! R ∈ [0, 2^S)                        (a range claim in the packed range buckets)
//! out + 2^31 ∈ [0, 2^32)              (a range claim there too, unless a model output)
//! ```
//!
//! The node's output is the Euclidean quotient of the accumulation, which must lie in `i32`
//! (an execution whose quotient does not is an error and has no trace); `R` is the rescaling
//! remainder, appended as advice ([`VirtualPoly::RescaleRemainder`]) and range-checked. The
//! arithmetic relation's input expression is [`input_expression`], so its initial claim is
//! the raw `acc(r)` while its `Σ L·R` body is unchanged. The sequence is the plan
//! [`fused_rescale_plan`](super::schedule::fused_rescale_plan).

use atlas_trace::Op;

use crate::{
    node_view::ComputationNode,
    onnx_proof::relation::{RelationExpr, RelationOpening},
    utils::opening_access::OpeningIdBuilder,
};
use atlas_algebra::field::Ring;
use atlas_claims::{constant, opening};
use common::VirtualPoly;

/// Rebase width `S` (in bits) of a node's fused rescaling division at the
/// model's log2 `scale`, or `None` when the operator does not fuse a
/// power-of-two rescale (`MeanOfSquares` divides by `count·2^S` and takes its
/// own path).
///
/// `Mul`/`Square` are at scale `2S` and rebase by `2^S`; `Cube` is at scale
/// `3S` and rebases by `2^(2S)`; einsum is a (sum of) products at scale `2S`.
pub fn op_rebase_bits(op: &Op, scale: u8) -> Option<i32> {
    let scale = i32::from(scale);
    match op {
        Op::Einsum(e) => Some(if e.deferred { 0 } else { scale }),
        Op::Mul(m) => Some(if m.rebase { scale } else { 0 }),
        Op::Square(s) => Some(if s.rebase { scale } else { 0 }),
        Op::Cube(c) => Some(if c.rebase { scale * 2 } else { 0 }),
        _ => None,
    }
}

/// [`op_rebase_bits`] of a node view (which carries the model scale).
pub fn rebase_bits(node: &ComputationNode) -> Option<i32> {
    op_rebase_bits(&node.operator, node.scale)
}

/// Whether a node fuses the rescaling division (`bits > 0`).
///
/// A `bits == 0` op is a raw product (no rebase, no output range check) — e.g. the
/// building-block `Mul`/`Square` used inside decompositions.
pub fn fuses_rebase(node: &ComputationNode) -> bool {
    rebase_bits(node).is_some_and(|b| b > 0)
}

/// The arithmetic relation's input expression: `out(r₀)·2^S + rem(r₀)` over the node's reduced
/// output (the rescaled quotient) and its remainder advice, or the reduced output opening
/// alone when the node does not fuse a rescale.
pub fn input_expression<F: Ring>(node: &ComputationNode) -> RelationExpr<F> {
    let ids = OpeningIdBuilder::new(node);
    match rebase_bits(node) {
        Some(bits) if bits > 0 => {
            let remainder = ids.advice(VirtualPoly::RescaleRemainder);
            constant(F::pow2(bits as usize)) * opening(RelationOpening::ReducedNodeOutput(node.idx))
                + opening(RelationOpening::Id(remainder))
        }
        _ => opening(RelationOpening::ReducedNodeOutput(node.idx)),
    }
}
