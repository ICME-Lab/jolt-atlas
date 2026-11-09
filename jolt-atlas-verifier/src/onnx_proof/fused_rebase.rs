//! Shared fused-rescaling (division + saturating clamp) sub-protocol .
//!
//! Several operators fuse an i64 accumulate, a rescaling division by `2^S`, and a
//! saturating clamp into a single node: einsum (`Σ_k L·R`), `Mul` (`left·right`),
//! `Square` (`x²`), and `Cube` (`x³`). They all share the same proof skeleton,
//! which lives here so each operator only has to supply its own *arithmetic*
//! sumcheck (`acc(r) = Σ … `):
//!
//! ```text
//! acc(r) = rescaled(r)·2^S + R(r)     (the operator's arithmetic sumcheck claim)
//! output = SatClamp(rescaled)         (clamp lookup, reuses `clamp_lookups`)
//! R ∈ [0, 2^S)                        (identity range-check on the remainder)
//! ```
//!
//! `rescaled` is the pre-clamp value ([`VirtualPoly::ClampAcc`]), recovered as the
//! floor-rebased accumulation; `R` is the rescaling remainder, appended as advice
//! ([`VirtualPoly::RescaleRemainder`]) and range-checked. The arithmetic relation's input
//! expression is [`input_expression`], so its initial claim is the raw `acc(r)` while its
//! `Σ L·R` body is unchanged. The sequence is the plan
//! [`fused_rescale_plan`](super::plan::fused_rescale_plan).
//!
//! Scalar nodes (`log_T = 0`) skip the one-hot reductions and open `rescaled` and
//! `R` in the clear (mirroring the `Add`/`Sub`/`Sum` clamp fallback).

use atlas_trace::Op;

use crate::{
    node_view::ComputationNode,
    onnx_proof::relation::{RelationExpr, RelationOpening},
    utils::opening_access::OpeningIdBuilder,
};
use common::{CommittedPoly, VirtualPoly};
use jolt_claims::{constant, opening};
use jolt_field::Ring;

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
        Op::Einsum(_) => Some(scale),
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
/// A `bits == 0` op is a raw product (no rebase, no clamp) — e.g. the
/// building-block `Mul`/`Square` used inside decompositions.
pub fn fuses_rebase(node: &ComputationNode) -> bool {
    rebase_bits(node).is_some_and(|b| b > 0)
}

/// The arithmetic relation's input expression: `q(r₀)·2^S + rem(r₀)` over the
/// node's pre-clamp quotient and remainder advice, or the reduced output opening when the node
/// does not fuse a rescale.
pub fn input_expression<F: Ring>(node: &ComputationNode) -> RelationExpr<F> {
    let ids = OpeningIdBuilder::new(node);
    match rebase_bits(node) {
        Some(bits) if bits > 0 => {
            let quotient = ids.advice(VirtualPoly::ClampAcc);
            let remainder = ids.advice(VirtualPoly::RescaleRemainder);
            constant(F::pow2(bits as usize)) * opening(RelationOpening::Id(quotient))
                + opening(RelationOpening::Id(remainder))
        }
        _ => opening(RelationOpening::ReducedNodeOutput(node.idx)),
    }
}

/// The committed one-hot decomposition polynomials a fused node must commit to:
/// the rescaling-remainder range check (`bits`-wide address) plus the saturating
/// clamp (64-bit address). Empty for non-fused or scalar nodes.
pub fn committed_polys(node: &ComputationNode) -> Vec<CommittedPoly> {
    // Both the clamp and the remainder one-hots are committed per packed
    // bucket at the model level (see `global_clamp`), never per node.
    let _ = node;
    Vec::new()
}
