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
//! `rescaled` is the pre-clamp value (`VirtualPoly::ClampAcc`), recovered as the
//! floor-rebased accumulation; `R` is the rescaling remainder, appended as advice
//! (`VirtualPoly::RescaleRemainder`) and range-checked. The prover's side of the plan is in
//! `ops::prove_plan`; this module holds the trace intermediates it reads.

use atlas_trace::{Tensor, Trace};
pub use jolt_atlas_verifier::onnx_proof::fused_rebase::*;

use crate::node_view::ComputationNode;
use crate::onnx_proof::trace_view::TraceView;
use atlas_algebra::utils::lookup_bits::LookupBits;

/// The operator's rescaling remainder `R = acc mod D`, padded to the node-output
/// cycle domain, where `D` is the operator-specific divisor (usually `2^S`, but
/// e.g. `MeanOfSquares` uses `N·2^S`). Dispatches to the per-operator re-execution
/// kernel (no trace change), mirroring [`clamp_lookups::clamp_intermediate`].
pub(crate) fn rebase_remainder(node: &ComputationNode, trace: &Trace) -> Tensor<i32> {
    trace
        .fused_intermediates(node.idx)
        .unwrap_or_else(|| {
            panic!(
                "fused_rebase: node {} ({}) has no trace intermediates",
                node.idx, node.operator
            )
        })
        .remainder
        .padded_pow2()
}

/// A (padded) rescaling remainder tensor as `bits`-bit lookup indices.
pub(crate) fn remainder_lookup_bits(remainder: &Tensor<i32>, bits: i32) -> Vec<LookupBits> {
    remainder
        .data()
        .iter()
        .map(|&v| LookupBits::new(v as u64, bits as usize))
        .collect()
}
