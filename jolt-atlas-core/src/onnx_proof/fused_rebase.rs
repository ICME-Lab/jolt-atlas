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
//! remainder, appended as advice (`VirtualPoly::RescaleRemainder`) and range-checked. The
//! prover's side of the plan is in `ops::prove_schedule`; this module holds the trace
//! intermediates it reads.

use atlas_trace::{NodeId, Tensor, Trace};
pub use jolt_atlas_verifier::onnx_proof::fused_rebase::*;

use crate::node_view::ComputationNode;
use crate::onnx_proof::trace_view::TraceView;

/// The operator's rescaling remainder `R = acc mod D`, padded to the node-output
/// cycle domain, where `D` is the operator-specific divisor (usually `2^S`, but
/// e.g. `MeanOfSquares` uses `N·2^S`), as the trace captured it.
pub(crate) fn rebase_remainder(node: &ComputationNode, trace: &Trace) -> Tensor<i64> {
    if let Some(remainder) = trace.rescale_remainder(NodeId(node.idx as u32)) {
        return remainder.padded_pow2();
    }
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
        .map(i64::from)
}
