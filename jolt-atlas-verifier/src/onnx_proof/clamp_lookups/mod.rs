//! The saturating-clamp accumulation of `Add`/`Sub`/`Sum`/`Einsum`/`Mul`/`Square`/`Cube`/
//! `MeanOfSquares`:
//!
//! ```text
//! output(x) = SatClamp(acc(x))
//! ```
//!
//! A clamped node opens its **pre-clamp i64 accumulation** `acc` (`VirtualPoly::ClampAcc`) at
//! its output point through the generic [`OpLookupProvider`](super::op_lookups::OpLookupProvider),
//! parameterized by [`SaturatingAccClampOperands`], which appends that witness claim and nothing
//! else. The clamp itself is proven in the node's packed clamp bucket after the node loop
//! ([`clamp_split`](super::clamp_split)), or by the tie `acc = out` for a model output whose
//! public value is unsaturated. The caller ties `acc` back to the
//! operands (`acc = left ± right` for `Add`/`Sub`, the axis reduction for `Sum`, the fused
//! rescale for the others).

use super::op_lookups::{DefaultLookupOperands, LookupOperandsTrait};
use crate::node_view::ComputationNode;
use atlas_algebra::{
    field::JoltField,
    poly::point::{OpeningPoint, BIG_ENDIAN},
    utils::errors::ProofVerifyError,
};
use atlas_sumcheck::opening_proof::{OpeningAccumulator, OpeningId, SumcheckId};
use common::VirtualPoly;

/// Widest address of the saturating-clamp lookup table (an i64 accumulation).
/// Individual nodes use `node.sat_clamp_bits ∈ {40, 48, 56, 64}` (see
/// `crate::node_view`); this is the default / upper bound.
pub const CLAMP_LOG_K: usize = 64;

/// Whether `node`'s (padded) output is a single element (`log_T = 0`).
///
/// A one-element `MeanOfSquares` node has its remainder bound checked in the clear instead of
/// by a lookup range check: the remainder opens at the empty point, so its claim is the value
/// (see [`recover_small_int`]). The saturating clamp makes no such distinction.
pub fn is_scalar(node: &ComputationNode) -> bool {
    node.pow2_padded_num_output_elements() == 1
}

/// Recover the signed integer encoded by a field element known to be a
/// small-magnitude `i32`/`i64` embedding (e.g. a rescale remainder).
///
/// Returns `None` if the element is neither a small non-negative value nor a
/// small negation — a malformed claim.
pub fn recover_small_int<F: JoltField>(x: F) -> Option<i64> {
    if let Some(u) = x.to_u64() {
        i64::try_from(u).ok()
    } else {
        (-x).to_u64()
            .and_then(|u| i64::try_from(u).ok())
            .map(|n| -n)
    }
}

/// Opening id for a node's accumulation (`ClampAcc`) polynomial.
pub fn acc_opening_id(node_idx: usize) -> OpeningId {
    OpeningId::new(
        VirtualPoly::ClampAcc(node_idx),
        SumcheckId::NodeExecution(node_idx),
    )
}

/// [`LookupOperandsTrait`] helper for the 64-bit accumulator-sourced saturating clamp, backed
/// by [`SaturationTable`].
pub struct SaturatingAccClampOperands {
    /// Lookup address width for this node (`node.sat_clamp_bits`).
    log_k: usize,
}

impl SaturatingAccClampOperands {
    /// Helper for `node`, using its statically derived clamp width.
    pub fn for_node(node: &ComputationNode) -> Self {
        Self {
            log_k: node.sat_clamp_bits,
        }
    }
}

impl LookupOperandsTrait for SaturatingAccClampOperands {
    const LOG_K: usize = CLAMP_LOG_K;

    fn rv_claim<F: JoltField>(
        node: &ComputationNode,
        accumulator: &dyn atlas_sumcheck::opening_proof::OpeningAccumulator<F>,
    ) -> Result<F, ProofVerifyError> {
        DefaultLookupOperands::rv_claim(node, accumulator)
    }

    fn witness_opening_id(node: &ComputationNode) -> OpeningId {
        acc_opening_id(node.idx)
    }

    fn r_cycle<F: JoltField>(
        node: &ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Result<OpeningPoint<BIG_ENDIAN, F>, ProofVerifyError> {
        DefaultLookupOperands::r_cycle(node, accumulator)
    }

    fn r_cycle_source(node_idx: usize) -> OpeningId {
        DefaultLookupOperands::r_cycle_source(node_idx)
    }

    fn log_k(&self) -> usize {
        self.log_k
    }
}
