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
//! ([`clamp_split`](super::clamp_split)), by the tie `acc = out` for a model output whose public
//! value is unsaturated, or in the clear for a scalar node. The caller ties `acc` back to the
//! operands (`acc = left ± right` for `Add`/`Sub`, the axis reduction for `Sum`, the fused
//! rescale for the others).

use super::op_lookups::{DefaultLookupOperands, LookupOperandsTrait};
use crate::node_view::ComputationNode;
use atlas_algebra::{
    field::JoltField,
    poly::opening_point::{OpeningPoint, BIG_ENDIAN},
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
/// Such "scalar" `Add`/`Sub` nodes skip the clamp lookup: the one-hot PCS
/// opening reduction degenerates with an empty cycle dimension. This is sound
/// because a single element is opened *in the clear* at the empty point — the
/// verifier recovers the (committed, hence valid-`i32`) operands and checks the
/// saturating identity `output == SatClamp(left ± right)` directly (see
/// [`recover_small_int`]).
pub fn is_scalar(node: &ComputationNode) -> bool {
    node.pow2_padded_num_output_elements() == 1
}

/// Recover the signed integer encoded by a field element known to be a
/// small-magnitude `i32`/`i64` embedding (e.g. `left ± right` for `i32`
/// operands, which lies in `[2·i32::MIN, 2·i32::MAX]`).
///
/// Returns `None` if the element is neither a small non-negative value nor a
/// small negation — a malformed claim. Operand claims are otherwise bound to
/// committed `i32` node outputs, so honest values always recover.
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

/// Verify a scalar node's clamp directly: `output_claim == SatClamp(combined)`,
/// where `combined` is the in-the-clear accumulation (`left ± right`, or the
/// axis sum). `op` names the operator for error messages.
pub fn verify_scalar_clamp<F: JoltField>(
    combined: F,
    output_claim: F,
    op: &str,
) -> Result<(), ProofVerifyError> {
    let value = recover_small_int(combined).ok_or_else(|| {
        ProofVerifyError::InvalidOpeningProof(format!(
            "{op} (scalar): accumulation claim is not a small signed-integer encoding"
        ))
    })?;
    let expected = F::from_i32(value.clamp(i32::MIN as i64, i32::MAX as i64) as i32);
    if output_claim != expected {
        return Err(ProofVerifyError::InvalidOpeningProof(format!(
            "{op} (scalar): output must equal SatClamp(input)"
        )));
    }
    Ok(())
}
