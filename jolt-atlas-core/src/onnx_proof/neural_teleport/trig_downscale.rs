//! Shared "downscale" (right-shift) stage for the Cos/Sin trig-table lookup.
//!
//! Reuses the full-domain remainder one-hot (over `XLEN` bits) established by
//! [`super::division`]'s teleportation remainder as the read address for a
//! [`RightShiftTable`] lookup, producing `downscaled = remainder >> TRIG_DOWNSCALE_BITS` —
//! the (much smaller) index the actual Cos/Sin table lookup reads. No separate
//! range-check is needed: the shift is proven purely by the one-hot's own correctness
//! (`RaVirtual`/`HammingWeight`/`Booleanity`) against the deterministic shift table.

use super::division::compute_division;
use crate::{
    onnx_proof::{op_lookups::LookupOperandsTrait, Prover, Verifier},
    utils::opening_access::AccOpeningAccessor,
};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    tensor::Tensor,
};
use common::{
    consts::{TRIG_DOWNSCALE_BITS, TRIG_PERIOD_MODULUS, XLEN},
    CommittedPoly, VirtualPoly,
};
use joltworks::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{OpeningAccumulator, OpeningId, OpeningPoint, SumcheckId, BIG_ENDIAN},
    },
    transcripts::Transcript,
    utils::lookup_bits::LookupBits,
};

/// `LookupOperandsTrait` helper for the trig downscale (right-shift) stage: reads the
/// existing full-domain `TeleportRemainder` advice as the lookup witness/address, and
/// registers the result under the shared `TrigDownscaleRa`/`TrigDownscaled` polynomials.
/// Shared by Cos and Sin (node index disambiguates).
#[derive(Default, Clone)]
pub struct TrigDownscaleOperands;

impl LookupOperandsTrait for TrigDownscaleOperands {
    const LOG_K: usize = XLEN;

    fn rv_claim<F: JoltField>(
        node: &ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> F {
        AccOpeningAccessor::new(accumulator, node)
            .get_advice(VirtualPoly::TrigDownscaled)
            .1
    }

    fn ra_virtual_poly(node_idx: usize) -> VirtualPoly {
        VirtualPoly::TrigDownscaleRa(node_idx)
    }

    fn ra_committed_poly(node_idx: usize, d: usize) -> CommittedPoly {
        CommittedPoly::TrigDownscaleRaD(node_idx, d)
    }

    fn witness_opening_id(node: &ComputationNode) -> OpeningId {
        OpeningId::new(
            VirtualPoly::TeleportRemainder(node.idx),
            SumcheckId::NodeExecution(node.idx),
        )
    }

    fn r_cycle<F: JoltField>(
        node: &ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        AccOpeningAccessor::new(accumulator, node)
            .get_advice(VirtualPoly::TeleportRemainder)
            .0
    }

    fn r_cycle_source(node_idx: usize) -> OpeningId {
        OpeningId::new(
            VirtualPoly::TeleportRemainder(node_idx),
            SumcheckId::NodeExecution(node_idx),
        )
    }

    /// The teleportation remainder, re-derived from the trace (not stored separately).
    fn witness(&self, node: &ComputationNode, trace: &Trace) -> Tensor<i64> {
        let LayerData { operands, .. } = Trace::layer_data(trace, node);
        let (_quotient, remainder) = compute_division(operands[0], TRIG_PERIOD_MODULUS as i32);
        remainder.map(|v| v as i64)
    }

    fn lookup_bits(witness: &Tensor<i64>) -> Vec<LookupBits> {
        witness
            .iter()
            .map(|&v| LookupBits::new(v as u64, XLEN))
            .collect()
    }
}

/// Appends the downscaled value (`remainder >> TRIG_DOWNSCALE_BITS`) as a
/// `VirtualPoly::TrigDownscaled` advice, evaluated at the teleportation remainder's own
/// opening point (mirrors `fused_rebase::cache_remainder_prove`). Returns the downscaled
/// tensor so the caller (the Cos/Sin table lookup) doesn't have to recompute it.
pub fn cache_downscaled_prove<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
    remainder: &Tensor<i32>,
) -> Tensor<i32> {
    let accessor = AccOpeningAccessor::new(&mut prover.accumulator, node);
    let r1 = accessor.get_advice(VirtualPoly::TeleportRemainder).0;
    let downscaled = remainder.map(|v| v >> TRIG_DOWNSCALE_BITS);
    let eval = MultilinearPolynomial::from(downscaled.clone()).evaluate(&r1.r);
    let mut provider = accessor.into_provider(&mut prover.transcript, r1);
    provider.append_advice(VirtualPoly::TrigDownscaled, eval);
    downscaled
}

/// Verifier counterpart of [`cache_downscaled_prove`].
pub fn cache_downscaled_verify<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    verifier: &mut Verifier<'_, F, T>,
) {
    let accessor = AccOpeningAccessor::new(&mut verifier.accumulator, node);
    let r1 = accessor.get_advice(VirtualPoly::TeleportRemainder).0;
    let mut provider = accessor.into_provider(&mut verifier.transcript, r1);
    provider.append_advice(VirtualPoly::TrigDownscaled);
}
