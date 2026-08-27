//! Shared "downscale" (right-shift) stage for the Cos/Sin trig-table lookup.
//!
//! Right-shifts the teleportation remainder via a [`RightShiftTable`] read-raf, producing
//! `downscaled = remainder >> TRIG_DOWNSCALE_BITS` — the smaller index the Cos/Sin table
//! lookup reads. Also establishes `r`'s claim (`VirtualPoly::TeleportRemainder`) for
//! `division.rs`'s `a = P·q + r` derivation.
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

/// `LookupOperandsTrait` helper for the trig downscale (right-shift) stage. Registers the
/// remainder witness/address claim under `VirtualPoly::TeleportRemainder`, and the
/// read-address one-hot under `VirtualPoly::TrigDownscaleRa`/`CommittedPoly::TrigDownscaleRaD`.
/// Shared by Cos and Sin.
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
            .get_reduced_opening()
            .0
    }

    fn r_cycle_source(node_idx: usize) -> OpeningId {
        OpeningId::new(
            VirtualPoly::NodeOutput(node_idx),
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

/// `LookupOperandsTrait` helper for the remainder range-check (`remainder < TRIG_PERIOD_MODULUS`).
/// Reuses [`TrigDownscaleOperands`]'s `ra`/witness/`r_cycle` identifiers: both lookups key off
/// the same `remainder`, batched with matching round counts, so their `ra` claims are
/// mathematically identical — letting the range-check share `TrigDownscaleRa`/`RaD` instead of
/// committing and one-hot-checking a second read-address polynomial.
#[derive(Default, Clone)]
pub struct TeleportRangeCheckOperands;

impl LookupOperandsTrait for TeleportRangeCheckOperands {
    const LOG_K: usize = XLEN;

    /// Every remainder must satisfy the range check, i.e. `Σ eq(r_cycle,·)·[remainder<τ] = 1`.
    fn rv_claim<F: JoltField>(
        _node: &ComputationNode,
        _accumulator: &dyn OpeningAccumulator<F>,
    ) -> F {
        F::one()
    }

    fn ra_virtual_poly(node_idx: usize) -> VirtualPoly {
        TrigDownscaleOperands::ra_virtual_poly(node_idx)
    }

    fn ra_committed_poly(node_idx: usize, d: usize) -> CommittedPoly {
        TrigDownscaleOperands::ra_committed_poly(node_idx, d)
    }

    fn witness_opening_id(node: &ComputationNode) -> OpeningId {
        TrigDownscaleOperands::witness_opening_id(node)
    }

    fn r_cycle<F: JoltField>(
        node: &ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        TrigDownscaleOperands::r_cycle(node, accumulator)
    }

    fn r_cycle_source(node_idx: usize) -> OpeningId {
        TrigDownscaleOperands::r_cycle_source(node_idx)
    }

    fn witness(&self, node: &ComputationNode, trace: &Trace) -> Tensor<i64> {
        TrigDownscaleOperands.witness(node, trace)
    }

    fn lookup_bits(witness: &Tensor<i64>) -> Vec<LookupBits> {
        TrigDownscaleOperands::lookup_bits(witness)
    }
}

/// Appends the downscaled value (`remainder >> TRIG_DOWNSCALE_BITS`) as a
/// `VirtualPoly::TrigDownscaled` advice, evaluated at the node's standard reduced
/// output-opening point (mirrors `fused_rebase::cache_remainder_prove`). Returns the
/// downscaled tensor so the caller (the Cos/Sin table lookup) doesn't have to recompute it.
pub fn cache_downscaled_prove<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
    remainder: &Tensor<i32>,
) -> Tensor<i32> {
    let accessor = AccOpeningAccessor::new(&mut prover.accumulator, node);
    let (r_node_output, _) = accessor.get_reduced_opening();
    let downscaled = remainder.map(|v| v >> TRIG_DOWNSCALE_BITS);
    let eval = MultilinearPolynomial::from(downscaled.clone()).evaluate(&r_node_output.r);
    let mut provider = accessor.into_provider(&mut prover.transcript, r_node_output);
    provider.append_advice(VirtualPoly::TrigDownscaled, eval);
    downscaled
}

/// Verifier counterpart of [`cache_downscaled_prove`].
pub fn cache_downscaled_verify<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    verifier: &mut Verifier<'_, F, T>,
) {
    let accessor = AccOpeningAccessor::new(&mut verifier.accumulator, node);
    let (r_node_output, _) = accessor.get_reduced_opening();
    let mut provider = accessor.into_provider(&mut verifier.transcript, r_node_output);
    provider.append_advice(VirtualPoly::TrigDownscaled);
}
