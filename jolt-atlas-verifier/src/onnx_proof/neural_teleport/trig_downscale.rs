//! Shared "downscale" (right-shift) stage for the Cos/Sin trig-table lookup.
//!
//! Right-shifts the teleportation remainder via a [`RightShiftTable`] read-raf, producing
//! `downscaled = remainder >> TRIG_DOWNSCALE_BITS` — the smaller index the Cos/Sin table
//! lookup reads. Also establishes `r`'s claim (`VirtualPoly::TeleportRemainder`) for
//! `division.rs`'s `a = P·q + r` derivation.
use crate::node_view::ComputationNode;
use crate::{
    onnx_proof::op_lookups::{LookupOperandsTrait, LookupRaPolys},
    utils::opening_access::AccOpeningAccessor,
};
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_algebra::{
    field::JoltField,
    poly::opening_point::{OpeningPoint, BIG_ENDIAN},
};
use atlas_sumcheck::opening_proof::{OpeningAccumulator, OpeningId, SumcheckId};
use common::{consts::XLEN, CommittedPoly, VirtualPoly};

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
    ) -> Result<F, ProofVerifyError> {
        Ok(AccOpeningAccessor::new(accumulator, node)
            .get_advice(VirtualPoly::TrigDownscaled)?
            .1)
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
    ) -> Result<OpeningPoint<BIG_ENDIAN, F>, ProofVerifyError> {
        Ok(AccOpeningAccessor::new(accumulator, node)
            .get_reduced_opening()?
            .0)
    }

    fn r_cycle_source(node_idx: usize) -> OpeningId {
        OpeningId::new(
            VirtualPoly::NodeOutput(node_idx),
            SumcheckId::NodeExecution(node_idx),
        )
    }
}

impl LookupRaPolys for TrigDownscaleOperands {
    fn ra_virtual_poly(node_idx: usize) -> VirtualPoly {
        VirtualPoly::TrigDownscaleRa(node_idx)
    }

    fn ra_committed_poly(node_idx: usize, d: usize) -> CommittedPoly {
        CommittedPoly::TrigDownscaleRaD(node_idx, d)
    }
}
