//! The softmax clamp lookup: `ũ = Clamp[u]` for the signed 32-bit address `u = c − x`, via
//! [`atlas_lookup_tables::clamp::SoftmaxClampTable`] (a negative address maps to 0, one above
//! the tables' last address to that address).
//!
//! The lookup's value is the table address `ũ` and its operand is `u`, both at the point the
//! softmax reads are left at; the address link in `mod.rs` ties `u` to the shift and the
//! operand.

use crate::node_view::ComputationNode;
use crate::{
    onnx_proof::op_lookups::{LookupOperandsTrait, LookupRaPolys},
    utils::opening_access::AccOpeningAccessor,
};
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_algebra::{
    field::JoltField,
    poly::point::{OpeningPoint, BIG_ENDIAN},
};
use atlas_sumcheck::opening_proof::{OpeningAccumulator, OpeningId, SumcheckId};
use common::{consts::XLEN, CommittedPoly, VirtualPoly};

/// [`LookupOperandsTrait`] helper for softmax's clamp of the address `u = c − x`.
#[derive(Default)]
pub struct SoftmaxSignificanceClampOperands;

impl LookupOperandsTrait for SoftmaxSignificanceClampOperands {
    const LOG_K: usize = XLEN;

    fn rv_claim<F: JoltField>(
        node: &ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Result<F, ProofVerifyError> {
        Ok(AccOpeningAccessor::new(accumulator, node)
            .get_advice(VirtualPoly::SoftmaxClampedAddr)?
            .1)
    }

    fn witness_opening_id(node: &ComputationNode) -> OpeningId {
        OpeningId::new(
            VirtualPoly::SoftmaxClampWitness(node.idx),
            SumcheckId::NodeExecution(node.idx),
        )
    }

    /// The point the softmax reads are left at, not this node's own output opening (the
    /// default every other helper uses).
    /// `Err` if the accumulator lacks a claim it reads.
    fn r_cycle<F: JoltField>(
        node: &ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Result<OpeningPoint<BIG_ENDIAN, F>, ProofVerifyError> {
        Ok(AccOpeningAccessor::new(accumulator, node)
            .get_advice(VirtualPoly::SoftmaxRead)?
            .0)
    }

    fn r_cycle_source(node_idx: usize) -> OpeningId {
        OpeningId::new(
            VirtualPoly::SoftmaxRead(node_idx),
            SumcheckId::NodeExecution(node_idx),
        )
    }
}

impl LookupRaPolys for SoftmaxSignificanceClampOperands {
    fn ra_virtual_poly(node_idx: usize) -> VirtualPoly {
        VirtualPoly::SoftmaxClampRa(node_idx)
    }

    fn ra_committed_poly(node_idx: usize, d: usize) -> CommittedPoly {
        CommittedPoly::SoftmaxClampRaD(node_idx, d)
    }
}
