//! [`LookupOperandsTrait`] helper proving the flat-exp softmax variant's saturating clamp
//! `z_c = min(z, z_bound - 1)` via [`joltworks::lookup_tables::clamp::SoftmaxSatClampTable`] —
//! the same table type the `softmax_last_axis_satclamp` variant uses (the flat table's padded
//! size happens to land on the same `SOFTMAX_SAT_CLAMP_BOUND` at every scale checked so far;
//! see the module doc on `mod.rs` for the size derivation).
//!
//! Unlike `softmax_last_axis_satclamp::sat_clamp::SoftmaxSatClampOperands` (which anchors at
//! `r2`, the point the exp-digit lookups converge to), this variant has no digit-split stage at
//! all, so `r_cycle`/`r_cycle_source` anchor directly at `r1` (`SoftmaxExpQ`'s opening point,
//! established by stage 1's `RecipMultProver`) — one fewer intermediate point than the
//! digit-decomposed variants need.
//!
//! `rv_claim` reads back the already-cached `SoftmaxFlatZC` advice (appended by `cache_z_c` in
//! `mod.rs`) rather than computing anything on the fly, since there's no digit split to combine.

use crate::{
    onnx_proof::op_lookups::{DefaultLookupOperands, LookupOperandsTrait},
    utils::opening_access::AccOpeningAccessor,
};
use atlas_onnx_tracer::{model::trace::Trace, node::ComputationNode, tensor::Tensor};
use common::{consts::XLEN, CommittedPoly, VirtualPoly};
use joltworks::{
    field::JoltField,
    poly::opening_proof::{OpeningAccumulator, OpeningId, OpeningPoint, SumcheckId, BIG_ENDIAN},
    utils::lookup_bits::LookupBits,
};

/// Precomputed-witness [`LookupOperandsTrait`] helper for the flat-exp variant's saturating
/// clamp. Mirrors `softmax_last_axis_satclamp::sat_clamp::SoftmaxSatClampOperands`.
#[derive(Default)]
pub(crate) struct FlatZClampOperands {
    z: Option<Tensor<i64>>,
}

impl FlatZClampOperands {
    pub(crate) fn new(z: Tensor<i64>) -> Self {
        Self { z: Some(z) }
    }
}

impl LookupOperandsTrait for FlatZClampOperands {
    const LOG_K: usize = XLEN;

    fn rv_claim<F: JoltField>(
        node: &ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> F {
        AccOpeningAccessor::new(accumulator, node)
            .get_advice(VirtualPoly::SoftmaxFlatZC)
            .1
    }

    fn transform_operand_claims<F: JoltField>(&self, claims: Vec<F>) -> (F, F) {
        DefaultLookupOperands.transform_operand_claims(claims)
    }

    fn transform_output_claim<F: JoltField>(&self, claim: F) -> F {
        DefaultLookupOperands.transform_output_claim(claim)
    }

    fn ra_virtual_poly(node_idx: usize) -> VirtualPoly {
        VirtualPoly::SoftmaxFlatClampRa(node_idx)
    }

    fn ra_committed_poly(node_idx: usize, d: usize) -> CommittedPoly {
        CommittedPoly::SoftmaxFlatClampRaD(node_idx, d)
    }

    fn witness_opening_id(node: &ComputationNode) -> OpeningId {
        OpeningId::new(
            VirtualPoly::SoftmaxFlatZWitness(node.idx),
            SumcheckId::NodeExecution(node.idx),
        )
    }

    /// Reads from `SoftmaxExpQ`'s already-cached opening (`r1`, established by stage 1's
    /// `RecipMultProver`) — this variant's only anchor point.
    fn r_cycle<F: JoltField>(
        node: &ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        AccOpeningAccessor::new(accumulator, node)
            .get_advice(VirtualPoly::SoftmaxExpQ)
            .0
    }

    fn r_cycle_source(node_idx: usize) -> OpeningId {
        OpeningId::new(
            VirtualPoly::SoftmaxExpQ(node_idx),
            SumcheckId::NodeExecution(node_idx),
        )
    }

    fn witness(&self, _node: &ComputationNode, _trace: &Trace) -> Tensor<i64> {
        self.z.clone().expect(
            "FlatZClampOperands::witness called without a precomputed z (verifier-only helper)",
        )
    }

    fn lookup_bits(witness: &Tensor<i64>) -> Vec<LookupBits> {
        DefaultLookupOperands::lookup_bits(witness)
    }
}
