//! [`LookupOperandsTrait`] helper proving softmax's saturating clamp
//! `z_c = min(z, z_bound - 1)` via [`joltworks::lookup_tables::clamp::SoftmaxSatClampTable`].
//!
//! This witness (`z = max_k - x`) is evaluated at the point softmax's exp-digit lookups
//! (`SoftmaxZHi`/`SoftmaxZLo`) already converge to (`r2`), not this node's own output opening —
//! `r_cycle`/`r_cycle_source` are overridden accordingly. `z` is always `>= 0` by construction, so
//! no offset trick is needed: this maps directly onto `ClampBoundedTable`'s floor-at-0 domain.
//!
//! The read-raf sumcheck's `rv_claim` is defined as `z_hi(r2)*base + z_lo(r2)` (computed
//! directly from the already-cached `SoftmaxZHi`/`SoftmaxZLo` claims — see `cache_z_hi_lo` in
//! `mod.rs`), rather than needing its own separate `z_c` witness/claim. This proves
//! `z_hi*base + z_lo == ClampTable[z]` as one relation: combined with `z_hi`/`z_lo` each being
//! independently bounded by the (untouched) exp-digit Shout lookups in stage 3, this forces the
//! digit split to be the true clamp of the real `z` — no separate link check needed for the
//! digit-split identity itself. `z` still needs tying back to the real input, done by
//! `operand_link` in `mod.rs`.

use crate::{
    onnx_proof::op_lookups::{DefaultLookupOperands, LookupOperandsTrait},
    utils::opening_access::AccOpeningAccessor,
};
use atlas_onnx_tracer::{
    model::trace::Trace, node::ComputationNode, ops::softmax::generate_exp_lut_decomposed,
    tensor::Tensor, utils::quantize::scale_to_multiplier,
};
use common::{
    consts::{MODEL_SCALE, XLEN},
    CommittedPoly, VirtualPoly,
};
use joltworks::{
    field::JoltField,
    poly::opening_proof::{OpeningAccumulator, OpeningId, OpeningPoint, SumcheckId, BIG_ENDIAN},
    utils::lookup_bits::LookupBits,
};

/// Precomputed-witness [`LookupOperandsTrait`] helper for softmax's saturating clamp.
///
/// `z` is computed by the caller (`softmax_z(x, max_k, last_dim)`) since the caller already has
/// both operands from `SoftmaxLastAxisTrace` — `witness()` returns it directly, ignoring the
/// `&Trace` parameter (mirrors `SaturatingAccClampOperands`'s `precomputed` field). `Option`
/// (rather than a bare `Tensor`) so `OpLookupProvider::new`/`.encoding()`'s `H: Default` bound is
/// satisfiable for verifier-only call sites, which never call `witness()`.
#[derive(Default)]
pub(crate) struct SoftmaxSatClampOperands {
    z: Option<Tensor<i64>>,
}

impl SoftmaxSatClampOperands {
    pub(crate) fn new(z: Tensor<i64>) -> Self {
        Self { z: Some(z) }
    }
}

impl LookupOperandsTrait for SoftmaxSatClampOperands {
    const LOG_K: usize = XLEN;

    fn rv_claim<F: JoltField>(
        node: &ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> F {
        let accessor = AccOpeningAccessor::new(accumulator, node);
        let z_hi_claim = accessor.get_advice(VirtualPoly::SoftmaxZHi).1;
        let z_lo_claim = accessor.get_advice(VirtualPoly::SoftmaxZLo).1;
        let base = generate_exp_lut_decomposed(scale_to_multiplier(MODEL_SCALE as i32) as i32).base;
        z_hi_claim * F::from_u64(base as u64) + z_lo_claim
    }

    fn transform_operand_claims<F: JoltField>(&self, claims: Vec<F>) -> (F, F) {
        DefaultLookupOperands.transform_operand_claims(claims)
    }

    fn transform_output_claim<F: JoltField>(&self, claim: F) -> F {
        DefaultLookupOperands.transform_output_claim(claim)
    }

    fn ra_virtual_poly(node_idx: usize) -> VirtualPoly {
        VirtualPoly::SoftmaxSatClampRa(node_idx)
    }

    fn ra_committed_poly(node_idx: usize, d: usize) -> CommittedPoly {
        CommittedPoly::SoftmaxSatClampRaD(node_idx, d)
    }

    fn witness_opening_id(node: &ComputationNode) -> OpeningId {
        OpeningId::new(
            VirtualPoly::SoftmaxSatClampWitness(node.idx),
            SumcheckId::NodeExecution(node.idx),
        )
    }

    /// Reads from `SoftmaxZHi`'s already-cached opening (`r2`, established by `cache_z_hi_lo`) —
    /// *not* this node's own output opening (the default every other helper uses).
    fn r_cycle<F: JoltField>(
        node: &ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        AccOpeningAccessor::new(accumulator, node)
            .get_advice(VirtualPoly::SoftmaxZHi)
            .0
    }

    fn r_cycle_source(node_idx: usize) -> OpeningId {
        OpeningId::new(
            VirtualPoly::SoftmaxZHi(node_idx),
            SumcheckId::NodeExecution(node_idx),
        )
    }

    fn witness(&self, _node: &ComputationNode, _trace: &Trace) -> Tensor<i64> {
        self.z
            .clone()
            .expect("SoftmaxSatClampOperands::witness called without a precomputed z (verifier-only helper)")
    }

    fn lookup_bits(witness: &Tensor<i64>) -> Vec<LookupBits> {
        DefaultLookupOperands::lookup_bits(witness)
    }
}
