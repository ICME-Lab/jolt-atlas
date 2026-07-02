use crate::utils::opening_access::AccOpeningAccessor;
use atlas_onnx_tracer::node::ComputationNode;
use common::{CommittedPoly, VirtualPoly};
use joltworks::{
    config::{OneHotConfig, OneHotParams},
    field::JoltField,
    poly::opening_proof::{OpeningAccumulator, OpeningId, OpeningPoint, SumcheckId, BIG_ENDIAN},
    subprotocols::{identity_range_check::IdentityRCProvider, shout::RaOneHotEncoding},
};

/// Bit width for the sat_diff range check at a given softmax scale (log₂ S).
///
/// The honest sat_diff is bounded by the attention-mask sentinel, not by 2^30:
/// with the quantize.rs extreme clamp at -(C(scale) << scale)
/// (`mask_sentinel_magnitude`, C(8)=8 / C(12)=11), masked inputs give
/// `sat_diff ≤ (score_spread + C(scale)·2^scale) − (z_bound − 1)`, which fits
/// in `log_scale + 4` bits — measured at scale 8: 429 on the whole-model
/// GPT-2 test workload (additive mask graph) and ≤ 2881 (12 bits) on the
/// replace/additive unit vehicles; at scale 12: 56888 (16 bits, GPT-2
/// layer-0 score magnitudes) — vs 30 bits under the old -2^30 clamp. This
/// drops the SatDiff one-hot commitment from d=8 to d=3 (scale 8) / d=4
/// (scale 12) committed polynomials per softmax node, each sized
/// ×(num scores).
///
/// Completeness gate: `2^bits` must exceed the model's max honest sat_diff
/// (`bits ≥ ⌈log₂(Σ+1)⌉`). Soundness does not depend on the bound: the
/// complementary-slackness sumcheck (sat_diff.rs) forcing the unique
/// `z = z_c + sat_diff` decomposition is bound-independent; only honest-prover
/// completeness needs the range to cover the witness.
pub const fn sat_diff_rc_bits(log_scale: usize) -> usize {
    log_scale + 4
}

// ---------------------------------------------------------------------------
// Unified IdentityRCProvider
// ---------------------------------------------------------------------------

/// Unified range-check provider for softmax sub-protocols.
///
/// All three softmax range checks (remainder, exp-remainder, sat-diff) follow
/// the same pattern: read a claim and cycle point from one virtual polynomial,
/// and emit an `ra` polynomial for the one-hot encoding.  This struct
/// parameterises that pattern via function pointers to the relevant
/// `VirtualPoly` constructors.
pub struct SoftmaxRCProvider {
    /// Computation node reference.
    pub node: ComputationNode,
    log_k: usize,
    /// Virtual polynomial for the claim value and cycle point.
    source_vp: fn(usize) -> VirtualPoly,
    /// Virtual polynomial for the ra decomposition.
    ra_vp: fn(usize) -> VirtualPoly,
}

impl SoftmaxRCProvider {
    /// Range-check for reciprocal-multiplication remainders (`R[k,j] ∈ [0, S)`).
    pub fn remainder(node: ComputationNode, scale: i32) -> Self {
        Self {
            node,
            log_k: scale as usize,
            source_vp: VirtualPoly::SoftmaxRecipMultRemainder,
            ra_vp: VirtualPoly::SoftmaxRemainderRa,
        }
    }

    /// Range-check for exponentiation remainders (`r_exp[k,j] ∈ [0, S)`).
    pub fn exp_remainder(node: ComputationNode, scale: i32) -> Self {
        Self {
            node,
            log_k: scale as usize,
            source_vp: VirtualPoly::SoftmaxExpRemainder,
            ra_vp: VirtualPoly::SoftmaxExpRemainderRa,
        }
    }

    /// Range-check for saturation-diff values (`sat_diff[k,j] ∈ [0, 2^D)`).
    pub fn sat_diff(node: ComputationNode, log_scale: i32) -> Self {
        Self {
            node,
            log_k: sat_diff_rc_bits(log_scale as usize),
            source_vp: VirtualPoly::SoftmaxSatDiff,
            ra_vp: VirtualPoly::SoftmaxSatDiffRa,
        }
    }
}

impl<F: JoltField> IdentityRCProvider<F> for SoftmaxRCProvider {
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let accessor = AccOpeningAccessor::new(accumulator, &self.node);
        accessor.get_advice(self.source_vp).1
    }

    fn log_K(&self) -> usize {
        self.log_k
    }

    fn r_cycle(&self, accumulator: &dyn OpeningAccumulator<F>) -> OpeningPoint<BIG_ENDIAN, F> {
        let accessor = AccOpeningAccessor::new(accumulator, &self.node);
        accessor.get_advice(self.source_vp).0
    }

    fn ra_poly(&self) -> (VirtualPoly, SumcheckId) {
        (
            (self.ra_vp)(self.node.idx),
            SumcheckId::NodeExecution(self.node.idx),
        )
    }
}

// ---------------------------------------------------------------------------
// Unified RaOneHotEncoding
// ---------------------------------------------------------------------------

/// Unified one-hot encoding for softmax `ra` polynomials.
///
/// All five softmax one-hot checks (remainder, exp-remainder, sat-diff,
/// exp-hi, exp-lo) follow the same pattern.  This struct captures the
/// differences via function pointers and a stored `SumcheckId` for the
/// cycle-point source.
pub struct SoftmaxRaEncoding {
    /// Index of the computation node.
    pub node_idx: usize,
    log_k: usize,
    committed_poly_fn: fn(usize, usize) -> CommittedPoly,
    r_cycle_vp: fn(usize) -> VirtualPoly,
    r_cycle_sc_id: SumcheckId,
    ra_vp: fn(usize) -> VirtualPoly,
}

impl SoftmaxRaEncoding {
    /// One-hot for reciprocal-mult remainder `ra`.
    /// Cycle point comes from the node output (`NodeExecution`).
    pub fn remainder(node_idx: usize, scale: i32) -> Self {
        Self {
            node_idx,
            log_k: scale as usize,
            committed_poly_fn: CommittedPoly::SoftmaxRemainderRaD,
            r_cycle_vp: VirtualPoly::NodeOutput,
            r_cycle_sc_id: SumcheckId::NodeExecution(node_idx),
            ra_vp: VirtualPoly::SoftmaxRemainderRa,
        }
    }

    /// One-hot for exponentiation remainder `ra`.
    pub fn exp_remainder(node_idx: usize, scale: i32) -> Self {
        Self {
            node_idx,
            log_k: scale as usize,
            committed_poly_fn: CommittedPoly::SoftmaxExpRemainderRaD,
            r_cycle_vp: VirtualPoly::SoftmaxExpQ,
            r_cycle_sc_id: SumcheckId::NodeExecution(node_idx),
            ra_vp: VirtualPoly::SoftmaxExpRemainderRa,
        }
    }

    /// One-hot for saturation-diff `ra`.
    pub fn sat_diff(node_idx: usize, log_scale: i32) -> Self {
        Self {
            node_idx,
            log_k: sat_diff_rc_bits(log_scale as usize),
            committed_poly_fn: CommittedPoly::SoftmaxSatDiffRaD,
            r_cycle_vp: VirtualPoly::SoftmaxSatDiff,
            r_cycle_sc_id: SumcheckId::NodeExecution(node_idx),
            ra_vp: VirtualPoly::SoftmaxSatDiffRa,
        }
    }

    /// One-hot for exp-hi Shout `ra`.
    pub fn exp_hi(node_idx: usize, log_table_size: usize) -> Self {
        Self {
            node_idx,
            log_k: log_table_size,
            committed_poly_fn: CommittedPoly::SoftmaxZHiRaD,
            r_cycle_vp: VirtualPoly::SoftmaxExpHi,
            r_cycle_sc_id: SumcheckId::NodeExecution(node_idx),
            ra_vp: VirtualPoly::SoftmaxZHiRa,
        }
    }

    /// One-hot for exp-lo Shout `ra`.
    pub fn exp_lo(node_idx: usize, log_table_size: usize) -> Self {
        Self {
            node_idx,
            log_k: log_table_size,
            committed_poly_fn: CommittedPoly::SoftmaxZLoRaD,
            r_cycle_vp: VirtualPoly::SoftmaxExpLo,
            r_cycle_sc_id: SumcheckId::NodeExecution(node_idx),
            ra_vp: VirtualPoly::SoftmaxZLoRa,
        }
    }
}

impl RaOneHotEncoding for SoftmaxRaEncoding {
    fn committed_poly(&self, d: usize) -> CommittedPoly {
        (self.committed_poly_fn)(self.node_idx, d)
    }

    fn r_cycle_source(&self) -> OpeningId {
        OpeningId::new((self.r_cycle_vp)(self.node_idx), self.r_cycle_sc_id)
    }

    fn ra_source(&self) -> OpeningId {
        OpeningId::new(
            (self.ra_vp)(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn log_k(&self) -> usize {
        self.log_k
    }

    fn one_hot_params(&self) -> OneHotParams {
        OneHotParams::from_config_and_log_K(&OneHotConfig::default(), self.log_k)
    }
}
