use common::{CommittedPolynomial, VirtualPolynomial};
use joltworks::{
    config::{OneHotConfig, OneHotParams},
    field::JoltField,
    poly::opening_proof::{OpeningAccumulator, OpeningPoint, SumcheckId, BIG_ENDIAN},
    subprotocols::{identity_range_check::IdentityRCProvider, shout::RaOneHotEncoding},
};

/// Bit width for sat_diff range check.
/// 2^32 > 2^30, covering causal-mask inputs (≈ −2^30).
pub const SAT_DIFF_RC_BITS: usize = 32;

// ---------------------------------------------------------------------------
// Unified IdentityRCProvider
// ---------------------------------------------------------------------------

/// Unified range-check provider for softmax sub-protocols.
///
/// All three softmax range checks (remainder, exp-remainder, sat-diff) follow
/// the same pattern: read a claim and cycle point from one virtual polynomial,
/// and emit an `ra` polynomial for the one-hot encoding.  This struct
/// parameterises that pattern via function pointers to the relevant
/// `VirtualPolynomial` constructors.
pub struct SoftmaxRCProvider {
    /// Index of the computation node.
    pub node_idx: usize,
    log_k: usize,
    /// Virtual polynomial for the claim value and cycle point.
    source_vp: fn(usize) -> VirtualPolynomial,
    /// Virtual polynomial for the ra decomposition.
    ra_vp: fn(usize) -> VirtualPolynomial,
}

impl SoftmaxRCProvider {
    /// Range-check for reciprocal-multiplication remainders (`R[k,j] ∈ [0, S)`).
    pub fn remainder(node_idx: usize, scale: i32) -> Self {
        Self {
            node_idx,
            log_k: scale as usize,
            source_vp: VirtualPolynomial::SoftmaxRecipMultRemainder,
            ra_vp: VirtualPolynomial::SoftmaxRemainderRa,
        }
    }

    /// Range-check for exponentiation remainders (`r_exp[k,j] ∈ [0, S)`).
    pub fn exp_remainder(node_idx: usize, scale: i32) -> Self {
        Self {
            node_idx,
            log_k: scale as usize,
            source_vp: VirtualPolynomial::SoftmaxExpRemainder,
            ra_vp: VirtualPolynomial::SoftmaxExpRemainderRa,
        }
    }

    /// Range-check for saturation-diff values (`sat_diff[k,j] ∈ [0, 2^D)`).
    pub fn sat_diff(node_idx: usize) -> Self {
        Self {
            node_idx,
            log_k: SAT_DIFF_RC_BITS,
            source_vp: VirtualPolynomial::SoftmaxSatDiff,
            ra_vp: VirtualPolynomial::SoftmaxSatDiffRa,
        }
    }
}

impl<F: JoltField> IdentityRCProvider<F> for SoftmaxRCProvider {
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        accumulator
            .get_virtual_polynomial_opening((self.source_vp)(self.node_idx), SumcheckId::Execution)
            .1
    }

    fn log_K(&self) -> usize {
        self.log_k
    }

    fn r_cycle(&self, accumulator: &dyn OpeningAccumulator<F>) -> OpeningPoint<BIG_ENDIAN, F> {
        accumulator
            .get_virtual_polynomial_opening((self.source_vp)(self.node_idx), SumcheckId::Execution)
            .0
    }

    fn ra_poly(&self) -> (VirtualPolynomial, SumcheckId) {
        ((self.ra_vp)(self.node_idx), SumcheckId::Execution)
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
    committed_poly_fn: fn(usize, usize) -> CommittedPolynomial,
    r_cycle_vp: fn(usize) -> VirtualPolynomial,
    r_cycle_sc_id: SumcheckId,
    ra_vp: fn(usize) -> VirtualPolynomial,
}

impl SoftmaxRaEncoding {
    /// One-hot for reciprocal-mult remainder `ra`.
    /// Cycle point comes from the node output (`NodeExecution`).
    pub fn remainder(node_idx: usize, scale: i32) -> Self {
        Self {
            node_idx,
            log_k: scale as usize,
            committed_poly_fn: CommittedPolynomial::SoftmaxRemainderRaD,
            r_cycle_vp: VirtualPolynomial::NodeOutput,
            r_cycle_sc_id: SumcheckId::NodeExecution(node_idx),
            ra_vp: VirtualPolynomial::SoftmaxRemainderRa,
        }
    }

    /// One-hot for exponentiation remainder `ra`.
    pub fn exp_remainder(node_idx: usize, scale: i32) -> Self {
        Self {
            node_idx,
            log_k: scale as usize,
            committed_poly_fn: CommittedPolynomial::SoftmaxExpRemainderRaD,
            r_cycle_vp: VirtualPolynomial::SoftmaxExpQ,
            r_cycle_sc_id: SumcheckId::Execution,
            ra_vp: VirtualPolynomial::SoftmaxExpRemainderRa,
        }
    }

    /// One-hot for saturation-diff `ra`.
    pub fn sat_diff(node_idx: usize) -> Self {
        Self {
            node_idx,
            log_k: SAT_DIFF_RC_BITS,
            committed_poly_fn: CommittedPolynomial::SoftmaxSatDiffRaD,
            r_cycle_vp: VirtualPolynomial::SoftmaxSatDiff,
            r_cycle_sc_id: SumcheckId::Execution,
            ra_vp: VirtualPolynomial::SoftmaxSatDiffRa,
        }
    }

    /// One-hot for exp-hi Shout `ra`.
    pub fn exp_hi(node_idx: usize, log_table_size: usize) -> Self {
        Self {
            node_idx,
            log_k: log_table_size,
            committed_poly_fn: CommittedPolynomial::SoftmaxExpZHiRaD,
            r_cycle_vp: VirtualPolynomial::SoftmaxExpHi,
            r_cycle_sc_id: SumcheckId::Execution,
            ra_vp: VirtualPolynomial::SoftmaxExpZHiRa,
        }
    }

    /// One-hot for exp-lo Shout `ra`.
    pub fn exp_lo(node_idx: usize, log_table_size: usize) -> Self {
        Self {
            node_idx,
            log_k: log_table_size,
            committed_poly_fn: CommittedPolynomial::SoftmaxExpZLoRaD,
            r_cycle_vp: VirtualPolynomial::SoftmaxExpLo,
            r_cycle_sc_id: SumcheckId::Execution,
            ra_vp: VirtualPolynomial::SoftmaxExpZLoRa,
        }
    }
}

impl RaOneHotEncoding for SoftmaxRaEncoding {
    fn committed_poly(&self, d: usize) -> CommittedPolynomial {
        (self.committed_poly_fn)(self.node_idx, d)
    }

    fn r_cycle_source(&self) -> (VirtualPolynomial, SumcheckId) {
        ((self.r_cycle_vp)(self.node_idx), self.r_cycle_sc_id)
    }

    fn ra_source(&self) -> (VirtualPolynomial, SumcheckId) {
        ((self.ra_vp)(self.node_idx), SumcheckId::Execution)
    }

    fn log_k(&self) -> usize {
        self.log_k
    }

    fn one_hot_params(&self) -> OneHotParams {
        OneHotParams::from_config_and_log_K(&OneHotConfig::default(), self.log_k)
    }
}
