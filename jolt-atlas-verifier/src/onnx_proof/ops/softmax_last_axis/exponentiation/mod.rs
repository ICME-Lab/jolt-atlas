use crate::node_view::ComputationNode;
use crate::utils::opening_access::AccOpeningAccessor;
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_algebra::{
    field::JoltField,
    poly::point::{OpeningPoint, BIG_ENDIAN},
    utils::math::Math,
};
use atlas_sumcheck::{
    config::{OneHotConfig, OneHotParams},
    opening_proof::{OpeningAccumulator, OpeningId, SumcheckId},
    shout::{RaOneHotEncoding, ReadRafProvider},
};
use common::{CommittedPoly, VirtualPoly};
pub use mult::MultParams;

/// Exp multiplication relation prover/verifier.
pub mod mult;

// ---------------------------------------------------------------------------
// Unified ReadRafProvider for exp sub-table lookups
// ---------------------------------------------------------------------------

/// Selects whether we are operating on the high or low digit of the
/// decomposed exponentiation.
#[derive(Clone, Copy)]
pub enum ExpDigit {
    /// High digit of the decomposed exponentiation.
    Hi,
    /// Low digit of the decomposed exponentiation.
    Lo,
}

impl ExpDigit {
    /// Virtual polynomial for the read-values value (`exp_hi` or `exp_lo`).
    fn rv_vp(self, node_idx: usize) -> VirtualPoly {
        match self {
            Self::Hi => VirtualPoly::SoftmaxExpHi(node_idx),
            Self::Lo => VirtualPoly::SoftmaxExpLo(node_idx),
        }
    }

    /// vp type for raf polynomials (`z_hi` or `z_lo`).
    fn raf_vp(self, node_idx: usize) -> VirtualPoly {
        match self {
            Self::Hi => VirtualPoly::SoftmaxZHi(node_idx),
            Self::Lo => VirtualPoly::SoftmaxZLo(node_idx),
        }
    }

    /// Virtual polynomial for the one-hot encoded read address.
    fn ra_vp(self, node_idx: usize) -> VirtualPoly {
        match self {
            Self::Hi => VirtualPoly::SoftmaxZHiRa(node_idx),
            Self::Lo => VirtualPoly::SoftmaxZLoRa(node_idx),
        }
    }
}

/// Unified Shout ReadRaf provider for exp sub-table lookups.
#[derive(Clone)]
pub struct ExpReadRafProvider {
    /// Computation node reference.
    pub node: ComputationNode,
    /// Number of entries in the sub-table (must be a power of two).
    pub table_size: usize,
    /// Which digit (hi or lo) this provider serves.
    pub digit: ExpDigit,
}

impl<F: JoltField> ReadRafProvider<F> for ExpReadRafProvider {
    fn log_K(&self) -> usize {
        self.table_size.log_2()
    }

    fn r(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Result<OpeningPoint<BIG_ENDIAN, F>, ProofVerifyError> {
        Ok(AccOpeningAccessor::new(accumulator, &self.node)
            .get_advice(|idx| self.digit.rv_vp(idx))?
            .0)
    }

    fn ra_poly(&self) -> (VirtualPoly, SumcheckId) {
        (
            self.digit.ra_vp(self.node.idx),
            SumcheckId::NodeExecution(self.node.idx),
        )
    }

    fn raf_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> Result<F, ProofVerifyError> {
        Ok(AccOpeningAccessor::new(accumulator, &self.node)
            .get_advice(|idx| self.digit.raf_vp(idx))?
            .1)
    }

    fn rv_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> Result<F, ProofVerifyError> {
        Ok(AccOpeningAccessor::new(accumulator, &self.node)
            .get_advice(|idx| self.digit.rv_vp(idx))?
            .1)
    }
}

// ---------------------------------------------------------------------------
// One-hot encodings of the exp sub-table read addresses
// ---------------------------------------------------------------------------

/// One-hot encoding of a softmax exp sub-table lookup's read addresses (`exp_hi` or `exp_lo`).
pub struct SoftmaxRaEncoding {
    /// Index of the computation node.
    pub node_idx: usize,
    log_k: usize,
    committed_poly_fn: fn(usize, usize) -> CommittedPoly,
    r_cycle_vp: fn(usize) -> VirtualPoly,
    ra_vp: fn(usize) -> VirtualPoly,
}

impl SoftmaxRaEncoding {
    /// One-hot for exp-hi Shout `ra`.
    pub fn exp_hi(node_idx: usize, log_table_size: usize) -> Self {
        Self {
            node_idx,
            log_k: log_table_size,
            committed_poly_fn: CommittedPoly::SoftmaxZHiRaD,
            r_cycle_vp: VirtualPoly::SoftmaxExpHi,
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
            ra_vp: VirtualPoly::SoftmaxZLoRa,
        }
    }
}

impl RaOneHotEncoding for SoftmaxRaEncoding {
    fn committed_poly(&self, d: usize) -> CommittedPoly {
        (self.committed_poly_fn)(self.node_idx, d)
    }

    fn r_cycle_source(&self) -> OpeningId {
        OpeningId::new(
            (self.r_cycle_vp)(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
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
