use common::VirtualPolynomial;
use joltworks::{
    field::JoltField,
    poly::opening_proof::{OpeningAccumulator, OpeningPoint, SumcheckId, BIG_ENDIAN},
    subprotocols::shout::ReadRafProvider,
    utils::math::Math,
};
pub use mult::{MultParams, MultProver, MultVerifier};

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
    /// Virtual polynomial for the looked-up value (`exp_hi` or `exp_lo`).
    fn rv_vp(self, node_idx: usize) -> VirtualPolynomial {
        match self {
            Self::Hi => VirtualPolynomial::SoftmaxExpHi(node_idx),
            Self::Lo => VirtualPolynomial::SoftmaxExpLo(node_idx),
        }
    }

    /// Virtual polynomial for the read-address-function output (`z_hi` or `z_lo`).
    fn raf_vp(self, node_idx: usize) -> VirtualPolynomial {
        match self {
            Self::Hi => VirtualPolynomial::SoftmaxExpZHi(node_idx),
            Self::Lo => VirtualPolynomial::SoftmaxExpZLo(node_idx),
        }
    }

    /// Virtual polynomial for the one-hot encoded read address.
    fn ra_vp(self, node_idx: usize) -> VirtualPolynomial {
        match self {
            Self::Hi => VirtualPolynomial::SoftmaxExpZHiRa(node_idx),
            Self::Lo => VirtualPolynomial::SoftmaxExpZLoRa(node_idx),
        }
    }
}

/// Unified Shout ReadRaf provider for exp sub-table lookups.
#[derive(Clone)]
pub struct ExpReadRafProvider {
    /// Node index in the computation graph.
    pub node_idx: usize,
    /// Number of entries in the sub-table (must be a power of two).
    pub table_size: usize,
    /// Which digit (hi or lo) this provider serves.
    pub digit: ExpDigit,
}

impl<F: JoltField> ReadRafProvider<F> for ExpReadRafProvider {
    fn log_K(&self) -> usize {
        self.table_size.log_2()
    }

    fn r(&self, accumulator: &dyn OpeningAccumulator<F>) -> OpeningPoint<BIG_ENDIAN, F> {
        accumulator
            .get_virtual_polynomial_opening(self.digit.rv_vp(self.node_idx), SumcheckId::Execution)
            .0
    }

    fn ra_poly(&self) -> (VirtualPolynomial, SumcheckId) {
        (self.digit.ra_vp(self.node_idx), SumcheckId::Execution)
    }

    fn raf_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        accumulator
            .get_virtual_polynomial_opening(self.digit.raf_vp(self.node_idx), SumcheckId::Execution)
            .1
    }

    fn rv_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        accumulator
            .get_virtual_polynomial_opening(self.digit.rv_vp(self.node_idx), SumcheckId::Execution)
            .1
    }
}
