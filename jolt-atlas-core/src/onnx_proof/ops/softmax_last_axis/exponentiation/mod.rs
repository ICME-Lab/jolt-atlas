use crate::utils::opening_access::AccOpeningAccessor;
use atlas_onnx_tracer::node::ComputationNode;
use common::VirtualPoly;
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

    fn r(&self, accumulator: &dyn OpeningAccumulator<F>) -> OpeningPoint<BIG_ENDIAN, F> {
        AccOpeningAccessor::new(accumulator, &self.node)
            .get_advice(|idx| self.digit.rv_vp(idx))
            .0
    }

    fn ra_poly(&self) -> (VirtualPoly, SumcheckId) {
        (
            self.digit.ra_vp(self.node.idx),
            SumcheckId::NodeExecution(self.node.idx),
        )
    }

    fn raf_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        AccOpeningAccessor::new(accumulator, &self.node)
            .get_advice(|idx| self.digit.raf_vp(idx))
            .1
    }

    fn rv_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        AccOpeningAccessor::new(accumulator, &self.node)
            .get_advice(|idx| self.digit.rv_vp(idx))
            .1
    }
}
