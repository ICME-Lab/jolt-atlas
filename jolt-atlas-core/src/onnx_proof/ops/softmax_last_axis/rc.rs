use common::{CommittedPolynomial, VirtualPolynomial};
use joltworks::{
    field::JoltField,
    poly::opening_proof::{OpeningAccumulator, OpeningPoint, SumcheckId, BIG_ENDIAN},
    subprotocols::identity_range_check::IdentityRCProvider,
};

/// Range-check provider for softmax reciprocal-multiplication remainders.
pub struct RemainderRCProvider {
    /// Index of the computation node being range-checked.
    pub node_idx: usize,
    /// Scale exponent (log₂ of the range upper bound).
    pub scale: i32,
}

impl<F: JoltField> IdentityRCProvider<F> for RemainderRCProvider {
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::SoftmaxRecipMultRemainder(self.node_idx),
                SumcheckId::Execution,
            )
            .1
    }

    fn log_K(&self) -> usize {
        self.scale as usize
    }

    fn r_cycle(&self, accumulator: &dyn OpeningAccumulator<F>) -> OpeningPoint<BIG_ENDIAN, F> {
        accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::SoftmaxRecipMultRemainder(self.node_idx),
                SumcheckId::Execution,
            )
            .0
    }

    fn ra_poly(&self) -> (VirtualPolynomial, SumcheckId) {
        (
            VirtualPolynomial::SoftmaxRemainderRa(self.node_idx),
            SumcheckId::Execution,
        )
    }
}

/// Stage 2: Inv-sum diff range check provider.
///
/// Proves `diff[k] = exp_sum_q[k] − 1 − r_inv[k] ∈ [0, N·S)` for all k,
/// which is equivalent to `r_inv[k] ∈ [0, exp_sum_q[k])`.
///
/// `log_k` = log₂(N·S) = log₂(N) + scale_exponent.
/// Range-check provider for inv-sum differences.
pub struct InvSumDiffRCProvider {
    /// Index of the computation node.
    pub node_idx: usize,
    /// Number of bits for range [0, N·S), i.e. log₂(N) + scale_exponent.
    pub log_k: usize,
}

impl<F: JoltField> IdentityRCProvider<F> for InvSumDiffRCProvider {
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::SoftmaxInvSumDiff(self.node_idx),
                SumcheckId::Execution,
            )
            .1
    }

    fn log_K(&self) -> usize {
        self.log_k
    }

    fn r_cycle(&self, accumulator: &dyn OpeningAccumulator<F>) -> OpeningPoint<BIG_ENDIAN, F> {
        accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::SoftmaxInvSumDiff(self.node_idx),
                SumcheckId::Execution,
            )
            .0
    }

    fn ra_poly(&self) -> (VirtualPolynomial, SumcheckId) {
        (
            VirtualPolynomial::SoftmaxInvSumDiffRa(self.node_idx),
            SumcheckId::Execution,
        )
    }
}

/// Stage 4b: Exp-remainder range check provider.
///
/// Proves `r_exp[k,j] ∈ [0, S)` for all k,j, where
/// `r_exp[k,j] = exp_hi[k,j]·exp_lo[k,j] − exp_q[k,j]·S`.
///
/// Identical in structure to [`RemainderRCProvider`] — constant S-bit bound
/// over F·N elements — but reads from `SoftmaxExpRemainder`.
pub struct ExpRemainderRCProvider {
    /// Index of the computation node.
    pub node_idx: usize,
    /// Scale exponent (log₂ of the range upper bound).
    pub scale: i32,
}

impl<F: JoltField> IdentityRCProvider<F> for ExpRemainderRCProvider {
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::SoftmaxExpRemainder(self.node_idx),
                SumcheckId::Execution,
            )
            .1
    }

    fn log_K(&self) -> usize {
        self.scale as usize
    }

    fn r_cycle(&self, accumulator: &dyn OpeningAccumulator<F>) -> OpeningPoint<BIG_ENDIAN, F> {
        accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::SoftmaxExpRemainder(self.node_idx),
                SumcheckId::Execution,
            )
            .0
    }

    fn ra_poly(&self) -> (VirtualPolynomial, SumcheckId) {
        (
            VirtualPolynomial::SoftmaxExpRemainderRa(self.node_idx),
            SumcheckId::Execution,
        )
    }
}
