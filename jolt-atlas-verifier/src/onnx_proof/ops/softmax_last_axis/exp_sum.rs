use crate::node_view::ComputationNode;
use crate::utils::opening_access::AccOpeningAccessor;
use atlas_algebra::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        opening_point::{OpeningPoint, BIG_ENDIAN, LITTLE_ENDIAN},
    },
    utils::math::Math,
};
use atlas_sumcheck::{
    opening_proof::{OpeningAccumulator, VerifierOpeningAccumulator},
    sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
};
use atlas_transcript::Transcript;
use common::VirtualPoly;

/// Degree of the exp-sum sumcheck.
pub const DEGREE_BOUND: usize = 2;

/// Shared prover/verifier parameters for softmax exponential sum proof.
#[derive(Clone)]
pub struct ExpSumParams<F: JoltField> {
    /// Computation node reference.
    pub node: ComputationNode,
    /// Random evaluation point.
    pub r0_k: Vec<F>,
    /// `[F, N]` — leading-dim product and last-axis size.
    pub F_N: [usize; 2],
}

impl<F: JoltField> ExpSumParams<F> {
    /// Create new parameters for exponential sum operation.
    pub fn new(
        node: ComputationNode,
        F_N: [usize; 2],
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut impl Transcript,
    ) -> Self {
        let log_f = F_N[0].log_2();
        let accessor = AccOpeningAccessor::new(accumulator, &node);
        let r = accessor.get_reduced_opening().0;
        let r0_k = r.split_at(log_f).0.r;
        Self { r0_k, node, F_N }
    }

    /// Returns log2 of the leading-dimension product `F`.
    fn log_F(&self) -> usize {
        self.F_N[0].log_2()
    }

    /// Returns log2 of the last-axis size `N`.
    pub fn log_N(&self) -> usize {
        self.F_N[1].log_2()
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for ExpSumParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let accessor = AccOpeningAccessor::new(accumulator, &self.node);
        accessor.get_advice(VirtualPoly::SoftmaxExpSum).1
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.log_F() + self.log_N()
    }

    // output = eq_eval * exp_q = Challenge(0) * Opening(exp_q)
}

/// Verifier for exponential sum in softmax.
pub struct ExpSumVerifier<F: JoltField> {
    params: ExpSumParams<F>,
}

impl<F: JoltField> ExpSumVerifier<F> {
    /// Create new verifier for exponential sum.
    pub fn new(
        node: ComputationNode,
        F_N: [usize; 2],
        accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = ExpSumParams::new(node, F_N, accumulator, transcript);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for ExpSumVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.node)
            .into_provider(transcript, opening_point);
        provider.append_advice(VirtualPoly::SoftmaxExpQ);
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r_sc = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening())
            .r;
        let accessor = AccOpeningAccessor::new(accumulator, &self.params.node);
        let exp_q_claim = accessor.get_advice(VirtualPoly::SoftmaxExpQ).1;
        // Evaluate inv_sum MLE at the leading part of the sumcheck challenge point.
        let r1_k = &r_sc[..self.params.log_F()];
        EqPolynomial::mle(&self.params.r0_k, r1_k) * (exp_q_claim)
    }
}
