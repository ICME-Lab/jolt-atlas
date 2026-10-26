use crate::node_view::ComputationNode;
use crate::utils::opening_access::AccOpeningAccessor;
use common::VirtualPoly;
use joltworks::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
    },
    subprotocols::sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    transcripts::Transcript,
    utils::math::Math,
};

/// Degree of the reciprocal-multiplication sumcheck.
pub const DEGREE_BOUND: usize = 3;

/// Shared prover/verifier parameters for softmax reciprocal multiplication proof.
#[derive(Clone)]
pub struct RecipMultParams<F: JoltField> {
    /// Computation node reference.
    pub node: ComputationNode,
    /// Random evaluation point.
    pub r: Vec<F>,
    /// Quantisation scale value S.
    pub S: i32,
    /// `[F, N]` — leading-dim product and last-axis size.
    pub F_N: [usize; 2],
}

impl<F: JoltField> RecipMultParams<F> {
    /// Create new parameters for reciprocal multiplication operation.
    pub fn new(
        node: ComputationNode,
        S: i32,
        F_N: [usize; 2],
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut impl Transcript,
    ) -> Self {
        let r = AccOpeningAccessor::new(accumulator, &node)
            .get_reduced_opening()
            .0
            .r;
        Self { r, S, node, F_N }
    }

    /// Returns `[F, N]`.
    pub fn F_N(&self) -> [usize; 2] {
        self.F_N
    }

    /// Returns the leading-dimension product `F`.
    pub fn F(&self) -> usize {
        self.F_N()[0]
    }

    /// Returns the last-axis size `N`.
    pub fn N(&self) -> usize {
        self.F_N()[1]
    }

    /// Returns log2 of the leading-dimension product `F`.
    fn log_F(&self) -> usize {
        self.F().log_2()
    }

    /// Returns log2 of the last-axis size `N`.
    pub fn log_N(&self) -> usize {
        self.N().log_2()
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for RecipMultParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let accessor = AccOpeningAccessor::new(accumulator, &self.node);
        let softmax_claim = accessor.get_reduced_opening().1;
        let R_claim = accessor
            .get_advice(VirtualPoly::SoftmaxRecipMultRemainder)
            .1;
        softmax_claim * F::from_i32(self.S) + R_claim
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.r.len()
    }

    // output = eq_eval * exp_q * inv_sum = Challenge(0) * Opening(exp_q)
    // Challenge(0) = eq_eval * inv_sum_claim (inv_sum is verifier-known)
}

/// Verifier for reciprocal multiplication in softmax.
pub struct RecipMultVerifier<F: JoltField> {
    params: RecipMultParams<F>,
    /// Full inv_sum vector (F elements) — used to evaluate at r_sc_leading.
    inv_sum_evals: Vec<F>,
}

impl<F: JoltField> RecipMultVerifier<F> {
    /// Create new verifier for reciprocal multiplication.
    pub fn new(
        node: ComputationNode,
        S: i32,
        F_N: [usize; 2],
        inv_sum_evals: Vec<F>,
        accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = RecipMultParams::new(node, S, F_N, accumulator, transcript);
        Self {
            params,
            inv_sum_evals,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for RecipMultVerifier<F> {
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
        // Only cache exp_q — inv_sum is verifier-known.
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
        let exp_q_claim = AccOpeningAccessor::new(accumulator, &self.params.node)
            .get_advice(VirtualPoly::SoftmaxExpQ)
            .1;
        // Evaluate inv_sum MLE at the leading part of the sumcheck challenge point.
        let r_sc_leading = &r_sc[..self.params.log_F()];
        let inv_sum_poly = MultilinearPolynomial::from(self.inv_sum_evals.clone());
        let inv_sum_claim = inv_sum_poly.evaluate(r_sc_leading);
        EqPolynomial::mle(&self.params.r, &r_sc) * (exp_q_claim * inv_sum_claim)
    }
}
