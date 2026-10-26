use crate::node_view::ComputationNode;
use crate::utils::opening_access::AccOpeningAccessor;
use atlas_algebra::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        opening_point::{OpeningPoint, BIG_ENDIAN, LITTLE_ENDIAN},
    },
};
use atlas_sumcheck::{
    opening_proof::{OpeningAccumulator, VerifierOpeningAccumulator},
    sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
};
use atlas_transcript::Transcript;
use common::VirtualPoly;

/// Degree of the exponentiation product sumcheck.
pub const DEGREE_BOUND: usize = 3;

// TODO: refactor code to use mul sum-check

/// Shared prover/verifier parameters for softmax exp multiplication proof.
#[derive(Clone)]
pub struct MultParams<F: JoltField> {
    /// Random evaluation point.
    pub r: Vec<F>,
    /// Quantisation scale exponent.
    pub S: i32,
    /// Computation node reference.
    pub node: ComputationNode,
}

impl<F: JoltField> MultParams<F> {
    /// Create new parameters for softmax exp multiplication operation.
    pub fn new(node: ComputationNode, S: i32, accumulator: &dyn OpeningAccumulator<F>) -> Self {
        let r = AccOpeningAccessor::new(accumulator, &node)
            .get_advice(VirtualPoly::SoftmaxExpQ)
            .0
            .r;
        Self { r, S, node }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for MultParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let accessor = AccOpeningAccessor::new(accumulator, &self.node);
        let exp_q_claim = accessor.get_advice(VirtualPoly::SoftmaxExpQ).1;
        let R_claim = accessor.get_advice(VirtualPoly::SoftmaxExpRemainder).1;
        exp_q_claim * F::from_i32(self.S) + R_claim
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.r.len()
    }

    // output = eq_eval * exp_hi * exp_lo = Challenge(0) * Opening(exp_hi) * Opening(exp_lo)
}

/// Verifier for softmax exp multiplication.
pub struct MultVerifier<F: JoltField> {
    params: MultParams<F>,
}

impl<F: JoltField> MultVerifier<F> {
    /// Create new verifier for softmax exp multiplication operation.
    pub fn new(node: ComputationNode, S: i32, accumulator: &VerifierOpeningAccumulator<F>) -> Self {
        let params = MultParams::new(node, S, accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for MultVerifier<F> {
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
        provider.append_advice(VirtualPoly::SoftmaxExpHi);
        provider.append_advice(VirtualPoly::SoftmaxExpLo);
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
        let exp_hi_claim = accessor.get_advice(VirtualPoly::SoftmaxExpHi).1;
        let exp_lo_claim = accessor.get_advice(VirtualPoly::SoftmaxExpLo).1;
        EqPolynomial::mle(&self.params.r, &r_sc) * exp_lo_claim * exp_hi_claim
    }
}
