use crate::node_view::ComputationNode;
use crate::utils::opening_access::AccOpeningAccessor;
use common::VirtualPoly;
use jolt_poly::{GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use joltworks::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, VerifierOpeningAccumulator,
            BIG_ENDIAN, LITTLE_ENDIAN,
        },
        split_eq_poly::SplitEqUnreducedFold,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
        upstream_sumcheck::round_check_failed,
    },
    transcripts::Transcript,
};

const DEGREE_BOUND: usize = 3;

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

/// Prover state for softmax exp multiplication.
pub struct MultProver<F: JoltField> {
    params: MultParams<F>,
    eq: GruenSplitEqPolynomial<F>,
    // log(F * N) variables
    exp_hi: MultilinearPolynomial<F>,
    // log(F * N) variables
    exp_lo: MultilinearPolynomial<F>,
}

impl<F: JoltField> MultProver<F> {
    /// Constructor for softmax exp multiplication prover.
    pub fn initialize(exp_hi: Vec<i32>, exp_lo: Vec<i32>, params: MultParams<F>) -> Self {
        let eq = GruenSplitEqPolynomial::new(&params.r, jolt_poly::BindingOrder::LowToHigh);
        let exp_hi = MultilinearPolynomial::from(exp_hi);
        let exp_lo = MultilinearPolynomial::from(exp_lo);
        Self {
            params,
            eq,
            exp_hi,
            exp_lo,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for MultProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn rounds(&mut self) -> Option<&mut dyn ProveRounds<F>> {
        Some(self)
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.node)
            .into_provider(transcript, opening_point);
        provider.append_advice(VirtualPoly::SoftmaxExpHi, self.exp_hi.final_claim());
        provider.append_advice(VirtualPoly::SoftmaxExpLo, self.exp_lo.final_claim());
    }
}

impl<F: JoltField> MultProver<F> {
    fn bind(&mut self, r: F) {
        let r_j = F::challenge_from_field(r);
        self.eq.bind(r);
        self.exp_hi.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.exp_lo.bind_parallel(r_j, BindingOrder::LowToHigh);
    }
}

impl<F: JoltField> ProveRounds<F> for MultProver<F> {
    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(r) = bind {
            self.bind(r);
        }
        let Self {
            eq, exp_hi, exp_lo, ..
        } = self;
        eq.gruen_round_deg_3(
            &|g| {
                let exp_hi_0 = exp_hi.get_bound_coeff(2 * g);
                let exp_hi_inf = exp_hi.get_bound_coeff(2 * g + 1) - exp_hi_0;
                let exp_lo_0 = exp_lo.get_bound_coeff(2 * g);
                let exp_lo_inf = exp_lo.get_bound_coeff(2 * g + 1) - exp_lo_0;
                [exp_lo_0 * exp_hi_0, exp_hi_inf * exp_lo_inf]
            },
            &|g| exp_lo.get_bound_coeff(2 * g + 1) * exp_hi.get_bound_coeff(2 * g + 1),
            previous_claim,
        )
        .map_err(round_check_failed(round, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
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
