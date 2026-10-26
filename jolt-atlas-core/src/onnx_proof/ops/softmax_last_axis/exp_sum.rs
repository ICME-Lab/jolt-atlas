use crate::node_view::ComputationNode;
use crate::utils::opening_access::AccOpeningAccessor;
use common::parallel::par_enabled;
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
    utils::{math::Math, thread::drop_in_background_thread},
};
use rayon::prelude::*;

const DEGREE_BOUND: usize = 2;

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
    fn log_N(&self) -> usize {
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

/// Prover for the exponential sum in softmax.
pub struct ExpSumProver<F: JoltField> {
    params: ExpSumParams<F>,
    gs_eq_r0_k: Option<GruenSplitEqPolynomial<F>>,
    eq_r0_k: Vec<F>,
    // log(F * N) variables
    exp_q: MultilinearPolynomial<F>,
}

impl<F: JoltField> ExpSumProver<F> {
    /// Constructor for exponential sum prover.
    pub fn initialize(exp_q: Vec<i32>, params: ExpSumParams<F>) -> Self {
        let eq_r0_k = EqPolynomial::evals(&params.r0_k);
        let exp_q = MultilinearPolynomial::from(exp_q);
        Self {
            params,
            eq_r0_k,
            gs_eq_r0_k: None,
            exp_q,
        }
    }

    fn compute_phase_1_message(
        &mut self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, F> {
        let m = round + 1;
        let Self { eq_r0_k, exp_q, .. } = self;
        let half_poly_len = exp_q.len() / 2;
        let eval_0 = (0..half_poly_len)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|kj| {
                let k = kj >> (self.params.log_N() - m);
                exp_q.get_bound_coeff(2 * kj) * eq_r0_k[k]
            })
            .sum();
        Ok(UnivariatePoly::from_evals_and_hint(
            previous_claim,
            &[eval_0],
        ))
    }

    fn compute_phase_2_message(&mut self, previous_claim: F) -> Result<UnivariatePoly<F>, F> {
        let Self { exp_q, .. } = self;
        let gs_eq_r0_k = self.gs_eq_r0_k.as_ref().unwrap();
        gs_eq_r0_k.gruen_round_deg_2(
            &|g| exp_q.get_bound_coeff(2 * g + 1),
            &|g| exp_q.get_bound_coeff(2 * g),
            previous_claim,
        )
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ExpSumProver<F> {
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
        provider.append_advice(VirtualPoly::SoftmaxExpQ, self.exp_q.final_claim());
    }
}

impl<F: JoltField> ExpSumProver<F> {
    /// Bind round `round`'s challenge.
    fn bind(&mut self, r: F, round: usize) {
        let r_j = F::challenge_from_field(r);
        self.exp_q.bind_parallel(r_j, BindingOrder::LowToHigh);

        if round == self.params.log_N() - 1 {
            self.gs_eq_r0_k = Some(GruenSplitEqPolynomial::new(
                &self.params.r0_k,
                jolt_poly::BindingOrder::LowToHigh,
            ));
            drop_in_background_thread(std::mem::take(&mut self.eq_r0_k));
        }
        if round >= self.params.log_N() {
            self.gs_eq_r0_k.as_mut().unwrap().bind(r);
        }
    }
}

impl<F: JoltField> ProveRounds<F> for ExpSumProver<F> {
    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    #[tracing::instrument(name = "ExpSumProver::prove_round", skip_all)]
    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(r) = bind {
            self.bind(r, round - 1);
        }
        if round < self.params.log_N() {
            self.compute_phase_1_message(round, previous_claim)
        } else {
            self.compute_phase_2_message(previous_claim)
        }
        .map_err(round_check_failed(round, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind, self.params.num_rounds() - 1);
        Ok(())
    }
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
