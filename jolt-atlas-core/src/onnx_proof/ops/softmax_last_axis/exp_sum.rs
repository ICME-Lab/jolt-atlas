use crate::onnx_proof::relation::{RelationSetup, RelationWitness};
use atlas_algebra::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        split_eq_poly::SplitEqUnreducedFold,
    },
    utils::thread::drop_in_background_thread,
};
use atlas_sumcheck::upstream_sumcheck::round_check_failed;
use common::parallel::par_enabled;
pub use jolt_atlas_verifier::onnx_proof::ops::softmax_last_axis::exp_sum::*;
use jolt_claims::SymbolicSumcheck;
use jolt_poly::{GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use rayon::prelude::*;

/// Prover for the exponential sum in softmax.
pub struct ExpSumProver<F: JoltField> {
    params: ExpSumParams<F>,
    setup: RelationSetup<F>,
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
            setup: params.relation_setup(),
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

impl<F: JoltField> RelationWitness<F> for ExpSumProver<F> {
    type Relation = ExpSumParams<F>;

    fn relation(&self) -> &ExpSumParams<F> {
        &self.params
    }

    fn setup(&self) -> &RelationSetup<F> {
        &self.setup
    }

    fn produced_claims(&self) -> Vec<F> {
        vec![self.exp_q.final_claim()]
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
        self.params.rounds()
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
        self.bind(bind, self.params.rounds() - 1);
        Ok(())
    }
}
