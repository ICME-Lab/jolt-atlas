use crate::onnx_proof::relation::{RelationSetup, RelationWitness};
use common::parallel::par_enabled;
pub use jolt_atlas_verifier::onnx_proof::ops::softmax_last_axis::max::*;
use jolt_claims::SymbolicSumcheck;
use std::array;

use atlas_algebra::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        split_eq_poly::SplitEqUnreducedFold,
    },
    utils::thread::drop_in_background_thread,
};
use atlas_sumcheck::upstream_sumcheck::round_check_failed;
use jolt_poly::{GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use rayon::prelude::*;

/// Prover for the max indicator sumcheck instance.
pub struct MaxIndicatorProver<F: JoltField> {
    params: MaxIndicatorParams<F>,
    setup: RelationSetup<F>,
    eq: Vec<F>,
    gs_eq: Option<GruenSplitEqPolynomial<F>>,
    // log(F * N) variables
    X: MultilinearPolynomial<F>,
    // log(F * N) variables
    e: MultilinearPolynomial<F>,
}

impl<F: JoltField> MaxIndicatorProver<F> {
    /// Constructor for softmax exp multiplication prover.
    pub fn initialize(X: Vec<i32>, mut params: MaxIndicatorParams<F>) -> Self {
        let eq = EqPolynomial::evals(&params.r1_k);
        let X = MultilinearPolynomial::from(X);
        // Take e out of params — the prover never reads params.e again
        let e = MultilinearPolynomial::from(std::mem::take(&mut params.e));
        Self {
            setup: params.relation_setup(),
            params,
            eq,
            gs_eq: None,
            X,
            e,
        }
    }

    fn compute_phase_1_message(
        &mut self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, F> {
        let m = round + 1;
        let Self { X, e, eq, .. } = self;
        let half_poly_len = X.len() / 2;
        let evals: [F; DEGREE_BOUND] = (0..half_poly_len)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|kj| {
                let k = kj >> (self.params.log_N() - m);
                let eq_val = eq[k];
                let e_vals = e.sumcheck_evals(kj, DEGREE_BOUND, BindingOrder::LowToHigh);
                let X_vals = X.sumcheck_evals(kj, DEGREE_BOUND, BindingOrder::LowToHigh);
                [
                    eq_val * X_vals[0] * e_vals[0],
                    eq_val * X_vals[1] * e_vals[1],
                    eq_val * X_vals[2] * e_vals[2],
                ]
            })
            .reduce(
                || [F::zero(); DEGREE_BOUND],
                |running, new| array::from_fn(|i| running[i] + new[i]),
            );
        Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals))
    }

    fn compute_phase_2_message(&mut self, previous_claim: F) -> Result<UnivariatePoly<F>, F> {
        let Self {
            gs_eq,
            X,
            e: indicator,
            ..
        } = self;
        let gs_eq = gs_eq.as_ref().unwrap();
        gs_eq.gruen_round_deg_3(
            &|g| {
                let i_0 = indicator.get_bound_coeff(2 * g);
                let i_inf = indicator.get_bound_coeff(2 * g + 1) - i_0;

                let X_0 = X.get_bound_coeff(2 * g);
                let X_inf = X.get_bound_coeff(2 * g + 1) - X_0;
                [i_0 * X_0, i_inf * X_inf]
            },
            &|g| indicator.get_bound_coeff(2 * g + 1) * X.get_bound_coeff(2 * g + 1),
            previous_claim,
        )
    }
}

impl<F: JoltField> RelationWitness<F> for MaxIndicatorProver<F> {
    type Relation = MaxIndicatorParams<F>;

    fn relation(&self) -> &MaxIndicatorParams<F> {
        &self.params
    }

    fn setup(&self) -> &RelationSetup<F> {
        &self.setup
    }

    fn produced_claims(&self) -> Vec<F> {
        vec![self.X.final_claim()]
    }
}

impl<F: JoltField> MaxIndicatorProver<F> {
    /// Bind round `round`'s challenge.
    fn bind(&mut self, r: F, round: usize) {
        let r_j = F::challenge_from_field(r);
        self.X.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.e.bind_parallel(r_j, BindingOrder::LowToHigh);

        if round == self.params.log_N() - 1 {
            self.gs_eq = Some(GruenSplitEqPolynomial::new(
                &self.params.r1_k,
                jolt_poly::BindingOrder::LowToHigh,
            ));
            drop_in_background_thread(std::mem::take(&mut self.eq));
        }
        if round >= self.params.log_N() {
            self.gs_eq.as_mut().unwrap().bind(r);
        }
    }
}

impl<F: JoltField> ProveRounds<F> for MaxIndicatorProver<F> {
    fn num_rounds(&self) -> usize {
        self.params.rounds()
    }

    #[tracing::instrument(name = "MaxIndicatorProver::prove_round", skip_all)]
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
