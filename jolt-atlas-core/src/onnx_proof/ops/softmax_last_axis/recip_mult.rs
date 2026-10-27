use crate::onnx_proof::relation::{RelationSetup, RelationWitness};
use atlas_algebra::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        split_eq_poly::SplitEqUnreducedFold,
    },
};
use atlas_sumcheck::upstream_sumcheck::round_check_failed;
pub use jolt_atlas_verifier::onnx_proof::ops::softmax_last_axis::recip_mult::*;
use jolt_claims::SymbolicSumcheck;
use jolt_poly::{GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};

/// Prover state for reciprocal multiplication in softmax.
///
/// Proves Relation A only: `exp_q[k,j] · inv_sum[k] = softmax_q[k,j] · S + R[k,j]`.
/// `inv_sum` is verifier-known (derived from prover-sent `exp_sum_q`), but
/// the prover still needs the polynomial for sumcheck round computation.
pub struct RecipMultProver<F: JoltField> {
    relation: RecipMultRelation<F>,
    setup: RelationSetup<F>,
    eq: GruenSplitEqPolynomial<F>,
    // log(F * N) variables
    exp_q: MultilinearPolynomial<F>,
    // log(F) variables — verifier-known, but prover needs for round computation
    inv_sum: MultilinearPolynomial<F>,
}

impl<F: JoltField> RecipMultProver<F> {
    /// Constructor for reciprocal multiplication prover.
    pub fn initialize(exp_q: Vec<i32>, inv_sum: Vec<i32>, params: RecipMultParams<F>) -> Self {
        let eq = GruenSplitEqPolynomial::new(&params.r, jolt_poly::BindingOrder::LowToHigh);
        let exp_q = MultilinearPolynomial::from(exp_q);
        let relation = RecipMultRelation::with_inv_sum(
            params,
            inv_sum.iter().map(|&v| F::from_i32(v)).collect(),
        );
        let inv_sum = MultilinearPolynomial::from(inv_sum);
        Self {
            setup: relation.relation_setup(),
            relation,
            eq,
            exp_q,
            inv_sum,
        }
    }

    fn compute_phase_1_message(
        &mut self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, F> {
        let m = round + 1;
        let log_N = self.relation.params().log_N();
        let Self {
            eq, exp_q, inv_sum, ..
        } = self;

        // Phase 1: binding j-variables (the low bits of exp_q).
        // Only exp_q depends on the current variable X; inv_sum is constant w.r.t. X.
        //
        //   s(X) = eq_lin(X) * q(X)
        //   q(X) = exp_q(k, X) · inv_sum(k)
        //
        // q is linear in X, so we only need q(1) = exp_q(k, 1) · inv_sum(k) (and q(0) when the
        // eq factor vanishes at 0).
        let (exp_q, inv_sum) = (&*exp_q, &*inv_sum);
        let at = |x: usize| {
            move |kj: usize| {
                let k = kj >> (log_N - m);
                exp_q.get_bound_coeff(2 * kj + x) * inv_sum.get_bound_coeff(k)
            }
        };
        eq.gruen_round_deg_2(&at(1), &at(0), previous_claim)
    }

    fn compute_phase_2_message(&mut self, previous_claim: F) -> Result<UnivariatePoly<F>, F> {
        let Self {
            eq, exp_q, inv_sum, ..
        } = self;

        // Phase 2: binding k-variables (the high bits shared by both polynomials).
        // Both exp_q and inv_sum depend on the current variable X.
        //
        //   s(X) = eq_lin(X) * q(X)
        //   q(X) = exp_q(X) · inv_sum(X)
        //
        // q is quadratic in X (product of two linear-in-X polynomials). Compute:
        //   c0 = q(0) = exp_q(0) · inv_sum(0)
        //   e  = leading coeff of q(X) = exp_q_∞ · inv_sum_∞
        eq.gruen_round_deg_3(
            &|g| {
                let exp_q_0 = exp_q.get_bound_coeff(2 * g);
                let exp_q_inf = exp_q.get_bound_coeff(2 * g + 1) - exp_q_0;

                let ise_0 = inv_sum.get_bound_coeff(2 * g);
                let ise_inf = inv_sum.get_bound_coeff(2 * g + 1) - ise_0;
                [exp_q_0 * ise_0, exp_q_inf * ise_inf]
            },
            &|g| exp_q.get_bound_coeff(2 * g + 1) * inv_sum.get_bound_coeff(2 * g + 1),
            previous_claim,
        )
    }
}

impl<F: JoltField> RelationWitness<F> for RecipMultProver<F> {
    type Relation = RecipMultRelation<F>;

    fn relation(&self) -> &RecipMultRelation<F> {
        &self.relation
    }

    fn setup(&self) -> &RelationSetup<F> {
        &self.setup
    }

    fn produced_claims(&self) -> Vec<F> {
        vec![self.exp_q.final_claim()]
    }
}

impl<F: JoltField> RecipMultProver<F> {
    /// Bind round `round`'s challenge.
    fn bind(&mut self, r: F, round: usize) {
        let r_j = F::challenge_from_field(r);
        self.eq.bind(r);
        self.exp_q.bind_parallel(r_j, BindingOrder::LowToHigh);
        if round >= self.relation.params().log_N() {
            self.inv_sum.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
    }
}

impl<F: JoltField> ProveRounds<F> for RecipMultProver<F> {
    fn num_rounds(&self) -> usize {
        self.relation.rounds()
    }

    #[tracing::instrument(name = "RecipMultProver::prove_round", skip_all)]
    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(r) = bind {
            self.bind(r, round - 1);
        }
        if round < self.relation.params().log_N() {
            self.compute_phase_1_message(round, previous_claim)
        } else {
            self.compute_phase_2_message(previous_claim)
        }
        .map_err(round_check_failed(round, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind, self.relation.rounds() - 1);
        Ok(())
    }
}
