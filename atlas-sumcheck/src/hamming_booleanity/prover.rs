//! The prover side of [`super`].

use super::*;
use crate::relation::prover::RelationWitness;
use crate::upstream_sumcheck::round_check_failed;
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use atlas_algebra::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        split_eq_poly::SplitEqUnreducedFold,
    },
};
use common::parallel::par_enabled;
use jolt_poly::{GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};

#[derive(Allocative)]
pub struct HammingBooleanitySumcheckProver<F: JoltField> {
    hw: Vec<MultilinearPolynomial<F>>,
    #[allocative(skip)]
    eq_r: GruenSplitEqPolynomial<F>,
    gamma_powers: Vec<F>,
    #[allocative(skip)]
    relation: HammingBooleanityRelation,
    #[allocative(skip)]
    setup: RelationSetup<F>,
}

impl<F: JoltField> HammingBooleanitySumcheckProver<F> {
    pub fn gen(params: HammingBooleanitySumcheckParams<F>, G: Vec<Vec<F>>) -> Self {
        let hw = G.into_iter().map(MultilinearPolynomial::from).collect();
        let eq_r = GruenSplitEqPolynomial::new(&params.r_cycle, jolt_poly::BindingOrder::LowToHigh);
        let gamma_powers = params.gamma_powers.clone();
        let (relation, setup) = params.into_parts();
        Self {
            hw,
            eq_r,
            gamma_powers,
            relation,
            setup,
        }
    }
}

impl<F: JoltField> RelationWitness<F> for HammingBooleanitySumcheckProver<F> {
    type Relation = HammingBooleanityRelation;

    fn relation(&self) -> &HammingBooleanityRelation {
        &self.relation
    }

    fn setup(&self) -> &RelationSetup<F> {
        &self.setup
    }

    fn produced_claims(&self) -> Vec<F> {
        self.hw.iter().map(|hw| hw.final_claim()).collect()
    }
}

impl<F: JoltField> HammingBooleanitySumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "HammingBooleanitySumcheckProver::bind", fields(variant = ?self.relation.sumcheck_id()))]
    fn bind(&mut self, r: F) {
        let r_j = F::challenge_from_field(r);
        self.hw
            .par_iter_mut()
            .with_min_len(par_enabled())
            .for_each(|ra| ra.bind_parallel(r_j, BindingOrder::LowToHigh));
        self.eq_r.bind(r);
    }
}

impl<F: JoltField> ProveRounds<F> for HammingBooleanitySumcheckProver<F> {
    fn num_rounds(&self) -> usize {
        self.relation.rounds()
    }

    #[tracing::instrument(skip_all, name = "HammingBooleanitySumcheckProver::prove_round", fields(variant = ?self.relation.sumcheck_id()))]
    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(r) = bind {
            self.bind(r);
        }
        let Self { hw, eq_r, .. } = self;
        let gamma_powers = &self.gamma_powers;

        let [q_constant, q_quadratic] = hw
            .par_iter()
            .zip(gamma_powers.par_iter())
            .with_min_len(par_enabled())
            .map(|(hw_d, &gamma)| {
                let [qd_c, qd_q] = eq_r.par_fold_out_in_unreduced::<2>(&|g| {
                    let hw0 = hw_d.get_bound_coeff(2 * g);
                    let hw1 = hw_d.get_bound_coeff(2 * g + 1);
                    let a = hw1 - hw0;
                    [hw0 * (hw0 - F::one()), a * a]
                });
                [qd_c * gamma, qd_q * gamma]
            })
            .reduce(
                || [F::zero(), F::zero()],
                |acc, new| [acc[0] + new[0], acc[1] + new[1]],
            );

        eq_r.gruen_round_deg_3_from(
            q_constant,
            q_quadratic,
            &|g| {
                hw.iter()
                    .zip(gamma_powers)
                    .map(|(hw_d, &gamma)| {
                        let hw1 = hw_d.get_bound_coeff(2 * g + 1);
                        gamma * hw1 * (hw1 - F::one())
                    })
                    .sum()
            },
            previous_claim,
        )
        .map_err(round_check_failed(round, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}
