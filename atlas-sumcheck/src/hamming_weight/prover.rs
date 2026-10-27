//! The prover side of [`super`].

use super::*;
use crate::relation::prover::RelationWitness;
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use ark_std::Zero;
use atlas_algebra::{
    field::JoltField,
    poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
};
use common::parallel::par_enabled;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};

#[derive(Allocative)]
pub struct HammingWeightSumcheckProver<F: JoltField> {
    ra: Vec<MultilinearPolynomial<F>>,
    gamma_powers: Vec<F>,
    #[allocative(skip)]
    relation: HammingWeightRelation,
    #[allocative(skip)]
    setup: RelationSetup<F>,
}

impl<F: JoltField> HammingWeightSumcheckProver<F> {
    pub fn gen(params: HammingWeightSumcheckParams<F>, G: Vec<Vec<F>>) -> Self {
        let ra = G.into_iter().map(MultilinearPolynomial::from).collect();
        let gamma_powers = params.gamma_powers.clone();
        let (relation, setup) = params.into_parts();
        Self {
            ra,
            gamma_powers,
            relation,
            setup,
        }
    }
}

impl<F: JoltField> RelationWitness<F> for HammingWeightSumcheckProver<F> {
    type Relation = HammingWeightRelation;

    fn relation(&self) -> &HammingWeightRelation {
        &self.relation
    }

    fn setup(&self) -> &RelationSetup<F> {
        &self.setup
    }

    fn produced_claims(&self) -> Vec<F> {
        self.ra.iter().map(|ra| ra.final_claim()).collect()
    }
}

crate::relation_witness_instance!([F: JoltField] HammingWeightSumcheckProver<F>);

impl<F: JoltField> HammingWeightSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "HammingWeightSumcheckProver::round_message", fields(variant = ?self.relation.sumcheck_id()))]
    fn round_message(&mut self, _round: usize, previous_claim: F) -> UnivariatePoly<F> {
        let prover_msg = self
            .ra
            .par_iter()
            .zip(self.gamma_powers.par_iter())
            .with_min_len(par_enabled())
            .map(|(ra, gamma)| {
                let ra_sum = (0..ra.len() / 2)
                    .into_par_iter()
                    .with_min_len(par_enabled())
                    .map(|i| ra.get_bound_coeff(2 * i))
                    .fold_with(F::UnreducedMulU64::zero(), |running, new| {
                        running + new.to_unreduced()
                    })
                    .reduce(F::UnreducedMulU64::zero, |running, new| running + new);
                F::reduce_mul_u64(ra_sum) * gamma
            })
            .reduce(F::zero, |running, new| running + new);

        let eval_at_0 = prover_msg;
        UnivariatePoly::from_evals_and_hint(previous_claim, &[eval_at_0])
    }

    #[tracing::instrument(skip_all, name = "HammingWeightSumcheckProver::bind", fields(variant = ?self.relation.sumcheck_id()))]
    fn bind(&mut self, r: F, _round: usize) {
        let r_j = F::challenge_from_field(r);
        self.ra
            .par_iter_mut()
            .with_min_len(par_enabled())
            .for_each(|ra| ra.bind_parallel(r_j, BindingOrder::LowToHigh));
    }
}

impl<F: JoltField> ProveRounds<F> for HammingWeightSumcheckProver<F> {
    fn num_rounds(&self) -> usize {
        self.relation.rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(r) = bind {
            self.bind(r, round - 1);
        }
        Ok(self.round_message(round, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind, self.relation.rounds() - 1);
        Ok(())
    }
}
