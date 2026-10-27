//! The prover side of [`super`].

use super::*;
use crate::relation::prover::RelationWitness;
use crate::{mles_product_sum::compute_mles_product_sum, upstream_sumcheck::round_check_failed};
use atlas_algebra::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, PolynomialBinding},
        ra_poly::RaPolynomial,
    },
};
use common::parallel::par_enabled;
use jolt_poly::{GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use std::sync::Arc;

pub struct RaSumcheckProver<F: JoltField> {
    ra_i_polys: Vec<RaPolynomial<u8, F>>,
    eq_poly: GruenSplitEqPolynomial<F>,
    relation: RaVirtualizationRelation,
    setup: RelationSetup<F>,
}

impl<F: JoltField> RaSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "RaSumcheckProver::gen")]
    pub fn gen(params: RaSumcheckParams<F>, H_indices: Vec<Vec<Option<u8>>>) -> Self {
        // Compute r_address_chunks with proper padding
        let r_address_chunks = params
            .one_hot_params
            .compute_r_address_chunks(&params.r_address.r);

        let ra_i_polys = H_indices
            .into_par_iter()
            .with_min_len(par_enabled())
            .enumerate()
            .map(|(i, lookup_indices)| {
                let eq_evals = EqPolynomial::evals(&r_address_chunks[i]);
                RaPolynomial::new(Arc::new(lookup_indices), eq_evals)
            })
            .collect();

        let eq_poly =
            GruenSplitEqPolynomial::new(&params.r_cycle.r, jolt_poly::BindingOrder::LowToHigh);
        let (relation, setup) = params.into_parts();
        Self {
            ra_i_polys,
            eq_poly,
            relation,
            setup,
        }
    }
}

impl<F: JoltField> RelationWitness<F> for RaSumcheckProver<F> {
    type Relation = RaVirtualizationRelation;

    fn relation(&self) -> &RaVirtualizationRelation {
        &self.relation
    }

    fn setup(&self) -> &RelationSetup<F> {
        &self.setup
    }

    fn produced_claims(&self) -> Vec<F> {
        self.ra_i_polys.iter().map(|ra| ra.final_claim()).collect()
    }
}

impl<F: JoltField> RaSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "RaSumcheckProver::bind")]
    fn bind(&mut self, r: F) {
        let r_j = F::challenge_from_field(r);
        self.ra_i_polys
            .iter_mut()
            .for_each(|p| p.bind_parallel(r_j, BindingOrder::LowToHigh));
        self.eq_poly.bind(r);
    }
}

impl<F: JoltField> ProveRounds<F> for RaSumcheckProver<F> {
    fn num_rounds(&self) -> usize {
        self.relation.rounds()
    }

    #[tracing::instrument(skip_all, name = "RaSumcheckProver::prove_round")]
    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(r) = bind {
            self.bind(r);
        }
        compute_mles_product_sum(&self.ra_i_polys, previous_claim, &self.eq_poly)
            .map_err(round_check_failed(round, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}
