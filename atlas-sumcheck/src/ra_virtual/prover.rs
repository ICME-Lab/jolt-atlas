//! The prover side of [`super`].

use super::*;
use crate::{
    mles_product_sum::compute_mles_product_sum,
    opening_proof::{prover::ProverOpeningAccumulator, SumcheckId},
    sumcheck::prover::SumcheckInstanceProver,
    sumcheck_verifier::SumcheckInstanceParams,
    upstream_sumcheck::round_check_failed,
};
use atlas_algebra::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, PolynomialBinding},
        ra_poly::RaPolynomial,
    },
};
use atlas_transcript::Transcript;
use common::parallel::par_enabled;
use jolt_poly::{GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use std::sync::Arc;

pub struct RaSumcheckProver<F: JoltField> {
    ra_i_polys: Vec<RaPolynomial<u8, F>>,
    eq_poly: GruenSplitEqPolynomial<F>,
    pub(super) params: RaSumcheckParams<F>,
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

        Self {
            ra_i_polys,
            eq_poly: GruenSplitEqPolynomial::new(
                &params.r_cycle.r,
                jolt_poly::BindingOrder::LowToHigh,
            ),
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for RaSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_cycle = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());

        // Compute r_address_chunks with proper padding
        let r_address_chunks = self
            .params
            .one_hot_params
            .compute_r_address_chunks(&self.params.r_address.r);

        for (i, r_address) in r_address_chunks.into_iter().enumerate() {
            let claim = self.ra_i_polys[i].final_claim();
            accumulator.append_sparse(
                transcript,
                vec![self.params.polynomial_types[i]],
                SumcheckId::RaVirtualization,
                r_address,
                r_cycle.r.clone(),
                vec![claim],
            );
        }
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
        self.params.num_rounds()
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
