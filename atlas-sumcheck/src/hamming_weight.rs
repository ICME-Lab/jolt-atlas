use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use ark_std::Zero;
use common::parallel::par_enabled;
use common::CommittedPoly;
use rayon::prelude::*;
use std::iter::zip;

use crate::{
    opening_proof::{
        OpeningAccumulator, OpeningId, ProverOpeningAccumulator, SumcheckId,
        VerifierOpeningAccumulator,
    },
    sumcheck_prover::SumcheckInstanceProver,
    sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
};
use atlas_algebra::{
    field::{IntoOpening, JoltField},
    poly::{
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_point::{OpeningPoint, BIG_ENDIAN, LITTLE_ENDIAN},
    },
};
use atlas_transcript::Transcript;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};

/// Degree bound of the sumcheck round polynomials in [`HammingWeightSumcheckVerifier`].
const DEGREE_BOUND: usize = 1;

pub struct HammingWeightSumcheckParams<F: JoltField> {
    pub d: usize,
    pub num_rounds: usize,
    pub gamma_powers: Vec<F>,
    pub polynomial_types: Vec<CommittedPoly>,
    pub sumcheck_id: SumcheckId,
    pub r_cycle: Vec<F>,
}

impl<F: JoltField> SumcheckInstanceParams<F> for HammingWeightSumcheckParams<F> {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        // No consumer constructs a `RamHammingWeight` instance. A consumer
        // whose input claim is another sumcheck's output should carry that
        // claim's `OpeningId` in these params explicitly.
        self.gamma_powers.iter().sum()
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    // output = Σ_{i=0}^{d-1} γ^i * ra_i
}

#[derive(Allocative)]
pub struct HammingWeightSumcheckProver<F: JoltField> {
    ra: Vec<MultilinearPolynomial<F>>,
    #[allocative(skip)]
    params: HammingWeightSumcheckParams<F>,
}

impl<F: JoltField> HammingWeightSumcheckProver<F> {
    pub fn gen(params: HammingWeightSumcheckParams<F>, G: Vec<Vec<F>>) -> Self {
        let ra = G.into_iter().map(MultilinearPolynomial::from).collect();
        Self { ra, params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for HammingWeightSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
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
        let claims: Vec<F> = self.ra.iter().map(|ra| ra.final_claim()).collect();

        accumulator.append_sparse(
            transcript,
            self.params.polynomial_types.clone(),
            self.params.sumcheck_id,
            opening_point.r,
            self.params.r_cycle.clone(),
            claims,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

impl<F: JoltField> HammingWeightSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "HammingWeightSumcheckProver::round_message", fields(variant = ?self.params.sumcheck_id))]
    fn round_message(&mut self, _round: usize, previous_claim: F) -> UnivariatePoly<F> {
        let prover_msg = self
            .ra
            .par_iter()
            .zip(self.params.gamma_powers.par_iter())
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

    #[tracing::instrument(skip_all, name = "HammingWeightSumcheckProver::bind", fields(variant = ?self.params.sumcheck_id))]
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
        SumcheckInstanceParams::num_rounds(&self.params)
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
        self.bind(bind, SumcheckInstanceParams::num_rounds(&self.params) - 1);
        Ok(())
    }
}

pub struct HammingWeightSumcheckVerifier<F: JoltField> {
    params: HammingWeightSumcheckParams<F>,
}

impl<F: JoltField> HammingWeightSumcheckVerifier<F> {
    pub fn new(params: HammingWeightSumcheckParams<F>) -> Self {
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for HammingWeightSumcheckVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        _sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let ra_claims = (0..self.params.d).map(|i| {
            let id = OpeningId::new(self.params.polynomial_types[i], self.params.sumcheck_id);
            accumulator.get_committed_polynomial_opening(id).1
        });

        // Compute batched claim: sum_{i=0}^{d-1} gamma^i * ra_i
        zip(ra_claims, &self.params.gamma_powers)
            .map(|(ra_claim, gamma)| ra_claim * gamma)
            .sum()
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening())
            .r
            .iter()
            .cloned()
            .chain(self.params.r_cycle.iter().cloned())
            .collect::<Vec<_>>();

        accumulator.append_sparse(
            transcript,
            self.params.polynomial_types.clone(),
            self.params.sumcheck_id,
            r,
        );
    }
}
