#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::CommittedPoly;
use rayon::prelude::*;
use std::iter::zip;

use crate::{
    opening_proof::{OpeningAccumulator, OpeningId, SumcheckId, VerifierOpeningAccumulator},
    sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
};
use atlas_algebra::{
    field::{IntoOpening, JoltField},
    poly::opening_point::{OpeningPoint, BIG_ENDIAN, LITTLE_ENDIAN},
};
use atlas_transcript::Transcript;

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

pub mod prover;
