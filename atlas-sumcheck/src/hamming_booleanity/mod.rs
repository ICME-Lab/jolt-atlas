#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::VirtualPoly;
use rayon::prelude::*;
use std::iter::zip;

use crate::{
    opening_proof::{OpeningAccumulator, OpeningId, SumcheckId, VerifierOpeningAccumulator},
    sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
};
use atlas_algebra::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        opening_point::{OpeningPoint, BIG_ENDIAN, LITTLE_ENDIAN},
    },
};
use atlas_transcript::Transcript;

/// Degree bound of the sumcheck round polynomials in [`HammingBooleanitySumcheckVerifier`].
const DEGREE_BOUND: usize = 3;

pub struct HammingBooleanitySumcheckParams<F: JoltField> {
    pub d: usize,
    pub num_rounds: usize,
    pub gamma_powers: Vec<F>,
    pub polynomial_types: Vec<VirtualPoly>,
    pub sumcheck_id: SumcheckId,
    pub r_cycle: Vec<F>,
}

impl<F: JoltField> SumcheckInstanceParams<F> for HammingBooleanitySumcheckParams<F> {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    // output = eq_eval * Σ_i gamma^i * hw_i * (hw_i - 1)
    //        = Σ_i [Challenge(2i) * Opening(hw_i)^2 + Challenge(2i+1) * Opening(hw_i)]
}

pub struct HammingBooleanitySumcheckVerifier<F: JoltField> {
    params: HammingBooleanitySumcheckParams<F>,
}

impl<F: JoltField> HammingBooleanitySumcheckVerifier<F> {
    pub fn new(params: HammingBooleanitySumcheckParams<F>) -> Self {
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for HammingBooleanitySumcheckVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let hw_claim = (0..self.params.d).map(|i| {
            let id = OpeningId::new(self.params.polynomial_types[i], self.params.sumcheck_id);
            accumulator.get_virtual_polynomial_opening(id).1
        });

        let r = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        // Compute batched claim: eq(r_cycle, r) * sum_{i=0}^{d-1} gamma^i * ( hw_i * (hw_i - 1) )
        EqPolynomial::mle(&self.params.r_cycle, &r.r)
            * zip(hw_claim, &self.params.gamma_powers)
                .map(|(hw_claim, gamma)| (hw_claim * (hw_claim - F::one())) * gamma)
                .sum::<F>()
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());

        self.params.polynomial_types.iter().for_each(|&poly_type| {
            let id = OpeningId::new(poly_type, self.params.sumcheck_id);
            accumulator.append_virtual(transcript, id, r.clone());
        });
    }
}

#[cfg(test)]
mod tests {
    use super::prover::*;
    use crate::opening_proof::prover::ProverOpeningAccumulator;
    use atlas_algebra::field::fp128::Fp128 as Fr;
    use jolt_field::Ring;
    use rand::{rngs::StdRng, RngCore, SeedableRng};

    use crate::sumcheck::Sumcheck;
    use atlas_transcript::{
        Blake2bProver, Blake2bVerifier, ProvingTranscript, Transcript, VerifyingTranscript,
    };

    use super::*;

    #[test]
    fn test_hamming_booleanity_sumcheck() {
        let mut rng = StdRng::seed_from_u64(0x123);
        let log_T = 5;
        let T = 1 << log_T;

        let mut prover_transcript = Blake2bProver::new(b"");
        let mut prover_accumulator = ProverOpeningAccumulator::new();
        let r_cycle = prover_transcript
            .challenge_vector_optimized::<Fr>(log_T)
            .into_opening();

        let params = HammingBooleanitySumcheckParams {
            d: 1,
            num_rounds: log_T,
            gamma_powers: vec![Fr::from_u8(1)],
            polynomial_types: vec![VirtualPoly::HammingWeight(0)],
            sumcheck_id: SumcheckId::RamHammingBooleanity,
            r_cycle,
        };

        let hw: Vec<Fr> = (0..T).map(|_| Fr::from_u32(rng.next_u32() % 2)).collect();

        let mut prover = HammingBooleanitySumcheckProver::gen(params, vec![hw]);

        let (proof, _) =
            Sumcheck::prove(&mut prover, &mut prover_accumulator, &mut prover_transcript);

        let mut verifier_transcript = Blake2bVerifier::new(b"", prover_transcript.narg().to_vec());
        let mut verifier_accumulator = VerifierOpeningAccumulator::new();
        let _r_cycle: Vec<Fr> = verifier_transcript
            .challenge_vector_optimized::<Fr>(log_T)
            .into_opening();

        // Take claims
        for (key, (_, value)) in &prover_accumulator.openings {
            let empty_point = OpeningPoint::<BIG_ENDIAN, Fr>::default();
            verifier_accumulator
                .openings
                .insert(*key, (empty_point, *value));
        }

        let verifier = HammingBooleanitySumcheckVerifier::new(prover.params);
        Sumcheck::verify(
            &proof,
            &verifier,
            &mut verifier_accumulator,
            &mut verifier_transcript,
        )
        .unwrap();
    }
}

pub mod prover;
