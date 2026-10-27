#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::CommittedPoly;
use rayon::prelude::*;
use std::{fmt::Debug, iter::zip};

use crate::{
    opening_proof::{OpeningAccumulator, OpeningId, SumcheckId, VerifierOpeningAccumulator},
    sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
};
use atlas_algebra::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        opening_point::{OpeningPoint, BIG_ENDIAN},
    },
};
use atlas_transcript::Transcript;

/// Degree bound of the sumcheck round polynomials in [`BooleanitySumcheckVerifier`].
const DEGREE_BOUND: usize = 3;

/// The input claim of a [`LinearTerm`]: `constant + Σ coef·claim(id)` over
/// virtual openings already in the accumulator.
#[derive(Clone, Debug)]
pub struct LinearClaim<F: JoltField> {
    pub constant: F,
    pub terms: Vec<(OpeningId, F)>,
}

impl<F: JoltField> LinearClaim<F> {
    pub fn constant(c: F) -> Self {
        Self {
            constant: c,
            terms: Vec::new(),
        }
    }

    pub fn evaluate(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.terms.iter().fold(self.constant, |acc, (id, coef)| {
            acc + *coef * accumulator.get_virtual_polynomial_opening(*id).1
        })
    }
}

/// An optional linear check folded into the Booleanity sumcheck, sharing its
/// `G` / `H` tables and its single opening of every chunk:
///
/// ```text
/// Σ_{k,t} Σ_d (γ'_d + w_d·k) · eq(r_cycle, t) · ra_d(k, t) = claim
/// ```
///
/// With `w = 0` this is the Hamming-weight check (`Σ_k ra_d(k, r_cycle)`
/// batched by `γ'`); with `w_d = β·2^{shift_d}` it also recovers the chunked
/// value `Σ_d 2^{shift_d}·Σ_k k·ra_d(k, r_cycle)`.
#[derive(Clone, Debug)]
pub struct LinearTerm<F: JoltField> {
    /// Per-chunk Hamming batching coefficients `γ'_d`.
    pub gammas: Vec<F>,
    /// Per-chunk value weights `w_d`.
    pub weights: Vec<F>,
    /// The claimed sum.
    pub claim: LinearClaim<F>,
}

pub struct BooleanitySumcheckParams<F: JoltField> {
    /// Optional linear check sharing the chunks' opening (see [`LinearTerm`]).
    pub linear: Option<LinearTerm<F>>,
    /// Number of address chunks
    pub d: usize,
    /// Log of chunk size
    pub log_k_chunk: usize,
    /// Log of trace length
    pub log_t: usize,
    /// Batching challenges
    pub gammas: Vec<F::Challenge>,
    /// Address binding point
    pub r_address: Vec<F>,
    /// Cycle binding point
    pub r_cycle: Vec<F>,
    /// Polynomial types for opening accumulator
    pub polynomial_types: Vec<CommittedPoly>,
    /// Sumcheck ID for opening accumulator
    pub sumcheck_id: SumcheckId,
}

impl<F: JoltField> SumcheckInstanceParams<F> for BooleanitySumcheckParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.log_k_chunk + self.log_t
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.linear
            .as_ref()
            .map_or(F::zero(), |l| l.claim.evaluate(accumulator))
    }

    fn normalize_opening_point(&self, sumcheck_challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        let mut opening_point = sumcheck_challenges.to_vec();
        opening_point[..self.log_k_chunk].reverse();
        opening_point[self.log_k_chunk..].reverse();
        opening_point.into()
    }

    // output = eq_eval * Σ_i γ_i * (ra_i² - ra_i)
    //        = Σ_i [ Challenge(2i) * Opening(ra_i) * Opening(ra_i)
    //              + Challenge(2i+1) * Opening(ra_i) ]
    // where Challenge(2i) = eq_eval * γ_i, Challenge(2i+1) = -eq_eval * γ_i
}

pub struct BooleanitySumcheckVerifier<F: JoltField> {
    params: BooleanitySumcheckParams<F>,
}

impl<F: JoltField> BooleanitySumcheckVerifier<F> {
    pub fn new(params: BooleanitySumcheckParams<F>) -> Self {
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for BooleanitySumcheckVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let ra_claims = self
            .params
            .polynomial_types
            .iter()
            .map(|poly_type| {
                let id = OpeningId::new(*poly_type, self.params.sumcheck_id);
                accumulator.get_committed_polynomial_opening(id).1
            })
            .collect::<Vec<F>>();

        let combined_r: Vec<F> = self
            .params
            .r_address
            .iter()
            .cloned()
            .rev()
            .chain(self.params.r_cycle.iter().cloned().rev())
            .collect();

        let booleanity = EqPolynomial::<F>::mle(sumcheck_challenges, &combined_r)
            * zip(&self.params.gammas, &ra_claims)
                .map(|(gamma, ra)| (ra.square() - *ra) * gamma)
                .sum::<F>();
        let Some(lin) = &self.params.linear else {
            return booleanity;
        };
        let (r_address, r_cycle) = sumcheck_challenges.split_at(self.params.log_k_chunk);
        // `k` at the (low-to-high bound) address point: Σ_j 2^j·r_j.
        let k: F = r_address
            .iter()
            .enumerate()
            .map(|(j, c)| F::from_u64(1u64 << j) * (*c).into())
            .sum();
        let r_cycle_rev: Vec<F> = self.params.r_cycle.iter().cloned().rev().collect();
        let eq_cycle = EqPolynomial::<F>::mle(r_cycle, &r_cycle_rev);
        let linear: F = ra_claims
            .iter()
            .zip(&lin.gammas)
            .zip(&lin.weights)
            .map(|((ra, gamma), weight)| *ra * (*gamma + *weight * k))
            .sum();
        booleanity + eq_cycle * linear
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        accumulator.append_sparse(
            transcript,
            self.params.polynomial_types.clone(),
            self.params.sumcheck_id,
            opening_point.r,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::prover::*;
    use crate::opening_proof::prover::ProverOpeningAccumulator;
    use atlas_algebra::field::fp128::Fp128 as Fr;
    use common::CommittedPoly;
    use rand::{rngs::StdRng, RngCore, SeedableRng};

    use crate::sumcheck::Sumcheck;
    use atlas_algebra::{field::IntoOpening, utils::thread::unsafe_allocate_zero_vec};
    use atlas_transcript::{
        Blake2bProver, Blake2bVerifier, ProvingTranscript, Transcript, VerifyingTranscript,
    };

    use super::*;

    #[test]
    fn test_small_bool_sumcheck() {
        let mut rng = StdRng::seed_from_u64(0x456);
        let log_lookups = 4;
        let log_words = 3;
        let words = 1 << log_words;
        let lookups = 1 << log_lookups;

        let mut prover_transcript = Blake2bProver::new(b"");
        let mut prover_accumulator = ProverOpeningAccumulator::new();
        let r_lookups: Vec<Fr> = prover_transcript
            .challenge_vector_optimized::<Fr>(log_lookups)
            .into_opening();
        let r_words: Vec<Fr> = prover_transcript
            .challenge_vector_optimized::<Fr>(log_words)
            .into_opening();

        let params = BooleanitySumcheckParams {
            linear: None,
            d: 1,
            log_k_chunk: log_words,
            log_t: log_lookups,
            r_cycle: r_lookups,
            r_address: r_words,
            gammas: vec![<Fr as JoltField>::Challenge::from(1)],
            polynomial_types: vec![CommittedPoly::NodeOutputRaD(0, 0)],
            sumcheck_id: SumcheckId::Booleanity,
        };

        // Generate read addresses and G vector
        let read_addresses: Vec<usize> = (0..lookups)
            .map(|_| (rng.next_u32() as usize) % words)
            .collect();

        let E: Vec<Fr> = EqPolynomial::evals(&params.r_cycle);
        let mut G = unsafe_allocate_zero_vec(words);
        for (i, &addr) in read_addresses.iter().enumerate() {
            G[addr] += E[i];
        }

        let H_indices: Vec<Option<u16>> = read_addresses
            .iter()
            .map(|&addr| Some(addr as u16))
            .collect();

        let mut prover = SmallBooleanitySumcheckProver::gen(params, vec![G], vec![H_indices]);

        let (proof, _) =
            Sumcheck::prove(&mut prover, &mut prover_accumulator, &mut prover_transcript);

        let mut verifier_transcript = Blake2bVerifier::new(b"", prover_transcript.narg().to_vec());
        let mut verifier_accumulator = VerifierOpeningAccumulator::new();
        let _r_x: Vec<<Fr as JoltField>::Challenge> =
            verifier_transcript.challenge_vector_optimized::<Fr>(log_lookups);
        let _r_words: Vec<<Fr as JoltField>::Challenge> =
            verifier_transcript.challenge_vector_optimized::<Fr>(log_words);

        // Take claims
        for (key, (_, value)) in &prover_accumulator.openings {
            let empty_point = OpeningPoint::<BIG_ENDIAN, Fr>::default();
            verifier_accumulator
                .openings
                .insert(*key, (empty_point, *value));
        }

        let verifier = BooleanitySumcheckVerifier::new(prover.params);
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
