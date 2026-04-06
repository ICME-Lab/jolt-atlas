#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

#[cfg(not(target_arch = "wasm32"))]
use crate::utils::profiling::print_current_memory_usage;
use crate::{
    field::JoltField,
    poly::{
        opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator},
        unipoly::{CompressedUniPoly, UniPoly},
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::{AppendToTranscript, Transcript},
    utils::errors::ProofVerifyError,
};

use ark_serialize::*;
use rayon::prelude::*;
use std::{
    marker::PhantomData,
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

#[derive(Clone, Debug, Default)]
pub struct BatchedSumcheckProveMetrics {
    pub calls: u64,
    pub instances_total: u64,
    pub max_instances: usize,
    pub rounds_total: u64,
    pub max_rounds: usize,
    pub total: Duration,
    pub append_input_claims: Duration,
    pub batching_coeffs: Duration,
    pub initialize_individual_claims: Duration,
    pub compute_messages: Duration,
    pub combine_univariate_polys: Duration,
    pub compress: Duration,
    pub transcript_and_challenge: Duration,
    pub update_individual_claims: Duration,
    pub ingest_challenges: Duration,
    pub finalize_instances: Duration,
    pub cache_openings: Duration,
}

static BATCHED_SUMCHECK_PROVE_METRICS: OnceLock<Mutex<BatchedSumcheckProveMetrics>> =
    OnceLock::new();

fn batched_sumcheck_prove_metrics() -> &'static Mutex<BatchedSumcheckProveMetrics> {
    BATCHED_SUMCHECK_PROVE_METRICS
        .get_or_init(|| Mutex::new(BatchedSumcheckProveMetrics::default()))
}

pub fn reset_batched_sumcheck_prove_metrics() {
    *batched_sumcheck_prove_metrics()
        .lock()
        .expect("batched sumcheck metrics mutex poisoned") = BatchedSumcheckProveMetrics::default();
}

pub fn snapshot_batched_sumcheck_prove_metrics() -> BatchedSumcheckProveMetrics {
    batched_sumcheck_prove_metrics()
        .lock()
        .expect("batched sumcheck metrics mutex poisoned")
        .clone()
}

/// Implements the standard technique for batching parallel sumchecks to reduce
/// verifier cost and proof size.
///
/// For details, refer to Jim Posen's ["Perspectives on Sumcheck Batching"](https://hackmd.io/s/HyxaupAAA).
/// We do what they describe as "front-loaded" batch sumcheck.
pub enum BatchedSumcheck {}
impl BatchedSumcheck {
    #[tracing::instrument(skip_all, name = "BatchedSumcheck::prove")]
    pub fn prove<F: JoltField, ProofTranscript: Transcript>(
        mut sumcheck_instances: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F::Challenge>) {
        let total_timing = Instant::now();
        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();
        let mut append_input_claims = Duration::ZERO;
        let mut batching_coeffs_time = Duration::ZERO;
        let mut initialize_individual_claims = Duration::ZERO;
        let mut compute_messages = Duration::ZERO;
        let mut combine_univariate_polys = Duration::ZERO;
        let mut compress_time = Duration::ZERO;
        let mut transcript_and_challenge = Duration::ZERO;
        let mut update_individual_claims = Duration::ZERO;
        let mut ingest_challenges = Duration::ZERO;
        let mut finalize_instances = Duration::ZERO;
        let mut cache_openings = Duration::ZERO;

        // Append input claims to transcript
        let timing = Instant::now();
        let input_claims: Vec<F> = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.input_claim(opening_accumulator))
            .collect();
        let num_rounds_per_instance: Vec<usize> = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .collect();
        input_claims.iter().for_each(|input_claim| {
            transcript.append_scalar(input_claim);
        });
        append_input_claims += timing.elapsed();

        let timing = Instant::now();
        let batching_coeffs: Vec<F> = transcript.challenge_vector(sumcheck_instances.len());
        batching_coeffs_time += timing.elapsed();

        // To see why we may need to scale by a power of two, consider a batch of
        // two sumchecks:
        //   claim_a = \sum_x P(x)             where x \in {0, 1}^M
        //   claim_b = \sum_{x, y} Q(x, y)     where x \in {0, 1}^M, y \in {0, 1}^N
        // Then the batched sumcheck is:
        //   \sum_{x, y} A * P(x) + B * Q(x, y)  where A and B are batching coefficients
        //   = A * \sum_y \sum_x P(x) + B * \sum_{x, y} Q(x, y)
        //   = A * \sum_y claim_a + B * claim_b
        //   = A * 2^N * claim_a + B * claim_b
        let timing = Instant::now();
        let mut individual_claims: Vec<F> = input_claims
            .iter()
            .zip(num_rounds_per_instance.iter())
            .map(|(input_claim, num_rounds)| {
                input_claim.mul_pow_2(max_num_rounds - *num_rounds)
            })
            .collect();
        initialize_individual_claims += timing.elapsed();

        #[cfg(test)]
        let mut batched_claim: F = individual_claims
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(claim, coeff)| *claim * coeff)
            .sum();

        let mut r_sumcheck: Vec<F::Challenge> = Vec::with_capacity(max_num_rounds);
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(max_num_rounds);

        for round in 0..max_num_rounds {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let label = format!("Sumcheck round {round}");
                print_current_memory_usage(label.as_str());
            }

            let remaining_rounds = max_num_rounds - round;

            let timing = Instant::now();
            let univariate_polys: Vec<UniPoly<F>> = sumcheck_instances
                .par_iter_mut()
                .zip(individual_claims.par_iter())
                .zip(num_rounds_per_instance.par_iter())
                .zip(input_claims.par_iter())
                .map(|(((sumcheck, previous_claim), num_rounds), input_claim)| {
                    if remaining_rounds > *num_rounds {
                        // We haven't gotten to this sumcheck's variables yet, so
                        // the univariate polynomial is just a constant equal to
                        // the input claim, scaled by a power of 2.
                        let scaled_input_claim =
                            input_claim.mul_pow_2(remaining_rounds - *num_rounds - 1);
                        // Constant polynomial
                        UniPoly::from_coeff(vec![scaled_input_claim])
                    } else {
                        let offset = max_num_rounds - *num_rounds;
                        sumcheck.compute_message(round - offset, *previous_claim)
                    }
                })
                .collect();
            compute_messages += timing.elapsed();

            // Linear combination of individual univariate polynomials
            let timing = Instant::now();
            let batched_univariate_poly: UniPoly<F> =
                univariate_polys.iter().zip(&batching_coeffs).fold(
                    UniPoly::from_coeff(vec![]),
                    |mut batched_poly, (poly, &coeff)| {
                        batched_poly += &(poly * coeff);
                        batched_poly
                    },
                );
            combine_univariate_polys += timing.elapsed();

            let timing = Instant::now();
            let compressed_poly = batched_univariate_poly.compress();
            compress_time += timing.elapsed();

            // append the prover's message to the transcript
            let timing = Instant::now();
            compressed_poly.append_to_transcript(transcript);
            let r_j = transcript.challenge_scalar_optimized::<F>();
            r_sumcheck.push(r_j);
            transcript_and_challenge += timing.elapsed();

            // Cache individual claims for this round
            let timing = Instant::now();
            individual_claims
                .iter_mut()
                .zip(univariate_polys.into_iter())
                .for_each(|(claim, poly)| *claim = poly.evaluate(&r_j));
            update_individual_claims += timing.elapsed();

            #[cfg(test)]
            {
                // Sanity check
                let h0 = batched_univariate_poly.evaluate::<F>(&F::zero());
                let h1 = batched_univariate_poly.evaluate::<F>(&F::one());
                assert_eq!(
                    h0 + h1,
                    batched_claim,
                    "round {round}: H(0) + H(1) = {h0} + {h1} != {batched_claim}"
                );
                batched_claim = batched_univariate_poly.evaluate(&r_j);
            }

            let timing = Instant::now();
            sumcheck_instances.par_iter_mut().for_each(|sumcheck| {
                // If a sumcheck instance has fewer than `max_num_rounds`,
                // we wait until there are <= `sumcheck.num_rounds()` left
                // before binding its variables.
                if remaining_rounds <= sumcheck.num_rounds() {
                    let offset = max_num_rounds - sumcheck.num_rounds();
                    sumcheck.ingest_challenge(r_j, round - offset);
                }
            });
            ingest_challenges += timing.elapsed();

            compressed_polys.push(compressed_poly);
        }

        // Allow each sumcheck instance to perform any end-of-protocol work (e.g. flushing
        // delayed bindings) after the final challenge has been ingested and before we cache
        // openings.
        let timing = Instant::now();
        for sumcheck in sumcheck_instances.iter_mut() {
            sumcheck.finalize();
        }
        finalize_instances += timing.elapsed();

        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        let timing = Instant::now();
        for sumcheck in sumcheck_instances.iter() {
            // If a sumcheck instance has fewer than `max_num_rounds`,
            // we wait until there are <= `sumcheck.num_rounds()` left
            // before binding its variables.
            // So, the sumcheck *actually* uses just the last `sumcheck.num_rounds()`
            // values of `r_sumcheck`.
            let r_slice = &r_sumcheck[max_num_rounds - sumcheck.num_rounds()..];

            // Cache polynomial opening claims, to be proven using either an
            // opening proof or sumcheck (in the case of virtual polynomials).
            sumcheck.cache_openings(opening_accumulator, transcript, r_slice);
        }
        cache_openings += timing.elapsed();

        let mut metrics = batched_sumcheck_prove_metrics()
            .lock()
            .expect("batched sumcheck metrics mutex poisoned");
        metrics.calls += 1;
        metrics.instances_total += sumcheck_instances.len() as u64;
        metrics.max_instances = metrics.max_instances.max(sumcheck_instances.len());
        metrics.rounds_total += max_num_rounds as u64;
        metrics.max_rounds = metrics.max_rounds.max(max_num_rounds);
        metrics.total += total_timing.elapsed();
        metrics.append_input_claims += append_input_claims;
        metrics.batching_coeffs += batching_coeffs_time;
        metrics.initialize_individual_claims += initialize_individual_claims;
        metrics.compute_messages += compute_messages;
        metrics.combine_univariate_polys += combine_univariate_polys;
        metrics.compress += compress_time;
        metrics.transcript_and_challenge += transcript_and_challenge;
        metrics.update_individual_claims += update_individual_claims;
        metrics.ingest_challenges += ingest_challenges;
        metrics.finalize_instances += finalize_instances;
        metrics.cache_openings += cache_openings;

        (SumcheckInstanceProof::new(compressed_polys), r_sumcheck)
    }

    #[tracing::instrument(skip_all, name = "BatchedSumcheck::verify")]
    pub fn verify<F: JoltField, ProofTranscript: Transcript>(
        proof: &SumcheckInstanceProof<F, ProofTranscript>,
        sumcheck_instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F::Challenge>, ProofVerifyError> {
        let max_degree = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.degree())
            .max()
            .unwrap();
        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        // Append input claims to transcript
        sumcheck_instances.iter().for_each(|sumcheck| {
            let input_claim = sumcheck.input_claim(opening_accumulator);
            transcript.append_scalar(&input_claim);
        });

        let batching_coeffs: Vec<F> = transcript.challenge_vector(sumcheck_instances.len());

        // To see why we may need to scale by a power of two, consider a batch of
        // two sumchecks:
        //   claim_a = \sum_x P(x)             where x \in {0, 1}^M
        //   claim_b = \sum_{x, y} Q(x, y)     where x \in {0, 1}^M, y \in {0, 1}^N
        // Then the batched sumcheck is:
        //   \sum_{x, y} A * P(x) + B * Q(x, y)  where A and B are batching coefficients
        //   = A * \sum_y \sum_x P(x) + B * \sum_{x, y} Q(x, y)
        //   = A * \sum_y claim_a + B * claim_b
        //   = A * 2^N * claim_a + B * claim_b
        let claim: F = sumcheck_instances
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(sumcheck, coeff)| {
                let num_rounds = sumcheck.num_rounds();
                let input_claim = sumcheck.input_claim(opening_accumulator);
                input_claim.mul_pow_2(max_num_rounds - num_rounds) * coeff
            })
            .sum();

        let (output_claim, r_sumcheck) =
            proof.verify(claim, max_num_rounds, max_degree, transcript)?;

        let expected_output_claim = sumcheck_instances
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(sumcheck, coeff)| {
                // If a sumcheck instance has fewer than `max_num_rounds`,
                // we wait until there are <= `sumcheck.num_rounds()` left
                // before binding its variables.
                // So, the sumcheck *actually* uses just the last `sumcheck.num_rounds()`
                // values of `r_sumcheck`.
                let r_slice = &r_sumcheck[max_num_rounds - sumcheck.num_rounds()..];

                // Cache polynomial opening claims, to be proven using either an
                // opening proof or sumcheck (in the case of virtual polynomials).
                sumcheck.cache_openings(opening_accumulator, transcript, r_slice);
                let claim = sumcheck.expected_output_claim(opening_accumulator, r_slice);

                claim * coeff
            })
            .sum();

        if output_claim != expected_output_claim {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        Ok(r_sumcheck)
    }
}

pub struct Sumcheck;

impl Sumcheck {
    #[tracing::instrument(skip_all, name = "Sumcheck::prove")]
    pub fn prove<F: JoltField, ProofTranscript: Transcript>(
        sumcheck_instance: &mut dyn SumcheckInstanceProver<F, ProofTranscript>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F::Challenge>) {
        let num_rounds = sumcheck_instance.num_rounds();

        // Append input claims to transcript
        let input_claim = sumcheck_instance.input_claim(opening_accumulator);
        transcript.append_scalar(&input_claim);
        let mut previous_claim = input_claim;
        let mut r_sumcheck: Vec<F::Challenge> = Vec::with_capacity(num_rounds);
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
        for round in 0..num_rounds {
            let univariate_poly = sumcheck_instance.compute_message(round, previous_claim);
            // append the prover's message to the transcript
            let compressed_poly = univariate_poly.compress();
            compressed_poly.append_to_transcript(transcript);
            let r_j = transcript.challenge_scalar_optimized::<F>();
            r_sumcheck.push(r_j);

            // Cache claim for this round
            previous_claim = univariate_poly.evaluate(&r_j);
            sumcheck_instance.ingest_challenge(r_j, round);
            compressed_polys.push(compressed_poly);
        }

        // Allow the sumcheck instance to perform any end-of-protocol work (e.g. flushing
        // delayed bindings) after the final challenge has been ingested and before we cache
        // openings.
        sumcheck_instance.finalize();

        sumcheck_instance.cache_openings(opening_accumulator, transcript, &r_sumcheck);
        (SumcheckInstanceProof::new(compressed_polys), r_sumcheck)
    }

    #[tracing::instrument(skip_all, name = "Sumcheck::verify")]
    pub fn verify<F: JoltField, ProofTranscript: Transcript>(
        proof: &SumcheckInstanceProof<F, ProofTranscript>,
        sumcheck_instance: &dyn SumcheckInstanceVerifier<F, ProofTranscript>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F::Challenge>, ProofVerifyError> {
        let num_rounds = sumcheck_instance.num_rounds();
        let input_claim = sumcheck_instance.input_claim(opening_accumulator);
        transcript.append_scalar(&input_claim);
        let degree_bound = sumcheck_instance.degree();
        let (final_claim, r_sumcheck) =
            proof.verify(input_claim, num_rounds, degree_bound, transcript)?;
        sumcheck_instance.cache_openings(opening_accumulator, transcript, &r_sumcheck);
        let expected_final_claim =
            sumcheck_instance.expected_output_claim(opening_accumulator, &r_sumcheck);
        if final_claim != expected_final_claim {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }
        Ok(r_sumcheck)
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct SumcheckInstanceProof<F: JoltField, ProofTranscript: Transcript> {
    pub compressed_polys: Vec<CompressedUniPoly<F>>,
    _marker: PhantomData<ProofTranscript>,
}

impl<F: JoltField, ProofTranscript: Transcript> SumcheckInstanceProof<F, ProofTranscript> {
    pub fn new(
        compressed_polys: Vec<CompressedUniPoly<F>>,
    ) -> SumcheckInstanceProof<F, ProofTranscript> {
        SumcheckInstanceProof {
            compressed_polys,
            _marker: PhantomData,
        }
    }

    /// Verify this sumcheck proof.
    /// Note: Verification does not execute the final check of sumcheck protocol: g_v(r_v) = oracle_g(r),
    /// as the oracle is not passed in. Expected that the caller will implement.
    ///
    /// Params
    /// - `claim`: Claimed evaluation
    /// - `num_rounds`: Number of rounds of sumcheck, or number of variables to bind
    /// - `degree_bound`: Maximum allowed degree of the combined univariate polynomial
    /// - `transcript`: Fiat-shamir transcript
    ///
    /// Returns (e, r)
    /// - `e`: Claimed evaluation at random point
    /// - `r`: Evaluation point
    pub fn verify(
        &self,
        claim: F,
        num_rounds: usize,
        degree_bound: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<(F, Vec<F::Challenge>), ProofVerifyError> {
        let mut e = claim;
        let mut r: Vec<F::Challenge> = Vec::new();

        // verify that there is a univariate polynomial for each round
        assert_eq!(self.compressed_polys.len(), num_rounds);
        for i in 0..self.compressed_polys.len() {
            // verify degree bound
            if self.compressed_polys[i].degree() > degree_bound {
                return Err(ProofVerifyError::InvalidInputLength(
                    degree_bound,
                    self.compressed_polys[i].degree(),
                ));
            }

            // append the prover's message to the transcript
            self.compressed_polys[i].append_to_transcript(transcript);

            //derive the verifier's challenge for the next round
            let r_i: F::Challenge = transcript.challenge_scalar_optimized::<F>();
            r.push(r_i);

            // evaluate the claimed degree-ell polynomial at r_i using the hint
            e = self.compressed_polys[i].eval_from_hint(&e, &r_i);
        }

        Ok((e, r))
    }
}
