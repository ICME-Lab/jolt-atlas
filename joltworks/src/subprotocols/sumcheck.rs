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
use std::marker::PhantomData;

/// Implements the standard technique for batching parallel sumchecks to reduce
/// verifier cost and proof size.
///
/// For details, refer to Jim Posen's ["Perspectives on Sumcheck Batching"](https://hackmd.io/s/HyxaupAAA).
/// We do what they describe as "front-loaded" batch sumcheck.
pub enum BatchedSumcheck {}
impl BatchedSumcheck {
    #[tracing::instrument(skip_all, name = "BatchedSumcheck::prove")]
    pub fn prove<F: JoltField, ProofTranscript: Transcript>(
        sumcheck_instances: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F::Challenge>) {
        Self::prove_inner(sumcheck_instances, opening_accumulator, transcript, false)
    }

    /// Like [`Self::prove`], but computes each round's per-instance messages
    /// *and* ingests each challenge in parallel across instances (nested with
    /// each instance's own parallelism).
    ///
    /// Only for batches of mutually independent instances: instances that
    /// share state behind an `RwLock` (e.g. the opening-reduction instances'
    /// shared eq tables) deadlock when a rayon thread holding a write guard
    /// steals a job that takes the same lock. Use [`Self::prove`] for those.
    #[tracing::instrument(skip_all, name = "BatchedSumcheck::prove_parallel_instances")]
    pub fn prove_parallel_instances<F: JoltField, ProofTranscript: Transcript>(
        sumcheck_instances: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F::Challenge>) {
        Self::prove_inner(sumcheck_instances, opening_accumulator, transcript, true)
    }

    fn prove_inner<F: JoltField, ProofTranscript: Transcript>(
        mut sumcheck_instances: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
        parallel_instances: bool,
    ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F::Challenge>) {
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
        let mut individual_claims: Vec<F> = sumcheck_instances
            .iter()
            .map(|sumcheck| {
                let num_rounds = sumcheck.num_rounds();
                let input_claim = sumcheck.input_claim(opening_accumulator);
                input_claim.mul_pow_2(max_num_rounds - num_rounds)
            })
            .collect();

        #[cfg(test)]
        let mut batched_claim: F = individual_claims
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(claim, coeff)| *claim * coeff)
            .sum();

        let mut r_sumcheck: Vec<F::Challenge> = Vec::with_capacity(max_num_rounds);
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(max_num_rounds);
        // Spread per-round message computation across instances rather than
        // inside each one (see `message` below) when asked to and worthwhile.
        const MANY_INSTANCES: usize = 8;
        let many_instances = parallel_instances && sumcheck_instances.len() >= MANY_INSTANCES;
        // Input claims are fixed for the whole protocol; precompute them so the
        // per-round (possibly parallel) message computation needs no accumulator.
        let input_claims: Vec<F> = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.input_claim(opening_accumulator))
            .collect();

        for round in 0..max_num_rounds {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let label = format!("Sumcheck round {round}");
                print_current_memory_usage(label.as_str());
            }

            let remaining_rounds = max_num_rounds - round;

            let message = |sumcheck: &mut &mut dyn SumcheckInstanceProver<F, ProofTranscript>,
                           previous_claim: &F,
                           input_claim: &F|
             -> UniPoly<F> {
                let num_rounds = sumcheck.num_rounds();
                if remaining_rounds > num_rounds {
                    // We haven't gotten to this sumcheck's variables yet, so
                    // the univariate polynomial is just a constant equal to
                    // the input claim, scaled by a power of 2.
                    let scaled_input_claim =
                        input_claim.mul_pow_2(remaining_rounds - num_rounds - 1);
                    // Constant polynomial
                    UniPoly::from_coeff(vec![scaled_input_claim])
                } else {
                    let offset = max_num_rounds - num_rounds;
                    sumcheck.compute_message(round - offset, *previous_claim)
                }
            };
            let univariate_polys: Vec<UniPoly<F>> = if many_instances {
                // Many instances: parallelize across instances. Each instance's
                // own parallelism is left on so the few large instances (which
                // dominate the batch) still fan out; rayon's work stealing
                // absorbs the nesting.
                sumcheck_instances
                    .par_iter_mut()
                    .zip(individual_claims.par_iter())
                    .zip(input_claims.par_iter())
                    .map(|((sumcheck, previous_claim), input_claim)| {
                        message(sumcheck, previous_claim, input_claim)
                    })
                    .collect()
            } else {
                sumcheck_instances
                    .iter_mut()
                    .zip(individual_claims.iter())
                    .zip(input_claims.iter())
                    .map(|((sumcheck, previous_claim), input_claim)| {
                        message(sumcheck, previous_claim, input_claim)
                    })
                    .collect()
            };

            // Linear combination of individual univariate polynomials
            let batched_univariate_poly: UniPoly<F> =
                univariate_polys.iter().zip(&batching_coeffs).fold(
                    UniPoly::from_coeff(vec![]),
                    |mut batched_poly, (poly, &coeff)| {
                        batched_poly += &(poly * coeff);
                        batched_poly
                    },
                );

            let compressed_poly = batched_univariate_poly.compress();

            // append the prover's message to the transcript
            compressed_poly.append_to_transcript(transcript);
            let r_j = transcript.challenge_scalar_optimized::<F>();
            r_sumcheck.push(r_j);

            // Cache individual claims for this round
            individual_claims
                .iter_mut()
                .zip(univariate_polys)
                .for_each(|(claim, poly)| *claim = poly.evaluate(&r_j));

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

            // If a sumcheck instance has fewer than `max_num_rounds`,
            // we wait until there are <= `sumcheck.num_rounds()` left
            // before binding its variables.
            let ingest = |sumcheck: &mut &mut dyn SumcheckInstanceProver<F, ProofTranscript>| {
                if remaining_rounds <= sumcheck.num_rounds() {
                    let offset = max_num_rounds - sumcheck.num_rounds();
                    sumcheck.ingest_challenge(r_j, round - offset);
                }
            };
            if many_instances {
                sumcheck_instances.par_iter_mut().for_each(ingest);
            } else {
                sumcheck_instances.iter_mut().for_each(ingest);
            }

            compressed_polys.push(compressed_poly);
        }

        // Allow each sumcheck instance to perform any end-of-protocol work (e.g. flushing
        // delayed bindings) after the final challenge has been ingested and before we cache
        // openings.
        for sumcheck in sumcheck_instances.iter_mut() {
            sumcheck.finalize();
        }

        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

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
        if self.compressed_polys.len() != num_rounds {
            return Err(ProofVerifyError::InvalidInputLength(
                num_rounds,
                self.compressed_polys.len(),
            ));
        }
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
