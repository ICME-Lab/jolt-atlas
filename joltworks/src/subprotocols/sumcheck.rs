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
        mut sumcheck_instances: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
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

        for round in 0..max_num_rounds {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let label = format!("Sumcheck round {round}");
                print_current_memory_usage(label.as_str());
            }

            let remaining_rounds = max_num_rounds - round;

            let univariate_polys: Vec<UniPoly<F>> = sumcheck_instances
                .iter_mut()
                .zip(individual_claims.iter())
                .map(|(sumcheck, previous_claim)| {
                    let num_rounds = sumcheck.num_rounds();
                    if remaining_rounds > num_rounds {
                        // We haven't gotten to this sumcheck's variables yet, so
                        // the univariate polynomial is just a constant equal to
                        // the input claim, scaled by a power of 2.
                        let num_rounds = sumcheck.num_rounds();
                        let scaled_input_claim = sumcheck
                            .input_claim(opening_accumulator)
                            .mul_pow_2(remaining_rounds - num_rounds - 1);
                        // Constant polynomial
                        UniPoly::from_coeff(vec![scaled_input_claim])
                    } else {
                        let offset = max_num_rounds - sumcheck.num_rounds();
                        sumcheck.compute_message(round - offset, *previous_claim)
                    }
                })
                .collect();

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
                .zip(univariate_polys.into_iter())
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

            for sumcheck in sumcheck_instances.iter_mut() {
                // If a sumcheck instance has fewer than `max_num_rounds`,
                // we wait until there are <= `sumcheck.num_rounds()` left
                // before binding its variables.
                if remaining_rounds <= sumcheck.num_rounds() {
                    let offset = max_num_rounds - sumcheck.num_rounds();
                    sumcheck.ingest_challenge(r_j, round - offset);
                }
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

    /// Prove a batched sumcheck in zero-knowledge mode.
    ///
    /// Instead of sending cleartext round polynomials, the prover commits to each
    /// round's batched polynomial using Pedersen commitments. The committed coefficients
    /// are later verified by BlindFold's split-committed R1CS.
    ///
    /// Returns (proof, challenges, initial_batched_claim).
    #[cfg(feature = "zk")]
    #[tracing::instrument(skip_all, name = "BatchedSumcheck::prove_zk")]
    pub fn prove_zk<
        F: JoltField,
        C: crate::curve::JoltCurve<F = F>,
        ProofTranscript: Transcript,
        R: rand_core::CryptoRngCore,
    >(
        mut sumcheck_instances: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        blindfold_accumulator: &mut crate::subprotocols::blindfold::BlindFoldAccumulator<F, C>,
        transcript: &mut ProofTranscript,
        pedersen_gens: &crate::poly::commitment::pedersen::PedersenGenerators<C>,
        rng: &mut R,
    ) -> (ZkSumcheckProof<F, C, ProofTranscript>, Vec<F::Challenge>, F) {
        use crate::poly::unipoly::UniPoly;
        use crate::subprotocols::blindfold::ZkStageData;

        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        // In ZK mode, don't absorb cleartext claims -- polynomial commitments provide binding.
        let batching_coeffs: Vec<F> = transcript.challenge_vector(sumcheck_instances.len());

        let mut individual_claims: Vec<F> = sumcheck_instances
            .iter()
            .map(|sumcheck| {
                let num_rounds = sumcheck.num_rounds();
                let input_claim = sumcheck.input_claim(opening_accumulator);
                input_claim.mul_pow_2(max_num_rounds - num_rounds)
            })
            .collect();

        let initial_batched_claim: F = individual_claims
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(claim, coeff)| *claim * coeff)
            .sum();

        let mut r_sumcheck: Vec<F::Challenge> = Vec::with_capacity(max_num_rounds);
        let mut round_commitments_g1: Vec<C::G1> = Vec::with_capacity(max_num_rounds);
        let mut poly_coeffs: Vec<Vec<F>> = Vec::with_capacity(max_num_rounds);
        let mut blinding_factors: Vec<F> = Vec::with_capacity(max_num_rounds);
        let mut poly_degrees: Vec<usize> = Vec::with_capacity(max_num_rounds);

        for round in 0..max_num_rounds {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let label = format!("Sumcheck round {round}");
                print_current_memory_usage(label.as_str());
            }

            let univariate_polys: Vec<UniPoly<F>> = sumcheck_instances
                .iter_mut()
                .zip(individual_claims.iter())
                .map(|(sumcheck, previous_claim)| {
                    let num_rounds = sumcheck.num_rounds();
                    let offset = sumcheck.round_offset(max_num_rounds);
                    let active = round >= offset && round < offset + num_rounds;
                    if active {
                        sumcheck.compute_message(round - offset, *previous_claim)
                    } else {
                        // Dummy round: polynomial is constant with H(0)=H(1)=previous_claim/2.
                        let two_inv = F::from_u64(2).inverse().unwrap();
                        UniPoly::from_coeff(vec![*previous_claim * two_inv])
                    }
                })
                .collect();

            let batched_univariate_poly: UniPoly<F> =
                univariate_polys.iter().zip(&batching_coeffs).fold(
                    UniPoly::from_coeff(vec![]),
                    |mut batched_poly, (poly, &coeff)| {
                        batched_poly += &(poly * coeff);
                        batched_poly
                    },
                );

            let blinding = F::random(rng);
            let commitment = pedersen_gens.commit(&batched_univariate_poly.coeffs, &blinding);

            transcript.append_serializable(&commitment);

            let r_j = transcript.challenge_scalar_optimized::<F>();
            r_sumcheck.push(r_j);

            individual_claims
                .iter_mut()
                .zip(univariate_polys.into_iter())
                .for_each(|(claim, poly)| *claim = poly.evaluate(&r_j));

            for sumcheck in sumcheck_instances.iter_mut() {
                let num_rounds = sumcheck.num_rounds();
                let offset = sumcheck.round_offset(max_num_rounds);
                let active = round >= offset && round < offset + num_rounds;
                if active {
                    sumcheck.ingest_challenge(r_j, round - offset);
                }
            }

            round_commitments_g1.push(commitment);
            poly_degrees.push(batched_univariate_poly.coeffs.len() - 1);
            poly_coeffs.push(batched_univariate_poly.coeffs.clone());
            blinding_factors.push(blinding);
        }

        for sumcheck in sumcheck_instances.iter_mut() {
            sumcheck.finalize();
        }

        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        for sumcheck in sumcheck_instances.iter() {
            let num_rounds = sumcheck.num_rounds();
            let offset = sumcheck.round_offset(max_num_rounds);
            let r_slice = &r_sumcheck[offset..offset + num_rounds];
            sumcheck.cache_openings(opening_accumulator, transcript, r_slice);
        }

        let output_claim_values = opening_accumulator.take_pending_claims();
        let output_claim_ids = opening_accumulator.take_pending_claim_ids();
        let oc_committed: Vec<_> = pedersen_gens.commit_chunked(&output_claim_values, rng);
        let output_claims: Vec<(crate::poly::opening_proof::OpeningId, F)> = output_claim_ids
            .into_iter()
            .zip(output_claim_values)
            .collect();
        let (output_claims_commitments, output_claims_blindings): (Vec<_>, Vec<_>) =
            oc_committed.into_iter().unzip();
        for com in &output_claims_commitments {
            transcript.append_serializable(com);
        }

        // Build output constraints, scaling each by its batching coefficient.
        // The sumcheck polynomials are batched (scaled by batching_coeff), so the
        // final claim = batching_coeff * expected_output. The raw constraint produces
        // expected_output (unbatched). We wrap each constraint with an extra challenge
        // that carries the batching coefficient so the R1CS checks:
        //   final_claim == batching_coeff * constraint_output
        let output_constraints: Vec<_> = sumcheck_instances
            .iter()
            .zip(&batching_coeffs)
            .map(|(sumcheck, _batch_coeff)| {
                sumcheck
                    .get_params()
                    .output_claim_constraint()
                    .map(|c| c.scale_by_new_challenge())
            })
            .collect();

        let constraint_challenge_values: Vec<Vec<F>> = sumcheck_instances
            .iter()
            .zip(&batching_coeffs)
            .map(|(sumcheck, &batch_coeff)| {
                let num_rounds = sumcheck.num_rounds();
                let offset = sumcheck.round_offset(max_num_rounds);
                let r_slice = &r_sumcheck[offset..offset + num_rounds];
                let mut vals = sumcheck
                    .get_params()
                    .output_constraint_challenge_values(r_slice);
                // Append the batching coefficient as the final challenge value,
                // matching the scale_by_new_challenge() wrapping above.
                vals.push(batch_coeff);
                vals
            })
            .collect();

        let input_constraints: Vec<_> = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.get_params().input_claim_constraint())
            .collect();

        let input_constraint_challenge_values: Vec<Vec<F>> = sumcheck_instances
            .iter()
            .map(|sumcheck| {
                sumcheck
                    .get_params()
                    .input_constraint_challenge_values(opening_accumulator)
            })
            .collect();

        let input_claim_scaling_exponents: Vec<usize> = sumcheck_instances
            .iter()
            .map(|sumcheck| max_num_rounds - sumcheck.num_rounds())
            .collect();

        blindfold_accumulator.push_stage_data(ZkStageData {
            initial_claim: initial_batched_claim,
            round_commitments: round_commitments_g1.clone(),
            poly_coeffs,
            blinding_factors,
            challenges: r_sumcheck.clone(),
            batching_coefficients: batching_coeffs.to_vec(),
            output_constraints,
            constraint_challenge_values,
            input_constraints,
            input_constraint_challenge_values,
            input_claim_scaling_exponents,
            output_claims,
            output_claims_blindings,
            output_claims_commitments: output_claims_commitments.clone(),
        });

        (
            ZkSumcheckProof {
                round_commitments: round_commitments_g1,
                poly_degrees,
                output_claims_commitments,
                _marker: PhantomData,
            },
            r_sumcheck,
            initial_batched_claim,
        )
    }

    /// Verify a ZK batched sumcheck proof.
    ///
    /// Absorbs commitments from the proof, derives challenges, but skips the
    /// output claim equality check (that is handled by BlindFold).
    #[cfg(feature = "zk")]
    #[tracing::instrument(skip_all, name = "BatchedSumcheck::verify_zk")]
    pub fn verify_zk<
        F: JoltField,
        C: crate::curve::JoltCurve<F = F>,
        ProofTranscript: Transcript,
    >(
        proof: &ZkSumcheckProof<F, C, ProofTranscript>,
        sumcheck_instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F::Challenge>, ProofVerifyError> {
        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        if proof.round_commitments.len() != max_num_rounds {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        // In ZK mode, don't absorb cleartext claims.
        // Derive batching coefficients to keep transcript in sync with prover.
        let _batching_coeffs: Vec<F> = transcript.challenge_vector(sumcheck_instances.len());

        // Absorb commitments and derive challenges
        let mut r_sumcheck: Vec<F::Challenge> = Vec::with_capacity(max_num_rounds);

        for commitment in &proof.round_commitments {
            transcript.append_serializable(commitment);
            let r_j = transcript.challenge_scalar_optimized::<F>();
            r_sumcheck.push(r_j);
        }

        // Cache openings for each instance
        for sumcheck in sumcheck_instances.iter() {
            let num_rounds = sumcheck.num_rounds();
            let offset = max_num_rounds - num_rounds;
            let r_slice = &r_sumcheck[offset..offset + num_rounds];
            sumcheck.cache_openings(opening_accumulator, transcript, r_slice);
        }

        // Absorb output claims commitments
        opening_accumulator.take_pending_claims();
        for com in &proof.output_claims_commitments {
            transcript.append_serializable(com);
        }

        // Skip output claim equality check -- BlindFold proves this
        Ok(r_sumcheck)
    }
}

/// ZK sumcheck proof containing Pedersen commitments instead of cleartext polynomials.
#[cfg(feature = "zk")]
#[derive(Clone, Debug)]
pub struct ZkSumcheckProof<
    F: JoltField,
    C: crate::curve::JoltCurve<F = F>,
    ProofTranscript: Transcript,
> {
    pub round_commitments: Vec<C::G1>,
    pub poly_degrees: Vec<usize>,
    pub output_claims_commitments: Vec<C::G1>,
    _marker: PhantomData<(F, ProofTranscript)>,
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
