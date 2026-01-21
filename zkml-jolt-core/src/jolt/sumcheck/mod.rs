use std::{cell::RefCell, rc::Rc};

use crate::jolt::pcs::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use jolt_core::{
    field::JoltField,
    poly::{
        commitment::{CommitmentScheme, HidingCommitmentScheme},
        multilinear_polynomial::MultilinearPolynomial,
        dense_mlpoly::DensePolynomial,
        opening_proof::{BIG_ENDIAN, OpeningPoint},
        unipoly::{CompressedUniPoly, UniPoly},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::{AppendToTranscript, Transcript},
    utils::errors::ProofVerifyError,
};

// ZK sumcheck types - defined locally until jolt-core is updated
pub mod zk_sumcheck;
pub use zk_sumcheck::{
    BatchOpeningProverState, BlindFoldProverState, ZKBatchedSumcheck,
    ZKSumcheckBatchOpeningProof, ZKSumcheckInstanceProof, ZKSumcheckProof,
};

/// Trait for a sumcheck instance that can be batched with other instances.
///
/// This trait defines the interface needed to participate in the `BatchedSumcheck` protocol,
/// which reduces verifier cost and proof size by batching multiple sumcheck protocols.
pub trait SumcheckInstance<F: JoltField>: Send + Sync {
    /// Returns the maximum degree of the sumcheck polynomial.
    fn degree(&self) -> usize;

    /// Returns the number of rounds/variables in this sumcheck instance.
    fn num_rounds(&self) -> usize;

    /// Returns the initial claim of this sumcheck instance, i.e.
    /// input_claim = \sum_{x \in \{0, 1}^N} P(x)
    fn input_claim(&self) -> F; // TODO(moodlezoup): maybe pass this an Option<Rc<RefCell<ProverOpeningAccumulator<F>>>>

    /// Computes the prover's message for a specific round of the sumcheck protocol.
    /// Returns the evaluations of the sumcheck polynomial at 0, 2, 3, ..., degree.
    /// The point evaluation at 1 can be interpolated using the previous round's claim.
    fn compute_prover_message(&mut self, round: usize, previous_claim: F) -> Vec<F>;

    /// Binds this sumcheck instance to the verifier's challenge from a specific round.
    /// This updates the internal state to prepare for the next round.
    fn bind(&mut self, r_j: F, round: usize);

    /// Computes the expected output claim given the verifier's challenges.
    /// This is used to verify the final result of the sumcheck protocol.
    fn expected_output_claim(
        &self,
        opening_accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F],
    ) -> F;

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F>;

    /// Caches polynomial opening claims needed after the sumcheck protocol completes.
    /// These openings will later be proven using either an opening proof or another sumcheck.
    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    );

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    );
}

pub enum SingleSumcheck {}
impl SingleSumcheck {
    /// Proves a single sumcheck instance.
    pub fn prove<F: JoltField, ProofTranscript: Transcript>(
        sumcheck_instance: &mut dyn SumcheckInstance<F>,
        opening_accumulator: Option<Rc<RefCell<ProverOpeningAccumulator<F>>>>,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>) {
        let num_rounds = sumcheck_instance.num_rounds();
        let mut r_sumcheck: Vec<F> = Vec::with_capacity(num_rounds);
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

        let mut previous_claim = sumcheck_instance.input_claim();
        for round in 0..num_rounds {
            let mut univariate_poly_evals =
                sumcheck_instance.compute_prover_message(round, previous_claim);
            univariate_poly_evals.insert(1, previous_claim - univariate_poly_evals[0]);
            let univariate_poly = UniPoly::from_evals(&univariate_poly_evals);

            // append the prover's message to the transcript
            let compressed_poly = univariate_poly.compress();
            compressed_poly.append_to_transcript(transcript);
            compressed_polys.push(compressed_poly);

            let r_j = transcript.challenge_scalar();
            r_sumcheck.push(r_j);

            // Cache claim for this round
            previous_claim = univariate_poly.evaluate(&r_j);

            sumcheck_instance.bind(r_j, round);
        }

        if let Some(opening_accumulator) = opening_accumulator {
            // Cache polynomial opening claims, to be proven using either an
            // opening proof or sumcheck (in the case of virtual polynomials).
            sumcheck_instance.cache_openings_prover(
                opening_accumulator,
                sumcheck_instance.normalize_opening_point(&r_sumcheck),
            );
        }

        (SumcheckInstanceProof::new(compressed_polys), r_sumcheck)
    }

    /// Verifies a single sumcheck instance.
    pub fn verify<F: JoltField, ProofTranscript: Transcript>(
        sumcheck_instance: &dyn SumcheckInstance<F>,
        proof: &SumcheckInstanceProof<F, ProofTranscript>,
        opening_accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F>, ProofVerifyError> {
        let (output_claim, r) = proof.verify(
            sumcheck_instance.input_claim(),
            sumcheck_instance.num_rounds(),
            sumcheck_instance.degree(),
            transcript,
        )?;
        if let Some(opening_accumulator) = &opening_accumulator {
            sumcheck_instance.cache_openings_verifier(
                opening_accumulator.clone(),
                sumcheck_instance.normalize_opening_point(&r),
            );
        }

        if output_claim != sumcheck_instance.expected_output_claim(opening_accumulator.clone(), &r)
        {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        Ok(r)
    }
}

/// Implements the standard technique for batching parallel sumchecks to reduce
/// verifier cost and proof size.
///
/// For details, refer to Jim Posen's ["Perspectives on Sumcheck Batching"](https://hackmd.io/s/HyxaupAAA).
/// We do what they describe as "front-loaded" batch sumcheck.
pub enum BatchedSumcheck {}
impl BatchedSumcheck {
    pub fn prove<F: JoltField, ProofTranscript: Transcript>(
        mut sumcheck_instances: Vec<&mut dyn SumcheckInstance<F>>,
        opening_accumulator: Option<Rc<RefCell<ProverOpeningAccumulator<F>>>>,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>) {
        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

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
                sumcheck
                    .input_claim()
                    .mul_pow_2(max_num_rounds - num_rounds)
            })
            .collect();

        #[cfg(test)]
        let mut batched_claim: F = individual_claims
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(claim, coeff)| *claim * coeff)
            .sum();

        let mut r_sumcheck: Vec<F> = Vec::with_capacity(max_num_rounds);
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(max_num_rounds);

        for round in 0..max_num_rounds {
            #[cfg(not(target_arch = "wasm32"))]
            {
                use jolt_core::utils::profiling::print_current_memory_usage;

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
                            .input_claim()
                            .mul_pow_2(remaining_rounds - num_rounds - 1);
                        // Constant polynomial
                        UniPoly::from_coeff(vec![scaled_input_claim])
                    } else {
                        let offset = max_num_rounds - sumcheck.num_rounds();
                        let mut univariate_poly_evals =
                            sumcheck.compute_prover_message(round - offset, *previous_claim);
                        univariate_poly_evals.insert(1, *previous_claim - univariate_poly_evals[0]);
                        UniPoly::from_evals(&univariate_poly_evals)
                    }
                })
                .collect();

            // Linear combination of individual univariate polynomials
            let batched_univariate_poly: UniPoly<F> =
                univariate_polys.iter().zip(batching_coeffs.iter()).fold(
                    UniPoly::from_coeff(vec![]),
                    |mut batched_poly, (poly, coeff)| {
                        batched_poly += &(poly * coeff);
                        batched_poly
                    },
                );

            let compressed_poly = batched_univariate_poly.compress();

            // append the prover's message to the transcript
            compressed_poly.append_to_transcript(transcript);
            let r_j = transcript.challenge_scalar();
            r_sumcheck.push(r_j);

            // Cache individual claims for this round
            individual_claims
                .iter_mut()
                .zip(univariate_polys.into_iter())
                .for_each(|(claim, poly)| *claim = poly.evaluate(&r_j));

            #[cfg(test)]
            {
                // Sanity check
                let h0 = batched_univariate_poly.evaluate(&F::zero());
                let h1 = batched_univariate_poly.evaluate(&F::one());
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
                    sumcheck.bind(r_j, round - offset);
                }
            }

            compressed_polys.push(compressed_poly);
        }

        if let Some(opening_accumulator) = opening_accumulator {
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
                sumcheck.cache_openings_prover(
                    opening_accumulator.clone(),
                    sumcheck.normalize_opening_point(r_slice),
                );
            }
        }

        (SumcheckInstanceProof::new(compressed_polys), r_sumcheck)
    }

    pub fn verify<F: JoltField, ProofTranscript: Transcript>(
        proof: &SumcheckInstanceProof<F, ProofTranscript>,
        sumcheck_instances: Vec<&dyn SumcheckInstance<F>>,
        opening_accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F>, ProofVerifyError> {
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
                sumcheck
                    .input_claim()
                    .mul_pow_2(max_num_rounds - num_rounds)
                    * coeff
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

                if let Some(opening_accumulator) = &opening_accumulator {
                    // Cache polynomial opening claims, to be proven using either an
                    // opening proof or sumcheck (in the case of virtual polynomials).
                    sumcheck.cache_openings_verifier(
                        opening_accumulator.clone(),
                        sumcheck.normalize_opening_point(r_slice),
                    );
                }
                let claim = sumcheck.expected_output_claim(opening_accumulator.clone(), r_slice);

                claim * coeff
            })
            .sum();

        if output_claim != expected_output_claim {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        Ok(r_sumcheck)
    }
}

/// ZK-aware batched sumcheck that wraps the joltworks ZK sumcheck.
///
/// Provides a unified interface for both ZK and non-ZK sumcheck proving/verifying.
pub struct ZKAwareBatchedSumcheck;

impl ZKAwareBatchedSumcheck {
    /// Proves a batch of sumcheck instances with optional ZK mode.
    ///
    /// # Arguments
    /// * `sumcheck_instances` - The sumcheck instances to prove
    /// * `opening_accumulator` - Optional prover opening accumulator
    /// * `transcript` - The Fiat-Shamir transcript
    /// * `zk_enabled` - Whether to use ZK sumcheck
    ///
    /// # Returns
    /// A tuple containing:
    /// - The standard sumcheck proof (for compatibility)
    /// - The sumcheck challenges
    /// - Optional ZK proof (if ZK mode enabled)
    ///
    /// # Note
    /// This currently uses hybrid ZK mode (polynomial coefficients in transcript).
    /// Full ZK mode with hiding commitments requires HidingCommitmentScheme support.
    #[allow(clippy::type_complexity)]
    pub fn prove<F, PCS, ProofTranscript>(
        sumcheck_instances: Vec<&mut dyn SumcheckInstance<F>>,
        opening_accumulator: Option<Rc<RefCell<ProverOpeningAccumulator<F>>>>,
        transcript: &mut ProofTranscript,
        zk_enabled: bool,
    ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, Option<ZKSumcheckProof<F, PCS>>)
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
        ProofTranscript: Transcript,
    {
        if zk_enabled {
            // Hybrid ZK mode: uses standard proving with ZK proof structure
            Self::prove_zk_hybrid(sumcheck_instances, opening_accumulator, transcript)
        } else {
            // Non-ZK mode
            let (proof, challenges) = BatchedSumcheck::prove(
                sumcheck_instances,
                opening_accumulator,
                transcript,
            );
            (proof, challenges, None)
        }
    }

    /// Proves a batch of sumcheck instances with full ZK mode using hiding commitments.
    ///
    /// This method requires a PCS that implements HidingCommitmentScheme.
    ///
    /// # Arguments
    /// * `sumcheck_instances` - The sumcheck instances to prove
    /// * `opening_accumulator` - Optional prover opening accumulator
    /// * `pcs_setup` - The PCS prover setup for creating hiding commitments
    /// * `transcript` - The Fiat-Shamir transcript
    ///
    /// # Returns
    /// A tuple containing:
    /// - The standard sumcheck proof (for compatibility)
    /// - The sumcheck challenges
    /// - The ZK proof with real commitments
    /// - The batch opening prover state (for generating batch opening proof)
    #[allow(clippy::type_complexity)]
    pub fn prove_full_zk<F, PCS, ProofTranscript>(
        mut sumcheck_instances: Vec<&mut dyn SumcheckInstance<F>>,
        opening_accumulator: Option<Rc<RefCell<ProverOpeningAccumulator<F>>>>,
        pcs_setup: &PCS::ProverSetup,
        transcript: &mut ProofTranscript,
    ) -> (
        SumcheckInstanceProof<F, ProofTranscript>,
        Vec<F>,
        ZKSumcheckProof<F, PCS>,
        BatchOpeningProverState<F, PCS>,
    )
    where
        F: JoltField,
        PCS: HidingCommitmentScheme<Field = F>,
        ProofTranscript: Transcript,
    {
        use rand::thread_rng;

        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        let batching_coeffs: Vec<F> = transcript.challenge_vector(sumcheck_instances.len());

        // Scale claims by power of two for batching
        let mut individual_claims: Vec<F> = sumcheck_instances
            .iter()
            .map(|sumcheck| {
                let num_rounds = sumcheck.num_rounds();
                sumcheck
                    .input_claim()
                    .mul_pow_2(max_num_rounds - num_rounds)
            })
            .collect();

        let mut r_sumcheck: Vec<F> = Vec::with_capacity(max_num_rounds);
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(max_num_rounds);
        let mut round_commitments: Vec<PCS::Commitment> = Vec::with_capacity(max_num_rounds);
        let mut opening_hints: Vec<PCS::OpeningProofHint> = Vec::with_capacity(max_num_rounds);
        let mut rng = thread_rng();

        // Track state for batch opening proof generation
        let mut batch_opening_state: BatchOpeningProverState<F, PCS> = BatchOpeningProverState::new();

        for round in 0..max_num_rounds {
            let remaining_rounds = max_num_rounds - round;

            let univariate_polys: Vec<UniPoly<F>> = sumcheck_instances
                .iter_mut()
                .zip(individual_claims.iter())
                .map(|(sumcheck, previous_claim)| {
                    let num_rounds = sumcheck.num_rounds();
                    if remaining_rounds > num_rounds {
                        let num_rounds = sumcheck.num_rounds();
                        let scaled_input_claim = sumcheck
                            .input_claim()
                            .mul_pow_2(remaining_rounds - num_rounds - 1);
                        UniPoly::from_coeff(vec![scaled_input_claim])
                    } else {
                        let offset = max_num_rounds - sumcheck.num_rounds();
                        let mut univariate_poly_evals =
                            sumcheck.compute_prover_message(round - offset, *previous_claim);
                        univariate_poly_evals.insert(1, *previous_claim - univariate_poly_evals[0]);
                        UniPoly::from_evals(&univariate_poly_evals)
                    }
                })
                .collect();

            // Linear combination of individual univariate polynomials
            let batched_univariate_poly: UniPoly<F> =
                univariate_polys.iter().zip(batching_coeffs.iter()).fold(
                    UniPoly::from_coeff(vec![]),
                    |mut batched_poly, (poly, coeff)| {
                        batched_poly += &(poly * coeff);
                        batched_poly
                    },
                );

            let compressed_poly = batched_univariate_poly.compress();

            // *** ZK CHANGE: Commit to the round polynomial with hiding ***
            let poly_coeffs = batched_univariate_poly.coeffs.clone();
            // Pad to power of 2 for commitment
            let padded_len = poly_coeffs.len().next_power_of_two();
            let mut padded_coeffs = poly_coeffs.clone();
            padded_coeffs.resize(padded_len, F::zero());

            let ml_poly = MultilinearPolynomial::LargeScalars(
                DensePolynomial::new(padded_coeffs)
            );

            // Sample blinding and create hiding commitment
            let blinding = PCS::sample_blinding(&mut rng);
            let (commitment, hint) = PCS::commit_hiding(&ml_poly, &blinding, pcs_setup);

            // *** Append COMMITMENT to transcript (not polynomial!) ***
            commitment.append_to_transcript(transcript);

            let r_j: F = transcript.challenge_scalar();
            r_sumcheck.push(r_j);

            // Store data for batch opening proof
            batch_opening_state.add_round(
                poly_coeffs,
                commitment.clone(),
                hint.clone(),
                blinding,
                r_j,
            );

            round_commitments.push(commitment);
            opening_hints.push(hint);

            // Cache individual claims for this round
            individual_claims
                .iter_mut()
                .zip(univariate_polys.into_iter())
                .for_each(|(claim, poly)| *claim = poly.evaluate(&r_j));

            for sumcheck in sumcheck_instances.iter_mut() {
                if remaining_rounds <= sumcheck.num_rounds() {
                    let offset = max_num_rounds - sumcheck.num_rounds();
                    sumcheck.bind(r_j, round - offset);
                }
            }

            compressed_polys.push(compressed_poly);
        }

        if let Some(opening_accumulator) = opening_accumulator {
            let max_num_rounds = sumcheck_instances
                .iter()
                .map(|sumcheck| sumcheck.num_rounds())
                .max()
                .unwrap();

            for sumcheck in sumcheck_instances.iter() {
                let r_slice = &r_sumcheck[max_num_rounds - sumcheck.num_rounds()..];

                sumcheck.cache_openings_prover(
                    opening_accumulator.clone(),
                    sumcheck.normalize_opening_point(r_slice),
                );
            }
        }

        let zk_proof = ZKSumcheckProof::with_hints(
            round_commitments,
            compressed_polys.clone(),
            opening_hints,
        );
        (
            SumcheckInstanceProof::new(compressed_polys),
            r_sumcheck,
            zk_proof,
            batch_opening_state,
        )
    }

    /// Proves a batch of sumcheck instances with full ZK mode and generates the batch opening proof.
    ///
    /// This is a convenience function that combines `prove_full_zk` with batch opening proof generation.
    ///
    /// # Returns
    /// A tuple containing:
    /// - The standard sumcheck proof
    /// - The sumcheck challenges
    /// - The ZK proof with commitments
    /// - The batch opening proof
    #[allow(clippy::type_complexity)]
    pub fn prove_full_zk_with_batch_opening<F, PCS, ProofTranscript>(
        sumcheck_instances: Vec<&mut dyn SumcheckInstance<F>>,
        opening_accumulator: Option<Rc<RefCell<ProverOpeningAccumulator<F>>>>,
        pcs_setup: &PCS::ProverSetup,
        transcript: &mut ProofTranscript,
    ) -> (
        SumcheckInstanceProof<F, ProofTranscript>,
        Vec<F>,
        ZKSumcheckProof<F, PCS>,
        ZKSumcheckBatchOpeningProof<F, PCS>,
    )
    where
        F: JoltField,
        PCS: HidingCommitmentScheme<Field = F>,
        ProofTranscript: Transcript,
    {
        let (proof, challenges, zk_proof, batch_state) =
            Self::prove_full_zk(sumcheck_instances, opening_accumulator, pcs_setup, transcript);

        // Generate the batch opening proof
        let batch_opening_proof = batch_state.generate_batch_opening_proof(pcs_setup, transcript);

        (proof, challenges, zk_proof, batch_opening_proof)
    }

    /// Hybrid ZK proving.
    ///
    /// Uses standard transcript operations but creates a ZK proof structure.
    /// This provides the ZK proof format for compatibility while using standard proving.
    #[allow(clippy::type_complexity)]
    fn prove_zk_hybrid<F, PCS, ProofTranscript>(
        sumcheck_instances: Vec<&mut dyn SumcheckInstance<F>>,
        opening_accumulator: Option<Rc<RefCell<ProverOpeningAccumulator<F>>>>,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, Option<ZKSumcheckProof<F, PCS>>)
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
        ProofTranscript: Transcript,
    {
        let (standard_proof, challenges) = BatchedSumcheck::prove(
            sumcheck_instances,
            opening_accumulator,
            transcript,
        );

        let zk_proof = ZKSumcheckProof::new(
            Vec::new(),
            standard_proof.compressed_polys.clone(),
        );

        (standard_proof, challenges, Some(zk_proof))
    }

    /// Verifies a batch of sumcheck instances with optional ZK mode.
    ///
    /// In hybrid ZK mode, this uses standard verification since the prover
    /// also uses standard transcript operations (with compressed_polys).
    ///
    /// # Arguments
    /// * `proof` - The standard sumcheck proof
    /// * `zk_proof` - Optional ZK sumcheck proof (unused in hybrid mode)
    /// * `sumcheck_instances` - The sumcheck instances to verify
    /// * `opening_accumulator` - Optional verifier opening accumulator
    /// * `transcript` - The Fiat-Shamir transcript
    /// * `zk_enabled` - Whether ZK mode is enabled (currently uses hybrid mode)
    ///
    /// # Returns
    /// The sumcheck challenges if verification succeeds.
    #[allow(clippy::type_complexity)]
    pub fn verify<F, PCS, ProofTranscript>(
        proof: &SumcheckInstanceProof<F, ProofTranscript>,
        _zk_proof: Option<&ZKSumcheckProof<F, PCS>>,
        sumcheck_instances: Vec<&dyn SumcheckInstance<F>>,
        opening_accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        transcript: &mut ProofTranscript,
        _zk_enabled: bool,
    ) -> Result<Vec<F>, ProofVerifyError>
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
        ProofTranscript: Transcript,
    {
        // In hybrid mode, we use standard verification since the prover
        // uses standard transcript operations (compressed_polys appended to transcript)
        BatchedSumcheck::verify(
            proof,
            sumcheck_instances,
            opening_accumulator,
            transcript,
        )
    }

    /// Verifies a batch of sumcheck instances with full ZK mode using hiding commitments.
    ///
    /// This method verifies a ZK sumcheck where the prover committed to round polynomials
    /// instead of sending coefficients directly.
    ///
    /// # Arguments
    /// * `proof` - The standard sumcheck proof (for polynomial evaluations)
    /// * `zk_proof` - The ZK proof containing round commitments
    /// * `sumcheck_instances` - The sumcheck instances to verify
    /// * `opening_accumulator` - Optional verifier opening accumulator
    /// * `transcript` - The Fiat-Shamir transcript
    ///
    /// # Returns
    /// The sumcheck challenges if verification succeeds.
    #[allow(clippy::type_complexity)]
    pub fn verify_full_zk<F, PCS, ProofTranscript>(
        proof: &SumcheckInstanceProof<F, ProofTranscript>,
        zk_proof: &ZKSumcheckProof<F, PCS>,
        sumcheck_instances: Vec<&dyn SumcheckInstance<F>>,
        opening_accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F>, ProofVerifyError>
    where
        F: JoltField,
        PCS: HidingCommitmentScheme<Field = F>,
        ProofTranscript: Transcript,
    {
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

        // Verify we have the right number of rounds
        if proof.compressed_polys.len() != max_num_rounds {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }
        if zk_proof.round_commitments.len() != max_num_rounds {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        // Get batching coefficients (same as prover)
        let batching_coeffs: Vec<F> = transcript.challenge_vector(sumcheck_instances.len());

        // Compute batched claim
        let batched_claim: F = sumcheck_instances
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(sumcheck, coeff)| {
                let num_rounds = sumcheck.num_rounds();
                sumcheck
                    .input_claim()
                    .mul_pow_2(max_num_rounds - num_rounds)
                    * coeff
            })
            .sum();

        let mut current_claim = batched_claim;
        let mut r_sumcheck: Vec<F> = Vec::with_capacity(max_num_rounds);

        for round in 0..max_num_rounds {
            // Decompress polynomial and verify degree
            let poly = proof.compressed_polys[round].decompress(&current_claim);
            if poly.degree() > max_degree {
                return Err(ProofVerifyError::SumcheckVerificationError);
            }

            // Verify sum check: h(0) + h(1) = current_claim
            let h0 = poly.evaluate(&F::zero());
            let h1 = poly.evaluate(&F::one());
            if h0 + h1 != current_claim {
                return Err(ProofVerifyError::SumcheckVerificationError);
            }

            // *** ZK CHANGE: Append COMMITMENT to transcript (not polynomial!) ***
            // This must match what the prover does
            zk_proof.round_commitments[round].append_to_transcript(transcript);

            // Get challenge
            let r_j: F = transcript.challenge_scalar();
            r_sumcheck.push(r_j);

            // Update claim for next round
            current_claim = poly.evaluate(&r_j);
        }

        // Verify final output claim
        let expected_output_claim: F = sumcheck_instances
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(sumcheck, coeff)| {
                let r_slice = &r_sumcheck[max_num_rounds - sumcheck.num_rounds()..];

                if let Some(opening_accumulator) = &opening_accumulator {
                    sumcheck.cache_openings_verifier(
                        opening_accumulator.clone(),
                        sumcheck.normalize_opening_point(r_slice),
                    );
                }
                let claim = sumcheck.expected_output_claim(opening_accumulator.clone(), r_slice);

                claim * coeff
            })
            .sum();

        if current_claim != expected_output_claim {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        Ok(r_sumcheck)
    }

    /// Verifies a full ZK sumcheck proof with batch opening proof verification.
    ///
    /// This is the complete zero-knowledge verification that:
    /// 1. Verifies sumcheck relations using commitments in transcript
    /// 2. Verifies the batch opening proof for the committed round polynomials
    ///
    /// # Arguments
    /// * `proof` - The standard sumcheck proof (for polynomial evaluations)
    /// * `zk_proof` - The ZK proof containing round commitments
    /// * `batch_opening` - The batch opening proof with evaluations and PCS proof
    /// * `sumcheck_instances` - The sumcheck instances to verify
    /// * `opening_accumulator` - Optional verifier opening accumulator
    /// * `pcs_setup` - The PCS verifier setup
    /// * `transcript` - The Fiat-Shamir transcript
    ///
    /// # Returns
    /// The sumcheck challenges if verification succeeds.
    #[allow(clippy::type_complexity)]
    pub fn verify_full_zk_with_batch_opening_instances<F, PCS, ProofTranscript>(
        proof: &SumcheckInstanceProof<F, ProofTranscript>,
        zk_proof: &ZKSumcheckProof<F, PCS>,
        batch_opening: &ZKSumcheckBatchOpeningProof<F, PCS>,
        sumcheck_instances: Vec<&dyn SumcheckInstance<F>>,
        opening_accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        pcs_setup: &PCS::VerifierSetup,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F>, ProofVerifyError>
    where
        F: JoltField,
        PCS: HidingCommitmentScheme<Field = F>,
        ProofTranscript: Transcript,
    {
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

        // Verify we have the right number of rounds
        if proof.compressed_polys.len() != max_num_rounds {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }
        if zk_proof.round_commitments.len() != max_num_rounds {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        // Get batching coefficients (same as prover)
        let batching_coeffs: Vec<F> = transcript.challenge_vector(sumcheck_instances.len());

        // Compute batched claim (initial claim for batch opening verification)
        let batched_claim: F = sumcheck_instances
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(sumcheck, coeff)| {
                let num_rounds = sumcheck.num_rounds();
                sumcheck
                    .input_claim()
                    .mul_pow_2(max_num_rounds - num_rounds)
                    * coeff
            })
            .sum();

        let mut current_claim = batched_claim;
        let mut r_sumcheck: Vec<F> = Vec::with_capacity(max_num_rounds);

        for round in 0..max_num_rounds {
            // Decompress polynomial and verify degree
            let poly = proof.compressed_polys[round].decompress(&current_claim);
            if poly.degree() > max_degree {
                return Err(ProofVerifyError::SumcheckVerificationError);
            }

            // Verify sum check: h(0) + h(1) = current_claim
            let h0 = poly.evaluate(&F::zero());
            let h1 = poly.evaluate(&F::one());
            if h0 + h1 != current_claim {
                return Err(ProofVerifyError::SumcheckVerificationError);
            }

            // *** ZK CHANGE: Append COMMITMENT to transcript (not polynomial!) ***
            // This must match what the prover does
            zk_proof.round_commitments[round].append_to_transcript(transcript);

            // Get challenge
            let r_j: F = transcript.challenge_scalar();
            r_sumcheck.push(r_j);

            // Update claim for next round
            current_claim = poly.evaluate(&r_j);
        }

        // Verify final output claim
        let expected_output_claim: F = sumcheck_instances
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(sumcheck, coeff)| {
                let r_slice = &r_sumcheck[max_num_rounds - sumcheck.num_rounds()..];

                if let Some(opening_accumulator) = &opening_accumulator {
                    sumcheck.cache_openings_verifier(
                        opening_accumulator.clone(),
                        sumcheck.normalize_opening_point(r_slice),
                    );
                }
                let claim = sumcheck.expected_output_claim(opening_accumulator.clone(), r_slice);

                claim * coeff
            })
            .sum();

        if current_claim != expected_output_claim {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        // *** Now verify the batch opening proof ***
        // The transcript is now at the same state as after the prover's sumcheck
        let _final_claim = batch_opening.verify(
            batched_claim,
            &zk_proof.round_commitments,
            &r_sumcheck,
            pcs_setup,
            transcript,
        )?;

        Ok(r_sumcheck)
    }

    /// Verifies a full ZK sumcheck proof with batch opening proof.
    ///
    /// This is the complete zero-knowledge verification that:
    /// 1. Reconstructs challenges from commitments in transcript
    /// 2. Verifies sumcheck relations using the batch opening proof
    /// 3. Verifies the PCS opening proof for the combined polynomial
    ///
    /// Unlike `verify_full_zk`, this does NOT require the compressed polynomials -
    /// all verification is done through the batch opening proof.
    ///
    /// # Arguments
    /// * `zk_proof` - The ZK proof containing round commitments
    /// * `batch_opening` - The batch opening proof with evaluations and PCS proof
    /// * `initial_claim` - The initial batched sumcheck claim
    /// * `expected_output_claim` - The expected final output claim
    /// * `pcs_setup` - The PCS verifier setup
    /// * `transcript` - The Fiat-Shamir transcript
    ///
    /// # Returns
    /// The sumcheck challenges if verification succeeds.
    #[allow(clippy::type_complexity)]
    pub fn verify_full_zk_with_batch_opening<F, PCS, ProofTranscript>(
        zk_proof: &ZKSumcheckProof<F, PCS>,
        batch_opening: &ZKSumcheckBatchOpeningProof<F, PCS>,
        initial_claim: F,
        expected_output_claim: F,
        pcs_setup: &PCS::VerifierSetup,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F>, ProofVerifyError>
    where
        F: JoltField,
        PCS: HidingCommitmentScheme<Field = F>,
        ProofTranscript: Transcript,
    {
        let num_rounds = zk_proof.round_commitments.len();

        // Reconstruct challenges from transcript (same as prover)
        let mut challenges = Vec::with_capacity(num_rounds);
        for commitment in &zk_proof.round_commitments {
            commitment.append_to_transcript(transcript);
            let r_j: F = transcript.challenge_scalar();
            challenges.push(r_j);
        }

        // Verify the batch opening proof
        let final_claim = batch_opening.verify(
            initial_claim,
            &zk_proof.round_commitments,
            &challenges,
            pcs_setup,
            transcript,
        )?;

        // Check the final claim matches expected
        if final_claim != expected_output_claim {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        Ok(challenges)
    }

    /// Internal ZK verification (hybrid mode) that matches the prover's transcript operations.
    ///
    /// This appends commitments (not raw polynomials) to the transcript to match
    /// what the ZK prover does, ensuring Fiat-Shamir challenges are consistent.
    #[allow(dead_code)]
    fn verify_zk_internal<F, PCS, ProofTranscript>(
        proof: &SumcheckInstanceProof<F, ProofTranscript>,
        zk_proof: &ZKSumcheckProof<F, PCS>,
        sumcheck_instances: Vec<&dyn SumcheckInstance<F>>,
        opening_accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F>, ProofVerifyError>
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
        ProofTranscript: Transcript,
    {
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

        // Verify we have the right number of rounds
        if proof.compressed_polys.len() != max_num_rounds {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        // Get batching coefficients (same as prover)
        let batching_coeffs: Vec<F> = transcript.challenge_vector(sumcheck_instances.len());

        // Compute batched claim
        let batched_claim: F = sumcheck_instances
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(sumcheck, coeff)| {
                let num_rounds = sumcheck.num_rounds();
                sumcheck
                    .input_claim()
                    .mul_pow_2(max_num_rounds - num_rounds)
                    * coeff
            })
            .sum();

        let mut current_claim = batched_claim;
        let mut r_sumcheck: Vec<F> = Vec::with_capacity(max_num_rounds);

        for round in 0..max_num_rounds {
            // Decompress polynomial and verify degree
            let poly = proof.compressed_polys[round].decompress(&current_claim);
            if poly.degree() > max_degree {
                return Err(ProofVerifyError::SumcheckVerificationError);
            }

            // Verify sum check: h(0) + h(1) = current_claim
            let h0 = poly.evaluate(&F::zero());
            let h1 = poly.evaluate(&F::one());
            if h0 + h1 != current_claim {
                return Err(ProofVerifyError::SumcheckVerificationError);
            }

            // *** ZK CHANGE: Append commitment to transcript (not polynomial!) ***
            // This must match what the prover does
            if round < zk_proof.round_commitments.len() {
                zk_proof.round_commitments[round].append_to_transcript(transcript);
            } else {
                // Fallback to compressed poly if no commitment available
                proof.compressed_polys[round].append_to_transcript(transcript);
            }

            // Get challenge
            let r_j: F = transcript.challenge_scalar();
            r_sumcheck.push(r_j);

            // Update claim for next round
            current_claim = poly.evaluate(&r_j);
        }

        // Verify final output claim
        let expected_output_claim: F = sumcheck_instances
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(sumcheck, coeff)| {
                let r_slice = &r_sumcheck[max_num_rounds - sumcheck.num_rounds()..];

                if let Some(opening_accumulator) = &opening_accumulator {
                    sumcheck.cache_openings_verifier(
                        opening_accumulator.clone(),
                        sumcheck.normalize_opening_point(r_slice),
                    );
                }
                let claim = sumcheck.expected_output_claim(opening_accumulator.clone(), r_slice);

                claim * coeff
            })
            .sum();

        if current_claim != expected_output_claim {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        Ok(r_sumcheck)
    }
}
