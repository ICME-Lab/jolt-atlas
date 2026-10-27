#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::{
    opening_proof::VerifierOpeningAccumulator,
    sumcheck_verifier::SumcheckInstanceVerifier,
    upstream_sumcheck::{compressed_to_upstream, expect_engine, FacadeTranscript},
};
use atlas_algebra::{
    field::JoltField,
    poly::unipoly::CompressedUniPoly,
    utils::{errors::ProofVerifyError, serialization::deserialize_vec},
};
use atlas_transcript::Transcript;
use jolt_sumcheck::{
    BatchMember, BatchPrelude, BooleanHypercube, ClearSumcheckRecorder, CompressedSumcheckProof,
    SumcheckClaim, SumcheckRecorder, SumcheckVerifier, SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};

use ark_serialize::*;

/// Implements the standard technique for batching parallel sumchecks to reduce
/// verifier cost and proof size.
///
/// For details, refer to Jim Posen's ["Perspectives on Sumcheck Batching"](https://hackmd.io/s/HyxaupAAA).
/// We do what they describe as "front-loaded" batch sumcheck.
pub enum BatchedSumcheck {}
impl BatchedSumcheck {
    #[tracing::instrument(skip_all, name = "BatchedSumcheck::verify")]
    pub fn verify<F: JoltField, ProofTranscript: Transcript>(
        proof: &SumcheckInstanceProof<F>,
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

        let input_claims: Vec<F> = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.input_claim(opening_accumulator))
            .collect();
        absorb_input_claims::<F, _>(
            &mut ClearSumcheckRecorder::<F>::new(),
            &input_claims,
            transcript,
        );

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
            .zip(input_claims.iter())
            .zip(batching_coeffs.iter())
            .map(|((sumcheck, input_claim), coeff)| {
                input_claim.mul_pow_2(max_num_rounds - sumcheck.num_rounds()) * coeff
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

/// Security level the per-round proof of work targets.
pub const ROUND_SECURITY_BITS: u32 = 128;

/// Proof-of-work difficulty before the challenge of a sumcheck round whose polynomial has degree
/// `degree`. A cheating round polynomial survives only if the challenge hits one of at most
/// `degree` roots, probability `degree / |C|` per query for challenge set `C`
/// ([`JoltField::challenge_set_size`]); the grind raises each query's cost so the bound reaches
/// `2^-ROUND_SECURITY_BITS` (`jolt_transcript::grind_bits_for_loss`).
pub fn round_pow_bits<F: JoltField>(degree: usize) -> u32 {
    jolt_transcript::grind_bits_for_loss(
        degree.max(1) as u64,
        F::challenge_set_size(),
        ROUND_SECURITY_BITS,
    )
    .expect("round degrees are far below the proof-of-work limit")
}

pub struct Sumcheck;

impl Sumcheck {
    #[tracing::instrument(skip_all, name = "Sumcheck::verify")]
    pub fn verify<F: JoltField, ProofTranscript: Transcript>(
        proof: &SumcheckInstanceProof<F>,
        sumcheck_instance: &dyn SumcheckInstanceVerifier<F, ProofTranscript>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F::Challenge>, ProofVerifyError> {
        let num_rounds = sumcheck_instance.num_rounds();
        let input_claim = sumcheck_instance.input_claim(opening_accumulator);
        absorb_input_claims::<F, _>(
            &mut ClearSumcheckRecorder::<F>::new(),
            &[input_claim],
            transcript,
        );
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

/// Absorb a batch's input claims the way upstream's clear recorder does (each under
/// `b"sumcheck_claim"`), on the prover and the verifier alike.
fn absorb_input_claims<F: JoltField, T: Transcript>(
    recorder: &mut ClearSumcheckRecorder<F>,
    input_claims: &[F],
    transcript: &mut T,
) {
    let ((), pow) = FacadeTranscript::<T, F>::scope(transcript, 0, |facade| {
        recorder.absorb_input_claims(input_claims, facade)
    });
    pow.expect("no proof of work runs while absorbing claims");
}

/// The upstream batch description of tail-aligned members (a member with fewer rounds is bound
/// in the last `rounds` rounds) with the given input claims and batching coefficients.
fn batch_prelude<F: JoltField>(
    rounds: &[usize],
    input_claims: &[F],
    coefficients: &[F],
    max_num_rounds: usize,
    max_degree: usize,
) -> BatchPrelude<F> {
    let members = rounds
        .iter()
        .zip(input_claims)
        .zip(coefficients)
        .map(|((&rounds, &input_claim), &coefficient)| BatchMember {
            input_claim,
            coefficient,
            rounds,
            offset: max_num_rounds - rounds,
        })
        .collect();
    expect_engine(BatchPrelude::try_new(members, max_num_rounds, max_degree))
}

#[derive(CanonicalSerialize, Debug, Clone)]
pub struct SumcheckInstanceProof<F: JoltField> {
    pub compressed_polys: Vec<CompressedUniPoly<F>>,
}

impl<F: JoltField> Valid for SumcheckInstanceProof<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.compressed_polys.check()
    }
}

impl<F: JoltField> CanonicalDeserialize for SumcheckInstanceProof<F> {
    /// The round count is prover-supplied, so the rounds decode through [`deserialize_vec`].
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(Self {
            compressed_polys: deserialize_vec(reader, compress, validate)?,
        })
    }
}

impl<F: JoltField> SumcheckInstanceProof<F> {
    pub fn new(compressed_polys: Vec<CompressedUniPoly<F>>) -> SumcheckInstanceProof<F> {
        SumcheckInstanceProof { compressed_polys }
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
    pub fn verify<ProofTranscript: Transcript>(
        &self,
        claim: F,
        num_rounds: usize,
        degree_bound: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<(F, Vec<F::Challenge>), ProofVerifyError> {
        if self.compressed_polys.len() != num_rounds {
            return Err(ProofVerifyError::InvalidInputLength(
                num_rounds,
                self.compressed_polys.len(),
            ));
        }
        let claim = SumcheckClaim {
            num_vars: num_rounds,
            degree: degree_bound,
            claimed_sum: claim,
        };
        let proof = CompressedSumcheckProof {
            round_polynomials: self
                .compressed_polys
                .iter()
                .map(compressed_to_upstream)
                .collect(),
        };
        let (res, pow) = FacadeTranscript::<ProofTranscript, F>::scope(
            transcript,
            round_pow_bits::<F>(degree_bound),
            |facade| {
                SumcheckVerifier::verify_compressed(
                    &claim,
                    &proof,
                    BooleanHypercube,
                    SUMCHECK_ROUND_TRANSCRIPT_LABEL,
                    facade,
                )
            },
        );
        pow?;
        let evaluation = res.map_err(|_| ProofVerifyError::SumcheckVerificationError)?;
        let r = evaluation
            .point
            .into_vec()
            .into_iter()
            .map(F::challenge_from_field)
            .collect();
        Ok((evaluation.value, r))
    }
}

pub mod prover;
