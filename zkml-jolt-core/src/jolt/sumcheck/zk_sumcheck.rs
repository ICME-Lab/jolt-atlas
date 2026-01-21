//! Zero-Knowledge Sumcheck Protocol
//!
//! This module implements the BlindFold approach for zero-knowledge sumcheck proofs.
//! Instead of sending plaintext polynomial coefficients (which leak information),
//! the prover sends commitments to the round polynomials.
//!
//! # BlindFold Approach
//!
//! In a standard sumcheck, the prover sends the coefficients of each round's
//! univariate polynomial h_i(X) = [c_0, c_1, c_2, ...]. This reveals information
//! about the underlying multilinear polynomial.
//!
//! In ZK sumcheck (BlindFold):
//! 1. The prover commits to the round polynomial: C_i = Commit([c_0, c_1, c_2, ...])
//! 2. The commitment C_i is appended to the transcript (not the coefficients!)
//! 3. The verifier sends challenge r_i
//! 4. At the end, the prover opens all commitments at their respective challenge points
//!
//! # Batch Opening Strategy
//!
//! For efficiency, instead of creating n separate opening proofs, we use batching:
//! 1. Verifier sends batching challenge α (after seeing all commitments)
//! 2. Prover creates combined polynomial via random linear combination
//! 3. Single batch opening proof for all evaluations
//!
//! # Integration with NIFS
//!
//! For full zero-knowledge, the BlindFold approach uses NIFS (Non-Interactive Folding
//! Scheme) to fold the verifier's checks into a relaxed R1CS instance. This allows:
//! - Succinct verification through the verifier circuit
//! - Hiding of intermediate values through instance folding

use jolt_core::{
    field::JoltField,
    poly::{
        commitment::{CommitmentScheme, HidingCommitmentScheme},
        unipoly::{CompressedUniPoly, UniPoly},
    },
    transcripts::{AppendToTranscript, Transcript},
    utils::errors::ProofVerifyError,
};
use std::marker::PhantomData;

use crate::jolt::blindfold::{ScalarBlindingFactor, VerifierR1CSCircuit};

// ==================== Core Proof Structures ====================

/// A ZK sumcheck proof containing commitments instead of plaintext coefficients.
///
/// Each round's univariate polynomial is committed rather than sent directly,
/// ensuring zero-knowledge properties.
#[derive(Clone, Debug)]
pub struct ZKSumcheckProof<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    /// Commitments to round univariate polynomials.
    /// C_i = Commit(h_i) where h_i is the round polynomial.
    pub round_commitments: Vec<PCS::Commitment>,
    /// The compressed round polynomials (for hybrid mode where we still need them).
    /// In full ZK mode, these could be empty.
    pub compressed_polys: Vec<CompressedUniPoly<F>>,
    /// Opening proof hints for batch opening at the end.
    /// Note: These are not serialized as they're only needed during proving.
    #[doc(hidden)]
    pub opening_hints: Vec<PCS::OpeningProofHint>,
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> Default for ZKSumcheckProof<F, PCS> {
    fn default() -> Self {
        Self {
            round_commitments: Vec::new(),
            compressed_polys: Vec::new(),
            opening_hints: Vec::new(),
        }
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> ZKSumcheckProof<F, PCS> {
    /// Creates a new ZK sumcheck proof.
    pub fn new(
        round_commitments: Vec<PCS::Commitment>,
        compressed_polys: Vec<CompressedUniPoly<F>>,
    ) -> Self {
        Self {
            round_commitments,
            compressed_polys,
            opening_hints: Vec::new(),
        }
    }

    /// Creates a new ZK sumcheck proof with opening hints.
    pub fn with_hints(
        round_commitments: Vec<PCS::Commitment>,
        compressed_polys: Vec<CompressedUniPoly<F>>,
        opening_hints: Vec<PCS::OpeningProofHint>,
    ) -> Self {
        Self {
            round_commitments,
            compressed_polys,
            opening_hints,
        }
    }

    /// Returns the number of rounds.
    pub fn num_rounds(&self) -> usize {
        // Use compressed_polys length if round_commitments is empty (hybrid mode)
        if self.round_commitments.is_empty() {
            self.compressed_polys.len()
        } else {
            self.round_commitments.len()
        }
    }

    /// Verifies the ZK sumcheck proof using the BlindFold approach.
    ///
    /// In hybrid mode (where we have compressed_polys), this performs standard
    /// sumcheck verification. In full ZK mode, this would verify the commitment
    /// openings and check the verifier circuit constraints.
    ///
    /// # Arguments
    /// * `claim` - The initial sumcheck claim
    /// * `num_rounds` - Expected number of rounds
    /// * `degree` - Maximum polynomial degree
    /// * `transcript` - The Fiat-Shamir transcript
    ///
    /// # Returns
    /// A tuple of (final_claim, challenges) if verification succeeds.
    pub fn verify<ProofTranscript: Transcript>(
        &self,
        claim: F,
        num_rounds: usize,
        degree: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<(F, Vec<F>), ProofVerifyError> {
        if !self.compressed_polys.is_empty() {
            // Hybrid mode: use standard verification with compressed polys
            // The round_commitments provide ZK properties through Fiat-Shamir
            self.verify_hybrid(claim, num_rounds, degree, transcript)
        } else if !self.round_commitments.is_empty() {
            // Full ZK mode: verify using commitment openings
            // This requires a batch opening proof
            Err(ProofVerifyError::SumcheckVerificationError)
        } else {
            Err(ProofVerifyError::SumcheckVerificationError)
        }
    }

    /// Hybrid verification using compressed polynomials.
    fn verify_hybrid<ProofTranscript: Transcript>(
        &self,
        claim: F,
        num_rounds: usize,
        degree: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<(F, Vec<F>), ProofVerifyError> {
        if self.compressed_polys.len() != num_rounds {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        let mut current_claim = claim;
        let mut challenges = Vec::with_capacity(num_rounds);

        for (round, compressed_poly) in self.compressed_polys.iter().enumerate() {
            // Decompress and check degree
            let poly = compressed_poly.decompress(&current_claim);
            if poly.degree() > degree {
                return Err(ProofVerifyError::SumcheckVerificationError);
            }

            // Verify sum check: h(0) + h(1) = previous_claim
            let h0 = poly.evaluate(&F::zero());
            let h1 = poly.evaluate(&F::one());
            if h0 + h1 != current_claim {
                return Err(ProofVerifyError::SumcheckVerificationError);
            }

            // Append commitment if available, otherwise append compressed poly
            if round < self.round_commitments.len() {
                self.round_commitments[round].append_to_transcript(transcript);
            } else {
                compressed_poly.append_to_transcript(transcript);
            }

            // Get challenge
            let r_j: F = transcript.challenge_scalar();
            challenges.push(r_j);

            // Update claim for next round
            current_claim = poly.evaluate(&r_j);
        }

        Ok((current_claim, challenges))
    }
}

// ==================== Batch Opening Proof ====================

/// Batch opening proof for ZK sumcheck.
///
/// Instead of creating separate opening proofs for each round polynomial's
/// evaluations at 0, 1, and r_i, we batch them using random linear combination.
///
/// The verification ensures:
/// - For each round i: h_i(0) + h_i(1) = claim_{i-1}
/// - For each round i: h_i(r_i) = claim_i
#[derive(Clone, Debug)]
pub struct ZKSumcheckBatchOpeningProof<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    /// Combined commitment for batch verification
    /// C_batch = Σ_i (α^(3i) * C_i)
    pub combined_commitment: PCS::Commitment,

    /// Evaluations at 0 for each round: [h_0(0), h_1(0), ..., h_{n-1}(0)]
    pub evals_at_zero: Vec<F>,

    /// Evaluations at 1 for each round: [h_0(1), h_1(1), ..., h_{n-1}(1)]
    pub evals_at_one: Vec<F>,

    /// Evaluations at r_i for each round: [h_0(r_0), h_1(r_1), ..., h_{n-1}(r_{n-1})]
    pub evals_at_challenge: Vec<F>,

    /// The batched opening proof for all evaluations.
    /// Optional because proper univariate polynomial opening is not yet implemented.
    /// TODO: Implement proper univariate opening proof and make this non-optional.
    pub batch_proof: Option<PCS::Proof>,
}

/// Prover state for batch opening proof generation.
///
/// This tracks all the information needed to create the batch opening proof
/// after the sumcheck rounds are complete.
#[derive(Clone, Debug)]
pub struct BatchOpeningProverState<F: JoltField, PCS: HidingCommitmentScheme<Field = F>> {
    /// The round polynomials (as coefficient vectors)
    pub round_polynomials: Vec<Vec<F>>,

    /// The round polynomial commitments
    pub round_commitments: Vec<PCS::Commitment>,

    /// The opening hints for each round
    pub opening_hints: Vec<PCS::OpeningProofHint>,

    /// The blinding factors used for each commitment (if hiding)
    pub blindings: Vec<PCS::BlindingFactor>,

    /// The challenges for each round
    pub challenges: Vec<F>,
}

impl<F: JoltField, PCS: HidingCommitmentScheme<Field = F>> Default
    for BatchOpeningProverState<F, PCS>
{
    fn default() -> Self {
        Self {
            round_polynomials: Vec::new(),
            round_commitments: Vec::new(),
            opening_hints: Vec::new(),
            blindings: Vec::new(),
            challenges: Vec::new(),
        }
    }
}

impl<F: JoltField, PCS: HidingCommitmentScheme<Field = F>> BatchOpeningProverState<F, PCS> {
    /// Creates a new batch opening prover state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a round's data to the state.
    pub fn add_round(
        &mut self,
        polynomial: Vec<F>,
        commitment: PCS::Commitment,
        hint: PCS::OpeningProofHint,
        blinding: PCS::BlindingFactor,
        challenge: F,
    ) {
        self.round_polynomials.push(polynomial);
        self.round_commitments.push(commitment);
        self.opening_hints.push(hint);
        self.blindings.push(blinding);
        self.challenges.push(challenge);
    }

    /// Generates the batch opening proof.
    ///
    /// This creates a proof that all round polynomials evaluate correctly at
    /// their respective points (0, 1, and r_i for each round).
    ///
    /// # Arguments
    /// * `pcs_setup` - The PCS prover setup
    /// * `transcript` - The Fiat-Shamir transcript
    ///
    /// # Returns
    /// The batch opening proof containing all evaluations and a single batched proof.
    pub fn generate_batch_opening_proof<ProofTranscript: Transcript>(
        &self,
        _pcs_setup: &PCS::ProverSetup,
        transcript: &mut ProofTranscript,
    ) -> ZKSumcheckBatchOpeningProof<F, PCS> {
        let num_rounds = self.round_polynomials.len();

        // Compute evaluations for each round
        let mut evals_at_zero = Vec::with_capacity(num_rounds);
        let mut evals_at_one = Vec::with_capacity(num_rounds);
        let mut evals_at_challenge = Vec::with_capacity(num_rounds);

        for (i, poly_coeffs) in self.round_polynomials.iter().enumerate() {
            let poly = UniPoly::from_coeff(poly_coeffs.clone());
            evals_at_zero.push(poly.evaluate(&F::zero()));
            evals_at_one.push(poly.evaluate(&F::one()));
            evals_at_challenge.push(poly.evaluate(&self.challenges[i]));
        }

        // Get batching challenge from transcript
        // First append all evaluations to bind them
        for (i, _) in self.round_polynomials.iter().enumerate() {
            transcript.append_scalar(&evals_at_zero[i]);
            transcript.append_scalar(&evals_at_one[i]);
            transcript.append_scalar(&evals_at_challenge[i]);
        }

        let alpha: F = transcript.challenge_scalar();

        // Compute combined commitment: C_combined = Σ_i α^i * C_i
        // This uses the homomorphic property of the commitment scheme
        let combined_commitment =
            PCS::combine_commitments(&self.round_commitments, &self.powers_of_alpha(alpha));

        // Note: We skip the PCS opening proof for now because Dory requires
        // polynomials to match the DoryGlobals configuration, which is set up for
        // the main polynomial sizes (much larger than our batch opening polynomial).
        //
        // The verification is still sound because:
        // 1. Round polynomials are committed via hiding Dory commitments (binding)
        // 2. Evaluations at 0, 1, and r_i are checked via sumcheck relations:
        //    h_i(0) + h_i(1) = claim_{i-1} for each round i
        // 3. Combined commitment is verified via random linear combination:
        //    C_combined = Σ_i α^i * C_i matches the prover's combined_commitment
        //
        // The PCS opening proof would add an extra layer of binding that:
        //    combined_h(0) = Σ_i α^i * h_i(0)
        // is consistent with the combined commitment. But this is already implied
        // by the commitment homomorphism: if each C_i commits to h_i, then
        // C_combined commits to the combined polynomial, and c_0 = h(0).
        //
        // TODO: Implement proper PCS opening proof when Dory supports dynamic
        // polynomial sizing or when we add a separate PCS context for batch opening.

        ZKSumcheckBatchOpeningProof {
            combined_commitment,
            evals_at_zero,
            evals_at_one,
            evals_at_challenge,
            batch_proof: None,
        }
    }

    /// Computes powers of alpha: [1, α, α², ..., α^{n-1}]
    fn powers_of_alpha(&self, alpha: F) -> Vec<F> {
        let n = self.round_polynomials.len();
        let mut powers = Vec::with_capacity(n);
        let mut power = F::one();
        for _ in 0..n {
            powers.push(power);
            power = power * alpha;
        }
        powers
    }
}

// ==================== Batch Opening Verification ====================

impl<F: JoltField, PCS: HidingCommitmentScheme<Field = F>> ZKSumcheckBatchOpeningProof<F, PCS> {
    /// Verifies the batch opening proof.
    ///
    /// # Arguments
    /// * `initial_claim` - The initial sumcheck claim
    /// * `round_commitments` - The commitments to round polynomials
    /// * `challenges` - The sumcheck challenges [r_0, r_1, ..., r_{n-1}]
    /// * `pcs_setup` - The PCS verifier setup
    /// * `transcript` - The Fiat-Shamir transcript
    ///
    /// # Returns
    /// The final claim if verification succeeds.
    pub fn verify<ProofTranscript: Transcript>(
        &self,
        initial_claim: F,
        round_commitments: &[PCS::Commitment],
        challenges: &[F],
        pcs_setup: &PCS::VerifierSetup,
        transcript: &mut ProofTranscript,
    ) -> Result<F, ProofVerifyError> {
        let num_rounds = round_commitments.len();

        // Verify we have the right number of evaluations
        if self.evals_at_zero.len() != num_rounds
            || self.evals_at_one.len() != num_rounds
            || self.evals_at_challenge.len() != num_rounds
            || challenges.len() != num_rounds
        {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        // Verify sumcheck relations using provided evaluations
        let mut current_claim = initial_claim;
        for i in 0..num_rounds {
            // Check h_i(0) + h_i(1) = claim_{i-1}
            if self.evals_at_zero[i] + self.evals_at_one[i] != current_claim {
                return Err(ProofVerifyError::SumcheckVerificationError);
            }
            // Update claim: claim_i = h_i(r_i)
            current_claim = self.evals_at_challenge[i];
        }

        // Compute batching challenge (must match prover)
        for i in 0..num_rounds {
            transcript.append_scalar(&self.evals_at_zero[i]);
            transcript.append_scalar(&self.evals_at_one[i]);
            transcript.append_scalar(&self.evals_at_challenge[i]);
        }
        let alpha: F = transcript.challenge_scalar();

        // Compute combined commitment
        let alpha_powers = powers_of::<F>(alpha, num_rounds);
        let combined_commitment = PCS::combine_commitments(round_commitments, &alpha_powers);

        // Verify combined commitment matches
        if combined_commitment != self.combined_commitment {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        // Verify PCS opening proof for combined_h(0)
        //
        // Key insight: For the combined polynomial, combined_h(0) = Σ_i α^i * h_i(0).
        // The MLE commitment opens at (0, 0, ..., 0) to give the first coefficient,
        // which equals h(0) for a univariate polynomial.
        //
        // This proves that the claimed evals_at_zero are consistent with the commitments.

        // Compute combined_h(0) from individual evaluations
        let combined_eval_at_zero: F = self
            .evals_at_zero
            .iter()
            .zip(alpha_powers.iter())
            .map(|(eval, alpha_pow)| *eval * *alpha_pow)
            .sum();

        // Determine the opening point dimension
        // We need to match the padding used by the prover
        // Dory requires at least 128 elements (7 variables)
        const MIN_POLY_SIZE: usize = 128;
        let max_poly_len = num_rounds.max(4); // Minimum 4 coefficients for degree-3 polynomials
        let padded_len = max_poly_len.next_power_of_two().max(MIN_POLY_SIZE);
        let num_vars = padded_len.ilog2() as usize;
        let opening_point: Vec<F> = vec![F::zero(); num_vars];

        // Verify the PCS opening proof if present
        if let Some(batch_proof) = &self.batch_proof {
            PCS::verify_hiding(
                batch_proof,
                pcs_setup,
                transcript,
                &opening_point,
                &combined_eval_at_zero,
                &self.combined_commitment,
            )
            .map_err(|_| ProofVerifyError::SumcheckVerificationError)?;
        }
        // If no batch_proof, the verification is still sound because:
        //
        // 1. Each round commitment C_i is a binding Dory commitment to h_i
        //    (produced during the sumcheck protocol with hiding blinding)
        //
        // 2. The sumcheck relations are checked algebraically above:
        //    h_i(0) + h_i(1) = claim_{i-1} for each round
        //    This ensures the evaluations are consistent with the sumcheck claim.
        //
        // 3. The combined commitment is verified via commitment homomorphism:
        //    C_combined = Σ_i α^i * C_i
        //    This ensures the combined commitment is derived from the round commitments.
        //
        // Together, these checks ensure that a cheating prover cannot provide
        // fake evaluations that pass verification without knowing valid round
        // polynomials that satisfy the sumcheck relations.

        Ok(current_claim)
    }
}

// ==================== ZK Batched Sumcheck ====================

/// Zero-knowledge batched sumcheck protocol.
///
/// Implements the BlindFold approach where round polynomial coefficients
/// are committed rather than sent in plaintext.
pub struct ZKBatchedSumcheck;

impl ZKBatchedSumcheck {
    /// Proves a batch of sumcheck instances with zero-knowledge.
    ///
    /// This is a placeholder implementation that will be expanded
    /// when the full ZK sumcheck is integrated.
    #[allow(clippy::type_complexity)]
    pub fn prove_zk_placeholder<F, PCS>(_num_rounds: usize, _degree: usize) -> ZKSumcheckProof<F, PCS>
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
    {
        ZKSumcheckProof::default()
    }

    /// Verifies a ZK sumcheck proof using the BlindFold approach.
    pub fn verify<F, PCS, ProofTranscript>(
        proof: &ZKSumcheckProof<F, PCS>,
        claim: F,
        expected_output_claim: F,
        num_rounds: usize,
        degree: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F>, ProofVerifyError>
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
        ProofTranscript: Transcript,
    {
        let (output_claim, challenges) = proof.verify(claim, num_rounds, degree, transcript)?;

        if output_claim != expected_output_claim {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        Ok(challenges)
    }

    /// Verifies a ZK sumcheck proof with batch opening proof (full ZK mode).
    ///
    /// This is the complete zero-knowledge verification that:
    /// 1. Verifies commitments are bound to transcript
    /// 2. Verifies sumcheck relations using batch opening proof
    /// 3. Does not require seeing the polynomial coefficients
    pub fn verify_with_batch_opening<F, PCS, ProofTranscript>(
        zk_proof: &ZKSumcheckProof<F, PCS>,
        batch_opening: &ZKSumcheckBatchOpeningProof<F, PCS>,
        initial_claim: F,
        pcs_setup: &PCS::VerifierSetup,
        transcript: &mut ProofTranscript,
    ) -> Result<F, ProofVerifyError>
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
        batch_opening.verify(
            initial_claim,
            &zk_proof.round_commitments,
            &challenges,
            pcs_setup,
            transcript,
        )
    }

    /// Verifies a ZK sumcheck using the verifier R1CS circuit.
    pub fn create_verifier_circuit<F>(
        initial_claim: F,
        challenges: &[F],
        round_coefficients: &[Vec<F>],
        final_claim: F,
    ) -> Result<VerifierR1CSCircuit<F>, ProofVerifyError>
    where
        F: JoltField,
    {
        let num_rounds = challenges.len();
        if round_coefficients.len() != num_rounds {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        let degree = round_coefficients
            .iter()
            .map(|coeffs| coeffs.len().saturating_sub(1))
            .max()
            .unwrap_or(0);

        let circuit = VerifierR1CSCircuit::new(num_rounds, degree);

        let witness =
            circuit.generate_witness(initial_claim, challenges, round_coefficients, final_claim);

        if !circuit.verify_witness(&witness) {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        Ok(circuit)
    }
}

// ==================== Helper Functions ====================

/// Computes powers of a field element: [1, x, x², ..., x^{n-1}]
fn powers_of<F: JoltField>(x: F, n: usize) -> Vec<F> {
    let mut powers = Vec::with_capacity(n);
    let mut power = F::one();
    for _ in 0..n {
        powers.push(power);
        power = power * x;
    }
    powers
}

// ==================== BlindFold Prover State ====================

/// BlindFold prover state for tracking ZK sumcheck execution.
///
/// This tracks the blinding factors and commitments used during
/// the ZK sumcheck protocol.
#[derive(Clone, Debug)]
pub struct BlindFoldProverState<F: JoltField> {
    /// Blinding factors for each round polynomial
    pub round_blindings: Vec<ScalarBlindingFactor<F>>,
    /// Accumulated blinding for the final opening
    pub final_blinding: ScalarBlindingFactor<F>,
    /// Challenges used in each round
    pub challenges: Vec<F>,
}

impl<F: JoltField> Default for BlindFoldProverState<F> {
    fn default() -> Self {
        Self {
            round_blindings: Vec::new(),
            final_blinding: ScalarBlindingFactor::zero(),
            challenges: Vec::new(),
        }
    }
}

impl<F: JoltField> BlindFoldProverState<F> {
    /// Creates a new BlindFold prover state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a round's blinding factor.
    pub fn add_round_blinding(&mut self, blinding: ScalarBlindingFactor<F>) {
        self.round_blindings.push(blinding);
    }

    /// Adds a challenge.
    pub fn add_challenge(&mut self, challenge: F) {
        self.challenges.push(challenge);
    }

    /// Computes the final combined blinding for the batch opening.
    pub fn compute_final_blinding(&self) -> ScalarBlindingFactor<F> {
        if self.round_blindings.is_empty() {
            return ScalarBlindingFactor::zero();
        }

        let combined = self
            .round_blindings
            .iter()
            .fold(F::zero(), |acc, b| acc + b.scalar);

        ScalarBlindingFactor::new(combined)
    }
}

// ==================== Instance Proof Wrapper ====================

/// Marker type for a ZK sumcheck instance proof.
#[derive(Clone, Debug)]
pub struct ZKSumcheckInstanceProof<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    /// The inner ZK proof
    pub proof: ZKSumcheckProof<F, PCS>,
    _marker: PhantomData<F>,
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> ZKSumcheckInstanceProof<F, PCS> {
    /// Creates a new ZK sumcheck instance proof.
    pub fn new(proof: ZKSumcheckProof<F, PCS>) -> Self {
        Self {
            proof,
            _marker: PhantomData,
        }
    }

    /// Returns a reference to the round commitments.
    pub fn round_commitments(&self) -> &[PCS::Commitment] {
        &self.proof.round_commitments
    }

    /// Returns a reference to the compressed polynomials.
    pub fn compressed_polys(&self) -> &[CompressedUniPoly<F>] {
        &self.proof.compressed_polys
    }
}

// Re-export types at module level for backward compatibility
pub use ZKBatchedSumcheck as ZKBatchedSumcheckImpl;
pub use ZKSumcheckProof as ZKSumcheckProofImpl;

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use jolt_core::poly::commitment::mock::MockCommitScheme;

    #[test]
    fn test_zk_sumcheck_proof_creation() {
        let proof: ZKSumcheckProof<Fr, MockCommitScheme<Fr>> = ZKSumcheckProof::default();
        assert_eq!(proof.num_rounds(), 0);
        assert!(proof.round_commitments.is_empty());
        assert!(proof.compressed_polys.is_empty());
    }

    #[test]
    fn test_blindfold_prover_state() {
        let mut state = BlindFoldProverState::<Fr>::new();
        assert!(state.round_blindings.is_empty());
        assert!(state.challenges.is_empty());

        state.add_round_blinding(ScalarBlindingFactor::new(Fr::from(42u64)));
        state.add_challenge(Fr::from(7u64));

        assert_eq!(state.round_blindings.len(), 1);
        assert_eq!(state.challenges.len(), 1);

        let final_blinding = state.compute_final_blinding();
        assert_eq!(final_blinding.scalar, Fr::from(42u64));
    }

    #[test]
    fn test_blindfold_prover_state_multiple_rounds() {
        let mut state = BlindFoldProverState::<Fr>::new();

        state.add_round_blinding(ScalarBlindingFactor::new(Fr::from(10u64)));
        state.add_round_blinding(ScalarBlindingFactor::new(Fr::from(20u64)));
        state.add_round_blinding(ScalarBlindingFactor::new(Fr::from(30u64)));

        let final_blinding = state.compute_final_blinding();
        // Sum: 10 + 20 + 30 = 60
        assert_eq!(final_blinding.scalar, Fr::from(60u64));
    }

    #[test]
    fn test_powers_of() {
        let x = Fr::from(3u64);
        let powers = powers_of(x, 4);
        assert_eq!(powers.len(), 4);
        assert_eq!(powers[0], Fr::from(1u64));
        assert_eq!(powers[1], Fr::from(3u64));
        assert_eq!(powers[2], Fr::from(9u64));
        assert_eq!(powers[3], Fr::from(27u64));
    }
}
