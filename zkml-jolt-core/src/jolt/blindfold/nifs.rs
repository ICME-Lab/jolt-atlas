//! Non-Interactive Folding Scheme (NIFS) for Zero-Knowledge.
//!
//! NIFS is a key component of the BlindFold approach. It allows folding a real
//! instance with a random satisfying instance to hide the witness while preserving
//! verifiability.
//!
//! # Folding Overview
//!
//! Given two relaxed R1CS instances (real and random):
//! - Instance_1: (W_1, E_1, u_1, x_1)
//! - Instance_2: (W_2, E_2, u_2, x_2)
//!
//! The folded instance is computed as:
//! - W' = W_1 + r·W_2
//! - E' = E_1 + r·T + r²·E_2  (where T is the cross-term)
//! - u' = u_1 + r·u_2
//! - x' = x_1 + r·x_2
//!
//! The cross-term T is committed before the folding challenge r is derived,
//! ensuring the prover cannot cheat by choosing W_2 adversarially.
//!
//! # Hiding Commitment Support
//!
//! When using a hiding commitment scheme, commitments include blinding factors:
//! - C(W) = Commit(W) + r_W * H
//! - C(E) = Commit(E) + r_E * H
//!
//! The folded blinding factors are computed as:
//! - r_W' = r_W_1 + r * r_W_2
//! - r_E' = r_E_1 + r * r_T + r² * r_E_2

use jolt_core::{
    field::JoltField,
    poly::{
        commitment::{CommitmentScheme, HidingCommitmentScheme},
        multilinear_polynomial::MultilinearPolynomial,
    },
    transcripts::Transcript,
};
use rand_core::{CryptoRng, RngCore};

use super::hiding_commitment::{fold_commitments, fold_error_commitments, ScalarBlindingFactor};
use super::relaxed_r1cs::{R1CSMatrices, RelaxedR1CSInstance, RelaxedR1CSWitness};

/// A NIFS folding proof.
///
/// Contains the commitment to the cross-term T, which is used to verify
/// that the folding was done correctly.
#[derive(Clone, Debug)]
pub struct NIFSProof<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    /// Commitment to the cross-term T
    pub T_commitment: PCS::Commitment,
}

/// A NIFS folding proof with hiding commitment support.
///
/// Extends the basic proof with blinding factors for zero-knowledge.
#[derive(Clone, Debug)]
pub struct HidingNIFSProof<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    /// Commitment to the cross-term T (with blinding)
    pub T_commitment: PCS::Commitment,
    /// Blinding factor for T commitment
    pub T_blinding: ScalarBlindingFactor<F>,
}

/// Blinding factors for a relaxed R1CS instance.
///
/// These are the randomness values used to make commitments hiding.
#[derive(Clone, Debug, Default)]
pub struct InstanceBlindingFactors<F: JoltField> {
    /// Blinding factor for W commitment
    pub W_blinding: ScalarBlindingFactor<F>,
    /// Blinding factor for E commitment
    pub E_blinding: ScalarBlindingFactor<F>,
}

impl<F: JoltField> InstanceBlindingFactors<F> {
    /// Creates new blinding factors.
    pub fn new(W_blinding: ScalarBlindingFactor<F>, E_blinding: ScalarBlindingFactor<F>) -> Self {
        Self {
            W_blinding,
            E_blinding,
        }
    }

    /// Creates zero (non-hiding) blinding factors.
    pub fn zero() -> Self {
        Self {
            W_blinding: ScalarBlindingFactor::zero(),
            E_blinding: ScalarBlindingFactor::zero(),
        }
    }

    /// Folds two blinding factors with a challenge scalar.
    ///
    /// r_W' = r_W_1 + r * r_W_2
    /// r_E' = r_E_1 + r * r_T + r² * r_E_2
    pub fn fold(
        blindings_1: &Self,
        blindings_2: &Self,
        T_blinding: &ScalarBlindingFactor<F>,
        r: F,
    ) -> Self {
        let r_squared = r * r;

        Self {
            W_blinding: ScalarBlindingFactor::new(
                blindings_1.W_blinding.scalar + r * blindings_2.W_blinding.scalar,
            ),
            E_blinding: ScalarBlindingFactor::new(
                blindings_1.E_blinding.scalar
                    + r * T_blinding.scalar
                    + r_squared * blindings_2.E_blinding.scalar,
            ),
        }
    }
}

/// Non-Interactive Folding Scheme.
///
/// Provides methods for folding relaxed R1CS instances in a zero-knowledge manner.
pub struct NIFS;

impl NIFS {
    /// Pads a vector to the next power of 2 length.
    fn pad_to_power_of_2<F: JoltField>(mut vec: Vec<F>) -> Vec<F> {
        if vec.is_empty() {
            return vec![F::zero()];
        }
        let next_pow2 = vec.len().next_power_of_two();
        vec.resize(next_pow2, F::zero());
        vec
    }

    /// Folds two relaxed R1CS instances.
    ///
    /// This is the core of the BlindFold zero-knowledge approach: by folding a real
    /// instance with a random satisfying instance, the witness is hidden while
    /// the verifier can still check correctness.
    ///
    /// # Arguments
    /// * `matrices` - The R1CS constraint matrices
    /// * `instance_real` - The real instance to fold
    /// * `witness_real` - The witness for the real instance
    /// * `instance_rand` - The random satisfying instance
    /// * `witness_rand` - The witness for the random instance
    /// * `transcript` - The Fiat-Shamir transcript
    /// * `pcs_setup` - The PCS prover setup
    ///
    /// # Returns
    /// A tuple containing:
    /// - The folding proof (with cross-term commitment)
    /// - The folded instance
    /// - The folded witness
    #[allow(clippy::type_complexity)]
    pub fn prove<F, PCS, ProofTranscript>(
        matrices: &R1CSMatrices<F>,
        instance_real: &RelaxedR1CSInstance<F, PCS>,
        witness_real: &RelaxedR1CSWitness<F>,
        instance_rand: &RelaxedR1CSInstance<F, PCS>,
        witness_rand: &RelaxedR1CSWitness<F>,
        transcript: &mut ProofTranscript,
        pcs_setup: &PCS::ProverSetup,
    ) -> (
        NIFSProof<F, PCS>,
        RelaxedR1CSInstance<F, PCS>,
        RelaxedR1CSWitness<F>,
    )
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
        PCS::Commitment: Clone,
        ProofTranscript: Transcript,
    {
        // Step 1: Compute the cross-term T
        // T = A(z_1) ∘ B(z_2) + A(z_2) ∘ B(z_1) - u_1·C(z_2) - u_2·C(z_1)
        let cross_term = Self::compute_cross_term(
            matrices,
            witness_real,
            instance_real.u,
            &instance_real.x,
            witness_rand,
            instance_rand.u,
            &instance_rand.x,
        );

        // Step 2: Commit to T (pad to power of 2 for polynomial commitment)
        // Note: For non-hiding NIFS, we use zero blinding (r_T = 0)
        let T_padded = Self::pad_to_power_of_2(cross_term.clone());
        let T_poly: MultilinearPolynomial<F> = T_padded.into();
        let (T_commitment, _T_hint) = PCS::commit(&T_poly, pcs_setup);
        let r_T = F::zero(); // Non-hiding version uses zero blinding

        // Step 3: Get folding challenge from transcript
        transcript.append_serializable(&T_commitment);
        let r: F = transcript.challenge_scalar();

        // Step 4: Fold instances (with proper commitment folding)
        let folded_instance =
            Self::fold_instances_with_t(instance_real, instance_rand, &T_commitment, r);

        // Step 5: Fold witnesses (now includes blinding factor folding)
        let folded_witness = Self::fold_witnesses(witness_real, witness_rand, &cross_term, r_T, r);

        let proof = NIFSProof { T_commitment };

        (proof, folded_instance, folded_witness)
    }

    /// Verifies a NIFS folding and computes the folded instance.
    ///
    /// The verifier doesn't have access to the witnesses, but can verify
    /// that the folding was done correctly given the commitments.
    ///
    /// # Arguments
    /// * `proof` - The folding proof
    /// * `instance_real` - The real instance
    /// * `instance_rand` - The random instance
    /// * `transcript` - The Fiat-Shamir transcript
    ///
    /// # Returns
    /// The folded instance
    pub fn verify<F, PCS, ProofTranscript>(
        proof: &NIFSProof<F, PCS>,
        instance_real: &RelaxedR1CSInstance<F, PCS>,
        instance_rand: &RelaxedR1CSInstance<F, PCS>,
        transcript: &mut ProofTranscript,
    ) -> RelaxedR1CSInstance<F, PCS>
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
        PCS::Commitment: Clone,
        ProofTranscript: Transcript,
    {
        // Get the same folding challenge as the prover
        transcript.append_serializable(&proof.T_commitment);
        let r: F = transcript.challenge_scalar();

        // Fold instances with proper commitment folding
        Self::fold_instances_with_t(instance_real, instance_rand, &proof.T_commitment, r)
    }

    /// Computes the cross-term T for NIFS folding.
    ///
    /// T = A(z_1) ∘ B(z_2) + A(z_2) ∘ B(z_1) - u_1·C(z_2) - u_2·C(z_1)
    pub fn compute_cross_term<F: JoltField>(
        matrices: &R1CSMatrices<F>,
        witness_1: &RelaxedR1CSWitness<F>,
        u_1: F,
        x_1: &[F],
        witness_2: &RelaxedR1CSWitness<F>,
        u_2: F,
        x_2: &[F],
    ) -> Vec<F> {
        // Construct z_1 = (1, x_1, W_1)
        let mut z_1 = vec![F::one()];
        z_1.extend_from_slice(x_1);
        z_1.extend_from_slice(&witness_1.W);

        // Construct z_2 = (1, x_2, W_2)
        let mut z_2 = vec![F::one()];
        z_2.extend_from_slice(x_2);
        z_2.extend_from_slice(&witness_2.W);

        // Compute matrix-vector products
        let az_1 = matrices.A.multiply(&z_1);
        let bz_1 = matrices.B.multiply(&z_1);
        let cz_1 = matrices.C.multiply(&z_1);

        let az_2 = matrices.A.multiply(&z_2);
        let bz_2 = matrices.B.multiply(&z_2);
        let cz_2 = matrices.C.multiply(&z_2);

        // T = A(z_1) ∘ B(z_2) + A(z_2) ∘ B(z_1) - u_1·C(z_2) - u_2·C(z_1)
        let num_constraints = az_1.len();
        let mut cross_term = vec![F::zero(); num_constraints];

        for i in 0..num_constraints {
            cross_term[i] = az_1[i] * bz_2[i] + az_2[i] * bz_1[i] - u_1 * cz_2[i] - u_2 * cz_1[i];
        }

        cross_term
    }

    /// Folds two instances using folding challenge r and T commitment.
    ///
    /// Computes commitments homomorphically:
    /// - W' = W_1 + r·W_2  =>  Commit(W') = Commit(W_1) + r·Commit(W_2)
    /// - E' = E_1 + r·T + r²·E_2  =>  Commit(E') = Commit(E_1) + r·Commit(T) + r²·Commit(E_2)
    fn fold_instances_with_t<F, PCS>(
        instance_1: &RelaxedR1CSInstance<F, PCS>,
        instance_2: &RelaxedR1CSInstance<F, PCS>,
        T_commitment: &PCS::Commitment,
        r: F,
    ) -> RelaxedR1CSInstance<F, PCS>
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
        PCS::Commitment: Clone,
    {
        // u' = u_1 + r·u_2
        let folded_u = instance_1.u + r * instance_2.u;

        // x' = x_1 + r·x_2
        let folded_x: Vec<F> = instance_1
            .x
            .iter()
            .zip(instance_2.x.iter())
            .map(|(x1, x2)| *x1 + r * x2)
            .collect();

        // Fold commitments homomorphically:
        // W' = W_1 + r·W_2
        let folded_W_commitment =
            fold_commitments::<F, PCS>(&instance_1.W_commitment, &instance_2.W_commitment, &r);

        // E' = E_1 + r·T + r²·E_2
        let folded_E_commitment = fold_error_commitments::<F, PCS>(
            &instance_1.E_commitment,
            T_commitment,
            &instance_2.E_commitment,
            &r,
        );

        RelaxedR1CSInstance::new(folded_W_commitment, folded_E_commitment, folded_u, folded_x)
    }

    /// Folds two witnesses using folding challenge r and cross-term T.
    ///
    /// This now delegates to RelaxedR1CSWitness::fold which handles all
    /// blinding factors internally.
    pub fn fold_witnesses<F: JoltField>(
        witness_1: &RelaxedR1CSWitness<F>,
        witness_2: &RelaxedR1CSWitness<F>,
        cross_term: &[F],
        r_T: F,
        r: F,
    ) -> RelaxedR1CSWitness<F> {
        RelaxedR1CSWitness::fold(witness_1, witness_2, cross_term, r_T, r)
    }

    /// Legacy fold_witnesses without blinding factor for T (uses zero).
    /// Kept for backwards compatibility with tests.
    pub fn fold_witnesses_simple<F: JoltField>(
        witness_1: &RelaxedR1CSWitness<F>,
        witness_2: &RelaxedR1CSWitness<F>,
        cross_term: &[F],
        r: F,
    ) -> RelaxedR1CSWitness<F> {
        RelaxedR1CSWitness::fold(witness_1, witness_2, cross_term, F::zero(), r)
    }
}

/// Hiding NIFS implementation using HidingCommitmentScheme.
///
/// This extends NIFS with proper commitment folding using blinding factors.
pub struct HidingNIFS;

impl HidingNIFS {
    /// Folds two relaxed R1CS instances with hiding commitments.
    ///
    /// This version uses the `HidingCommitmentScheme` trait for proper
    /// homomorphic commitment folding with blinding factors.
    ///
    /// The blinding factors are now integrated into `RelaxedR1CSWitness`, so
    /// this function no longer takes separate blinding factor parameters.
    ///
    /// # Arguments
    /// * `matrices` - The R1CS constraint matrices
    /// * `instance_real` - The real instance to fold
    /// * `witness_real` - The witness for the real instance (includes blinding factors)
    /// * `instance_rand` - The random satisfying instance
    /// * `witness_rand` - The witness for the random instance (includes blinding factors)
    /// * `transcript` - The Fiat-Shamir transcript
    /// * `pcs_setup` - The PCS prover setup
    /// * `rng` - Cryptographic random number generator for blinding
    ///
    /// # Returns
    /// A tuple containing:
    /// - The folding proof (with cross-term commitment and blinding)
    /// - The folded instance
    /// - The folded witness (with folded blinding factors)
    #[allow(clippy::type_complexity)]
    #[allow(clippy::too_many_arguments)]
    pub fn prove_hiding<F, PCS, ProofTranscript, R>(
        matrices: &R1CSMatrices<F>,
        instance_real: &RelaxedR1CSInstance<F, PCS>,
        witness_real: &RelaxedR1CSWitness<F>,
        instance_rand: &RelaxedR1CSInstance<F, PCS>,
        witness_rand: &RelaxedR1CSWitness<F>,
        transcript: &mut ProofTranscript,
        pcs_setup: &PCS::ProverSetup,
        rng: &mut R,
    ) -> (
        HidingNIFSProof<F, PCS>,
        RelaxedR1CSInstance<F, PCS>,
        RelaxedR1CSWitness<F>,
    )
    where
        F: JoltField,
        PCS: HidingCommitmentScheme<Field = F>,
        PCS::Commitment: Clone,
        ProofTranscript: Transcript,
        R: RngCore + CryptoRng,
    {
        // Step 1: Compute the cross-term T
        let cross_term = NIFS::compute_cross_term(
            matrices,
            witness_real,
            instance_real.u,
            &instance_real.x,
            witness_rand,
            instance_rand.u,
            &instance_rand.x,
        );

        // Step 2: Sample random blinding for T and commit
        let T_blinding_pcs = PCS::sample_blinding(rng);

        // Convert cross-term to multilinear polynomial for hiding commitment
        let cross_term_padded = NIFS::pad_to_power_of_2(cross_term.clone());
        let cross_term_poly: MultilinearPolynomial<F> = cross_term_padded.into();
        let (T_commitment, _hint) =
            PCS::commit_hiding(&cross_term_poly, &T_blinding_pcs, pcs_setup);

        // Sample scalar blinding factor for T (for witness folding)
        let r_T = F::random(rng);
        let T_blinding = ScalarBlindingFactor::new(r_T);

        // Step 3: Get folding challenge from transcript
        transcript.append_serializable(&T_commitment);
        let r: F = transcript.challenge_scalar();

        // Step 4: Fold instances with homomorphic commitment operations
        let folded_instance =
            Self::fold_instances_hiding::<F, PCS>(instance_real, instance_rand, &T_commitment, r);

        // Step 5: Fold witnesses (including blinding factors)
        // The new RelaxedR1CSWitness::fold handles all blinding factor folding internally
        let folded_witness = NIFS::fold_witnesses(witness_real, witness_rand, &cross_term, r_T, r);

        let proof = HidingNIFSProof {
            T_commitment,
            T_blinding,
        };

        (proof, folded_instance, folded_witness)
    }

    /// Verifies a hiding NIFS folding and computes the folded instance.
    ///
    /// The verifier doesn't have access to the witnesses or blinding factors,
    /// but can verify that the commitment folding was done correctly.
    pub fn verify_hiding<F, PCS, ProofTranscript>(
        proof: &HidingNIFSProof<F, PCS>,
        instance_real: &RelaxedR1CSInstance<F, PCS>,
        instance_rand: &RelaxedR1CSInstance<F, PCS>,
        transcript: &mut ProofTranscript,
    ) -> RelaxedR1CSInstance<F, PCS>
    where
        F: JoltField,
        PCS: HidingCommitmentScheme<Field = F>,
        PCS::Commitment: Clone,
        ProofTranscript: Transcript,
    {
        // Get the same folding challenge as the prover
        transcript.append_serializable(&proof.T_commitment);
        let r: F = transcript.challenge_scalar();

        // Fold instances with homomorphic commitment operations
        Self::fold_instances_hiding::<F, PCS>(instance_real, instance_rand, &proof.T_commitment, r)
    }

    /// Folds two instances using homomorphic commitment operations.
    fn fold_instances_hiding<F, PCS>(
        instance_1: &RelaxedR1CSInstance<F, PCS>,
        instance_2: &RelaxedR1CSInstance<F, PCS>,
        T_commitment: &PCS::Commitment,
        r: F,
    ) -> RelaxedR1CSInstance<F, PCS>
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
        PCS::Commitment: Clone,
    {
        // u' = u_1 + r·u_2
        let folded_u = instance_1.u + r * instance_2.u;

        // x' = x_1 + r·x_2
        let folded_x: Vec<F> = instance_1
            .x
            .iter()
            .zip(instance_2.x.iter())
            .map(|(x1, x2)| *x1 + r * x2)
            .collect();

        // Commitment folding using homomorphic operations:
        // W' = W_1 + r·W_2  =>  C(W') = C(W_1) + r·C(W_2)
        let folded_W_commitment =
            fold_commitments::<F, PCS>(&instance_1.W_commitment, &instance_2.W_commitment, &r);

        // E' = E_1 + r·T + r²·E_2  =>  C(E') = C(E_1) + r·C(T) + r²·C(E_2)
        let folded_E_commitment = fold_error_commitments::<F, PCS>(
            &instance_1.E_commitment,
            T_commitment,
            &instance_2.E_commitment,
            &r,
        );

        RelaxedR1CSInstance::new(folded_W_commitment, folded_E_commitment, folded_u, folded_x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn test_fold_witnesses() {
        let witness_1 = RelaxedR1CSWitness::new_simple(
            vec![Fr::from(1u64), Fr::from(2u64)],  // W
            Fr::from(10u64),                        // r_W
            vec![Fr::from(0u64), Fr::from(0u64)],  // E
            Fr::from(0u64),                         // r_E
        );
        let witness_2 = RelaxedR1CSWitness::new_simple(
            vec![Fr::from(3u64), Fr::from(4u64)],  // W
            Fr::from(30u64),                        // r_W
            vec![Fr::from(0u64), Fr::from(0u64)],  // E
            Fr::from(0u64),                         // r_E
        );
        let cross_term = vec![Fr::from(0u64), Fr::from(0u64)];
        let r_T = Fr::from(0u64);
        let r = Fr::from(2u64);

        let folded = NIFS::fold_witnesses(&witness_1, &witness_2, &cross_term, r_T, r);

        // W' = W_1 + r·W_2 = [1, 2] + 2·[3, 4] = [7, 10]
        assert_eq!(folded.W[0], Fr::from(7u64));
        assert_eq!(folded.W[1], Fr::from(10u64));

        // r_W' = r_W_1 + r·r_W_2 = 10 + 2*30 = 70
        assert_eq!(folded.r_W, Fr::from(70u64));
    }

    #[test]
    fn test_instance_blinding_factors_creation() {
        let W_blinding = ScalarBlindingFactor::<Fr>::new(Fr::from(10u64));
        let E_blinding = ScalarBlindingFactor::<Fr>::new(Fr::from(20u64));

        let blindings = InstanceBlindingFactors::new(W_blinding.clone(), E_blinding.clone());

        assert_eq!(blindings.W_blinding.scalar, Fr::from(10u64));
        assert_eq!(blindings.E_blinding.scalar, Fr::from(20u64));
    }

    #[test]
    fn test_instance_blinding_factors_zero() {
        let blindings = InstanceBlindingFactors::<Fr>::zero();

        assert_eq!(blindings.W_blinding.scalar, Fr::from(0u64));
        assert_eq!(blindings.E_blinding.scalar, Fr::from(0u64));
    }

    #[test]
    fn test_instance_blinding_factors_fold() {
        // blindings_1: W=10, E=20
        let blindings_1 = InstanceBlindingFactors::new(
            ScalarBlindingFactor::new(Fr::from(10u64)),
            ScalarBlindingFactor::new(Fr::from(20u64)),
        );

        // blindings_2: W=30, E=40
        let blindings_2 = InstanceBlindingFactors::new(
            ScalarBlindingFactor::new(Fr::from(30u64)),
            ScalarBlindingFactor::new(Fr::from(40u64)),
        );

        // T_blinding: 50
        let T_blinding = ScalarBlindingFactor::new(Fr::from(50u64));

        // r = 2
        let r = Fr::from(2u64);

        let folded = InstanceBlindingFactors::fold(&blindings_1, &blindings_2, &T_blinding, r);

        // r_W' = r_W_1 + r * r_W_2 = 10 + 2 * 30 = 70
        assert_eq!(folded.W_blinding.scalar, Fr::from(70u64));

        // r_E' = r_E_1 + r * r_T + r² * r_E_2
        //      = 20 + 2 * 50 + 4 * 40
        //      = 20 + 100 + 160
        //      = 280
        assert_eq!(folded.E_blinding.scalar, Fr::from(280u64));
    }

    #[test]
    fn test_witness_folding_with_nonzero_cross_term() {
        let witness_1 = RelaxedR1CSWitness::new_simple(
            vec![Fr::from(1u64)],   // W
            Fr::from(10u64),         // r_W
            vec![Fr::from(5u64)],   // E_1 = 5
            Fr::from(20u64),         // r_E
        );
        let witness_2 = RelaxedR1CSWitness::new_simple(
            vec![Fr::from(2u64)],   // W
            Fr::from(30u64),         // r_W
            vec![Fr::from(3u64)],   // E_2 = 3
            Fr::from(40u64),         // r_E
        );
        let cross_term = vec![Fr::from(7u64)]; // T = 7
        let r_T = Fr::from(50u64);
        let r = Fr::from(2u64);

        let folded = NIFS::fold_witnesses(&witness_1, &witness_2, &cross_term, r_T, r);

        // W' = W_1 + r·W_2 = 1 + 2*2 = 5
        assert_eq!(folded.W[0], Fr::from(5u64));

        // E' = E_1 + r·T + r²·E_2
        //    = 5 + 2*7 + 4*3
        //    = 5 + 14 + 12
        //    = 31
        assert_eq!(folded.E[0], Fr::from(31u64));

        // r_E' = r_E_1 + r·r_T + r²·r_E_2 = 20 + 2*50 + 4*40 = 280
        assert_eq!(folded.r_E, Fr::from(280u64));
    }

    #[test]
    fn test_blinding_factor_homomorphism() {
        // Test that blinding factors fold correctly for commitment homomorphism
        // If C = Commit(m) + r * H, then folding should preserve this structure

        let r_W_1 = Fr::from(11u64);
        let r_E_1 = Fr::from(13u64);
        let r_W_2 = Fr::from(17u64);
        let r_E_2 = Fr::from(19u64);
        let r_T = Fr::from(23u64);
        let r = Fr::from(3u64);

        let blindings_1 = InstanceBlindingFactors::new(
            ScalarBlindingFactor::new(r_W_1),
            ScalarBlindingFactor::new(r_E_1),
        );

        let blindings_2 = InstanceBlindingFactors::new(
            ScalarBlindingFactor::new(r_W_2),
            ScalarBlindingFactor::new(r_E_2),
        );

        let T_blinding = ScalarBlindingFactor::new(r_T);

        let folded = InstanceBlindingFactors::fold(&blindings_1, &blindings_2, &T_blinding, r);

        // Verify W blinding: r_W' = r_W_1 + r * r_W_2
        let expected_W = r_W_1 + r * r_W_2;
        assert_eq!(folded.W_blinding.scalar, expected_W);

        // Verify E blinding: r_E' = r_E_1 + r * r_T + r² * r_E_2
        let r_squared = r * r;
        let expected_E = r_E_1 + r * r_T + r_squared * r_E_2;
        assert_eq!(folded.E_blinding.scalar, expected_E);
    }

    #[test]
    fn test_multiple_sequential_folds() {
        // Test that multiple sequential folds work correctly

        // Initial witnesses (using new_simple for convenience)
        let witness_1 = RelaxedR1CSWitness::new_simple(
            vec![Fr::from(1u64)], Fr::from(1u64),
            vec![Fr::from(0u64)], Fr::from(0u64),
        );
        let witness_2 = RelaxedR1CSWitness::new_simple(
            vec![Fr::from(2u64)], Fr::from(2u64),
            vec![Fr::from(0u64)], Fr::from(0u64),
        );
        let witness_3 = RelaxedR1CSWitness::new_simple(
            vec![Fr::from(3u64)], Fr::from(3u64),
            vec![Fr::from(0u64)], Fr::from(0u64),
        );

        // Fold 1 and 2
        let cross_term_12 = vec![Fr::from(0u64)];
        let r1 = Fr::from(2u64);
        let folded_12 = NIFS::fold_witnesses_simple(&witness_1, &witness_2, &cross_term_12, r1);

        // W' = 1 + 2*2 = 5
        assert_eq!(folded_12.W[0], Fr::from(5u64));

        // Fold (1+2) and 3
        let cross_term_123 = vec![Fr::from(0u64)];
        let r2 = Fr::from(3u64);
        let folded_123 = NIFS::fold_witnesses_simple(&folded_12, &witness_3, &cross_term_123, r2);

        // W'' = 5 + 3*3 = 14
        assert_eq!(folded_123.W[0], Fr::from(14u64));
    }
}
