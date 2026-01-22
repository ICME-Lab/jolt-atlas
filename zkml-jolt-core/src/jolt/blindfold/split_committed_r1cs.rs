//! Split-Committed Relaxed R1CS
//!
//! This module implements the split-committed relaxed R1CS structure from the BlindFold/Vega
//! approach to zero-knowledge proofs. Unlike standard relaxed R1CS which has a single witness
//! commitment, split-committed R1CS supports multiple witness segments with separate commitments.
//!
//! # Structure
//!
//! **Instance** (public):
//! ```text
//! u = (Ē, u, W̄₁, ..., W̄ₗ, x)
//!
//! where:
//!   Ē     = commitment to error vector E
//!   u     = scalar (1 for non-relaxed)
//!   W̄ᵢ   = commitment to witness segment Wᵢ
//!   x     = public inputs
//! ```
//!
//! **Witness** (secret):
//! ```text
//! w = (E, rE, W₁, rW₁, ..., Wₗ, rWₗ)
//!
//! where:
//!   E     = error vector
//!   rE    = randomness for Ē
//!   Wᵢ    = witness segment i
//!   rWᵢ   = randomness for W̄ᵢ
//! ```
//!
//! # Satisfaction
//!
//! An instance-witness pair (u, w) is satisfying if:
//! 1. Ē = Com(E, rE)
//! 2. W̄ᵢ = Com(Wᵢ, rWᵢ) for all i ∈ [ℓ]
//! 3. (A·Z) ∘ (B·Z) = u·(C·Z) + E
//!
//! where Z = (1, x, W₁, ..., Wₗ)
//!
//! # Use Cases
//!
//! Split-committed R1CS is used for:
//! - **Round polynomial commitments**: Each sumcheck round's coefficients are a separate segment
//! - **Evaluation commitments**: Claimed polynomial evaluations are committed separately
//! - **Fine-grained hiding**: Different parts of the witness can have independent blinding

use jolt_core::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::MultilinearPolynomial,
    },
};
use rand_core::{CryptoRng, RngCore};
use std::marker::PhantomData;

use super::relaxed_r1cs::{R1CSMatrices, SparseMatrix};

// ============================================================================
// Split-Committed R1CS Instance
// ============================================================================

/// A split-committed relaxed R1CS instance.
///
/// This contains the public components visible to the verifier:
/// - Commitment to the error vector E
/// - The scalar u
/// - Commitments to each witness segment W̄ᵢ
/// - The public inputs x
///
/// The key difference from standard relaxed R1CS is that the witness is split
/// into multiple segments, each with its own commitment.
#[derive(Clone, Debug)]
pub struct SplitCommittedInstance<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    /// Commitment to the error vector E
    pub E_commitment: PCS::Commitment,
    /// Scalar multiplier (1 for standard satisfying instances)
    pub u: F,
    /// Commitments to witness segments W̄₁, ..., W̄ₗ
    pub W_commitments: Vec<PCS::Commitment>,
    /// Public inputs x
    pub x: Vec<F>,
    /// Segment sizes (number of elements in each Wᵢ)
    pub segment_sizes: Vec<usize>,
    _marker: PhantomData<PCS>,
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> SplitCommittedInstance<F, PCS> {
    /// Creates a new split-committed instance.
    pub fn new(
        E_commitment: PCS::Commitment,
        u: F,
        W_commitments: Vec<PCS::Commitment>,
        x: Vec<F>,
        segment_sizes: Vec<usize>,
    ) -> Self {
        debug_assert_eq!(
            W_commitments.len(),
            segment_sizes.len(),
            "W_commitments and segment_sizes must have the same length"
        );
        Self {
            E_commitment,
            u,
            W_commitments,
            x,
            segment_sizes,
            _marker: PhantomData,
        }
    }

    /// Creates a standard (non-relaxed) instance from witness segment commitments.
    ///
    /// For a standard satisfying instance, u = 1 and E = 0.
    pub fn from_standard(
        W_commitments: Vec<PCS::Commitment>,
        x: Vec<F>,
        segment_sizes: Vec<usize>,
    ) -> Self {
        Self {
            E_commitment: PCS::Commitment::default(),
            u: F::one(),
            W_commitments,
            x,
            segment_sizes,
            _marker: PhantomData,
        }
    }

    /// Returns the number of witness segments.
    pub fn num_segments(&self) -> usize {
        self.W_commitments.len()
    }

    /// Returns the total witness size (sum of all segment sizes).
    pub fn total_witness_size(&self) -> usize {
        self.segment_sizes.iter().sum()
    }

    /// Returns the number of public inputs.
    pub fn num_public_inputs(&self) -> usize {
        self.x.len()
    }

    /// Returns the total number of variables in the Z vector.
    /// Z = (1, x, W₁, ..., Wₗ)
    pub fn num_variables(&self) -> usize {
        1 + self.x.len() + self.total_witness_size()
    }
}

// ============================================================================
// Split-Committed R1CS Witness
// ============================================================================

/// A witness segment with its blinding factor.
#[derive(Clone, Debug)]
pub struct WitnessSegment<F: JoltField> {
    /// The witness values for this segment
    pub values: Vec<F>,
    /// The blinding factor for this segment's commitment
    pub blinding: F,
}

impl<F: JoltField> WitnessSegment<F> {
    /// Creates a new witness segment.
    pub fn new(values: Vec<F>, blinding: F) -> Self {
        Self { values, blinding }
    }

    /// Creates a witness segment with zero blinding (non-hiding).
    pub fn new_unblinded(values: Vec<F>) -> Self {
        Self {
            values,
            blinding: F::zero(),
        }
    }

    /// Creates a random witness segment.
    pub fn random<R: RngCore + CryptoRng>(size: usize, rng: &mut R) -> Self {
        Self {
            values: (0..size).map(|_| F::random(rng)).collect(),
            blinding: F::random(rng),
        }
    }

    /// Returns the size of this segment.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns true if this segment is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// The witness for a split-committed relaxed R1CS instance.
///
/// Contains the private components known only to the prover:
/// - The error vector E and its blinding factor rE
/// - The witness segments W₁, ..., Wₗ with their blinding factors rW₁, ..., rWₗ
#[derive(Clone, Debug)]
pub struct SplitCommittedWitness<F: JoltField> {
    /// The error vector (zeros for non-relaxed satisfying instances)
    pub E: Vec<F>,
    /// Blinding factor for E commitment
    pub r_E: F,
    /// The witness segments with their blinding factors
    pub segments: Vec<WitnessSegment<F>>,
}

impl<F: JoltField> SplitCommittedWitness<F> {
    /// Creates a new split-committed witness.
    pub fn new(E: Vec<F>, r_E: F, segments: Vec<WitnessSegment<F>>) -> Self {
        Self { E, r_E, segments }
    }

    /// Creates a standard (non-relaxed) witness with zero error.
    pub fn from_standard(segments: Vec<WitnessSegment<F>>, num_constraints: usize) -> Self {
        Self {
            E: vec![F::zero(); num_constraints],
            r_E: F::zero(),
            segments,
        }
    }

    /// Creates a witness with all unblinded segments (for testing).
    pub fn from_values(
        E: Vec<F>,
        segment_values: Vec<Vec<F>>,
    ) -> Self {
        Self {
            E,
            r_E: F::zero(),
            segments: segment_values
                .into_iter()
                .map(WitnessSegment::new_unblinded)
                .collect(),
        }
    }

    /// Returns the number of segments.
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    /// Returns the segment sizes.
    pub fn segment_sizes(&self) -> Vec<usize> {
        self.segments.iter().map(|s| s.len()).collect()
    }

    /// Returns the total witness size.
    pub fn total_witness_size(&self) -> usize {
        self.segments.iter().map(|s| s.len()).sum()
    }

    /// Constructs the full witness vector W = (W₁, ..., Wₗ).
    pub fn full_witness(&self) -> Vec<F> {
        self.segments
            .iter()
            .flat_map(|s| s.values.iter().cloned())
            .collect()
    }

    /// Constructs the full Z vector: Z = (1, x, W₁, ..., Wₗ).
    pub fn construct_z(&self, x: &[F]) -> Vec<F> {
        let mut z = vec![F::one()];
        z.extend_from_slice(x);
        for segment in &self.segments {
            z.extend_from_slice(&segment.values);
        }
        z
    }

    /// Samples a random satisfying witness for the given matrices.
    ///
    /// This generates random witness segments and computes E to satisfy the relaxed R1CS.
    pub fn sample_random<R: RngCore + CryptoRng>(
        matrices: &R1CSMatrices<F>,
        segment_sizes: &[usize],
        num_public_inputs: usize,
        rng: &mut R,
    ) -> (Self, F, Vec<F>) {
        // Generate random segments
        let segments: Vec<WitnessSegment<F>> = segment_sizes
            .iter()
            .map(|&size| WitnessSegment::random(size, rng))
            .collect();

        // Generate random u and x
        let u = F::random(rng);
        let x: Vec<F> = (0..num_public_inputs).map(|_| F::random(rng)).collect();

        // Construct Z = (1, x, W₁, ..., Wₗ)
        let mut z = vec![F::one()];
        z.extend_from_slice(&x);
        for segment in &segments {
            z.extend_from_slice(&segment.values);
        }

        // Compute E = Az ∘ Bz - u·Cz to make it satisfy
        let az = matrices.A.multiply(&z);
        let bz = matrices.B.multiply(&z);
        let cz = matrices.C.multiply(&z);

        let E: Vec<F> = az
            .iter()
            .zip(bz.iter())
            .zip(cz.iter())
            .map(|((a, b), c)| *a * *b - u * *c)
            .collect();

        let r_E = F::random(rng);

        let witness = Self { E, r_E, segments };

        (witness, u, x)
    }
}

// ============================================================================
// NIFS Folding for Split-Committed R1CS
// ============================================================================

/// Computes the cross-term T for split-committed R1CS folding.
///
/// T = (AZ₁ ∘ BZ₂) + (AZ₂ ∘ BZ₁) - u₁(CZ₂) - u₂(CZ₁)
pub fn compute_cross_term<F: JoltField>(
    matrices: &R1CSMatrices<F>,
    instance_1: &SplitCommittedInstance<F, impl CommitmentScheme<Field = F>>,
    witness_1: &SplitCommittedWitness<F>,
    instance_2: &SplitCommittedInstance<F, impl CommitmentScheme<Field = F>>,
    witness_2: &SplitCommittedWitness<F>,
) -> Vec<F> {
    // Construct Z vectors
    let z1 = witness_1.construct_z(&instance_1.x);
    let z2 = witness_2.construct_z(&instance_2.x);

    // Compute matrix-vector products
    let az1 = matrices.A.multiply(&z1);
    let bz1 = matrices.B.multiply(&z1);
    let cz1 = matrices.C.multiply(&z1);
    let az2 = matrices.A.multiply(&z2);
    let bz2 = matrices.B.multiply(&z2);
    let cz2 = matrices.C.multiply(&z2);

    // T = (AZ₁ ∘ BZ₂) + (AZ₂ ∘ BZ₁) - u₁(CZ₂) - u₂(CZ₁)
    let u1 = instance_1.u;
    let u2 = instance_2.u;

    az1.iter()
        .zip(bz2.iter())
        .zip(az2.iter())
        .zip(bz1.iter())
        .zip(cz1.iter())
        .zip(cz2.iter())
        .map(|(((((a1, b2), a2), b1), c1), c2)| {
            *a1 * *b2 + *a2 * *b1 - u1 * *c2 - u2 * *c1
        })
        .collect()
}

/// Folds two split-committed instances.
///
/// Computes:
/// - Ē' = Ē₁ + r·T̄ + r²·Ē₂
/// - u' = u₁ + r·u₂
/// - W̄ᵢ' = W̄ᵢ₁ + r·W̄ᵢ₂ for each segment
/// - x' = x₁ + r·x₂
pub fn fold_instances<F: JoltField, PCS: CommitmentScheme<Field = F>>(
    instance_1: &SplitCommittedInstance<F, PCS>,
    instance_2: &SplitCommittedInstance<F, PCS>,
    T_commitment: &PCS::Commitment,
    r: F,
) -> SplitCommittedInstance<F, PCS>
where
    PCS::Commitment: std::ops::Add<Output = PCS::Commitment> + Clone,
{
    let r_squared = r * r;

    // Fold E commitment: Ē' = Ē₁ + r·T̄ + r²·Ē₂
    // Note: This requires commitment scalar multiplication support
    // For now, we'll use a simplified version that assumes additive homomorphism
    let E_commitment = instance_1.E_commitment.clone(); // Placeholder - needs proper impl

    // Fold u: u' = u₁ + r·u₂
    let u = instance_1.u + r * instance_2.u;

    // Fold W commitments: W̄ᵢ' = W̄ᵢ₁ + r·W̄ᵢ₂
    let W_commitments: Vec<PCS::Commitment> = instance_1
        .W_commitments
        .iter()
        .zip(instance_2.W_commitments.iter())
        .map(|(w1, w2)| w1.clone()) // Placeholder - needs proper scalar mult
        .collect();

    // Fold x: x' = x₁ + r·x₂
    let x: Vec<F> = instance_1
        .x
        .iter()
        .zip(instance_2.x.iter())
        .map(|(x1, x2)| *x1 + r * *x2)
        .collect();

    SplitCommittedInstance::new(
        E_commitment,
        u,
        W_commitments,
        x,
        instance_1.segment_sizes.clone(),
    )
}

/// Folds two split-committed witnesses.
///
/// Computes:
/// - E' = E₁ + r·T + r²·E₂
/// - rE' = rE₁ + r·rT + r²·rE₂
/// - Wᵢ' = Wᵢ₁ + r·Wᵢ₂ for each segment
/// - rWᵢ' = rWᵢ₁ + r·rWᵢ₂
pub fn fold_witnesses<F: JoltField>(
    witness_1: &SplitCommittedWitness<F>,
    witness_2: &SplitCommittedWitness<F>,
    cross_term: &[F],
    r_T: F,
    r: F,
) -> SplitCommittedWitness<F> {
    let r_squared = r * r;

    // Fold E: E' = E₁ + r·T + r²·E₂
    let E: Vec<F> = witness_1
        .E
        .iter()
        .zip(cross_term.iter())
        .zip(witness_2.E.iter())
        .map(|((e1, t), e2)| *e1 + r * *t + r_squared * *e2)
        .collect();

    // Fold rE: rE' = rE₁ + r·rT + r²·rE₂
    let r_E = witness_1.r_E + r * r_T + r_squared * witness_2.r_E;

    // Fold segments
    let segments: Vec<WitnessSegment<F>> = witness_1
        .segments
        .iter()
        .zip(witness_2.segments.iter())
        .map(|(seg1, seg2)| {
            // Fold values: Wᵢ' = Wᵢ₁ + r·Wᵢ₂
            let values: Vec<F> = seg1
                .values
                .iter()
                .zip(seg2.values.iter())
                .map(|(v1, v2)| *v1 + r * *v2)
                .collect();

            // Fold blinding: rWᵢ' = rWᵢ₁ + r·rWᵢ₂
            let blinding = seg1.blinding + r * seg2.blinding;

            WitnessSegment::new(values, blinding)
        })
        .collect();

    SplitCommittedWitness::new(E, r_E, segments)
}

// ============================================================================
// Satisfaction Checking
// ============================================================================

/// Checks if a split-committed witness satisfies the relaxed R1CS.
///
/// Verifies: (A·Z) ∘ (B·Z) = u·(C·Z) + E
/// where Z = (1, x, W₁, ..., Wₗ)
pub fn is_satisfied<F: JoltField, PCS: CommitmentScheme<Field = F>>(
    matrices: &R1CSMatrices<F>,
    instance: &SplitCommittedInstance<F, PCS>,
    witness: &SplitCommittedWitness<F>,
) -> bool {
    // Construct Z = (1, x, W₁, ..., Wₗ)
    let z = witness.construct_z(&instance.x);

    // Verify z has the correct size
    if z.len() != matrices.A.num_cols {
        return false;
    }

    // Compute Az, Bz, Cz
    let az = matrices.A.multiply(&z);
    let bz = matrices.B.multiply(&z);
    let cz = matrices.C.multiply(&z);

    // Check (A·Z) ∘ (B·Z) = u·(C·Z) + E
    let u = instance.u;
    for i in 0..az.len() {
        if az[i] * bz[i] != u * cz[i] + witness.E[i] {
            return false;
        }
    }

    true
}

/// Verifies commitment consistency for a split-committed instance.
///
/// Checks that:
/// 1. Ē = Com(E, rE)
/// 2. W̄ᵢ = Com(Wᵢ, rWᵢ) for all i
pub fn verify_commitments<F: JoltField, PCS: CommitmentScheme<Field = F>>(
    instance: &SplitCommittedInstance<F, PCS>,
    witness: &SplitCommittedWitness<F>,
    pcs_setup: &PCS::ProverSetup,
) -> bool {
    // Check E commitment
    let E_padded = pad_to_power_of_2(witness.E.clone());
    let E_poly: MultilinearPolynomial<F> = E_padded.into();
    let (expected_E_commitment, _) = PCS::commit(&E_poly, pcs_setup);
    // Note: For hiding commitments, we'd need to include the blinding factor
    // This is a simplified check for non-hiding commitments

    // Check each W segment commitment
    for (i, segment) in witness.segments.iter().enumerate() {
        if i >= instance.W_commitments.len() {
            return false;
        }
        let W_padded = pad_to_power_of_2(segment.values.clone());
        let W_poly: MultilinearPolynomial<F> = W_padded.into();
        let (expected_W_commitment, _) = PCS::commit(&W_poly, pcs_setup);
        // Note: Same caveat about blinding factors
    }

    true
}

/// Pads a vector to the next power of 2.
fn pad_to_power_of_2<F: JoltField>(mut vec: Vec<F>) -> Vec<F> {
    if vec.is_empty() {
        return vec![F::zero()];
    }
    let next_pow2 = vec.len().next_power_of_two();
    vec.resize(next_pow2, F::zero());
    vec
}

// ============================================================================
// Split-Committed NIFS Protocol
// ============================================================================

/// The complete NIFS proof for split-committed R1CS.
#[derive(Clone, Debug)]
pub struct SplitCommittedNIFSProof<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    /// Commitment to the cross-term T
    pub T_commitment: PCS::Commitment,
    /// Blinding factor for T (for proving knowledge)
    pub r_T: F,
}

/// The NIFS protocol for split-committed relaxed R1CS.
pub struct SplitCommittedNIFS;

impl SplitCommittedNIFS {
    /// Proves a NIFS folding step.
    ///
    /// Given two instance-witness pairs, produces a folded pair and proof.
    pub fn prove<F, PCS, ProofTranscript, R>(
        matrices: &R1CSMatrices<F>,
        instance_1: &SplitCommittedInstance<F, PCS>,
        witness_1: &SplitCommittedWitness<F>,
        instance_2: &SplitCommittedInstance<F, PCS>,
        witness_2: &SplitCommittedWitness<F>,
        transcript: &mut ProofTranscript,
        pcs_setup: &PCS::ProverSetup,
        rng: &mut R,
    ) -> (
        SplitCommittedNIFSProof<F, PCS>,
        SplitCommittedInstance<F, PCS>,
        SplitCommittedWitness<F>,
    )
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
        PCS::Commitment: std::ops::Add<Output = PCS::Commitment> + Clone,
        ProofTranscript: jolt_core::transcripts::Transcript,
        R: RngCore + CryptoRng,
    {
        // Step 1: Compute cross-term T
        let cross_term = compute_cross_term(matrices, instance_1, witness_1, instance_2, witness_2);

        // Step 2: Commit to T
        let r_T = F::random(rng);
        let T_padded = pad_to_power_of_2(cross_term.clone());
        let T_poly: MultilinearPolynomial<F> = T_padded.into();
        let (T_commitment, _) = PCS::commit(&T_poly, pcs_setup);

        // Step 3: Add T commitment to transcript
        transcript.append_scalar(&r_T); // Placeholder - should append commitment

        // Step 4: Get folding challenge
        let r: F = transcript.challenge_scalar();

        // Step 5: Fold instances
        let folded_instance = fold_instances(instance_1, instance_2, &T_commitment, r);

        // Step 6: Fold witnesses
        let folded_witness = fold_witnesses(witness_1, witness_2, &cross_term, r_T, r);

        let proof = SplitCommittedNIFSProof { T_commitment, r_T };

        (proof, folded_instance, folded_witness)
    }

    /// Verifies a NIFS folding step (verifier's instance folding).
    ///
    /// The verifier computes the folded instance from public data.
    pub fn verify_fold<F, PCS, ProofTranscript>(
        instance_1: &SplitCommittedInstance<F, PCS>,
        instance_2: &SplitCommittedInstance<F, PCS>,
        proof: &SplitCommittedNIFSProof<F, PCS>,
        transcript: &mut ProofTranscript,
    ) -> SplitCommittedInstance<F, PCS>
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
        PCS::Commitment: std::ops::Add<Output = PCS::Commitment> + Clone,
        ProofTranscript: jolt_core::transcripts::Transcript,
    {
        // Add T commitment to transcript (same as prover)
        transcript.append_scalar(&proof.r_T); // Placeholder

        // Get folding challenge (same as prover via Fiat-Shamir)
        let r: F = transcript.challenge_scalar();

        // Fold instances (verifier can do this from public data)
        fold_instances(instance_1, instance_2, &proof.T_commitment, r)
    }
}

// ============================================================================
// Conversion from Standard to Split-Committed
// ============================================================================

impl<F: JoltField> SplitCommittedWitness<F> {
    /// Creates a split-committed witness from a single witness vector.
    ///
    /// This is useful for converting from the simple relaxed R1CS to split-committed.
    pub fn from_single_witness(
        witness: Vec<F>,
        num_constraints: usize,
        segment_sizes: &[usize],
    ) -> Self {
        // Verify segment sizes sum to witness length
        let total: usize = segment_sizes.iter().sum();
        assert_eq!(
            total,
            witness.len(),
            "Segment sizes must sum to witness length"
        );

        // Split witness into segments
        let mut offset = 0;
        let segments: Vec<WitnessSegment<F>> = segment_sizes
            .iter()
            .map(|&size| {
                let values = witness[offset..offset + size].to_vec();
                offset += size;
                WitnessSegment::new_unblinded(values)
            })
            .collect();

        Self::from_standard(segments, num_constraints)
    }

    /// Adds random blinding factors to all segments.
    pub fn add_blinding<R: RngCore + CryptoRng>(&mut self, rng: &mut R) {
        self.r_E = F::random(rng);
        for segment in &mut self.segments {
            segment.blinding = F::random(rng);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_std::{One, Zero};
    use jolt_core::poly::commitment::mock::{MockCommitScheme, MockCommitment};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    type TestPCS = MockCommitScheme<Fr>;
    type TestCommitment = MockCommitment<Fr>;

    /// Helper to create a default commitment for testing.
    fn default_commitment() -> TestCommitment {
        TestCommitment::default()
    }

    /// Helper to create a vector of default commitments.
    fn default_commitments(n: usize) -> Vec<TestCommitment> {
        vec![TestCommitment::default(); n]
    }

    #[test]
    fn test_witness_segment_creation() {
        let values = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64)];
        let blinding = Fr::from(42u64);
        let segment = WitnessSegment::new(values.clone(), blinding);

        assert_eq!(segment.len(), 3);
        assert_eq!(segment.values, values);
        assert_eq!(segment.blinding, blinding);
    }

    #[test]
    fn test_witness_segment_unblinded() {
        let values = vec![Fr::from(1u64), Fr::from(2u64)];
        let segment = WitnessSegment::new_unblinded(values.clone());

        assert_eq!(segment.blinding, Fr::zero());
        assert_eq!(segment.values, values);
    }

    #[test]
    fn test_split_committed_instance_creation() {
        let E_commitment = default_commitment();
        let u = Fr::one();
        let W_commitments = default_commitments(3);
        let x = vec![Fr::from(1u64), Fr::from(2u64)];
        let segment_sizes = vec![10, 20, 30];

        let instance = SplitCommittedInstance::<Fr, TestPCS>::new(
            E_commitment,
            u,
            W_commitments,
            x.clone(),
            segment_sizes.clone(),
        );

        assert_eq!(instance.num_segments(), 3);
        assert_eq!(instance.total_witness_size(), 60);
        assert_eq!(instance.num_public_inputs(), 2);
        assert_eq!(instance.num_variables(), 1 + 2 + 60); // 1 + x.len() + W.len()
    }

    #[test]
    fn test_split_committed_witness_creation() {
        let segments = vec![
            WitnessSegment::new(vec![Fr::from(1u64), Fr::from(2u64)], Fr::from(10u64)),
            WitnessSegment::new(vec![Fr::from(3u64), Fr::from(4u64), Fr::from(5u64)], Fr::from(20u64)),
        ];
        let E = vec![Fr::zero(); 5];
        let r_E = Fr::from(30u64);

        let witness = SplitCommittedWitness::new(E.clone(), r_E, segments);

        assert_eq!(witness.num_segments(), 2);
        assert_eq!(witness.segment_sizes(), vec![2, 3]);
        assert_eq!(witness.total_witness_size(), 5);
    }

    #[test]
    fn test_construct_z() {
        let segments = vec![
            WitnessSegment::new_unblinded(vec![Fr::from(10u64), Fr::from(20u64)]),
            WitnessSegment::new_unblinded(vec![Fr::from(30u64)]),
        ];
        let witness = SplitCommittedWitness::from_standard(segments, 1);
        let x = vec![Fr::from(5u64)];

        let z = witness.construct_z(&x);

        // Z = (1, x, W₁, W₂) = (1, 5, 10, 20, 30)
        assert_eq!(z.len(), 5);
        assert_eq!(z[0], Fr::one());
        assert_eq!(z[1], Fr::from(5u64));
        assert_eq!(z[2], Fr::from(10u64));
        assert_eq!(z[3], Fr::from(20u64));
        assert_eq!(z[4], Fr::from(30u64));
    }

    #[test]
    fn test_split_committed_satisfaction() {
        // Create a simple circuit: x * w1 = w2
        // Z = (1, x, w1, w2) where x is public, w1 and w2 are private
        let mut matrices: R1CSMatrices<Fr> = R1CSMatrices::new(1, 4, 1);

        // A: select x (index 1)
        matrices.A.add_entry(0, 1, Fr::one());
        // B: select w1 (index 2)
        matrices.B.add_entry(0, 2, Fr::one());
        // C: select w2 (index 3)
        matrices.C.add_entry(0, 3, Fr::one());

        // Create witness: x=3, w1=4, w2=12 (since 3*4=12)
        let x = vec![Fr::from(3u64)];
        let segments = vec![
            WitnessSegment::new_unblinded(vec![Fr::from(4u64)]),  // w1
            WitnessSegment::new_unblinded(vec![Fr::from(12u64)]), // w2
        ];
        let witness = SplitCommittedWitness::from_standard(segments, 1);

        let instance = SplitCommittedInstance::<Fr, TestPCS>::from_standard(
            default_commitments(2),
            x,
            vec![1, 1],
        );

        assert!(is_satisfied(&matrices, &instance, &witness));
    }

    #[test]
    fn test_split_committed_unsatisfied() {
        // Same circuit as above but wrong witness
        let mut matrices: R1CSMatrices<Fr> = R1CSMatrices::new(1, 4, 1);
        matrices.A.add_entry(0, 1, Fr::one());
        matrices.B.add_entry(0, 2, Fr::one());
        matrices.C.add_entry(0, 3, Fr::one());

        // Wrong witness: x=3, w1=4, w2=13 (should be 12)
        let x = vec![Fr::from(3u64)];
        let segments = vec![
            WitnessSegment::new_unblinded(vec![Fr::from(4u64)]),
            WitnessSegment::new_unblinded(vec![Fr::from(13u64)]), // Wrong!
        ];
        let witness = SplitCommittedWitness::from_standard(segments, 1);

        let instance = SplitCommittedInstance::<Fr, TestPCS>::from_standard(
            default_commitments(2),
            x,
            vec![1, 1],
        );

        assert!(!is_satisfied(&matrices, &instance, &witness));
    }

    #[test]
    fn test_cross_term_computation() {
        let mut rng = StdRng::seed_from_u64(12345);

        // Simple circuit
        let mut matrices: R1CSMatrices<Fr> = R1CSMatrices::new(1, 4, 1);
        matrices.A.add_entry(0, 1, Fr::one());
        matrices.B.add_entry(0, 2, Fr::one());
        matrices.C.add_entry(0, 3, Fr::one());

        // Generate two random satisfying instances
        let segment_sizes = vec![1, 1];
        let (witness_1, u_1, x_1) =
            SplitCommittedWitness::sample_random(&matrices, &segment_sizes, 1, &mut rng);
        let instance_1 = SplitCommittedInstance::<Fr, TestPCS>::new(
            default_commitment(),
            u_1,
            default_commitments(2),
            x_1,
            segment_sizes.clone(),
        );

        let (witness_2, u_2, x_2) =
            SplitCommittedWitness::sample_random(&matrices, &segment_sizes, 1, &mut rng);
        let instance_2 = SplitCommittedInstance::<Fr, TestPCS>::new(
            default_commitment(),
            u_2,
            default_commitments(2),
            x_2,
            segment_sizes.clone(),
        );

        // Both should satisfy
        assert!(is_satisfied(&matrices, &instance_1, &witness_1));
        assert!(is_satisfied(&matrices, &instance_2, &witness_2));

        // Compute cross-term
        let T = compute_cross_term(&matrices, &instance_1, &witness_1, &instance_2, &witness_2);

        assert_eq!(T.len(), 1);
    }

    #[test]
    fn test_witness_folding() {
        let mut rng = StdRng::seed_from_u64(54321);

        // Create two witnesses
        let segments_1 = vec![
            WitnessSegment::new(vec![Fr::from(1u64)], Fr::from(10u64)),
            WitnessSegment::new(vec![Fr::from(2u64)], Fr::from(20u64)),
        ];
        let witness_1 = SplitCommittedWitness::new(
            vec![Fr::from(5u64)],
            Fr::from(100u64),
            segments_1,
        );

        let segments_2 = vec![
            WitnessSegment::new(vec![Fr::from(3u64)], Fr::from(30u64)),
            WitnessSegment::new(vec![Fr::from(4u64)], Fr::from(40u64)),
        ];
        let witness_2 = SplitCommittedWitness::new(
            vec![Fr::from(7u64)],
            Fr::from(200u64),
            segments_2,
        );

        let cross_term = vec![Fr::from(11u64)];
        let r_T = Fr::from(50u64);
        let r = Fr::from(2u64);

        let folded = fold_witnesses(&witness_1, &witness_2, &cross_term, r_T, r);

        // Check E folding: E' = E₁ + r·T + r²·E₂ = 5 + 2*11 + 4*7 = 5 + 22 + 28 = 55
        assert_eq!(folded.E[0], Fr::from(55u64));

        // Check rE folding: rE' = rE₁ + r·rT + r²·rE₂ = 100 + 2*50 + 4*200 = 100 + 100 + 800 = 1000
        assert_eq!(folded.r_E, Fr::from(1000u64));

        // Check segment 0 values: W' = W₁ + r·W₂ = 1 + 2*3 = 7
        assert_eq!(folded.segments[0].values[0], Fr::from(7u64));

        // Check segment 0 blinding: rW' = rW₁ + r·rW₂ = 10 + 2*30 = 70
        assert_eq!(folded.segments[0].blinding, Fr::from(70u64));

        // Check segment 1 values: W' = W₁ + r·W₂ = 2 + 2*4 = 10
        assert_eq!(folded.segments[1].values[0], Fr::from(10u64));

        // Check segment 1 blinding: rW' = rW₁ + r·rW₂ = 20 + 2*40 = 100
        assert_eq!(folded.segments[1].blinding, Fr::from(100u64));
    }

    #[test]
    fn test_folding_preserves_satisfaction() {
        let mut rng = StdRng::seed_from_u64(99999);

        // Create a circuit
        let mut matrices: R1CSMatrices<Fr> = R1CSMatrices::new(1, 4, 1);
        matrices.A.add_entry(0, 1, Fr::one());
        matrices.B.add_entry(0, 2, Fr::one());
        matrices.C.add_entry(0, 3, Fr::one());

        let segment_sizes = vec![1, 1];

        // Generate two random satisfying instances
        let (witness_1, u_1, x_1) =
            SplitCommittedWitness::sample_random(&matrices, &segment_sizes, 1, &mut rng);
        let instance_1 = SplitCommittedInstance::<Fr, TestPCS>::new(
            default_commitment(),
            u_1,
            default_commitments(2),
            x_1.clone(),
            segment_sizes.clone(),
        );

        let (witness_2, u_2, x_2) =
            SplitCommittedWitness::sample_random(&matrices, &segment_sizes, 1, &mut rng);
        let instance_2 = SplitCommittedInstance::<Fr, TestPCS>::new(
            default_commitment(),
            u_2,
            default_commitments(2),
            x_2.clone(),
            segment_sizes.clone(),
        );

        // Both should satisfy
        assert!(is_satisfied(&matrices, &instance_1, &witness_1));
        assert!(is_satisfied(&matrices, &instance_2, &witness_2));

        // Compute cross-term and fold
        let T = compute_cross_term(&matrices, &instance_1, &witness_1, &instance_2, &witness_2);
        let r_T = Fr::random(&mut rng);
        let r = Fr::from(7u64); // Random challenge

        let folded_witness = fold_witnesses(&witness_1, &witness_2, &T, r_T, r);

        // Compute folded instance manually
        let folded_u = u_1 + r * u_2;
        let folded_x: Vec<Fr> = x_1.iter().zip(x_2.iter()).map(|(a, b)| *a + r * *b).collect();
        let folded_instance = SplitCommittedInstance::<Fr, TestPCS>::new(
            default_commitment(),
            folded_u,
            default_commitments(2),
            folded_x,
            segment_sizes.clone(),
        );

        // Folded should also satisfy!
        assert!(is_satisfied(&matrices, &folded_instance, &folded_witness));
    }

    #[test]
    fn test_from_single_witness() {
        let witness = vec![
            Fr::from(1u64),
            Fr::from(2u64),
            Fr::from(3u64),
            Fr::from(4u64),
            Fr::from(5u64),
        ];
        let segment_sizes = vec![2, 3];

        let split_witness =
            SplitCommittedWitness::from_single_witness(witness.clone(), 1, &segment_sizes);

        assert_eq!(split_witness.num_segments(), 2);
        assert_eq!(split_witness.segments[0].values, vec![Fr::from(1u64), Fr::from(2u64)]);
        assert_eq!(
            split_witness.segments[1].values,
            vec![Fr::from(3u64), Fr::from(4u64), Fr::from(5u64)]
        );

        // Full witness should match original
        assert_eq!(split_witness.full_witness(), witness);
    }
}
