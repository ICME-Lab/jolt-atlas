//! Random Satisfying Instance Generator
//!
//! This module generates random satisfying instances for NIFS folding.
//! These random instances are used to blind the real witness when folded
//! together, achieving zero-knowledge.
//!
//! # Security Property
//!
//! The random instance must satisfy the R1CS constraints but have
//! uniformly random witness values. When folded with a real instance,
//! the folded witness is computationally indistinguishable from random.
//!
//! # Generation Strategy
//!
//! For a relaxed R1CS: Az ∘ Bz = u·Cz + E
//!
//! We generate instances that satisfy the constraints by:
//! 1. Generating random witness W and public inputs x
//! 2. Computing z = (1, x, W)
//! 3. Computing E = Az ∘ Bz - Cz (the "slack" that makes it satisfy)
//! 4. Setting u = 1
//!
//! This always produces a valid relaxed R1CS instance.

use jolt_core::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::MultilinearPolynomial,
    },
};
use rand_core::{CryptoRng, RngCore};

use super::relaxed_r1cs::{R1CSMatrices, RelaxedR1CSInstance, RelaxedR1CSWitness};

/// Generator for random satisfying instances.
///
/// Creates R1CS instances with random witnesses that satisfy the constraints,
/// used for blinding in NIFS folding.
pub struct RandomInstanceGenerator<F: JoltField> {
    _marker: std::marker::PhantomData<F>,
}

impl<F: JoltField> RandomInstanceGenerator<F> {
    /// Creates a new random instance generator.
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }

    /// Pads a vector to the next power of 2 length.
    fn pad_to_power_of_2(mut vec: Vec<F>) -> Vec<F> {
        if vec.is_empty() {
            return vec![F::zero()];
        }
        let next_pow2 = vec.len().next_power_of_two();
        vec.resize(next_pow2, F::zero());
        vec
    }

    /// Generates a random satisfying instance for the given R1CS.
    ///
    /// The generated instance satisfies: Az ∘ Bz = u·Cz + E
    /// with u = 1 and E computed as the "slack" to make the equation hold.
    ///
    /// # Arguments
    /// * `matrices` - The R1CS constraint matrices
    /// * `rng` - Random number generator
    /// * `pcs_setup` - The PCS prover setup for computing commitments
    ///
    /// # Returns
    /// A tuple of (instance, witness) that satisfies the relaxed R1CS.
    pub fn generate_random_satisfying<PCS, R>(
        matrices: &R1CSMatrices<F>,
        rng: &mut R,
        pcs_setup: &PCS::ProverSetup,
    ) -> (RelaxedR1CSInstance<F, PCS>, RelaxedR1CSWitness<F>)
    where
        PCS: CommitmentScheme<Field = F>,
        R: RngCore + CryptoRng,
    {
        let num_vars = matrices.A.num_cols;
        let num_public = matrices.num_public_inputs;
        let num_private = num_vars - 1 - num_public; // -1 for the leading 1

        // Generate random private witness
        let witness: Vec<F> = (0..num_private).map(|_| F::random(rng)).collect();

        // Generate random public inputs
        let public_inputs: Vec<F> = (0..num_public).map(|_| F::random(rng)).collect();

        // Construct full assignment z = (1, x, W)
        let mut z = vec![F::one()];
        z.extend_from_slice(&public_inputs);
        z.extend_from_slice(&witness);

        // Compute Az, Bz, Cz
        let az = matrices.A.multiply(&z);
        let bz = matrices.B.multiply(&z);
        let cz = matrices.C.multiply(&z);

        // Compute E = Az ∘ Bz - Cz (the error/slack vector)
        // This ensures Az ∘ Bz = 1·Cz + E is satisfied
        let error: Vec<F> = az
            .iter()
            .zip(bz.iter())
            .zip(cz.iter())
            .map(|((a, b), c)| *a * *b - *c)
            .collect();

        // Pad vectors to power of 2 for polynomial commitment
        let W_padded = Self::pad_to_power_of_2(witness.clone());
        let E_padded = Self::pad_to_power_of_2(error.clone());

        // Compute actual commitments to W and E
        let W_poly: MultilinearPolynomial<F> = W_padded.into();
        let E_poly: MultilinearPolynomial<F> = E_padded.into();

        let (W_commitment, _W_hint) = PCS::commit(&W_poly, pcs_setup);
        let (E_commitment, _E_hint) = PCS::commit(&E_poly, pcs_setup);

        // Create the relaxed witness with computed error
        // For non-hiding version, use zero blinding factors
        let relaxed_witness = RelaxedR1CSWitness::new_simple(
            witness,
            F::zero(), // r_W = 0 for non-hiding
            error,
            F::zero(), // r_E = 0 for non-hiding
        );

        // Create the instance with actual commitments
        let instance = RelaxedR1CSInstance::new(
            W_commitment,
            E_commitment,
            F::one(), // u = 1
            public_inputs,
        );

        (instance, relaxed_witness)
    }

    /// Generates a random instance that truly satisfies standard R1CS (E = 0).
    ///
    /// This is more constrained than `generate_random_satisfying` and may
    /// not always succeed for arbitrary R1CS matrices. It works for:
    /// - Identity-like matrices
    /// - Matrices with specific structure
    ///
    /// For general matrices, use `generate_random_satisfying` which always works.
    pub fn try_generate_standard_satisfying<PCS, R>(
        matrices: &R1CSMatrices<F>,
        rng: &mut R,
        _pcs_setup: &PCS::ProverSetup,
        max_attempts: usize,
    ) -> Option<(RelaxedR1CSInstance<F, PCS>, RelaxedR1CSWitness<F>)>
    where
        PCS: CommitmentScheme<Field = F>,
        R: RngCore + CryptoRng,
    {
        let num_constraints = matrices.A.num_rows;
        let num_vars = matrices.A.num_cols;
        let num_public = matrices.num_public_inputs;
        let num_private = num_vars - 1 - num_public;

        for _ in 0..max_attempts {
            // Generate random witness and public inputs
            let witness: Vec<F> = (0..num_private).map(|_| F::random(rng)).collect();
            let public_inputs: Vec<F> = (0..num_public).map(|_| F::random(rng)).collect();

            // Construct z = (1, x, W)
            let mut z = vec![F::one()];
            z.extend_from_slice(&public_inputs);
            z.extend_from_slice(&witness);

            // Check if this satisfies Az ∘ Bz = Cz
            let az = matrices.A.multiply(&z);
            let bz = matrices.B.multiply(&z);
            let cz = matrices.C.multiply(&z);

            let satisfies = az
                .iter()
                .zip(bz.iter())
                .zip(cz.iter())
                .all(|((a, b), c)| *a * *b == *c);

            if satisfies {
                let relaxed_witness =
                    RelaxedR1CSWitness::from_standard(witness, num_constraints);

                let instance = RelaxedR1CSInstance::new(
                    PCS::Commitment::default(),
                    PCS::Commitment::default(),
                    F::one(),
                    public_inputs,
                );

                return Some((instance, relaxed_witness));
            }
        }

        None
    }

    /// Generates a random satisfying instance that specifically satisfies
    /// the sumcheck verifier circuit.
    ///
    /// This is used for ZK sumcheck where we need random instances that
    /// satisfy the verifier's checks.
    pub fn generate_for_sumcheck_verifier<PCS, R>(
        num_rounds: usize,
        degree: usize,
        rng: &mut R,
        pcs_setup: &PCS::ProverSetup,
    ) -> (RelaxedR1CSInstance<F, PCS>, RelaxedR1CSWitness<F>)
    where
        PCS: CommitmentScheme<Field = F>,
        R: RngCore + CryptoRng,
    {
        use super::verifier_circuit::VerifierR1CSCircuit;

        // Create the verifier circuit
        let circuit: VerifierR1CSCircuit<F> = VerifierR1CSCircuit::new(num_rounds, degree);

        // Generate random instance for this circuit
        Self::generate_random_satisfying(&circuit.matrices, rng, pcs_setup)
    }
}

impl<F: JoltField> Default for RandomInstanceGenerator<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to sample random field elements.
pub fn sample_random_field_elements<F: JoltField, R: RngCore + CryptoRng>(
    count: usize,
    rng: &mut R,
) -> Vec<F> {
    (0..count).map(|_| F::random(rng)).collect()
}

/// Helper function to sample a random non-zero field element.
pub fn sample_random_nonzero<F: JoltField, R: RngCore + CryptoRng>(rng: &mut R) -> F {
    loop {
        let r = F::random(rng);
        if r != F::zero() {
            return r;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use jolt_core::poly::commitment::mock::MockCommitScheme;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_sample_random_field_elements() {
        let mut rng = StdRng::seed_from_u64(12345);
        let elements: Vec<Fr> = sample_random_field_elements(10, &mut rng);
        assert_eq!(elements.len(), 10);

        // Check that they're not all zero (probability is negligible)
        assert!(elements.iter().any(|e| *e != Fr::from(0u64)));
    }

    #[test]
    fn test_sample_random_nonzero() {
        let mut rng = StdRng::seed_from_u64(12345);
        let element: Fr = sample_random_nonzero(&mut rng);
        assert_ne!(element, Fr::from(0u64));
    }

    #[test]
    fn test_random_instance_generator_creation() {
        let _generator: RandomInstanceGenerator<Fr> = RandomInstanceGenerator::new();
        // Just verify it can be created
    }

    #[test]
    fn test_generate_random_satisfying_instance() {
        let mut rng = StdRng::seed_from_u64(42);

        // Create a simple R1CS: x * y = z
        // Variables: [1, x, y, z] where x is public, y and z are private
        // z = (1, x, w) where w = [y, z]
        let mut matrices: R1CSMatrices<Fr> = R1CSMatrices::new(1, 4, 1);

        // A: select x (index 1)
        matrices.A.add_entry(0, 1, Fr::from(1u64));
        // B: select y (index 2)
        matrices.B.add_entry(0, 2, Fr::from(1u64));
        // C: select z (index 3)
        matrices.C.add_entry(0, 3, Fr::from(1u64));

        // Generate random satisfying instance
        let (_instance, witness) =
            RandomInstanceGenerator::<Fr>::generate_random_satisfying::<MockCommitScheme<Fr>, _>(
                &matrices, &mut rng, &(),
            );

        // Verify it satisfies the relaxed R1CS: Az ∘ Bz = u·Cz + E
        assert!(matrices.is_relaxed_satisfied(&witness, Fr::from(1u64), &_instance.x));
    }

    #[test]
    fn test_generate_random_satisfying_identity_circuit() {
        let mut rng = StdRng::seed_from_u64(123);

        // Create an identity circuit: 1 * 1 = 1 (always satisfied)
        let mut matrices: R1CSMatrices<Fr> = R1CSMatrices::new(1, 3, 1);

        // A: constant 1
        matrices.A.add_entry(0, 0, Fr::from(1u64));
        // B: constant 1
        matrices.B.add_entry(0, 0, Fr::from(1u64));
        // C: constant 1
        matrices.C.add_entry(0, 0, Fr::from(1u64));

        // Generate random satisfying instance
        let (_instance, witness) =
            RandomInstanceGenerator::<Fr>::generate_random_satisfying::<MockCommitScheme<Fr>, _>(
                &matrices, &mut rng, &(),
            );

        // This should always satisfy with E = 0 since 1 * 1 = 1
        assert!(matrices.is_relaxed_satisfied(&witness, Fr::from(1u64), &_instance.x));

        // For identity circuit, E should be zero
        assert!(witness.E.iter().all(|e| *e == Fr::from(0u64)));
    }

    #[test]
    fn test_generate_multiple_random_instances() {
        let mut rng = StdRng::seed_from_u64(999);

        // Create a more complex circuit
        let mut matrices: R1CSMatrices<Fr> = R1CSMatrices::new(2, 5, 1);

        // Constraint 1: x * w1 = w2
        matrices.A.add_entry(0, 1, Fr::from(1u64)); // x
        matrices.B.add_entry(0, 2, Fr::from(1u64)); // w1
        matrices.C.add_entry(0, 3, Fr::from(1u64)); // w2

        // Constraint 2: w1 * w1 = w3
        matrices.A.add_entry(1, 2, Fr::from(1u64)); // w1
        matrices.B.add_entry(1, 2, Fr::from(1u64)); // w1
        matrices.C.add_entry(1, 4, Fr::from(1u64)); // w3

        // Generate multiple instances and verify all satisfy
        for _ in 0..10 {
            let (_instance, witness) =
                RandomInstanceGenerator::<Fr>::generate_random_satisfying::<MockCommitScheme<Fr>, _>(
                    &matrices, &mut rng, &(),
                );

            assert!(matrices.is_relaxed_satisfied(&witness, Fr::from(1u64), &_instance.x));
        }
    }

    #[test]
    fn test_random_instances_are_different() {
        let mut rng = StdRng::seed_from_u64(777);

        // Create a simple circuit
        let mut matrices: R1CSMatrices<Fr> = R1CSMatrices::new(1, 3, 1);
        matrices.A.add_entry(0, 1, Fr::from(1u64));
        matrices.B.add_entry(0, 0, Fr::from(1u64));
        matrices.C.add_entry(0, 2, Fr::from(1u64));

        // Generate two instances
        let (instance1, witness1) =
            RandomInstanceGenerator::<Fr>::generate_random_satisfying::<MockCommitScheme<Fr>, _>(
                &matrices, &mut rng, &(),
            );
        let (instance2, witness2) =
            RandomInstanceGenerator::<Fr>::generate_random_satisfying::<MockCommitScheme<Fr>, _>(
                &matrices, &mut rng, &(),
            );

        // They should be different (with overwhelming probability)
        assert_ne!(witness1.W, witness2.W);
        assert_ne!(instance1.x, instance2.x);
    }
}
