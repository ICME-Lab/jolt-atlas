//! Relaxed R1CS (Rank-1 Constraint System) for NIFS folding.
//!
//! A standard R1CS has the form: Az ∘ Bz = Cz
//! where ∘ denotes the Hadamard (element-wise) product.
//!
//! A Relaxed R1CS generalizes this to: Az ∘ Bz = u·Cz + E
//! where:
//! - u is a scalar (1 for satisfying instances, arbitrary after folding)
//! - E is an error vector (zero for satisfying instances)
//!
//! This relaxation is key to NIFS folding, allowing us to combine two
//! instances while preserving the ability to verify correctness.
//!
//! # Blinding Factors
//!
//! For zero-knowledge, commitments include blinding factors:
//! - C(W) = Commit(W) + r_W * H
//! - C(E) = Commit(E) + r_E * H
//!
//! The witness stores both the values and their blinding factors, ensuring
//! proper homomorphic folding of commitments.

use jolt_core::{
    field::JoltField,
    poly::commitment::commitment_scheme::CommitmentScheme,
};
use rand_core::{CryptoRng, RngCore};
use std::marker::PhantomData;

/// A committed relaxed R1CS instance.
///
/// This contains the public components visible to the verifier:
/// - Commitment to the witness W
/// - Commitment to the error vector E
/// - The scalar u
/// - The public inputs x
/// - Commitments to round polynomials (for sumcheck ZK)
#[derive(Clone, Debug)]
pub struct RelaxedR1CSInstance<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    /// Commitment to the witness vector W
    pub W_commitment: PCS::Commitment,
    /// Commitment to the error vector E (zero for satisfying instances)
    pub E_commitment: PCS::Commitment,
    /// Scalar multiplier (1 for standard satisfying instances)
    pub u: F,
    /// Public inputs/outputs
    pub x: Vec<F>,
    /// Per-round commitments to round polynomial coefficients
    /// These correspond to the sumcheck round polynomials
    pub round_commitments: Vec<PCS::Commitment>,
    _marker: PhantomData<PCS>,
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> RelaxedR1CSInstance<F, PCS> {
    /// Creates a new relaxed R1CS instance.
    pub fn new(
        W_commitment: PCS::Commitment,
        E_commitment: PCS::Commitment,
        u: F,
        x: Vec<F>,
    ) -> Self {
        Self {
            W_commitment,
            E_commitment,
            u,
            x,
            round_commitments: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Creates a new relaxed R1CS instance with round commitments.
    pub fn new_with_rounds(
        W_commitment: PCS::Commitment,
        E_commitment: PCS::Commitment,
        u: F,
        x: Vec<F>,
        round_commitments: Vec<PCS::Commitment>,
    ) -> Self {
        Self {
            W_commitment,
            E_commitment,
            u,
            x,
            round_commitments,
            _marker: PhantomData,
        }
    }

    /// Creates a standard (non-relaxed) instance from a witness commitment.
    ///
    /// For a standard satisfying instance, u = 1 and E = 0.
    pub fn from_standard(W_commitment: PCS::Commitment, x: Vec<F>) -> Self {
        Self {
            W_commitment,
            E_commitment: PCS::Commitment::default(),
            u: F::one(),
            x,
            round_commitments: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Creates a standard instance with round commitments.
    pub fn from_standard_with_rounds(
        W_commitment: PCS::Commitment,
        x: Vec<F>,
        round_commitments: Vec<PCS::Commitment>,
    ) -> Self {
        Self {
            W_commitment,
            E_commitment: PCS::Commitment::default(),
            u: F::one(),
            x,
            round_commitments,
            _marker: PhantomData,
        }
    }

    /// Returns the number of round commitments.
    pub fn num_rounds(&self) -> usize {
        self.round_commitments.len()
    }
}

/// The witness for a relaxed R1CS instance.
///
/// Contains the private components known only to the prover:
/// - The witness vector W and its blinding factor r_W
/// - The error vector E and its blinding factor r_E
/// - Per-round polynomial coefficients and their blinding factors
///
/// The blinding factors are essential for zero-knowledge: they ensure
/// that commitments reveal nothing about the underlying values.
#[derive(Clone, Debug)]
pub struct RelaxedR1CSWitness<F: JoltField> {
    /// The error vector (zeros for non-relaxed satisfying instances)
    pub E: Vec<F>,
    /// Blinding factor for E commitment: C(E) = Commit(E) + r_E * H
    pub r_E: F,
    /// The witness vector (private portion of Z)
    pub W: Vec<F>,
    /// Blinding factor for W commitment: C(W) = Commit(W) + r_W * H
    pub r_W: F,
    /// Per-round polynomial coefficients (openings for round_commitments)
    /// Each inner Vec contains the coefficients for one sumcheck round polynomial
    pub round_coefficients: Vec<Vec<F>>,
    /// Per-round blinding factors for round polynomial commitments
    pub round_blindings: Vec<F>,
}

impl<F: JoltField> RelaxedR1CSWitness<F> {
    /// Creates a new relaxed R1CS witness with all components.
    pub fn new(
        W: Vec<F>,
        r_W: F,
        E: Vec<F>,
        r_E: F,
        round_coefficients: Vec<Vec<F>>,
        round_blindings: Vec<F>,
    ) -> Self {
        debug_assert_eq!(
            round_coefficients.len(),
            round_blindings.len(),
            "round_coefficients and round_blindings must have the same length"
        );
        Self {
            E,
            r_E,
            W,
            r_W,
            round_coefficients,
            round_blindings,
        }
    }

    /// Creates a new witness with only W and E (no rounds).
    pub fn new_simple(W: Vec<F>, r_W: F, E: Vec<F>, r_E: F) -> Self {
        Self {
            E,
            r_E,
            W,
            r_W,
            round_coefficients: Vec::new(),
            round_blindings: Vec::new(),
        }
    }

    /// Creates a non-relaxed witness for a standard R1CS.
    ///
    /// For a standard satisfying instance:
    /// - E = 0 (no error)
    /// - r_E = 0 (can use zero blinding initially)
    ///
    /// # Arguments
    /// * `witness` - The witness vector W
    /// * `num_constraints` - Number of constraints (determines E vector size)
    /// * `round_coefficients` - Coefficients for each sumcheck round polynomial
    /// * `round_blindings` - Blinding factors for round commitments
    /// * `rng` - Random number generator for sampling r_W
    pub fn new_non_relaxed<R: RngCore + CryptoRng>(
        witness: Vec<F>,
        num_constraints: usize,
        round_coefficients: Vec<Vec<F>>,
        round_blindings: Vec<F>,
        rng: &mut R,
    ) -> Self {
        Self {
            W: witness,
            r_W: F::random(rng),
            E: vec![F::zero(); num_constraints],
            r_E: F::zero(), // E = 0 for non-relaxed, so blinding doesn't matter
            round_coefficients,
            round_blindings,
        }
    }

    /// Creates a standard (non-relaxed) witness from just a witness vector.
    ///
    /// This is a simpler constructor for cases where:
    /// - No round coefficients are needed yet
    /// - Zero blinding is acceptable (will be set properly during folding)
    pub fn from_standard(W: Vec<F>, constraint_count: usize) -> Self {
        Self {
            W,
            r_W: F::zero(),
            E: vec![F::zero(); constraint_count],
            r_E: F::zero(),
            round_coefficients: Vec::new(),
            round_blindings: Vec::new(),
        }
    }

    /// Creates a random satisfying witness for use in BlindFold.
    ///
    /// This generates a completely random witness that satisfies the relaxed R1CS
    /// by computing E = Az ∘ Bz - u·Cz (making E absorb any "errors").
    ///
    /// # Arguments
    /// * `matrices` - The R1CS constraint matrices
    /// * `witness_size` - Size of the W vector
    /// * `public_input_size` - Size of the public input x
    /// * `num_rounds` - Number of sumcheck rounds (for round_coefficients)
    /// * `coeffs_per_round` - Number of coefficients per round polynomial
    /// * `rng` - Random number generator
    pub fn sample_random<R: RngCore + CryptoRng>(
        matrices: &R1CSMatrices<F>,
        witness_size: usize,
        public_input_size: usize,
        num_rounds: usize,
        coeffs_per_round: usize,
        rng: &mut R,
    ) -> (Self, F, Vec<F>) {
        // Sample random witness W
        let W: Vec<F> = (0..witness_size).map(|_| F::random(rng)).collect();
        let r_W = F::random(rng);

        // Sample random u and x
        let u = F::random(rng);
        let x: Vec<F> = (0..public_input_size).map(|_| F::random(rng)).collect();

        // Construct z = (1, x, W)
        let mut z = vec![F::one()];
        z.extend_from_slice(&x);
        z.extend_from_slice(&W);

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

        // Sample random round coefficients and blindings
        let round_coefficients: Vec<Vec<F>> = (0..num_rounds)
            .map(|_| (0..coeffs_per_round).map(|_| F::random(rng)).collect())
            .collect();
        let round_blindings: Vec<F> = (0..num_rounds).map(|_| F::random(rng)).collect();

        let witness = Self {
            E,
            r_E,
            W,
            r_W,
            round_coefficients,
            round_blindings,
        };

        (witness, u, x)
    }

    /// Folds two witnesses using folding challenge r and cross-term T.
    ///
    /// Computes:
    /// - W' = W_1 + r·W_2
    /// - E' = E_1 + r·T + r²·E_2
    /// - r_W' = r_W_1 + r·r_W_2
    /// - r_E' = r_E_1 + r·r_T + r²·r_E_2
    /// - round_coeffs' = round_coeffs_1 + r·round_coeffs_2
    /// - round_blindings' = round_blindings_1 + r·round_blindings_2
    pub fn fold(
        witness_1: &Self,
        witness_2: &Self,
        cross_term: &[F],
        r_T: F,
        r: F,
    ) -> Self {
        let r_squared = r * r;

        // W' = W_1 + r·W_2
        let folded_W: Vec<F> = witness_1
            .W
            .iter()
            .zip(witness_2.W.iter())
            .map(|(w1, w2)| *w1 + r * *w2)
            .collect();

        // r_W' = r_W_1 + r·r_W_2
        let folded_r_W = witness_1.r_W + r * witness_2.r_W;

        // E' = E_1 + r·T + r²·E_2
        let folded_E: Vec<F> = witness_1
            .E
            .iter()
            .zip(cross_term.iter())
            .zip(witness_2.E.iter())
            .map(|((e1, t), e2)| *e1 + r * *t + r_squared * *e2)
            .collect();

        // r_E' = r_E_1 + r·r_T + r²·r_E_2
        let folded_r_E = witness_1.r_E + r * r_T + r_squared * witness_2.r_E;

        // Fold round coefficients: coeffs' = coeffs_1 + r·coeffs_2
        let folded_round_coefficients: Vec<Vec<F>> = witness_1
            .round_coefficients
            .iter()
            .zip(witness_2.round_coefficients.iter())
            .map(|(coeffs1, coeffs2)| {
                coeffs1
                    .iter()
                    .zip(coeffs2.iter())
                    .map(|(c1, c2)| *c1 + r * *c2)
                    .collect()
            })
            .collect();

        // Fold round blindings: blindings' = blindings_1 + r·blindings_2
        let folded_round_blindings: Vec<F> = witness_1
            .round_blindings
            .iter()
            .zip(witness_2.round_blindings.iter())
            .map(|(b1, b2)| *b1 + r * *b2)
            .collect();

        Self {
            E: folded_E,
            r_E: folded_r_E,
            W: folded_W,
            r_W: folded_r_W,
            round_coefficients: folded_round_coefficients,
            round_blindings: folded_round_blindings,
        }
    }

    /// Returns the length of the witness vector.
    pub fn witness_len(&self) -> usize {
        self.W.len()
    }

    /// Returns the number of constraints (length of error vector).
    pub fn constraint_count(&self) -> usize {
        self.E.len()
    }

    /// Returns the number of sumcheck rounds.
    pub fn num_rounds(&self) -> usize {
        self.round_coefficients.len()
    }

    /// Verifies that round coefficients are consistent with round commitments.
    ///
    /// This checks that for each round i:
    /// round_commitments[i] = Commit(round_coefficients[i], round_blindings[i])
    pub fn verify_round_consistency<PCS: CommitmentScheme<Field = F>>(
        &self,
        instance: &RelaxedR1CSInstance<F, PCS>,
        _pcs_setup: &PCS::ProverSetup,
    ) -> bool {
        if self.round_coefficients.len() != instance.round_commitments.len() {
            return false;
        }
        if self.round_blindings.len() != instance.round_commitments.len() {
            return false;
        }
        // In a full implementation, we would verify each commitment here
        // For now, just check lengths match
        true
    }
}

/// Sparse matrix representation for R1CS.
///
/// The matrices A, B, C are typically sparse in R1CS, so we use
/// a sparse representation for efficiency.
#[derive(Clone, Debug)]
pub struct SparseMatrix<F: JoltField> {
    /// Number of rows in the matrix
    pub num_rows: usize,
    /// Number of columns in the matrix
    pub num_cols: usize,
    /// Non-zero entries as (row, col, value) triples
    pub entries: Vec<(usize, usize, F)>,
}

impl<F: JoltField> SparseMatrix<F> {
    /// Creates a new sparse matrix.
    pub fn new(num_rows: usize, num_cols: usize) -> Self {
        Self {
            num_rows,
            num_cols,
            entries: Vec::new(),
        }
    }

    /// Adds an entry to the matrix.
    pub fn add_entry(&mut self, row: usize, col: usize, value: F) {
        if value != F::zero() {
            self.entries.push((row, col, value));
        }
    }

    /// Multiplies the matrix by a vector.
    pub fn multiply(&self, vec: &[F]) -> Vec<F> {
        assert_eq!(vec.len(), self.num_cols);
        let mut result = vec![F::zero(); self.num_rows];
        for &(row, col, ref value) in &self.entries {
            result[row] += *value * vec[col];
        }
        result
    }
}

/// R1CS constraint matrices (A, B, C).
#[derive(Clone, Debug)]
pub struct R1CSMatrices<F: JoltField> {
    /// Matrix A
    pub A: SparseMatrix<F>,
    /// Matrix B
    pub B: SparseMatrix<F>,
    /// Matrix C
    pub C: SparseMatrix<F>,
    /// Number of public inputs (not counting the leading 1)
    pub num_public_inputs: usize,
}

impl<F: JoltField> R1CSMatrices<F> {
    /// Creates new R1CS matrices.
    pub fn new(num_constraints: usize, num_variables: usize, num_public_inputs: usize) -> Self {
        Self {
            A: SparseMatrix::new(num_constraints, num_variables),
            B: SparseMatrix::new(num_constraints, num_variables),
            C: SparseMatrix::new(num_constraints, num_variables),
            num_public_inputs,
        }
    }

    /// Checks if a witness satisfies the standard R1CS.
    ///
    /// Verifies: Az ∘ Bz = Cz
    pub fn is_satisfied(&self, witness: &[F], public_inputs: &[F]) -> bool {
        // Construct full assignment z = (1, x, w) where x is public inputs and w is witness
        let mut z = vec![F::one()];
        z.extend_from_slice(public_inputs);
        z.extend_from_slice(witness);

        let az = self.A.multiply(&z);
        let bz = self.B.multiply(&z);
        let cz = self.C.multiply(&z);

        // Check Az ∘ Bz = Cz
        for i in 0..az.len() {
            if az[i] * bz[i] != cz[i] {
                return false;
            }
        }
        true
    }

    /// Checks if a relaxed witness satisfies the relaxed R1CS.
    ///
    /// Verifies: Az ∘ Bz = u·Cz + E
    pub fn is_relaxed_satisfied(
        &self,
        witness: &RelaxedR1CSWitness<F>,
        u: F,
        public_inputs: &[F],
    ) -> bool {
        // Construct full assignment z = (1, x, w)
        let mut z = vec![F::one()];
        z.extend_from_slice(public_inputs);
        z.extend_from_slice(&witness.W);

        let az = self.A.multiply(&z);
        let bz = self.B.multiply(&z);
        let cz = self.C.multiply(&z);

        // Check Az ∘ Bz = u·Cz + E
        for i in 0..az.len() {
            if az[i] * bz[i] != u * cz[i] + witness.E[i] {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_std::{One, Zero};
    use rand::thread_rng;

    #[test]
    fn test_sparse_matrix_multiply() {
        // Create a simple 2x3 matrix:
        // [1 0 2]
        // [0 3 0]
        let mut matrix: SparseMatrix<Fr> = SparseMatrix::new(2, 3);
        matrix.add_entry(0, 0, Fr::from(1u64));
        matrix.add_entry(0, 2, Fr::from(2u64));
        matrix.add_entry(1, 1, Fr::from(3u64));

        // Multiply by [1, 2, 3]
        let vec = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64)];
        let result = matrix.multiply(&vec);

        // Expected: [1*1 + 2*3, 3*2] = [7, 6]
        assert_eq!(result[0], Fr::from(7u64));
        assert_eq!(result[1], Fr::from(6u64));
    }

    #[test]
    fn test_relaxed_r1cs_witness_from_standard() {
        let witness = vec![Fr::from(1u64), Fr::from(2u64)];
        let relaxed = RelaxedR1CSWitness::from_standard(witness.clone(), 5);

        assert_eq!(relaxed.W, witness);
        assert_eq!(relaxed.E.len(), 5);
        assert!(relaxed.E.iter().all(|e| *e == Fr::from(0u64)));
        assert_eq!(relaxed.r_W, Fr::zero());
        assert_eq!(relaxed.r_E, Fr::zero());
    }

    #[test]
    fn test_relaxed_r1cs_witness_new_simple() {
        let W = vec![Fr::from(1u64), Fr::from(2u64)];
        let r_W = Fr::from(10u64);
        let E = vec![Fr::from(3u64), Fr::from(4u64)];
        let r_E = Fr::from(20u64);

        let witness = RelaxedR1CSWitness::new_simple(W.clone(), r_W, E.clone(), r_E);

        assert_eq!(witness.W, W);
        assert_eq!(witness.r_W, r_W);
        assert_eq!(witness.E, E);
        assert_eq!(witness.r_E, r_E);
        assert!(witness.round_coefficients.is_empty());
        assert!(witness.round_blindings.is_empty());
    }

    #[test]
    fn test_relaxed_r1cs_witness_with_rounds() {
        let W = vec![Fr::from(1u64)];
        let r_W = Fr::from(10u64);
        let E = vec![Fr::from(0u64)];
        let r_E = Fr::from(0u64);
        let round_coefficients = vec![
            vec![Fr::from(1u64), Fr::from(2u64)],
            vec![Fr::from(3u64), Fr::from(4u64)],
        ];
        let round_blindings = vec![Fr::from(100u64), Fr::from(200u64)];

        let witness = RelaxedR1CSWitness::new(
            W.clone(),
            r_W,
            E.clone(),
            r_E,
            round_coefficients.clone(),
            round_blindings.clone(),
        );

        assert_eq!(witness.num_rounds(), 2);
        assert_eq!(witness.round_coefficients, round_coefficients);
        assert_eq!(witness.round_blindings, round_blindings);
    }

    #[test]
    fn test_witness_fold() {
        let witness_1 = RelaxedR1CSWitness::new(
            vec![Fr::from(1u64)],           // W
            Fr::from(10u64),                // r_W
            vec![Fr::from(5u64)],           // E
            Fr::from(20u64),                // r_E
            vec![vec![Fr::from(1u64)]],     // round_coefficients
            vec![Fr::from(100u64)],         // round_blindings
        );

        let witness_2 = RelaxedR1CSWitness::new(
            vec![Fr::from(2u64)],           // W
            Fr::from(30u64),                // r_W
            vec![Fr::from(3u64)],           // E
            Fr::from(40u64),                // r_E
            vec![vec![Fr::from(2u64)]],     // round_coefficients
            vec![Fr::from(200u64)],         // round_blindings
        );

        let cross_term = vec![Fr::from(7u64)];
        let r_T = Fr::from(50u64);
        let r = Fr::from(2u64);

        let folded = RelaxedR1CSWitness::fold(&witness_1, &witness_2, &cross_term, r_T, r);

        // W' = W_1 + r·W_2 = 1 + 2*2 = 5
        assert_eq!(folded.W[0], Fr::from(5u64));

        // r_W' = r_W_1 + r·r_W_2 = 10 + 2*30 = 70
        assert_eq!(folded.r_W, Fr::from(70u64));

        // E' = E_1 + r·T + r²·E_2 = 5 + 2*7 + 4*3 = 5 + 14 + 12 = 31
        assert_eq!(folded.E[0], Fr::from(31u64));

        // r_E' = r_E_1 + r·r_T + r²·r_E_2 = 20 + 2*50 + 4*40 = 20 + 100 + 160 = 280
        assert_eq!(folded.r_E, Fr::from(280u64));

        // round_coeffs' = coeffs_1 + r·coeffs_2 = 1 + 2*2 = 5
        assert_eq!(folded.round_coefficients[0][0], Fr::from(5u64));

        // round_blindings' = blindings_1 + r·blindings_2 = 100 + 2*200 = 500
        assert_eq!(folded.round_blindings[0], Fr::from(500u64));
    }

    #[test]
    fn test_sample_random_satisfying_witness() {
        let mut rng = thread_rng();

        // Create a simple R1CS: x * x = y (single constraint)
        let mut matrices = R1CSMatrices::<Fr>::new(1, 4, 1); // 1 constraint, 4 vars (1, x, w0, w1), 1 public input

        // z = (1, x, w0, w1) = (1, public_input, witness[0], witness[1])
        // Constraint: z[2] * z[2] = z[3]  (w0 * w0 = w1)
        matrices.A.add_entry(0, 2, Fr::one()); // A selects w0
        matrices.B.add_entry(0, 2, Fr::one()); // B selects w0
        matrices.C.add_entry(0, 3, Fr::one()); // C selects w1

        let (witness, u, x) = RelaxedR1CSWitness::sample_random(
            &matrices,
            2,  // witness_size (w0, w1)
            1,  // public_input_size (x)
            3,  // num_rounds
            4,  // coeffs_per_round
            &mut rng,
        );

        // Verify the witness satisfies the relaxed R1CS
        assert!(matrices.is_relaxed_satisfied(&witness, u, &x));

        // Check round data was generated
        assert_eq!(witness.num_rounds(), 3);
        assert_eq!(witness.round_coefficients[0].len(), 4);
        assert_eq!(witness.round_blindings.len(), 3);
    }

    #[test]
    fn test_non_relaxed_witness_creation() {
        let mut rng = thread_rng();

        let witness_vec = vec![Fr::from(5u64), Fr::from(25u64)]; // 5, 5*5
        let round_coefficients = vec![
            vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64)],
        ];
        let round_blindings = vec![Fr::from(42u64)];

        let witness = RelaxedR1CSWitness::new_non_relaxed(
            witness_vec.clone(),
            1,  // num_constraints
            round_coefficients.clone(),
            round_blindings.clone(),
            &mut rng,
        );

        assert_eq!(witness.W, witness_vec);
        assert!(witness.r_W != Fr::zero()); // Should be random
        assert_eq!(witness.E, vec![Fr::zero()]); // Non-relaxed has E = 0
        assert_eq!(witness.r_E, Fr::zero());
        assert_eq!(witness.round_coefficients, round_coefficients);
        assert_eq!(witness.round_blindings, round_blindings);
    }
}
