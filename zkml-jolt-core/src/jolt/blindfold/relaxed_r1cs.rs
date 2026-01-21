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

use jolt_core::{
    field::JoltField,
    poly::commitment::commitment_scheme::CommitmentScheme,
};
use std::marker::PhantomData;

/// A committed relaxed R1CS instance.
///
/// This contains the public components visible to the verifier:
/// - Commitment to the witness W
/// - Commitment to the error vector E
/// - The scalar u
/// - The public inputs x
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
            _marker: PhantomData,
        }
    }
}

/// The witness for a relaxed R1CS instance.
///
/// Contains the private components known only to the prover:
/// - The witness vector W
/// - The error vector E
#[derive(Clone, Debug)]
pub struct RelaxedR1CSWitness<F: JoltField> {
    /// The witness vector
    pub W: Vec<F>,
    /// The error vector (zero for satisfying instances)
    pub E: Vec<F>,
}

impl<F: JoltField> RelaxedR1CSWitness<F> {
    /// Creates a new relaxed R1CS witness.
    pub fn new(W: Vec<F>, E: Vec<F>) -> Self {
        Self { W, E }
    }

    /// Creates a standard (non-relaxed) witness from a witness vector.
    ///
    /// For a standard satisfying instance, E = 0.
    pub fn from_standard(W: Vec<F>, constraint_count: usize) -> Self {
        Self {
            W,
            E: vec![F::zero(); constraint_count],
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
    use jolt_core::field::JoltField;

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
    }
}
