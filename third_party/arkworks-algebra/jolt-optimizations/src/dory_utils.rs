//! Dory-specific optimized utilities for vector scalar multiplication

use ark_bn254::{Fr, G2Projective};
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::decomp_4d::{decompose_scalar_table_based, fr_to_bigint, u128_to_fr};
use crate::frobenius::frobenius_psi_power_projective;
use crate::glv_four::{
    shamir_glv_mul_4d_precomputed, PrecomputedShamir4Data, PrecomputedShamir4Table,
};

/// Helper function to decompose a scalar into 4D GLV form
fn decompose_scalar(scalar: Fr) -> ([<Fr as PrimeField>::BigInt; 4], [bool; 4]) {
    let scalar_bigint = fr_to_bigint(scalar);
    let (coeffs, signs) = decompose_scalar_table_based(&scalar_bigint);

    let k_bigint = [
        u128_to_fr(coeffs[0]).into_bigint(),
        u128_to_fr(coeffs[1]).into_bigint(),
        u128_to_fr(coeffs[2]).into_bigint(),
        u128_to_fr(coeffs[3]).into_bigint(),
    ];

    (k_bigint, signs)
}

/// Precomputed data for efficient vector scalar multiplication with a fixed scalar
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct VectorScalarMulData {
    /// Decomposed scalar coefficients
    pub scalar_coeffs: [<Fr as PrimeField>::BigInt; 4],
    /// Signs for each coefficient
    pub scalar_signs: [bool; 4],
    /// Precomputed Shamir tables for each generator
    pub precomputed_data: PrecomputedShamir4Data,
}

impl VectorScalarMulData {
    /// Create precomputed data for vector scalar multiplication
    pub fn new(generators: &[G2Projective], scalar: Fr) -> Self {
        // Decompose the scalar once
        let (scalar_coeffs, scalar_signs) = decompose_scalar(scalar);

        // Precompute Shamir tables for all generators
        let precomputed_data = PrecomputedShamir4Data::new(generators);

        Self {
            scalar_coeffs,
            scalar_signs,
            precomputed_data,
        }
    }

    /// Get the number of generators this data was created for
    pub fn num_generators(&self) -> usize {
        self.precomputed_data.shamir_tables.len()
    }
}

/// Perform vector scalar multiplication and addition using precomputed data
pub fn vector_scalar_mul_add_precomputed(v: &mut [G2Projective], data: &VectorScalarMulData) {
    assert_eq!(
        v.len(),
        data.num_generators(),
        "Vector length must match number of precomputed generators"
    );

    use rayon::prelude::*;

    // Perform scalar multiplication and addition in parallel
    v.par_iter_mut().enumerate().for_each(|(i, v_point)| {
        let scalar_mul_result = shamir_glv_mul_4d_precomputed(
            &data.precomputed_data.shamir_tables[i],
            &data.scalar_coeffs,
            &data.scalar_signs,
        );
        *v_point += scalar_mul_result;
    });
}

/// Perform vector scalar multiplication and addition online (without precomputation)
pub fn vector_scalar_mul_add_online(
    v: &mut [G2Projective],
    generators: &[G2Projective],
    scalar: Fr,
) {
    assert_eq!(
        v.len(),
        generators.len(),
        "Vector and generators must have the same length"
    );

    use rayon::prelude::*;

    // Decompose the scalar once
    let (scalar_coeffs, scalar_signs) = decompose_scalar(scalar);

    // Perform scalar multiplication and addition in parallel
    v.par_iter_mut()
        .zip(generators.par_iter())
        .for_each(|(v_point, generator)| {
            // Compute Frobenius bases for this generator
            let frobenius_bases = [
                *generator,
                frobenius_psi_power_projective(generator, 1),
                frobenius_psi_power_projective(generator, 2),
                frobenius_psi_power_projective(generator, 3),
            ];

            // Create temporary Shamir table
            let shamir_table = PrecomputedShamir4Table::new(&frobenius_bases);

            // Perform scalar multiplication and add to existing value
            let scalar_mul_result =
                shamir_glv_mul_4d_precomputed(&shamir_table, &scalar_coeffs, &scalar_signs);
            *v_point += scalar_mul_result;
        });
}

/// Convenience function to create and use precomputed data in one call
pub fn vector_scalar_mul_add(v: &mut [G2Projective], generators: &[G2Projective], scalar: Fr) {
    let data = VectorScalarMulData::new(generators, scalar);
    vector_scalar_mul_add_precomputed(v, &data);
}

/// Precomputed data for efficient vector scalar multiplication where we scale the vector elements
/// and add generators: v[i] = scalar * v[i] + generators[i]
#[derive(Clone, Debug)]
pub struct VectorScalarMulVData {
    /// Decomposed scalar coefficients
    pub scalar_coeffs: [<Fr as PrimeField>::BigInt; 4],
    /// Signs for each coefficient
    pub scalar_signs: [bool; 4],
}

impl VectorScalarMulVData {
    /// Create precomputed scalar decomposition for vector element scaling
    pub fn new(scalar: Fr) -> Self {
        let (scalar_coeffs, scalar_signs) = decompose_scalar(scalar);

        Self {
            scalar_coeffs,
            scalar_signs,
        }
    }
}

/// Perform vector scalar multiplication with vector scaling using precomputed data
pub fn vector_scalar_mul_v_add_g_precomputed(
    v: &mut [G2Projective],
    generators: &[G2Projective],
    data: &VectorScalarMulVData,
) {
    assert_eq!(
        v.len(),
        generators.len(),
        "Vector and generators must have the same length"
    );

    use rayon::prelude::*;

    // Perform scalar multiplication and addition in parallel
    v.par_iter_mut()
        .zip(generators.par_iter())
        .for_each(|(v_point, generator)| {
            // Compute Frobenius bases for current vector element
            let frobenius_bases = [
                *v_point,
                frobenius_psi_power_projective(v_point, 1),
                frobenius_psi_power_projective(v_point, 2),
                frobenius_psi_power_projective(v_point, 3),
            ];

            // Create temporary Shamir table for v_point
            let shamir_table = PrecomputedShamir4Table::new(&frobenius_bases);

            // Perform scalar multiplication: scalar * v[i] + generators[i]
            let v_scaled = shamir_glv_mul_4d_precomputed(
                &shamir_table,
                &data.scalar_coeffs,
                &data.scalar_signs,
            );
            *v_point = v_scaled + generator;
        });
}

/// Perform vector scalar multiplication with vector scaling online (without precomputation)
pub fn vector_scalar_mul_v_add_g_online(
    v: &mut [G2Projective],
    generators: &[G2Projective],
    scalar: Fr,
) {
    assert_eq!(
        v.len(),
        generators.len(),
        "Vector and generators must have the same length"
    );

    let data = VectorScalarMulVData::new(scalar);
    vector_scalar_mul_v_add_g_precomputed(v, generators, &data);
}
