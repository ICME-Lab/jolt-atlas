//! Dory protocol vector operations for G1
//! Implements v[i] = v[i] + scalar * g[i] ; v[i] = scalar * v[i] + gamma[i]
//! Using GLV + strauss-shamir's trick extended to 4 scalars
//! https://crypto.stackexchange.com/questions/99975/strauss-shamir-trick-on-ec-multiplication-by-scalar

use ark_bn254::{Fr, G1Projective};
use rayon::prelude::*;

use crate::decomp_2d::{decompose_scalar_2d, glv_endomorphism};
use crate::glv_two::{shamir_glv_mul_2d, shamir_glv_mul_2d_precomputed};
use crate::{
    glv_two_precompute, glv_two_precompute_windowed2_signed, glv_two_scalar_mul_online,
    glv_two_scalar_mul_windowed2_signed, PrecomputedShamir2Data, PrecomputedShamir2Table,
    Windowed2Signed2Data,
};

// ============================================================================
// Operation 1: v[i] = v[i] + scalar * g[i]
// ============================================================================

/// Online version
pub fn vector_add_scalar_mul_g1_online(
    v: &mut [G1Projective],
    generators: &[G1Projective],
    scalar: Fr,
) {
    assert_eq!(v.len(), generators.len());
    let (coeffs, signs) = decompose_scalar_2d(scalar);

    v.par_iter_mut()
        .zip(generators.par_iter())
        .for_each(|(vi, gen)| {
            let bases = [*gen, glv_endomorphism(gen)];
            *vi += shamir_glv_mul_2d(&bases, &coeffs, &signs);
        });
}

/// Precomputed full
pub fn vector_add_scalar_mul_g1_precomputed(
    v: &mut [G1Projective],
    scalar: Fr,
    precomputed_tables: &[PrecomputedShamir2Table],
) {
    assert_eq!(v.len(), precomputed_tables.len());
    let (coeffs, signs) = decompose_scalar_2d(scalar);

    v.par_iter_mut()
        .zip(precomputed_tables.par_iter())
        .for_each(|(vi, table)| {
            *vi += shamir_glv_mul_2d_precomputed(table, &coeffs, &signs);
        });
}

/// 2-bit signed precomputed
pub fn vector_add_scalar_mul_g1_windowed2_signed(
    v: &mut [G1Projective],
    scalar: Fr,
    precomputed_generators: &Windowed2Signed2Data,
) {
    assert_eq!(v.len(), precomputed_generators.windowed2_tables.len());

    // Use the GLV scalar multiplication on the generators
    let products = glv_two_scalar_mul_windowed2_signed(precomputed_generators, scalar);

    // Add products to v
    v.par_iter_mut()
        .zip(products.par_iter())
        .for_each(|(vi, &prod)| {
            *vi += prod;
        });
}

// ============================================================================
// Operation 2: v[i] = scalar * v[i] + gamma[i]
// ============================================================================

/// Online
pub fn vector_scalar_mul_add_gamma_g1_online(
    v: &mut [G1Projective],
    scalar: Fr,
    gamma: &[G1Projective],
) {
    assert_eq!(v.len(), gamma.len());
    let (coeffs, signs) = decompose_scalar_2d(scalar);

    v.par_iter_mut()
        .zip(gamma.par_iter())
        .for_each(|(vi, &gamma_i)| {
            let bases = [*vi, glv_endomorphism(vi)];
            *vi = shamir_glv_mul_2d(&bases, &coeffs, &signs) + gamma_i;
        });
}

/// Precomputed method
/// Note: We can't precompute on v since it changes, so this just uses online method
pub fn vector_scalar_mul_add_gamma_g1_precomputed(
    v: &mut [G1Projective],
    scalar: Fr,
    gamma: &[G1Projective],
) {
    // For this operation, we can't precompute on v since it's being modified
    // So we just use the online version
    vector_scalar_mul_add_gamma_g1_online(v, scalar, gamma);
}

/// 2-bit signed precomputed
/// Note: We can't precompute on v since it changes, so this uses online scalar mul
pub fn vector_scalar_mul_add_gamma_g1_windowed2_signed(
    v: &mut [G1Projective],
    scalar: Fr,
    gamma: &[G1Projective],
) {
    assert_eq!(v.len(), gamma.len());

    // Compute scalar * v[i] for all i using online method
    let products = glv_two_scalar_mul_online(scalar, v);

    // Replace v with products + gamma
    v.par_iter_mut()
        .zip(products.par_iter())
        .zip(gamma.par_iter())
        .for_each(|((vi, &prod), &gamma_i)| {
            *vi = prod + gamma_i;
        });
}

// ============================================================================
// Helper functions for precomputation
// ============================================================================

/// Precompute Shamir tables for a set of G1 generators
pub fn precompute_g1_generators(generators: &[G1Projective]) -> PrecomputedShamir2Data {
    glv_two_precompute(generators)
}

/// Precompute 2-bit signed tables for a set of G1 generators
pub fn precompute_g1_generators_windowed2_signed(
    generators: &[G1Projective],
) -> Windowed2Signed2Data {
    glv_two_precompute_windowed2_signed(generators)
}
