//! Dory protocol vector operations for G2
//! Implements v[i] = v[i] + scalar * g[i] and v[i] = scalar * v[i] + gamma[i]

use ark_bn254::{Fr, G2Projective};
use rayon::prelude::*;

use crate::decomp_4d::decompose_scalar_4d;
use crate::frobenius::frobenius_psi_power_projective;
use crate::glv_four::{shamir_glv_mul_4d, shamir_glv_mul_4d_precomputed};
use crate::{
    glv_four_precompute, glv_four_precompute_windowed2_signed, glv_four_scalar_mul_online,
    glv_four_scalar_mul_windowed2_signed, PrecomputedShamir4Data, PrecomputedShamir4Table,
    Windowed2Signed4Data,
};

// ============================================================================
// Operation 1: v[i] = v[i] + scalar * g[i]
// ============================================================================

/// Online version:
pub fn vector_add_scalar_mul_g2_online(
    v: &mut [G2Projective],
    generators: &[G2Projective],
    scalar: Fr,
) {
    assert_eq!(v.len(), generators.len());
    let (coeffs, signs) = decompose_scalar_4d(scalar);

    v.par_iter_mut()
        .zip(generators.par_iter())
        .for_each(|(vi, gen)| {
            let bases = [
                *gen,
                frobenius_psi_power_projective(gen, 1),
                frobenius_psi_power_projective(gen, 2),
                frobenius_psi_power_projective(gen, 3),
            ];
            *vi += shamir_glv_mul_4d(&bases, &coeffs, &signs);
        });
}

/// Precomputed full
pub fn vector_add_scalar_mul_g2_precomputed(
    v: &mut [G2Projective],
    scalar: Fr,
    precomputed_tables: &[PrecomputedShamir4Table],
) {
    assert_eq!(v.len(), precomputed_tables.len());
    let (coeffs, signs) = decompose_scalar_4d(scalar);

    v.par_iter_mut()
        .zip(precomputed_tables.par_iter())
        .for_each(|(vi, table)| {
            *vi += shamir_glv_mul_4d_precomputed(table, &coeffs, &signs);
        });
}

/// 2-bit signed precomputed
pub fn vector_add_scalar_mul_g2_windowed2_signed(
    v: &mut [G2Projective],
    scalar: Fr,
    precomputed_generators: &Windowed2Signed4Data,
) {
    assert_eq!(v.len(), precomputed_generators.windowed2_tables.len());

    // Use the GLV scalar multiplication on the generators
    let products = glv_four_scalar_mul_windowed2_signed(precomputed_generators, scalar);

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
pub fn vector_scalar_mul_add_gamma_g2_online(
    v: &mut [G2Projective],
    scalar: Fr,
    gamma: &[G2Projective],
) {
    assert_eq!(v.len(), gamma.len());
    let (coeffs, signs) = decompose_scalar_4d(scalar);

    v.par_iter_mut()
        .zip(gamma.par_iter())
        .for_each(|(vi, &gamma_i)| {
            let bases = [
                *vi,
                frobenius_psi_power_projective(vi, 1),
                frobenius_psi_power_projective(vi, 2),
                frobenius_psi_power_projective(vi, 3),
            ];
            *vi = shamir_glv_mul_4d(&bases, &coeffs, &signs) + gamma_i;
        });
}

/// Precomputed full
pub fn vector_scalar_mul_add_gamma_g2_precomputed(
    v: &mut [G2Projective],
    scalar: Fr,
    gamma: &[G2Projective],
) {
    // For this operation, we can't precompute on v since it's being modified
    // So we just use the online version
    vector_scalar_mul_add_gamma_g2_online(v, scalar, gamma);
}

/// 2-bit signed
/// Note: We can't precompute on v since it changes, so this uses online scalar mul
pub fn vector_scalar_mul_add_gamma_g2_windowed2_signed(
    v: &mut [G2Projective],
    scalar: Fr,
    gamma: &[G2Projective],
) {
    assert_eq!(v.len(), gamma.len());

    // Compute scalar * v[i] for all i using online method
    let products = glv_four_scalar_mul_online(scalar, v);

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

/// Precompute Shamir tables for a set of G2 generators
pub fn precompute_g2_generators(generators: &[G2Projective]) -> PrecomputedShamir4Data {
    glv_four_precompute(generators)
}

/// Precompute 2-bit signed tables for a set of G2 generators
pub fn precompute_g2_generators_windowed2_signed(
    generators: &[G2Projective],
) -> Windowed2Signed4Data {
    glv_four_precompute_windowed2_signed(generators)
}
