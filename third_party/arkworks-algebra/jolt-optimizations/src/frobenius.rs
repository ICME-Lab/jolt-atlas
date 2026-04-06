//! Frobenius endomorphism operations for BN254 G2
//!
//! This module provides functionality for computing Frobenius endomorphisms
//! ψ^k on BN254 G2 curve points.

use crate::constants::get_frobenius_coefficients;
use ark_bn254::{G2Affine, G2Projective};
use ark_ec::{AffineRepr, CurveGroup};
use ark_std::Zero;

/// Compute the Frobenius endomorphism ψ^k for BN254 G2 (projective version)
///
/// Implements the Frobenius endomorphism manually as conjugation in Fq2
/// followed by multiplication by precomputed coefficients.
pub fn frobenius_psi_power_projective(p: &G2Projective, k: usize) -> G2Projective {
    if p.is_zero() {
        return *p;
    }

    let mut res = *p;
    let coeffs = get_frobenius_coefficients();

    // Apply Frobenius map to coordinates (conjugation in Fq2)
    if (k & 1) == 1 {
        // Odd power - apply conjugation
        res.x.conjugate_in_place();
        res.y.conjugate_in_place();
        res.z.conjugate_in_place();
    }
    // Even power - identity (no conjugation)

    // Apply the coefficients based on power
    match k % 4 {
        0 => res,
        1 => {
            res.x *= coeffs.psi1_coef2;
            res.y *= coeffs.psi1_coef3;
            res
        },
        2 => {
            res.x *= coeffs.psi2_coef2;
            res.y *= coeffs.psi2_coef3;
            res
        },
        3 => {
            res.x *= coeffs.psi3_coef2;
            res.y *= coeffs.psi3_coef3;
            res
        },
        _ => unreachable!(),
    }
}

/// Compute the Frobenius endomorphism ψ^k for BN254 G2
pub fn frobenius_psi_power_affine(p: &G2Affine, k: usize) -> G2Affine {
    let projective_result = frobenius_psi_power_projective(&p.into_group(), k);
    projective_result.into_affine()
}
