//! 4-dimensional scalar decomposition for BN254 G2
//!
//! This module implements the table-based scalar decomposition algorithm
//! that decomposes a scalar k into k = k0 + k1*λ + k2*λ² + k3*λ³
//! where λ is the Frobenius powers endo eigenvalue.

use crate::constants::POWER_OF_2_DECOMPOSITIONS;
use ark_bn254::Fr;
use ark_ff::{BigInteger, PrimeField};
use num_bigint::{BigInt, Sign};

/// Convert u128 to Fr field element
pub fn u128_to_fr(val: u128) -> Fr {
    let bytes = val.to_be_bytes();
    Fr::from_be_bytes_mod_order(&bytes)
}

/// Convert Fr field element to BigInt
pub fn fr_to_bigint(fr: Fr) -> BigInt {
    let bytes = fr.into_bigint().to_bytes_be();
    BigInt::from_bytes_be(Sign::Plus, &bytes)
}

/// Convert BigInt to Fr field element
pub fn bigint_to_fr(big: &BigInt) -> Fr {
    let (sign, bytes) = big.to_bytes_be();

    if sign == Sign::Minus {
        let r_bytes = Fr::MODULUS.to_bytes_be();
        let r_big = BigInt::from_bytes_be(Sign::Plus, &r_bytes);
        let positive_big = r_big - BigInt::from_bytes_be(Sign::Plus, &bytes);
        let (_sign, positive_bytes) = positive_big.to_bytes_be();
        Fr::from_be_bytes_mod_order(&positive_bytes)
    } else {
        Fr::from_be_bytes_mod_order(&bytes)
    }
}

/// Table-based 4-dimensional scalar decomposition
///
/// Decomposes a scalar k into (k0, k1, k2, k3) such that:
/// k ≡ k0 + k1*λ + k2*λ² + k3*λ³ (mod r)
///
/// Returns the coefficients and their sign flags.
/// Each coefficient is guaranteed to be at most ~66 bits.
/// If we do lattice reduction directly we may be able to save 1 bit on the decomposition
pub fn decompose_scalar_table_based(scalar: &BigInt) -> ([u128; 4], [bool; 4]) {
    let mut k0 = 0u128;
    let mut k1 = 0u128;
    let mut k2 = 0u128;
    let mut k3 = 0u128;

    // Find all set bits in the scalar
    let mut temp_scalar = scalar.clone();
    let mut bit_position = 0;

    while temp_scalar > BigInt::from(0) && bit_position < 254 {
        if &temp_scalar & BigInt::from(1) == BigInt::from(1) {
            // This bit is set, add the corresponding decomposition
            let (decomp_k0, decomp_k1, decomp_k2, decomp_k3, neg0, neg1, neg2, neg3) =
                POWER_OF_2_DECOMPOSITIONS[bit_position];

            if neg0 {
                k0 = k0.wrapping_sub(decomp_k0);
            } else {
                k0 = k0.wrapping_add(decomp_k0);
            }
            if neg1 {
                k1 = k1.wrapping_sub(decomp_k1);
            } else {
                k1 = k1.wrapping_add(decomp_k1);
            }
            if neg2 {
                k2 = k2.wrapping_sub(decomp_k2);
            } else {
                k2 = k2.wrapping_add(decomp_k2);
            }
            if neg3 {
                k3 = k3.wrapping_sub(decomp_k3);
            } else {
                k3 = k3.wrapping_add(decomp_k3);
            }
        }

        temp_scalar >>= 1;
        bit_position += 1;
    }

    // Handle signs by making coefficients positive and tracking negation flags
    let (final_k0, neg_flag0) = if (k0 as i128) < 0 {
        (k0.wrapping_neg(), true)
    } else {
        (k0, false)
    };

    let (final_k1, neg_flag1) = if (k1 as i128) < 0 {
        (k1.wrapping_neg(), true)
    } else {
        (k1, false)
    };

    let (final_k2, neg_flag2) = if (k2 as i128) < 0 {
        (k2.wrapping_neg(), true)
    } else {
        (k2, false)
    };

    let (final_k3, neg_flag3) = if (k3 as i128) < 0 {
        (k3.wrapping_neg(), true)
    } else {
        (k3, false)
    };

    (
        [final_k0, final_k1, final_k2, final_k3],
        [neg_flag0, neg_flag1, neg_flag2, neg_flag3],
    )
}

/// Get the maximum bit length of the decomposed coefficients
pub fn get_max_coefficient_bits(coeffs: &[u128; 4]) -> usize {
    coeffs
        .iter()
        .map(|k| (128 - k.leading_zeros()) as usize)
        .max()
        .unwrap_or(0)
}

/// Decompose scalar for 4D GLV multiplication
/// Returns coefficients as BigInts and their signs
pub fn decompose_scalar_4d(scalar: Fr) -> ([<Fr as PrimeField>::BigInt; 4], [bool; 4]) {
    let scalar_bigint = fr_to_bigint(scalar);
    let (coeffs_u128, signs) = decompose_scalar_table_based(&scalar_bigint);

    // Convert u128 coefficients to BigInt
    let coeffs = [
        Fr::from(coeffs_u128[0]).into_bigint(),
        Fr::from(coeffs_u128[1]).into_bigint(),
        Fr::from(coeffs_u128[2]).into_bigint(),
        Fr::from(coeffs_u128[3]).into_bigint(),
    ];

    (coeffs, signs)
}
