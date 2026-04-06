//! 2D GLV scalar multiplication implementations for BN254 G1
//!
//! This module provides optimized scalar multiplication algorithms for BN254 G1 using
//! 2-dimensional GLV decomposition with the Shamir trick and precomputed lookup tables.

use ark_bn254::{Fq, Fr, G1Projective};
use ark_ff::{BigInteger, MontFp, PrimeField};
use ark_std::ops::{AddAssign, Neg};
use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{One, Signed};

/// GLV lambda for BN254 G1
const _LAMBDA: Fr =
    MontFp!("21888242871839275217838484774961031246154997185409878258781734729429964517155");

/// GLV endomorphism coefficient for BN254 G1
const ENDO_COEFF: Fq =
    MontFp!("21888242871839275220042445260109153167277707414472061641714758635765020556616");

/// GLV scalar decomposition coefficients for BN254 G1
const SCALAR_DECOMP_COEFFS: [(bool, <Fr as PrimeField>::BigInt); 4] = [
    (
        false,
        ark_ff::BigInt!("147946756881789319000765030803803410728"),
    ),
    (true, ark_ff::BigInt!("9931322734385697763")),
    (false, ark_ff::BigInt!("9931322734385697763")),
    (
        false,
        ark_ff::BigInt!("147946756881789319010696353538189108491"),
    ),
];

/// Helper function to decompose a scalar using 2D GLV
pub fn decompose_scalar_2d(scalar: Fr) -> ([<Fr as PrimeField>::BigInt; 2], [bool; 2]) {
    // Convert to num_bigint::BigInt for arithmetic
    let scalar_bytes = scalar.into_bigint().to_bytes_be();
    let scalar_bigint = BigInt::from_bytes_be(Sign::Plus, &scalar_bytes);

    let coeff_bigints: [BigInt; 4] = SCALAR_DECOMP_COEFFS.map(|x| {
        let bytes = x.1.to_bytes_be();
        BigInt::from_bytes_be(x.0.then_some(Sign::Plus).unwrap_or(Sign::Minus), &bytes)
    });

    let [n11, n12, n21, n22] = coeff_bigints;

    let r_bytes = Fr::MODULUS.to_bytes_be();
    let r = BigInt::from_bytes_be(Sign::Plus, &r_bytes);

    // beta = vector([k,0]) * self.curve.N_inv
    // The inverse of N is 1/r * Matrix([[n22, -n12], [-n21, n11]]).
    // so β = (k*n22, -k*n12)/r

    let beta_1 = {
        let (mut div, rem) = (&scalar_bigint * &n22).div_rem(&r);
        if (&rem + &rem) > r {
            div.add_assign(BigInt::one());
        }
        div
    };
    let beta_2 = {
        let (mut div, rem) = (&scalar_bigint * &n12.clone().neg()).div_rem(&r);
        if (&rem + &rem) > r {
            div.add_assign(BigInt::one());
        }
        div
    };

    // b = vector([int(beta[0]), int(beta[1])]) * self.curve.N
    // b = (β1N11 + β2N21, β1N12 + β2N22) with the signs!
    //   = (b11   + b12  , b21   + b22)   with the signs!

    // b1
    let b11 = &beta_1 * &n11;
    let b12 = &beta_2 * &n21;
    let b1 = b11 + b12;

    // b2
    let b21 = &beta_1 * &n12;
    let b22 = &beta_2 * &n22;
    let b2 = b21 + b22;

    let k1 = &scalar_bigint - b1;
    let k1_abs = BigUint::try_from(k1.abs()).unwrap();

    // k2
    let k2 = -b2;
    let k2_abs = BigUint::try_from(k2.abs()).unwrap();

    // Convert back to arkworks BigInt
    let k1_fr = Fr::from(k1_abs);
    let k2_fr = Fr::from(k2_abs);

    let k_bigint = [k1_fr.into_bigint(), k2_fr.into_bigint()];

    let signs = [k1.sign() == Sign::Plus, k2.sign() == Sign::Plus];

    (k_bigint, signs)
}

/// Apply GLV endomorphism to G1 point
pub fn glv_endomorphism(point: &G1Projective) -> G1Projective {
    let mut res = *point;
    res.x *= ENDO_COEFF;
    res
}
