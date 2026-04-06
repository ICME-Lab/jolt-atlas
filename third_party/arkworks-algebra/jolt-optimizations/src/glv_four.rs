//! 4D GLV scalar multiplication for BN254 G2
//! Three methods: (1) online, (2) precomputed full, (3) signed table

use ark_bn254::{Fr, G2Projective};
use ark_ec::AdditiveGroup;
use ark_ff::{BigInteger, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::Zero;
use rayon::prelude::*;

use crate::decomp_4d::decompose_scalar_4d;
use crate::frobenius::frobenius_psi_power_projective;

/// Online 4D GLV scalar multiplication
pub fn glv_four_scalar_mul_online(scalar: Fr, points: &[G2Projective]) -> Vec<G2Projective> {
    let (coeffs, signs) = decompose_scalar_4d(scalar);

    points
        .par_iter()
        .map(|point| {
            // Compute Frobenius powers: P, ψ(P), ψ²(P), ψ³(P)
            let bases = [
                *point,
                frobenius_psi_power_projective(point, 1),
                frobenius_psi_power_projective(point, 2),
                frobenius_psi_power_projective(point, 3),
            ];

            // Shamir's trick with sign handling
            shamir_glv_mul_4d(&bases, &coeffs, &signs)
        })
        .collect()
}

/// Shamir's trick for 4-point scalar multiplication with signs
pub(crate) fn shamir_glv_mul_4d(
    bases: &[G2Projective; 4],
    coeffs: &[<Fr as PrimeField>::BigInt; 4],
    signs: &[bool; 4],
) -> G2Projective {
    let mut result = G2Projective::zero();
    let max_bits = coeffs
        .iter()
        .map(|c| c.num_bits() as usize)
        .max()
        .unwrap_or(0);

    for bit_idx in (0..max_bits).rev() {
        result = result.double();

        // Check bits and accumulate
        for (i, coeff) in coeffs.iter().enumerate() {
            if coeff.get_bit(bit_idx) {
                if signs[i] {
                    // signs[i] = true means negative in 4D decomposition
                    result -= bases[i];
                } else {
                    // signs[i] = false means positive in 4D decomposition
                    result += bases[i];
                }
            }
        }
    }

    result
}

/// Precomputed data for 4D GLV with Shamir table
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PrecomputedShamir4Data {
    pub shamir_tables: Vec<PrecomputedShamir4Table>,
}

/// Shamir lookup table: all 256 combinations for [P, ψ(P), ψ²(P), ψ³(P)] with signs
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PrecomputedShamir4Table {
    pub table: Vec<G2Projective>, // 2^4 points × 2^4 signs (256 elements)
}

impl PrecomputedShamir4Table {
    /// Create table for [P, ψ(P), ψ²(P), ψ³(P)] with all sign combinations
    pub fn new(bases: &[G2Projective; 4]) -> Self {
        let mut table = vec![G2Projective::zero(); 256];

        table.par_iter_mut().enumerate().for_each(|(idx, point)| {
            let point_mask = idx & 0xF; // Which points to include
            let sign_mask = idx >> 4; // Which points to negate

            *point = G2Projective::zero();
            for i in 0..4 {
                if (point_mask >> i) & 1 == 1 {
                    if (sign_mask >> i) & 1 == 1 {
                        *point -= bases[i];
                    } else {
                        *point += bases[i];
                    }
                }
            }
        });

        Self { table }
    }

    #[inline]
    pub fn get(&self, point_mask: usize, sign_mask: usize) -> G2Projective {
        self.table[point_mask | (sign_mask << 4)]
    }
}

impl PrecomputedShamir4Data {
    pub fn new(points: &[G2Projective]) -> Self {
        let shamir_tables = points
            .par_iter()
            .map(|point| {
                let frobenius_bases = [
                    *point,
                    frobenius_psi_power_projective(point, 1),
                    frobenius_psi_power_projective(point, 2),
                    frobenius_psi_power_projective(point, 3),
                ];
                PrecomputedShamir4Table::new(&frobenius_bases)
            })
            .collect();

        Self { shamir_tables }
    }
}

/// Precompute for multiple points
pub fn glv_four_precompute(points: &[G2Projective]) -> PrecomputedShamir4Data {
    PrecomputedShamir4Data::new(points)
}

/// Scalar multiplication using precomputed data
pub fn glv_four_scalar_mul(data: &PrecomputedShamir4Data, scalar: Fr) -> Vec<G2Projective> {
    let (coeffs, signs) = decompose_scalar_4d(scalar);

    data.shamir_tables
        .par_iter()
        .map(|table| shamir_glv_mul_4d_precomputed(table, &coeffs, &signs))
        .collect()
}

/// Shamir's trick using precomputed table
pub(crate) fn shamir_glv_mul_4d_precomputed(
    table: &PrecomputedShamir4Table,
    coeffs: &[<Fr as PrimeField>::BigInt; 4],
    signs: &[bool; 4],
) -> G2Projective {
    let mut result = G2Projective::zero();
    let max_bits = coeffs
        .iter()
        .map(|c| c.num_bits() as usize)
        .max()
        .unwrap_or(0);

    for bit_idx in (0..max_bits).rev() {
        result = result.double();

        // Build masks for table lookup
        let mut point_mask = 0;
        let mut sign_mask = 0;

        for i in 0..4 {
            if coeffs[i].get_bit(bit_idx) {
                point_mask |= 1 << i;
                if signs[i] {
                    // signs[i] = true means negative in 4D decomposition
                    sign_mask |= 1 << i;
                }
            }
        }

        if point_mask != 0 {
            result += table.get(point_mask, sign_mask);
        }
    }

    result
}

/// Precomputed data for 2-bit windowed signed method
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct Windowed2Signed4Data {
    pub windowed2_tables: Vec<Windowed2Signed4Table>,
}

/// 2-bit signed table: stores [±P, ±2P, ±3P] for each base
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct Windowed2Signed4Table {
    pub signed_multiples: Vec<G2Projective>, // 4 bases × 6 variants (24 elements)
}

impl Windowed2Signed4Table {
    /// Create signed multiples for 2-bit processing
    pub fn new(bases: &[G2Projective; 4]) -> Self {
        let mut signed_multiples = vec![G2Projective::zero(); 24];

        for (base_idx, &base) in bases.iter().enumerate() {
            let offset = base_idx * 6;
            signed_multiples[offset] = base; // 1*base
            signed_multiples[offset + 1] = base.double(); // 2*base
            signed_multiples[offset + 2] = base + base.double(); // 3*base
            signed_multiples[offset + 3] = -base; // -1*base
            signed_multiples[offset + 4] = -base.double(); // -2*base
            signed_multiples[offset + 5] = -(base + base.double()); // -3*base
        }

        Self { signed_multiples }
    }

    /// Get 2-bit windowed combination
    #[inline]
    pub fn get_windowed2(&self, coeffs: &[i8; 4]) -> G2Projective {
        let mut result = G2Projective::zero();

        for (i, &coeff) in coeffs.iter().enumerate() {
            if coeff != 0 {
                let abs_coeff = coeff.abs() as usize;
                if abs_coeff <= 3 {
                    let idx = i * 6 + abs_coeff - 1 + if coeff < 0 { 3 } else { 0 };
                    result += self.signed_multiples[idx];
                }
            }
        }

        result
    }
}

impl Windowed2Signed4Data {
    pub fn new(points: &[G2Projective]) -> Self {
        let windowed2_tables = points
            .par_iter()
            .map(|point| {
                let frobenius_bases = [
                    *point,
                    frobenius_psi_power_projective(point, 1),
                    frobenius_psi_power_projective(point, 2),
                    frobenius_psi_power_projective(point, 3),
                ];
                Windowed2Signed4Table::new(&frobenius_bases)
            })
            .collect();

        Self { windowed2_tables }
    }
}

/// Precompute for 2-bit windowed signed method
pub fn glv_four_precompute_windowed2_signed(points: &[G2Projective]) -> Windowed2Signed4Data {
    Windowed2Signed4Data::new(points)
}

/// Scalar multiplication using 2-bit windowed signed method
pub fn glv_four_scalar_mul_windowed2_signed(
    data: &Windowed2Signed4Data,
    scalar: Fr,
) -> Vec<G2Projective> {
    let (coeffs, signs) = decompose_scalar_4d(scalar);

    data.windowed2_tables
        .par_iter()
        .map(|table| glv_four_scalar_mul_windowed2_signed_single(table, &coeffs, &signs))
        .collect()
}

/// 2-bit windowed signed multiplication for single point
fn glv_four_scalar_mul_windowed2_signed_single(
    table: &Windowed2Signed4Table,
    coeffs: &[<Fr as PrimeField>::BigInt; 4],
    signs: &[bool; 4],
) -> G2Projective {
    // Convert scalars to signed coefficients for 2-bit windowed processing
    let scalar_coeffs: Vec<Vec<i8>> = coeffs
        .par_iter()
        .zip(signs.par_iter())
        .map(|(scalar, &is_negative)| {
            let mut coeffs = Vec::new();
            let scalar_ref = scalar.as_ref();

            // Convert to base-4 coefficients (2 bits at a time)
            for limb in scalar_ref {
                for window_idx in 0..32 {
                    // 64 bits / 2 = 32 windows per limb
                    let window_bits = (limb >> (window_idx * 2)) & 0x3;
                    let signed_coeff = if is_negative {
                        -(window_bits as i8)
                    } else {
                        window_bits as i8
                    };
                    coeffs.push(signed_coeff);
                }
            }

            coeffs
        })
        .collect();

    // Find maximum coefficient length
    let max_coeffs = scalar_coeffs
        .iter()
        .map(|coeffs| {
            coeffs
                .iter()
                .rposition(|&c| c != 0)
                .map(|pos| pos + 1)
                .unwrap_or(0)
        })
        .max()
        .unwrap_or(0);

    // 2-bit windowed processing: process 2 bits at a time
    let mut result = G2Projective::zero();

    for coeff_idx in (0..max_coeffs).rev() {
        // Quadruple (process 2 bits)
        result = result.double().double();

        // Extract coefficients for this window
        let mut window_coeffs = [0i8; 4];
        for i in 0..4 {
            if coeff_idx < scalar_coeffs[i].len() {
                window_coeffs[i] = scalar_coeffs[i][coeff_idx];
            }
        }

        // Use signed lookup if any coefficient is non-zero
        if window_coeffs.iter().any(|&c| c != 0) {
            result += table.get_windowed2(&window_coeffs);
        }
    }

    result
}

/// Scalar multiplication using precomputed data and decomposed scalar
pub fn glv_four_scalar_mul_decomposed(
    data: &PrecomputedShamir4Data,
    coeffs: &[<Fr as PrimeField>::BigInt; 4],
    signs: &[bool; 4],
) -> Vec<G2Projective> {
    data.shamir_tables
        .par_iter()
        .map(|table| shamir_glv_mul_4d_precomputed(table, coeffs, signs))
        .collect()
}
