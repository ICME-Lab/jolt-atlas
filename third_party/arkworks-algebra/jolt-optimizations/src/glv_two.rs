//! 2D GLV scalar multiplication for BN254 G1
//! Three methods: (1) online, (2) precomputed full, (3) signed table

use ark_bn254::{Fr, G1Projective};
use ark_ec::AdditiveGroup;
use ark_ff::{BigInteger, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::Zero;
use rayon::prelude::*;

use crate::decomp_2d::{decompose_scalar_2d, glv_endomorphism};

/// Online 2D GLV scalar multiplication
pub fn glv_two_scalar_mul_online(scalar: Fr, points: &[G1Projective]) -> Vec<G1Projective> {
    let (coeffs, signs) = decompose_scalar_2d(scalar);

    points
        .par_iter()
        .map(|point| {
            // Compute bases: P and λ(P)
            let bases = [*point, glv_endomorphism(point)];

            // Shamir's trick with sign handling
            shamir_glv_mul_2d(&bases, &coeffs, &signs)
        })
        .collect()
}

/// Shamir's trick for 2-point scalar multiplication with signs
pub(crate) fn shamir_glv_mul_2d(
    bases: &[G1Projective; 2],
    coeffs: &[<Fr as PrimeField>::BigInt; 2],
    signs: &[bool; 2],
) -> G1Projective {
    let mut result = G1Projective::zero();
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
                    result += bases[i];
                } else {
                    result -= bases[i];
                }
            }
        }
    }

    result
}

/// Precomputed data for 2D GLV with Shamir table
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PrecomputedShamir2Data {
    pub shamir_tables: Vec<PrecomputedShamir2Table>,
}

/// Shamir lookup table: all 16 combinations for [P, λ(P)] with signs
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PrecomputedShamir2Table {
    pub table: [G1Projective; 16], // 2^2 points × 2^2 signs
}

impl PrecomputedShamir2Table {
    /// Create table for [P, λ(P)] with all sign combinations
    pub fn new(bases: &[G1Projective; 2]) -> Self {
        let mut table = [G1Projective::zero(); 16];

        table.par_iter_mut().enumerate().for_each(|(idx, point)| {
            let point_mask = idx & 0x3; // Which points to include
            let sign_mask = idx >> 2; // Which points to negate

            *point = G1Projective::zero();
            for i in 0..2 {
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
    pub fn get(&self, point_mask: usize, sign_mask: usize) -> G1Projective {
        self.table[point_mask | (sign_mask << 2)]
    }
}

impl PrecomputedShamir2Data {
    pub fn new(points: &[G1Projective]) -> Self {
        let shamir_tables = points
            .par_iter()
            .map(|point| {
                let glv_bases = [*point, glv_endomorphism(point)];
                PrecomputedShamir2Table::new(&glv_bases)
            })
            .collect();

        Self { shamir_tables }
    }
}

/// Precompute for multiple points
pub fn glv_two_precompute(points: &[G1Projective]) -> PrecomputedShamir2Data {
    PrecomputedShamir2Data::new(points)
}

/// Scalar multiplication using precomputed data
pub fn glv_two_scalar_mul(data: &PrecomputedShamir2Data, scalar: Fr) -> Vec<G1Projective> {
    let (coeffs, signs) = decompose_scalar_2d(scalar);

    data.shamir_tables
        .par_iter()
        .map(|table| shamir_glv_mul_2d_precomputed(table, &coeffs, &signs))
        .collect()
}

/// Shamir's trick using precomputed table
pub(crate) fn shamir_glv_mul_2d_precomputed(
    table: &PrecomputedShamir2Table,
    coeffs: &[<Fr as PrimeField>::BigInt; 2],
    signs: &[bool; 2],
) -> G1Projective {
    let mut result = G1Projective::zero();
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

        for i in 0..2 {
            if coeffs[i].get_bit(bit_idx) {
                point_mask |= 1 << i;
                if !signs[i] {
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
pub struct Windowed2Signed2Data {
    pub windowed2_tables: Vec<Windowed2Signed2Table>,
}

/// 2-bit signed table: stores [±P, ±2P, ±3P] for each base
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct Windowed2Signed2Table {
    pub signed_multiples: [G1Projective; 12], // 2 bases × 6 variants
}

impl Windowed2Signed2Table {
    /// Create signed multiples for 2-bit processing
    pub fn new(bases: &[G1Projective; 2]) -> Self {
        let mut signed_multiples = [G1Projective::zero(); 12];

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
    pub fn get_windowed2(&self, coeffs: &[i8; 2]) -> G1Projective {
        let mut result = G1Projective::zero();

        for (i, &coeff) in coeffs.iter().enumerate() {
            if coeff != 0 {
                let abs_coeff = coeff.unsigned_abs() as usize;
                let idx = i * 6 + abs_coeff - 1 + if coeff < 0 { 3 } else { 0 };
                result += self.signed_multiples[idx];
            }
        }

        result
    }
}

impl Windowed2Signed2Data {
    pub fn new(points: &[G1Projective]) -> Self {
        let windowed2_tables = points
            .par_iter()
            .map(|point| {
                let glv_bases = [*point, glv_endomorphism(point)];
                Windowed2Signed2Table::new(&glv_bases)
            })
            .collect();

        Self { windowed2_tables }
    }
}

/// Precompute for 2-bit windowed signed method
pub fn glv_two_precompute_windowed2_signed(points: &[G1Projective]) -> Windowed2Signed2Data {
    Windowed2Signed2Data::new(points)
}

/// Scalar multiplication using 2-bit windowed signed method
pub fn glv_two_scalar_mul_windowed2_signed(
    data: &Windowed2Signed2Data,
    scalar: Fr,
) -> Vec<G1Projective> {
    let (coeffs, signs) = decompose_scalar_2d(scalar);

    data.windowed2_tables
        .par_iter()
        .map(|table| glv_two_scalar_mul_windowed2_signed_single(table, &coeffs, &signs))
        .collect()
}

/// 2-bit windowed signed multiplication for single point
fn glv_two_scalar_mul_windowed2_signed_single(
    table: &Windowed2Signed2Table,
    coeffs: &[<Fr as PrimeField>::BigInt; 2],
    signs: &[bool; 2],
) -> G1Projective {
    // Convert scalars to signed coefficients for 2-bit windowed processing
    let scalar_coeffs: Vec<Vec<i8>> = coeffs
        .par_iter()
        .zip(signs.par_iter())
        .map(|(scalar, &is_positive)| {
            let mut coeffs = Vec::new();
            let scalar_ref = scalar.as_ref();

            // Convert to base-4 coefficients (2 bits at a time)
            for limb in scalar_ref {
                for window_idx in 0..32 {
                    // 64 bits / 2 = 32 windows per limb
                    let window_bits = (limb >> (window_idx * 2)) & 0x3;
                    // In 2D decomposition, signs[i] = true means positive
                    let signed_coeff = if is_positive {
                        window_bits as i8
                    } else {
                        -(window_bits as i8)
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
    let mut result = G1Projective::zero();

    for coeff_idx in (0..max_coeffs).rev() {
        // Quadruple (process 2 bits)
        result = result.double().double();

        // Extract coefficients for this window
        let mut window_coeffs = [0i8; 2];
        for i in 0..2 {
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
pub fn glv_two_scalar_mul_decomposed(
    data: &PrecomputedShamir2Data,
    coeffs: &[<Fr as PrimeField>::BigInt; 2],
    signs: &[bool; 2],
) -> Vec<G1Projective> {
    data.shamir_tables
        .par_iter()
        .map(|table| shamir_glv_mul_2d_precomputed(table, coeffs, signs))
        .collect()
}

// ============================================================================
// Fixed-base MSM utilities
// ============================================================================

/// Decomposed scalar for separate handling
pub struct DecomposedScalar2D {
    pub coeffs: [<Fr as PrimeField>::BigInt; 2],
    pub signs: [bool; 2],
}

impl DecomposedScalar2D {
    pub fn from_scalar(scalar: Fr) -> Self {
        let (coeffs, signs) = decompose_scalar_2d(scalar);
        Self { coeffs, signs }
    }
}

/// Precomputed data for fixed-base scalar multiplication
pub struct FixedBasePrecomputedG1 {
    pub glv_bases: [G1Projective; 2],
    pub shamir_table: PrecomputedShamir2Table,
}

impl FixedBasePrecomputedG1 {
    pub fn new(base: &G1Projective) -> Self {
        let glv_bases = [*base, glv_endomorphism(base)];
        let shamir_table = PrecomputedShamir2Table::new(&glv_bases);
        Self {
            glv_bases,
            shamir_table,
        }
    }

    /// Single scalar multiplication
    pub fn mul_scalar(&self, scalar: Fr) -> G1Projective {
        let decomposed = DecomposedScalar2D::from_scalar(scalar);
        self.mul_scalar_decomposed(&decomposed)
    }

    /// Single scalar multiplication with decomposed scalar
    pub fn mul_scalar_decomposed(&self, decomposed: &DecomposedScalar2D) -> G1Projective {
        shamir_glv_mul_2d_precomputed(&self.shamir_table, &decomposed.coeffs, &decomposed.signs)
    }

    /// Multiple scalar multiplications
    pub fn mul_scalars(&self, scalars: &[Fr]) -> Vec<G1Projective> {
        scalars
            .par_iter()
            .map(|&scalar| self.mul_scalar(scalar))
            .collect()
    }

    /// Multiple scalar multiplications with decomposed scalars
    pub fn mul_scalars_decomposed(&self, decomposed: &[DecomposedScalar2D]) -> Vec<G1Projective> {
        decomposed
            .par_iter()
            .map(|d| self.mul_scalar_decomposed(d))
            .collect()
    }
}

/// Fixed-base vector MSM: compute base * scalars[i] for all i
/// Used in Dory for the g2_scaling by g_fin in eval_vmv_re
pub fn fixed_base_vector_msm_g1(base: &G1Projective, scalars: &[Fr]) -> Vec<G1Projective> {
    let precomputed = FixedBasePrecomputedG1::new(base);
    precomputed.mul_scalars(scalars)
}
