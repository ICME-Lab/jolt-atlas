//! Precomputed constants for BN254 G2 4-dimensional scalar decomposition
//!
//! This module contains the precomputed lookup table for powers of 2 decompositions for G2
//! and Frobenius endomorphism coefficients.

use ark_bn254::Fq2;
use ark_bn254::Fr;
use ark_ff::MontFp;

// Include the precomputed power of 2 decompositions table
include!("power_of_2_decompositions.rs");

/// Frobenius endomorphism coefficients for BN254 G2
pub struct FrobeniusCoefficients {
    pub psi1_coef2: Fq2,
    pub psi1_coef3: Fq2,
    pub psi2_coef2: Fq2,
    pub psi2_coef3: Fq2,
    pub psi3_coef2: Fq2,
    pub psi3_coef3: Fq2,
}

pub fn get_frobenius_coefficients() -> FrobeniusCoefficients {
    FrobeniusCoefficients {
        // psi^1 coefficients
        psi1_coef2: Fq2::new(
            MontFp!(
                "21575463638280843010398324269430826099269044274347216827212613867836435027261"
            ),
            MontFp!(
                "10307601595873709700152284273816112264069230130616436755625194854815875713954"
            ),
        ),
        psi1_coef3: Fq2::new(
            MontFp!("2821565182194536844548159561693502659359617185244120367078079554186484126554"),
            MontFp!("3505843767911556378687030309984248845540243509899259641013678093033130930403"),
        ),
        // psi^2 coefficients
        psi2_coef2: Fq2::new(
            MontFp!(
                "21888242871839275220042445260109153167277707414472061641714758635765020556616"
            ),
            MontFp!("0"),
        ),
        psi2_coef3: Fq2::new(
            MontFp!(
                "21888242871839275222246405745257275088696311157297823662689037894645226208582"
            ),
            MontFp!("0"),
        ),
        // psi^3 coefficients
        psi3_coef2: Fq2::new(
            MontFp!("3772000881919853776433695186713858239009073593817195771773381919316419345261"),
            MontFp!("2236595495967245188281701248203181795121068902605861227855261137820944008926"),
        ),
        psi3_coef3: Fq2::new(
            MontFp!(
                "19066677689644738377698246183563772429336693972053703295610958340458742082029"
            ),
            MontFp!(
                "18382399103927718843559375435273026243156067647398564021675359801612095278180"
            ),
        ),
    }
}

/// Get the Frobenius eigenvalue λ_ψ for BN254
pub fn get_bn254_frobenius_eigenvalue() -> Fr {
    use ark_ff::PrimeField;
    Fr::from_be_bytes_mod_order(&[
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x6f, 0x4d, 0x82, 0x48, 0xee, 0xb8, 0x59, 0xfb, 0xf8, 0x3e, 0x96, 0x82, 0xe8, 0x7c,
        0xfd, 0x46,
    ])
}
