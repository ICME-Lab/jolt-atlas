//! Optimized BN254 G2 scalar multiplication using 4D GLV decomposition
//!
//! This crate provides optimized scalar multiplication algorithms for BN254 G2
//! using 4-dimensional decomposition combining GLV and Frobenius endomorphisms.
//!
//! The main optimization reduces a 256-bit scalar multiplication to four ~66-bit
//! scalar multiplications.
//! Also provides BN254 G1 equivalents.
//! Uses Strauss-shamir batched scalar multiplication to maximally take advantage of GLV.

pub mod batch_addition;
pub mod constants;
pub mod decomp_2d;
pub mod decomp_4d;
pub mod dory_g1;
pub mod dory_g2;
pub mod dory_utils;
pub mod fq12_poly;
pub mod frobenius;
pub mod glv_two;
pub mod msm_bucket;
pub mod small_row;
pub mod witness_gen;

mod glv_four;
pub use glv_four::{
    glv_four_precompute, glv_four_precompute_windowed2_signed, glv_four_scalar_mul,
    glv_four_scalar_mul_decomposed, glv_four_scalar_mul_online,
    glv_four_scalar_mul_windowed2_signed, PrecomputedShamir4Data, PrecomputedShamir4Table,
    Windowed2Signed4Data, Windowed2Signed4Table,
};

pub use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};

pub use glv_two::{
    fixed_base_vector_msm_g1, glv_two_precompute, glv_two_precompute_windowed2_signed,
    glv_two_scalar_mul, glv_two_scalar_mul_decomposed, glv_two_scalar_mul_online,
    glv_two_scalar_mul_windowed2_signed, DecomposedScalar2D, FixedBasePrecomputedG1,
    PrecomputedShamir2Data, PrecomputedShamir2Table, Windowed2Signed2Data, Windowed2Signed2Table,
};

pub use dory_utils::{
    vector_scalar_mul_add, vector_scalar_mul_add_online, vector_scalar_mul_add_precomputed,
    vector_scalar_mul_v_add_g_online, vector_scalar_mul_v_add_g_precomputed, VectorScalarMulData,
    VectorScalarMulVData,
};

pub use frobenius::frobenius_psi_power_projective;

pub use dory_g1::{
    vector_add_scalar_mul_g1_online, vector_add_scalar_mul_g1_precomputed,
    vector_add_scalar_mul_g1_windowed2_signed, vector_scalar_mul_add_gamma_g1_online,
};

pub use dory_g2::{
    vector_add_scalar_mul_g2_online, vector_add_scalar_mul_g2_precomputed,
    vector_add_scalar_mul_g2_windowed2_signed, vector_scalar_mul_add_gamma_g2_online,
};

pub use batch_addition::{batch_g1_additions, batch_g1_additions_multi};

pub use msm_bucket::{
    batch_addition_matrix, batch_addition_matrix_u8, batch_addition_matrix_u8_variable,
    msm_rows_bucket_affine, msm_rows_bucket_projective,
};
pub use small_row::SmallRow;

pub use fq12_poly::{
    eq_weights, eval_multilinear, fq12_to_multilinear_evals, fq12_to_poly12_coeffs, g_coeffs,
    g_eval, to_multilinear_evals,
};

pub use witness_gen::{get_g_mle, h_tilde_at_point, ExponentiationSteps};
