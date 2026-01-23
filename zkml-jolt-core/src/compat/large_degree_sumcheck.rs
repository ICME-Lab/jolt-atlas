//! Stub implementations for large_degree_sumcheck functions
//!
//! These functions were removed from jolt-core and need to be reimplemented
//! or the calling code needs to be updated.

use jolt_core::{
    field::JoltField,
    poly::{
        multilinear_polynomial::MultilinearPolynomial,
        unipoly::UniPoly,
    },
};

/// Computes the univariate polynomial from eq_mle product.
///
/// TODO: This is a stub that needs proper implementation.
pub fn compute_eq_mle_product_univariate<F: JoltField>(
    _mle_product_coeffs: Vec<F>,
    _round: usize,
    _r_cycle: &[F::Challenge],
) -> UniPoly<F> {
    todo!("compute_eq_mle_product_univariate needs implementation")
}

/// Computes MLE product coefficients using Karatsuba algorithm.
///
/// TODO: This is a stub that needs proper implementation.
pub fn compute_mle_product_coeffs_katatsuba<F: JoltField, const N: usize, const M: usize>(
    _ra_i_polys: &[MultilinearPolynomial<F>],
    _round: usize,
    _num_rounds: usize,
    _eq_factor: &F,
    _E_table: &[Vec<F>],
) -> Vec<F> {
    todo!("compute_mle_product_coeffs_katatsuba needs implementation")
}
