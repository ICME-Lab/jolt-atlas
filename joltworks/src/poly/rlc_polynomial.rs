use crate::{
    field::JoltField,
    poly::{dense_mlpoly::DensePolynomial, multilinear_polynomial::MultilinearPolynomial},
};
use common::CommittedPolynomial;
use rayon::prelude::*;
use std::collections::BTreeMap;

/// Build materialized dense polynomial from this state (for HyperKZG).
/// Regenerates witness polynomials from trace and combines them homomorphically.
#[tracing::instrument(skip_all)]
pub fn build_materialized_rlc<F: JoltField>(
    coeffs: &[F],
    polynomials: &BTreeMap<CommittedPolynomial, MultilinearPolynomial<F>>,
) -> MultilinearPolynomial<F> {
    // Partition into dense and one-hot polynomials (like RLCPolynomial::linear_combination)
    let (dense, one_hot): (Vec<_>, Vec<_>) = polynomials
        .iter()
        .zip(coeffs.iter())
        .partition(|((_, p), _)| !matches!(p, MultilinearPolynomial::OneHot(_)));

    // Compute joint polynomial length from dense polynomials
    let dense_len = dense
        .iter()
        .map(|((_, p), _)| p.original_len())
        .max()
        .unwrap_or(0);

    // Compute one-hot length (K * T)
    let one_hot_len = one_hot
        .iter()
        .map(|((_, p), _)| match p {
            MultilinearPolynomial::OneHot(oh) => oh.K * oh.nonzero_indices.len(),
            _ => unreachable!(),
        })
        .max()
        .unwrap_or(0);

    let joint_len = dense_len.max(one_hot_len);

    // Homomorphically combine dense polynomials: joint[i] = Σ coeff_j * poly_j[i]
    let mut joint_coeffs: Vec<F> = (0..joint_len)
        .into_par_iter()
        .map(|i| {
            dense
                .iter()
                .map(|((_, poly), coeff)| {
                    if i < poly.original_len() {
                        **coeff * poly.get_scaled_coeff(i, F::one())
                    } else {
                        F::zero()
                    }
                })
                .sum()
        })
        .collect();

    // Add one-hot polynomials directly (sparse - only set nonzero entries)
    // Index formula matches HyperKZG::commit_one_hot: k * T + t (no bit reversal)
    for ((_, poly), coeff) in one_hot.iter() {
        match poly {
            MultilinearPolynomial::OneHot(oh) => {
                let T = oh.nonzero_indices.len();
                for (t, k_opt) in oh.nonzero_indices.iter().enumerate() {
                    if let Some(k) = k_opt {
                        let idx = *k as usize * T + t;
                        joint_coeffs[idx] += **coeff;
                    }
                }
            }
            _ => unreachable!(),
        }
    }

    MultilinearPolynomial::LargeScalars(DensePolynomial::new(joint_coeffs))
}
