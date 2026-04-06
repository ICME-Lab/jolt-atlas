use crate::{
    field::JoltField,
    poly::{dense_mlpoly::DensePolynomial, multilinear_polynomial::MultilinearPolynomial},
    utils::small_scalar::SmallScalar,
};
use common::CommittedPolynomial;
use rayon::prelude::*;
use std::collections::BTreeMap;

const RLC_CHUNK_SIZE: usize = 1 << 14;

#[inline]
fn accumulate_dense_chunk<F: JoltField>(
    out: &mut [F],
    start: usize,
    poly: &MultilinearPolynomial<F>,
    coeff: F,
) {
    match poly {
        MultilinearPolynomial::LargeScalars(p) => {
            let coeffs = &p.Z[start..start + out.len()];
            out.iter_mut()
                .zip(coeffs.iter())
                .for_each(|(dst, src)| *dst += *src * coeff);
        }
        MultilinearPolynomial::BoolScalars(p) => {
            let coeffs = &p.coeffs[start..start + out.len()];
            out.iter_mut().zip(coeffs.iter()).for_each(|(dst, src)| {
                if *src {
                    *dst += coeff;
                }
            });
        }
        MultilinearPolynomial::U8Scalars(p) => {
            let coeffs = &p.coeffs[start..start + out.len()];
            out.iter_mut()
                .zip(coeffs.iter())
                .for_each(|(dst, src)| *dst += src.field_mul(coeff));
        }
        MultilinearPolynomial::U16Scalars(p) => {
            let coeffs = &p.coeffs[start..start + out.len()];
            out.iter_mut()
                .zip(coeffs.iter())
                .for_each(|(dst, src)| *dst += src.field_mul(coeff));
        }
        MultilinearPolynomial::U32Scalars(p) => {
            let coeffs = &p.coeffs[start..start + out.len()];
            out.iter_mut()
                .zip(coeffs.iter())
                .for_each(|(dst, src)| *dst += src.field_mul(coeff));
        }
        MultilinearPolynomial::U64Scalars(p) => {
            let coeffs = &p.coeffs[start..start + out.len()];
            out.iter_mut()
                .zip(coeffs.iter())
                .for_each(|(dst, src)| *dst += src.field_mul(coeff));
        }
        MultilinearPolynomial::U128Scalars(p) => {
            let coeffs = &p.coeffs[start..start + out.len()];
            out.iter_mut()
                .zip(coeffs.iter())
                .for_each(|(dst, src)| *dst += src.field_mul(coeff));
        }
        MultilinearPolynomial::I32Scalars(p) => {
            let coeffs = &p.coeffs[start..start + out.len()];
            out.iter_mut()
                .zip(coeffs.iter())
                .for_each(|(dst, src)| *dst += src.field_mul(coeff));
        }
        MultilinearPolynomial::I64Scalars(p) => {
            let coeffs = &p.coeffs[start..start + out.len()];
            out.iter_mut()
                .zip(coeffs.iter())
                .for_each(|(dst, src)| *dst += src.field_mul(coeff));
        }
        MultilinearPolynomial::I128Scalars(p) => {
            let coeffs = &p.coeffs[start..start + out.len()];
            out.iter_mut()
                .zip(coeffs.iter())
                .for_each(|(dst, src)| *dst += src.field_mul(coeff));
        }
        MultilinearPolynomial::S128Scalars(p) => {
            let coeffs = &p.coeffs[start..start + out.len()];
            out.iter_mut()
                .zip(coeffs.iter())
                .for_each(|(dst, src)| *dst += src.field_mul(coeff));
        }
        MultilinearPolynomial::OneHot(_) => unreachable!(),
    }
}

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

    // Homomorphically combine dense polynomials in cache-friendly chunks.
    let mut joint_coeffs: Vec<F> = vec![F::zero(); joint_len];
    joint_coeffs
        .par_chunks_mut(RLC_CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let start = chunk_idx * RLC_CHUNK_SIZE;
            for ((_, poly), coeff) in dense.iter() {
                let poly_len = poly.original_len();
                if start >= poly_len {
                    continue;
                }
                let overlap_len = chunk.len().min(poly_len - start);
                accumulate_dense_chunk(&mut chunk[..overlap_len], start, poly, **coeff);
            }
        });

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
