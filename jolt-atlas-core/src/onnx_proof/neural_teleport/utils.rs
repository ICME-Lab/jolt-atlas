//! Shared utilities for neural teleportation operations.
//!
//! This module centralizes helper functions used by activation ops (erf, tanh,
//! cos, sin) to build read-address one-hot evaluations.

use super::n_bits_to_usize;
use atlas_onnx_tracer::tensor::Tensor;
use joltworks::{
    field::JoltField,
    poly::eq_poly::EqPolynomial,
    utils::thread::unsafe_allocate_zero_vec,
};
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator},
    slice::ParallelSlice,
};

/// Compute one-hot read-address evaluations from direct non-negative indices.
///
/// This variant is used by trigonometric teleportation ops (`cos` and `sin`),
/// where inputs are remainders already in `[0, table_size)`.
pub fn compute_ra_evals_direct<F>(
    r: &[F::Challenge],
    indexes: &Tensor<i32>,
    table_size: usize,
) -> Vec<F>
where
    F: JoltField,
{
    let indexes_usize = indexes.par_iter().map(|&x| x as usize).collect::<Vec<usize>>();
    compute_ra_evals_from_usize_indices(r, &indexes_usize, table_size)
}

/// Compute one-hot read-address evaluations from signed n-bit two's-complement values.
///
/// This variant is used by lookup-table ops (`erf` and `tanh`), where signed
/// quotient values are mapped into `[0, 2^log_table_size)` via two's-complement.
pub fn compute_ra_evals_nbits_2comp<F>(
    r: &[F::Challenge],
    input: &Tensor<i32>,
    log_table_size: usize,
) -> Vec<F>
where
    F: JoltField,
{
    let table_size = 1 << log_table_size;
    let input_usize = input
        .par_iter()
        .map(|&x| n_bits_to_usize(x, log_table_size))
        .collect::<Vec<usize>>();

    compute_ra_evals_from_usize_indices(r, &input_usize, table_size)
}

fn compute_ra_evals_from_usize_indices<F>(
    r: &[F::Challenge],
    indices_usize: &[usize],
    table_size: usize,
) -> Vec<F>
where
    F: JoltField,
{
    let e = EqPolynomial::evals(r);
    let num_threads = rayon::current_num_threads();
    let chunk_size = indices_usize.len().div_ceil(num_threads);

    let partial_results: Vec<Vec<F>> = indices_usize
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let mut local_ra = unsafe_allocate_zero_vec::<F>(table_size);
            let base_idx = chunk_idx * chunk_size;
            chunk.iter().enumerate().for_each(|(local_j, &k)| {
                let global_j = base_idx + local_j;
                local_ra[k] += e[global_j];
            });
            local_ra
        })
        .collect();

    let mut ra = unsafe_allocate_zero_vec::<F>(table_size);
    for partial in partial_results {
        ra.par_iter_mut()
            .zip(partial.par_iter())
            .for_each(|(dest, &src)| *dest += src);
    }
    ra
}
