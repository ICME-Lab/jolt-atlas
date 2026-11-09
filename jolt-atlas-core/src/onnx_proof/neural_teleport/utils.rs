//! Shared utilities for neural teleportation operations.
//!
//! This module centralizes helper functions used by activation ops, and periodic functions.

pub use jolt_atlas_verifier::onnx_proof::neural_teleport::utils::*;

use super::n_bits_to_usize;
use atlas_algebra::{
    field::{FieldChallengeOps, JoltField},
    poly::eq_poly::EqPolynomial,
    utils::thread::unsafe_allocate_zero_vec,
};
use atlas_trace::Tensor;
use common::parallel::par_enabled;
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    },
    slice::ParallelSlice,
};

/// Compute one-hot read-address evaluations from signed n-bit two's-complement values.
///
/// This variant is used by lookup-table ops where signed
/// quotient values are mapped into `[0, 2^log_table_size)` via two's-complement.
pub fn compute_ra_evals_nbits_2comp<F, U>(
    r: &[U],
    input: &Tensor<i32>,
    log_table_size: usize,
) -> Vec<F>
where
    U: Copy + Send + Sync + Into<F>,
    F: JoltField + FieldChallengeOps<U>,
{
    let table_size = 1 << log_table_size;
    let input_usize = input
        .par_iter()
        .with_min_len(par_enabled())
        .map(|&x| n_bits_to_usize(x, log_table_size))
        .collect::<Vec<usize>>();

    compute_ra_evals_from_usize_indices(r, &input_usize, table_size)
}

/// Compute one-hot read-address evaluations from already-computed non-negative indices.
pub fn compute_ra_evals_from_usize_indices<F, U>(
    r: &[U],
    indices_usize: &[usize],
    table_size: usize,
) -> Vec<F>
where
    U: Copy + Send + Sync + Into<F>,
    F: JoltField + FieldChallengeOps<U>,
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
            .with_min_len(par_enabled())
            .for_each(|(dest, &src)| *dest += src);
    }
    ra
}
