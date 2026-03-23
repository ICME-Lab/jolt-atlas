//! Shared utilities for neural teleportation operations.
//!
//! This module centralizes helper functions used by activation ops, and periodic functions.

use super::{n_bits_to_usize, usize_to_n_bits, SCALE};
use atlas_onnx_tracer::tensor::Tensor;
use joltworks::{
    field::{FieldChallengeOps, JoltField},
    poly::eq_poly::EqPolynomial,
    utils::thread::unsafe_allocate_zero_vec,
};
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    },
    slice::ParallelSlice,
};

/// Compute one-hot read-address evaluations from direct non-negative indices.
///
/// This variant is used by trigonometric teleportation ops (`cos` and `sin`),
/// where inputs are remainders already in `[0, table_size)`.
pub fn compute_ra_evals_direct<F, U>(r: &[U], indexes: &Tensor<i32>, table_size: usize) -> Vec<F>
where
    U: Copy + Send + Sync + Into<F>,
    F: JoltField + FieldChallengeOps<U>,
{
    let indexes_usize = indexes
        .par_iter()
        .map(|&x| x as usize)
        .collect::<Vec<usize>>();
    compute_ra_evals_from_usize_indices(r, &indexes_usize, table_size)
}

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
        .map(|&x| n_bits_to_usize(x, log_table_size))
        .collect::<Vec<usize>>();

    compute_ra_evals_from_usize_indices(r, &input_usize, table_size)
}

/// Build a signed two's-complement lookup table for a unary activation.
///
/// The table domain is `[-2^(n-1), 2^(n-1))` encoded in two's complement, and
/// outputs are quantized using the shared neural-teleport `SCALE`.
/// Each lookup to the table at index `x` corresponds to the activation of `x * τ`.
pub fn materialize_signed_activation_table(
    log_table_size: usize,
    tau: i32,
    activation: fn(&Tensor<i32>, f64) -> Tensor<i32>,
) -> Vec<i32> {
    let table_size = 1 << log_table_size;
    let indices: Vec<i32> = (0..table_size)
        .map(|i| {
            usize_to_n_bits(i, log_table_size)
                .checked_mul(tau)
                .expect("overflow in activation table index")
        })
        .collect();
    let indices_tensor = Tensor::new(Some(&indices), &[1, table_size])
        .expect("failed to build activation LUT input tensor");
    let result = activation(&indices_tensor, SCALE);
    result.data().to_vec()
}

macro_rules! define_signed_activation_table {
    ($table:ident, $activation:path) => {
        #[doc = "Lookup table for a signed neural-teleport activation."]
        #[derive(Debug, Clone, Copy)]
        pub struct $table {
            log_table_size: usize,
            tau: i32,
        }

        impl $table {
            /// Create a new lookup table with the specified bit width.
            pub fn new(log_table_size: usize, tau: i32) -> Self {
                Self {
                    log_table_size,
                    tau,
                }
            }

            /// Returns the size of the table (2^log_table_size).
            pub fn table_size(&self) -> usize {
                1 << self.log_table_size
            }

            /// Returns the log2 of the table size.
            pub fn log_table_size(&self) -> usize {
                self.log_table_size
            }

            /// Materialize the lookup table values.
            pub fn materialize(&self) -> Vec<i32> {
                crate::onnx_proof::neural_teleport::utils::materialize_signed_activation_table(
                    self.log_table_size,
                    self.tau,
                    $activation,
                )
            }
        }
    };
}

pub(crate) use define_signed_activation_table;

fn compute_ra_evals_from_usize_indices<F, U>(
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
            .for_each(|(dest, &src)| *dest += src);
    }
    ra
}
