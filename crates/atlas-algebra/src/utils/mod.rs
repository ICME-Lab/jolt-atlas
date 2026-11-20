use crate::field::{ChallengeFieldOps, JoltField};
use common::parallel::par_enabled;

use rayon::prelude::*;

pub mod accumulation;
pub mod errors;
pub mod expanding_table;
pub mod lookup_bits;
pub mod math;
pub mod profiling;
pub mod small_scalar;
pub mod thread;

/// Converts an integer value to a bitvector (all values {0,1}) of field elements.
/// Note: ordering has the MSB in the highest index. All of the following represent the integer 1:
/// - [1]
/// - [0, 0, 1]
/// - [0, 0, 0, 0, 0, 0, 0, 1]
/// ```ignore
/// use jolt_core::utils::index_to_field_bitvector;
/// # use crate::field::fp128::Fp128 as Fr;
/// # use num_traits::{One, Zero};
/// let zero = Fr::zero();
/// let one = Fr::one();
///
/// assert_eq!(index_to_field_bitvector::<Fr>(1, 1), vec![one]);
/// assert_eq!(index_to_field_bitvector::<Fr>(1, 3), vec![zero, zero, one]);
/// assert_eq!(index_to_field_bitvector::<Fr>(1, 7), vec![zero, zero, zero, zero, zero, zero, one]);
/// ```
pub fn index_to_field_bitvector<F: JoltField + ChallengeFieldOps<F>>(
    value: u64,
    bits: usize,
) -> Vec<F> {
    assert!((value as u128) < 1 << bits);
    let mut bitvector: Vec<F> = Vec::with_capacity(bits);

    for i in (0..bits).rev() {
        if (value >> i) & 1 == 1 {
            bitvector.push(F::one());
        } else {
            bitvector.push(F::zero());
        }
    }
    bitvector
}

#[tracing::instrument(skip_all)]
pub fn compute_dotproduct<F: JoltField>(a: &[F], b: &[F]) -> F {
    a.par_iter()
        .zip_eq(b.par_iter())
        .with_min_len(par_enabled())
        .map(|(a_i, b_i)| *a_i * *b_i)
        .sum()
}

/// The fixed-seed RNG the test suites draw from (the seed arkworks' `test_rng` used, so every
/// test sees the stream it always has).
pub fn test_rng() -> rand::rngs::StdRng {
    use rand::SeedableRng;
    let seed = [
        1, 0, 0, 0, 23, 0, 0, 0, 200, 1, 0, 0, 210, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ];
    rand::rngs::StdRng::from_seed(seed)
}
