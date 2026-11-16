//! Utilities; the model-only ones live in `jolt-atlas-verifier`.

pub use jolt_atlas_verifier::utils::*;

pub mod opening_access;

use atlas_algebra::utils::lookup_bits::LookupBits;
use atlas_trace::Tensor;
use common::{consts::XLEN, parallel::par_enabled};
use rayon::prelude::*;

/// The lookup index of each element of `operand`: its value as an `XLEN`-bit address.
pub fn compute_lookup_indices(operand: &Tensor<i32>) -> Vec<LookupBits> {
    debug_assert!(
        operand.dims().iter().all(|d| d.is_power_of_two()),
        "operand dims must all be power-of-two, got {:?}",
        operand.dims()
    );
    operand
        .data()
        .par_iter()
        .with_min_len(par_enabled())
        .map(|&value| LookupBits::new(value as u32 as u64, XLEN))
        .collect()
}
