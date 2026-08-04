use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// Suffix that evaluates to 1 iff all bits with significance >= 2^BOUND are zero.
pub enum HigherAllZeroSuffix<const XLEN: usize, const BOUND: usize> {}

impl<const XLEN: usize, const BOUND: usize> SparseDenseSuffix for HigherAllZeroSuffix<XLEN, BOUND> {
    fn suffix_mle(bits: LookupBits) -> u32 {
        let bound_index = XLEN - BOUND - 1;
        let len = bits.len();
        let bits_u64: u64 = bits.into();

        let suffix_start_index = XLEN - len;
        for pos in 0..len {
            let global_index = suffix_start_index + pos;
            if global_index <= bound_index {
                let bit = (bits_u64 >> (len - 1 - pos)) & 1;
                if bit == 1 {
                    return 0;
                }
            }
        }

        1
    }
}

use crate::lookup_tables::clamp::{CLAMP_BOUND, SIGN_BIT_I32};
use common::consts::{ACTIVATION_BOUND, SOFTMAX_CLAMP_BOUND};
pub type ClampHigherAllZeroSuffix<const XLEN: usize> = HigherAllZeroSuffix<XLEN, CLAMP_BOUND>;

pub type SatClampHigherAllZeroSuffix<const XLEN: usize> = HigherAllZeroSuffix<XLEN, SIGN_BIT_I32>;

pub type ActivationHigherAllZeroSuffix<const XLEN: usize> =
    HigherAllZeroSuffix<XLEN, ACTIVATION_BOUND>;

/// Its own nominal type rather than a `HigherAllZeroSuffix<XLEN, SOFTMAX_CLAMP_BOUND>` alias:
/// `SOFTMAX_CLAMP_BOUND` and `ACTIVATION_BOUND` are derived from the same `MODEL_SCALE`
/// by unrelated formulas and can coincide (e.g. both `16` at `MODEL_SCALE=12`), which would make
/// this the same concrete type as `ActivationHigherAllZeroSuffix` and collide on `SuffixVariant`
/// (suffixes have no per-alias tag like prefixes' `CP_INDEX` to disambiguate).
pub enum SoftmaxClampHigherAllZeroSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for SoftmaxClampHigherAllZeroSuffix<XLEN> {
    fn suffix_mle(bits: LookupBits) -> u32 {
        HigherAllZeroSuffix::<XLEN, SOFTMAX_CLAMP_BOUND>::suffix_mle(bits)
    }
}
