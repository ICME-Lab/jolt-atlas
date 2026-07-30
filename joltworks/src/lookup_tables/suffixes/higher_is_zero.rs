use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// Suffix that evaluates to 1 iff all bits with significance >= 2^BOUND are zero.
pub enum HigherIsZeroSuffix<const XLEN: usize, const BOUND: usize> {}

impl<const XLEN: usize, const BOUND: usize> SparseDenseSuffix for HigherIsZeroSuffix<XLEN, BOUND> {
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

use crate::lookup_tables::clamp::CLAMP_TABLE_BOUND;
use common::consts::{ACTIVATION_TABLE_BOUND, SOFTMAX_SAT_CLAMP_BOUND};
pub type ClampHigherIsZeroSuffix<const XLEN: usize> = HigherIsZeroSuffix<XLEN, CLAMP_TABLE_BOUND>;

pub type SatClampHigherIsZeroSuffix<const XLEN: usize> = HigherIsZeroSuffix<XLEN, 32>;

pub type ActivationHigherIsZeroSuffix<const XLEN: usize> =
    HigherIsZeroSuffix<XLEN, ACTIVATION_TABLE_BOUND>;

/// Its own nominal type rather than a `HigherIsZeroSuffix<XLEN, SOFTMAX_SAT_CLAMP_BOUND>` alias:
/// `SOFTMAX_SAT_CLAMP_BOUND` and `ACTIVATION_TABLE_BOUND` are derived from the same `MODEL_SCALE`
/// by unrelated formulas and can coincide (e.g. both `16` at `MODEL_SCALE=12`), which would make
/// this the same concrete type as `ActivationHigherIsZeroSuffix` and collide on `SuffixVariant`
/// (suffixes have no per-alias tag like prefixes' `CP_INDEX` to disambiguate).
pub enum SoftmaxSatClampHigherIsZeroSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for SoftmaxSatClampHigherIsZeroSuffix<XLEN> {
    fn suffix_mle(bits: LookupBits) -> u32 {
        HigherIsZeroSuffix::<XLEN, SOFTMAX_SAT_CLAMP_BOUND>::suffix_mle(bits)
    }
}
