use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// Suffix that evaluates to `lower_word(bits)` only when all bits with
/// significance >= 2^BOUND are one; otherwise evaluates to 0. Mirrors
/// [`super::hzero_mul_lword::HZeroMulLWordSuffix`], checking for 1 instead of 0.
pub enum HOneMulLWordSuffix<const XLEN: usize, const BOUND: usize> {}

impl<const XLEN: usize, const BOUND: usize> SparseDenseSuffix for HOneMulLWordSuffix<XLEN, BOUND> {
    fn suffix_mle(bits: LookupBits) -> u32 {
        let bound_index = XLEN - BOUND - 1;
        let len = bits.len();
        let bits_u64: u64 = bits.into();

        let suffix_start_index = XLEN - len;
        let mut lower_word = 0u32;

        for pos in 0..len {
            let global_index = suffix_start_index + pos;
            let bit = ((bits_u64 >> (len - 1 - pos)) & 1) as u32;

            if global_index <= bound_index && bit == 0 {
                return 0;
            }

            if global_index > bound_index {
                let exponent = XLEN - global_index - 1;
                lower_word += bit << exponent;
            }
        }

        lower_word
    }
}

use crate::lookup_tables::clamp::{CLAMP_BOUND, SIGN_BIT_I32};
use common::consts::ACTIVATION_BOUND;

pub type ClampHOneMulLWordSuffix<const XLEN: usize> = HOneMulLWordSuffix<XLEN, CLAMP_BOUND>;

pub type SatClampHOneMulLWordSuffix<const XLEN: usize> = HOneMulLWordSuffix<XLEN, SIGN_BIT_I32>;

pub type ActivationHOneMulLWordSuffix<const XLEN: usize> =
    HOneMulLWordSuffix<XLEN, ACTIVATION_BOUND>;
