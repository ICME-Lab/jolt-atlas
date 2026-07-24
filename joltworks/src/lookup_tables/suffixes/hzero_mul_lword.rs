use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// Suffix that evaluates to `lower_word(bits)` only when all bits with
/// significance >= 2^BOUND are zero; otherwise evaluates to 0.
pub enum HZeroMulLWordSuffix<const XLEN: usize, const BOUND: usize> {}

impl<const XLEN: usize, const BOUND: usize> SparseDenseSuffix for HZeroMulLWordSuffix<XLEN, BOUND> {
    fn suffix_mle(bits: LookupBits) -> u32 {
        let bound_index = XLEN - BOUND - 1;
        let len = bits.len();
        let bits_u64: u64 = bits.into();

        let suffix_start_index = XLEN - len;
        let mut lower_word = 0u32;

        for pos in 0..len {
            let global_index = suffix_start_index + pos;
            let bit = ((bits_u64 >> (len - 1 - pos)) & 1) as u32;

            if global_index <= bound_index && bit == 1 {
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

use crate::lookup_tables::clamp::CLAMP_TABLE_BOUND;
pub type ClampHZeroMulLWordSuffix<const XLEN: usize> = HZeroMulLWordSuffix<XLEN, CLAMP_TABLE_BOUND>;

pub type SatClampHZeroMulLWordSuffix<const XLEN: usize> = HZeroMulLWordSuffix<XLEN, 32>;
