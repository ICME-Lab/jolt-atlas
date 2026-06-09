use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

pub enum WordLtBoundSuffix<const XLEN: usize, const U: usize> {}

impl<const XLEN: usize, const U: usize> SparseDenseSuffix for WordLtBoundSuffix<XLEN, U> {
    fn suffix_mle(bits: LookupBits) -> u32 {
        let ubound_index = 2 * XLEN - U - 1;
        let len = bits.len();
        let bits_u64: u64 = bits.into();

        let suffix_start_index = 2 * XLEN - len;
        let mut result = 0u32;
        for pos in 0..len {
            let global_index = suffix_start_index + pos;
            if global_index > ubound_index {
                let exponent = 2 * XLEN - global_index - 1;
                let bit = ((bits_u64 >> (len - 1 - pos)) & 1) as u32;
                result += bit << exponent;
            }
        }

        result
    }
}
