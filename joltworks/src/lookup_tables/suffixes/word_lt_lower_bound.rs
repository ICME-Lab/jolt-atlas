use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

pub enum WordLtLowerBoundSuffix<const XLEN: usize, const L: usize> {}

impl<const XLEN: usize, const L: usize> SparseDenseSuffix for WordLtLowerBoundSuffix<XLEN, L> {
    fn suffix_mle(bits: LookupBits) -> u32 {
        let lbound_index = 2 * XLEN - L - 1;
        let len = bits.len();
        let bits_u64: u64 = bits.into();

        let suffix_start_index = 2 * XLEN - len;
        let mut result = 0u32;
        for pos in 0..len {
            let global_index = suffix_start_index + pos;
            if global_index > lbound_index {
                let exponent = 2 * XLEN - global_index - 1;
                let bit = ((bits_u64 >> (len - 1 - pos)) & 1) as u32;
                result += bit << exponent;
            }
        }

        result
    }
}
