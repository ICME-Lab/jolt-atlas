use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// Bitwise XOR suffix
pub enum XorSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for XorSuffix<XLEN> {
    fn suffix_mle(b: LookupBits) -> u32 {
        let (x, y) = b.uninterleave();
        u32::from(x) ^ u32::from(y)
    }
}
