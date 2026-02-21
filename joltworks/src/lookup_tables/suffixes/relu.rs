use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// Suffix component for ReLU activation lookup table decomposition.
pub enum ReluSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for ReluSuffix<XLEN> {
    fn suffix_mle(bits: LookupBits) -> u32 {
        let mut b: u64 = bits.into();
        b %= 1 << XLEN;
        if bits.len() >= XLEN {
            let sign_bit = b >> (XLEN - 1);
            (b as u32) * (1 - sign_bit as u32)
        } else {
            b as u32
        }
    }
}
