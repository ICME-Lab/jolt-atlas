use super::SparseDenseSuffix;
use crate::utils::lookup_bits::LookupBits;

/// Suffix component for word without MSB lookup table decomposition.
pub enum WordNoMsbSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for WordNoMsbSuffix<XLEN> {
    fn suffix_mle(bits: LookupBits) -> u32 {
        let mut b: u64 = bits.into();
        b %= 1 << (XLEN - 1);
        b as u32
    }
}
