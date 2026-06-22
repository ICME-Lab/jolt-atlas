use super::SparseDenseSuffix;
use crate::utils::lookup_bits::LookupBits;

/// Suffix component for lower word without MSB lookup table decomposition.
pub enum LowerWordNoMsbSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for LowerWordNoMsbSuffix<XLEN> {
    fn suffix_mle(bits: LookupBits) -> u32 {
        assert!(bits.len() < 32);
        let b: u64 = bits.into();
        b as u32
    }
}
