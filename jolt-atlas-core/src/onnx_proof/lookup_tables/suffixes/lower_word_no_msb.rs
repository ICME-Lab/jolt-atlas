use super::SparseDenseSuffix;
use joltworks::utils::lookup_bits::LookupBits;

pub enum LowerWordNoMsbSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for LowerWordNoMsbSuffix<XLEN> {
    fn suffix_mle(bits: LookupBits) -> u32 {
        let mut b: u64 = bits.into();
        b %= 1 << (XLEN - 1);
        b as u32
    }
}
