use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// The constant 1.
pub enum OneSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for OneSuffix<XLEN> {
    fn suffix_mle(_: LookupBits) -> u32 {
        1
    }
}
