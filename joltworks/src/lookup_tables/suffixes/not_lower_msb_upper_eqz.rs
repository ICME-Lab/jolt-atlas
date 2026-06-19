use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// Computes `(1 − msb) * eqz(upper_bits_in_suffix)`.
///
/// - b.len() < 32:  1  — m and eqz are fully in prefix
/// - b.len() == 32: (1 − bit[31])
/// - b.len() > 32:  (1 − bit[31]) * (bits 32..b.len()-1 all zero)
pub enum NotLowerMsbUpperEqzSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for NotLowerMsbUpperEqzSuffix<XLEN> {
    fn suffix_mle(b: LookupBits) -> u32 {
        let bits: u64 = b.into();

        // upper bits and word sign-bit are fully processed in the prefix
        if b.len() < 32 {
            return 1;
        }

        let m = (bits >> 31) & 1;
        let not_m = 1 - m;

        if b.len() == 32 {
            return not_m as u32;
        }

        // contains upper bits: check they are all zero
        let eqz: u64 = ((bits >> 32) == 0).into();
        (eqz * not_m) as u32
    }
}
