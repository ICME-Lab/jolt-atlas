use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// Computes `(1 − msb) * eqz(upper_bits_in_suffix) * lower_mag(suffix)`.
///
/// Captures the suffix contribution to the `(1−m)*a*low` product without multiplying
/// suffix evaluations against each other (the prefix handles the remaining `a_pre` factor).
///
/// - b.len() > 32:  (1 − bit[31]) * (bits 32..b.len()-1 all zero) * (bits 0..30)
/// - b.len() == 32: (1 − bit[31]) * (bits 0..30)
/// - b.len() < 32:  (bits 0..30 of b)   — m and eqz are fully in prefix
pub enum NotLowerMsbUpperEqzLowSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for NotLowerMsbUpperEqzLowSuffix<XLEN> {
    fn suffix_mle(b: LookupBits) -> u32 {
        let b_val: u64 = b.into();
        let low = (b_val % (1u64 << 31)) as u32;
        if b.len() < 32 {
            return low;
        }
        let m = ((b_val >> 31) & 1) as u32;
        let not_m = 1 - m;
        if b.len() == 32 {
            return not_m * low;
        }
        let upper_len = b.len() - 32;
        let upper_mask = (1u64 << upper_len) - 1;
        let upper = (b_val >> 32) & upper_mask;
        let eqz = (upper == 0) as u32;
        not_m * eqz * low
    }
}
