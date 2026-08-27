use crate::utils::lookup_bits::LookupBits;
use common::consts::TRIG_PERIOD_MODULUS;

use super::SparseDenseSuffix;

/// The `i`-th bit of `K` (global bit index, 0 = MSB), truncated to `XLEN` bits.
fn k_bit<const XLEN: usize, const K: u64>(i: usize) -> bool {
    (K >> (XLEN - 1 - i)) & 1 == 1
}

/// `Σ_{i: k_i=1} (1-x_i) · Π_{j<i} EQ(x_j,k_j)`, restricted to this suffix's bit window
/// (same per-bit accumulation as `LessThanConstPrefix`, with a local running `eq`).
pub enum LessThanConstSuffix<const XLEN: usize, const K: u64> {}

impl<const XLEN: usize, const K: u64> SparseDenseSuffix for LessThanConstSuffix<XLEN, K> {
    fn suffix_mle(b: LookupBits) -> u32 {
        let b_len = b.len();
        let global_start = XLEN - b_len;
        let b_u64: u64 = b.into();

        let mut lt = 0u32;
        let mut eq = 1u32;
        for pos in 0..b_len {
            let global_index = global_start + pos;
            let bit = ((b_u64 >> (b_len - 1 - pos)) & 1) as u32;
            if k_bit::<XLEN, K>(global_index) {
                lt += eq * (1 - bit);
            }
            eq *= if k_bit::<XLEN, K>(global_index) {
                bit
            } else {
                1 - bit
            };
        }
        lt
    }
}

/// `LessThanConstSuffix` specialized to `K = TRIG_PERIOD_MODULUS`.
pub type TrigLessThanConstSuffix<const XLEN: usize> =
    LessThanConstSuffix<XLEN, { TRIG_PERIOD_MODULUS as u64 }>;
