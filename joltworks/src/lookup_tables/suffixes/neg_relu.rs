use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// Suffix component for Relu(-x) lookup table MLE decomposition.
pub enum NegReluSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for NegReluSuffix<XLEN> {
    fn suffix_mle(bits: LookupBits) -> u32 {
        let b: u64 = bits.into();
        let num_bits = bits.len();

        if num_bits >= XLEN {
            // Two's complement negation, gated by sign bit: neg_relu(x) = is_negative * |x|
            let is_negative = (b >> (XLEN - 1)) & 1;
            let abs_value = (!b % (1 << (XLEN - 1))) as u32 + 1;
            abs_value * is_negative as u32
        } else {
            // Partial suffix: flip the available bits (!b truncated to num_bits).
            (!b % (1 << num_bits)) as u32
        }
    }
}
