use jolt_core::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// Returns the lower WORD_SIZE - 1 bits. Used to range-check values to be in
/// the range [0, 2^WORD_SIZE).
pub enum ReluSuffix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize> SparseDenseSuffix for ReluSuffix<WORD_SIZE> {
    fn suffix_mle(b: LookupBits) -> u32 {
        let sign_bit = if b.len() < WORD_SIZE {
            // Suffix is too small, return 0 (prefix will handle)
            0
        } else {
            // Extract bit at position (half_word_size), which is the sign bit
            let bits = u64::from(b);
            let sign_bit_position = WORD_SIZE - 1;
            let sign_bit = (bits >> sign_bit_position) & 1;
            sign_bit as u32
        };

        (1 - sign_bit) * (u64::from(b) % (1 << (WORD_SIZE - 1))) as u32
    }
}
