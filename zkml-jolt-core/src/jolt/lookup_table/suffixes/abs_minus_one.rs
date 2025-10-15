use super::LookupBits;

use super::SparseDenseSuffix;

/// Suffix for the Abs lookup, handles the negative case.
/// Returns -x for negative x, and 0 for positive x. If sign bit is not in the suffix, returns -x for all x.
pub enum AbsIncrSuffix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize> SparseDenseSuffix for AbsIncrSuffix<WORD_SIZE> {
    fn suffix_mle(b: LookupBits) -> u32 {
        if b.len() == 0 {
            // Suffix is empty, prefix will handle whole word, we just cover the "+1" from (-x = !x + 1)
            return 0;
        }
        let sign_bit = if b.len() < WORD_SIZE {
            // Suffix is too small, set to 1 (prefix will handle sign bit)
            1
        } else {
            // Extract bit at position (half_word_size), which is the sign bit
            let bits = u64::from(b);
            let sign_bit_position = WORD_SIZE - 1;
            let sign_bit = (bits >> sign_bit_position) & 1;
            sign_bit as u32
        };

        // mask must equal b length, but should not exceed exceed WORD_SIZE-1 to not include sign bit
        let mask = std::cmp::min(WORD_SIZE - 1, b.len());
        sign_bit * ((!u64::from(b)).wrapping_add(0) % (1 << mask)) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    #[ignore]
    fn test_suffix_mle() {
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let number = rng.next_u32() as i32;

            // phase 0
            let bitlookup = LookupBits::new(number as u64, 48);
            let mle = AbsIncrSuffix::<32>::suffix_mle(bitlookup);
            let expected = if number < 0 { (-number) as u32 } else { 0 };
            assert_eq!(mle, expected);

            // phase 1
            let bitlookup = LookupBits::new(number as u64, 32);
            let mle = AbsIncrSuffix::<32>::suffix_mle(bitlookup);
            let expected = if number < 0 { (-number) as u32 } else { 0 };
            assert_eq!(mle, expected);

            // phase 2
            let bitlookup = LookupBits::new(number as u64, 16);
            let mle = AbsIncrSuffix::<32>::suffix_mle(bitlookup);
            // sign bit is out of range, so always produces opposite of number's suffix
            let expected = (-(number as i16)) as u16 as u32;
            assert_eq!(mle, expected);

            // phase 3
            let bitlookup = LookupBits::new(0, 0);
            let mle = AbsIncrSuffix::<32>::suffix_mle(bitlookup);
            // all word is handled by prefix, we just cover the +1 for negative case
            assert_eq!(mle, 1);
        }
    }
}
