//! Bit vectors and their packing into `F_{2^128}`.
//!
//! A bit vector is stored as `u64` words, bit `b` at position `b mod 64` of word `b / 64`. Each
//! 128 consecutive bits form one field element in the basis `1, x, …, x^127`: bit `u` of a
//! group is the coefficient of `x^u`. As a multilinear polynomial, the low seven variables of
//! a bit's index select its position inside the element and the rest select the element.

use super::gf128::Gf128;

/// The number of variables packing removes: `log2` of the bits per field element.
pub const LOG_PACKING: usize = 7;

/// Bit `index` of a word-packed bit vector.
#[inline]
pub fn bit(words: &[u64], index: usize) -> bool {
    (words[index / 64] >> (index % 64)) & 1 == 1
}

/// Packs a bit vector whose length is a multiple of 128 into field elements.
pub fn pack(words: &[u64]) -> Vec<Gf128> {
    assert!(
        words.len().is_multiple_of(2),
        "a packed bit vector is whole elements"
    );
    words
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| Gf128(pair[0] as u128 | (pair[1] as u128) << 64))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_elements_hold_their_bits_as_coefficients() {
        let words = [0x8000_0000_0000_0001u64, 0x2, 0, 1 << 63];
        let packed = pack(&words);
        assert_eq!(packed.len(), 2);
        for (e, element) in packed.iter().enumerate() {
            for u in 0..128 {
                assert_eq!((element.0 >> u) & 1 == 1, bit(&words, 128 * e + u));
            }
        }
        assert_eq!(packed[0], Gf128(1 | 1 << 63 | 1 << 65));
        assert_eq!(packed[1], Gf128(1 << 127));
    }
}
