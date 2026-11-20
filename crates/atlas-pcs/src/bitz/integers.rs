//! Integer arithmetic for BitZ's projection of a claim over Fp128 onto a random prime.
//!
//! The claim is lifted to the integers, where it is below `2^320`, and reduced modulo a prime
//! drawn from the transcript. Arithmetic modulo that prime is on `u128` values below `2^127`,
//! so sums never overflow.
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::unimplemented,
    clippy::todo,
    clippy::indexing_slicing
)]

use atlas_algebra::field::fp128::Fp128;
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_transcript::Transcript;

/// The modulus of Fp128, `2^128 − C`.
pub const FP128_MODULUS: u128 = Fp128::C.wrapping_neg();

/// An unsigned integer below `2^320`, as little-endian 64-bit limbs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Wide(pub [u64; 5]);

impl Wide {
    /// `a · b` for `a < 2^192` (low and high parts) and `b < 2^128`.
    pub fn product(a: (u128, u64), b: u128) -> Self {
        let a = [a.0 as u64, (a.0 >> 64) as u64, a.1];
        let b = [b as u64, (b >> 64) as u64];
        let mut limbs = [0u64; 5];
        for (i, &x) in a.iter().enumerate() {
            let mut carry = 0u128;
            for (j, &y) in b.iter().enumerate() {
                if let Some(limb) = limbs.get_mut(i + j) {
                    let t = x as u128 * y as u128 + *limb as u128 + carry;
                    *limb = t as u64;
                    carry = t >> 64;
                }
            }
            if let Some(limb) = limbs.get_mut(i + b.len()) {
                // The partial product fits: this limb was zero before the carry.
                *limb = carry as u64;
            }
        }
        Self(limbs)
    }

    /// `self + other`, or `None` on overflow of `2^320`.
    pub fn checked_add(&self, other: &Self) -> Option<Self> {
        let mut limbs = [0u64; 5];
        let mut carry = false;
        for ((out, &x), &y) in limbs.iter_mut().zip(&self.0).zip(&other.0) {
            let (sum, c1) = x.overflowing_add(y);
            let (sum, c2) = sum.overflowing_add(carry as u64);
            *out = sum;
            carry = c1 || c2;
        }
        (!carry).then_some(Self(limbs))
    }

    /// `self << shift` for `shift < 64`, or `None` if bits are lost.
    pub fn checked_shl(&self, shift: u32) -> Option<Self> {
        if shift >= 64 {
            return None;
        }
        if shift == 0 {
            return Some(*self);
        }
        let mut limbs = [0u64; 5];
        let mut carry = 0u64;
        for (out, &x) in limbs.iter_mut().zip(&self.0) {
            *out = x << shift | carry;
            carry = x >> (64 - shift);
        }
        (carry == 0).then_some(Self(limbs))
    }

    /// Whether `self ≤ other`.
    pub fn le(&self, other: &Self) -> bool {
        for (x, y) in self.0.iter().rev().zip(other.0.iter().rev()) {
            if x != y {
                return x < y;
            }
        }
        true
    }

    /// The residue modulo the Fp128 modulus, as a field element.
    pub fn to_fp128(&self) -> Fp128 {
        let base = Fp128::from(1u128 << 64);
        self.0.iter().rev().fold(Fp128::from(0u128), |acc, &limb| {
            acc * base + Fp128::from(limb as u128)
        })
    }

    /// The residue modulo `q < 2^127`.
    pub fn rem(&self, q: u128) -> u128 {
        let base = (1u128 << 64) % q;
        self.0.iter().rev().fold(0, |acc, &limb| {
            add_mod(mul_mod(acc, base, q), limb as u128 % q, q)
        })
    }
}

/// `(a + b) mod q` for `a, b < q < 2^127`.
#[inline]
pub fn add_mod(a: u128, b: u128, q: u128) -> u128 {
    let sum = a + b;
    if sum >= q {
        sum - q
    } else {
        sum
    }
}

/// `(a · b) mod q` for `a, b < q < 2^127`, by doubling and adding.
pub fn mul_mod(mut a: u128, mut b: u128, q: u128) -> u128 {
    let mut acc = 0;
    while b != 0 {
        if b & 1 == 1 {
            acc = add_mod(acc, a, q);
        }
        a = add_mod(a, a, q);
        b >>= 1;
    }
    acc
}

fn pow_mod(mut base: u128, mut exponent: u128, q: u128) -> u128 {
    let mut acc = 1 % q;
    while exponent != 0 {
        if exponent & 1 == 1 {
            acc = mul_mod(acc, base, q);
        }
        base = mul_mod(base, base, q);
        exponent >>= 1;
    }
    acc
}

/// The first 64 primes: trial divisors, and the Miller–Rabin bases.
const SMALL_PRIMES: [u128; 64] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307,
    311,
];

/// Whether `n < 2^127` is prime: trial division, then Miller–Rabin to 64 fixed bases. A
/// composite passes one base with probability at most 1/4 over the choice of base; the bases
/// here are fixed, so for an adversarially chosen `n` this is a heuristic, and the candidates
/// it is run on are transcript outputs.
pub fn is_prime(n: u128) -> bool {
    if n < 2 {
        return false;
    }
    for &p in &SMALL_PRIMES {
        if n == p {
            return true;
        }
        if n.is_multiple_of(p) {
            return false;
        }
    }
    let twos = (n - 1).trailing_zeros();
    let odd = (n - 1) >> twos;
    'bases: for &base in &SMALL_PRIMES {
        let mut x = pow_mod(base, odd, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..twos {
            x = mul_mod(x, x, n);
            if x == n - 1 {
                continue 'bases;
            }
        }
        return false;
    }
    true
}

/// The most candidates drawn before giving up: a prime is found within a few hundred except
/// with negligible probability.
const MAX_PRIME_CANDIDATES: usize = 1 << 16;

/// Draws a uniform prime in `[2^low_bits, 2^high_bits)` from the transcript, by rejection.
pub fn sample_prime<T: Transcript>(
    low_bits: u32,
    high_bits: u32,
    transcript: &mut T,
) -> Result<u128, ProofVerifyError> {
    if !(1..127).contains(&low_bits) || low_bits >= high_bits || high_bits > 127 {
        return Err(ProofVerifyError::Malformed("prime interval".into()));
    }
    let mask = (1u128 << high_bits) - 1;
    for _ in 0..MAX_PRIME_CANDIDATES {
        let candidate = transcript.challenge_u128() & mask;
        if candidate >> low_bits != 0 && candidate & 1 == 1 && is_prime(candidate) {
            return Ok(candidate);
        }
    }
    Err(ProofVerifyError::InvalidOpeningProof(
        "no prime among the transcript's candidates".into(),
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use atlas_transcript::{Blake2bProver, ProvingTranscript};

    /// Schoolbook product of little-endian limb vectors, the reference for `Wide::product`.
    fn reference_product(a: &[u64], b: &[u64]) -> Vec<u64> {
        let mut out = vec![0u64; a.len() + b.len()];
        for (i, &x) in a.iter().enumerate() {
            let mut carry = 0u128;
            for (j, &y) in b.iter().enumerate() {
                let t = x as u128 * y as u128 + out[i + j] as u128 + carry;
                out[i + j] = t as u64;
                carry = t >> 64;
            }
            out[i + b.len()] = carry as u64;
        }
        out
    }

    #[test]
    fn wide_arithmetic_matches_schoolbook() {
        let cases = [
            ((u128::MAX, u64::MAX), u128::MAX),
            ((0, 0), 5),
            ((1 << 100 | 7, 3), 1 << 127 | 12345),
            (
                (0x1234_5678_9abc_def0_0fed_cba9_8765_4321, 0xffff),
                0xdead_beef,
            ),
        ];
        for (a, b) in cases {
            let expected = reference_product(
                &[a.0 as u64, (a.0 >> 64) as u64, a.1],
                &[b as u64, (b >> 64) as u64],
            );
            assert_eq!(Wide::product(a, b).0.to_vec(), expected);
        }
        let x = Wide([u64::MAX, 1, 0, 0, 0]);
        assert_eq!(
            x.checked_add(&Wide([1, 0, 0, 0, 0])),
            Some(Wide([0, 2, 0, 0, 0]))
        );
        assert_eq!(
            Wide([0, 0, 0, 0, u64::MAX]).checked_add(&Wide([0, 0, 0, 0, 1])),
            None
        );
        assert_eq!(x.checked_shl(4), Some(Wide([u64::MAX << 4, 0x1f, 0, 0, 0])));
        assert_eq!(Wide([0, 0, 0, 0, 1 << 60]).checked_shl(4), None);
        assert!(x.le(&x) && Wide([5, 0, 0, 0, 0]).le(&x) && !x.le(&Wide([5, 0, 0, 0, 0])));
    }

    #[test]
    fn residues_match_small_integers() {
        // 2^64·3 + 10, modulo small numbers and as a field element.
        let x = Wide([10, 3, 0, 0, 0]);
        let value = (3u128 << 64) + 10;
        for q in [97u128, 65_537, (1 << 61) - 1] {
            assert_eq!(x.rem(q), value % q);
        }
        assert_eq!(x.to_fp128(), Fp128::from(value));
        // The modulus itself is zero in the field, and 2^128 is the offset.
        let modulus = Wide([FP128_MODULUS as u64, (FP128_MODULUS >> 64) as u64, 0, 0, 0]);
        assert_eq!(modulus.to_fp128(), Fp128::from(0u128));
        assert_eq!(Wide([0, 0, 1, 0, 0]).to_fp128(), Fp128::from(Fp128::C));
    }

    #[test]
    fn modular_multiplication_matches_wide_products() {
        let q = (1u128 << 113) - 133;
        for (a, b) in [
            (q - 1, q - 1),
            (123_456_789, q - 2),
            (1 << 100, 1 << 100),
            (0, 5),
        ] {
            assert_eq!(mul_mod(a, b, q), Wide::product((a, 0), b).rem(q));
        }
    }

    #[test]
    fn primality_agrees_with_known_primes_and_composites() {
        let mersenne_61 = (1u128 << 61) - 1;
        let mersenne_89 = (1u128 << 89) - 1;
        let mersenne_107 = (1u128 << 107) - 1;
        for p in [
            2,
            3,
            311,
            313,
            65_537,
            mersenne_61,
            mersenne_89,
            mersenne_107,
        ] {
            assert!(is_prime(p), "{p}");
        }
        // Carmichael numbers, a square of a prime, and a product of two large primes.
        for n in [
            0,
            1,
            561,
            41_041,
            825_265,
            313 * 313,
            mersenne_61 * 65_537,
            (1u128 << 67) - 1,
        ] {
            assert!(!is_prime(n), "{n}");
        }
    }

    #[test]
    fn sampled_primes_are_primes_in_the_interval() {
        let mut transcript = Blake2bProver::new(b"prime-test");
        for _ in 0..4 {
            let q = sample_prime(100, 105, &mut transcript).unwrap();
            assert!(q >> 100 != 0 && q >> 105 == 0 && is_prime(q));
        }
        assert!(sample_prime(105, 105, &mut transcript).is_err());
        assert!(sample_prime(100, 128, &mut transcript).is_err());
    }
}
