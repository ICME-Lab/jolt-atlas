//! The binary field `F_{2^128} = F_2[x] / (x^128 + x^7 + x^2 + x + 1)`.
//!
//! An element is its coefficient vector: bit `i` of the `u128` is the coefficient of `x^i`.
//! Addition is XOR. Multiplication is a carry-less 128 × 128 product followed by reduction with
//! `x^128 = x^7 + x^2 + x + 1`; the 64 × 64 carry-less products use `PMULL` on aarch64 and
//! `PCLMULQDQ` on x86-64 when the CPU has them, and a bitwise loop otherwise.
//!
//! [`GENERATOR`] generates the multiplicative group, of order `2^128 − 1`: BitZ proves integer
//! claims in its exponent.

use atlas_algebra::serialize::{Canonical, Error};
use std::io::{Read, Write};
use std::ops::{Add, AddAssign, Mul, MulAssign};

/// An element of `F_{2^128}`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Gf128(pub u128);

/// A generator of the multiplicative group `F_{2^128}^*`.
pub const GENERATOR: Gf128 = Gf128(GENERATOR_BITS);

/// The smallest element, as an integer, whose order is `2^128 − 1` (pinned by
/// `generator_has_full_order`).
const GENERATOR_BITS: u128 = 0x2;

/// The prime factors of `2^128 − 1`, the order of the multiplicative group.
pub const GROUP_ORDER_FACTORS: [u128; 9] = [
    3,
    5,
    17,
    257,
    641,
    65_537,
    274_177,
    6_700_417,
    67_280_421_310_721,
];

impl Gf128 {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(1);

    #[inline]
    pub fn is_zero(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub fn square(self) -> Self {
        self * self
    }

    /// `self^e`, by square and multiply from the low bit.
    pub fn pow(self, mut e: u128) -> Self {
        let mut base = self;
        let mut acc = Self::ONE;
        while e != 0 {
            if e & 1 == 1 {
                acc *= base;
            }
            base = base.square();
            e >>= 1;
        }
        acc
    }

    /// The multiplicative inverse, `self^(2^128 − 2)`; `None` for zero.
    pub fn inverse(self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            Some(self.pow(u128::MAX - 1))
        }
    }

    /// The product computed without CPU carry-less multiply instructions.
    pub fn mul_portable(self, rhs: Self) -> Self {
        let (hi, lo) = clmul128(self.0, rhs.0, clmul64_portable);
        Self(reduce(hi, lo))
    }
}

impl Add for Gf128 {
    type Output = Self;
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}

impl AddAssign for Gf128 {
    #[inline]
    #[allow(clippy::suspicious_op_assign_impl)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl Mul for Gf128 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let (hi, lo) = clmul128(self.0, rhs.0, clmul64);
        Self(reduce(hi, lo))
    }
}

impl MulAssign for Gf128 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Canonical for Gf128 {
    fn encode<W: Write>(&self, writer: W) -> Result<(), Error> {
        self.0.encode(writer)
    }

    fn encoded_len(&self) -> usize {
        16
    }

    fn decode<R: Read>(reader: R) -> Result<Self, Error> {
        Ok(Self(u128::decode(reader)?))
    }
}

/// The 256-bit carry-less product of two 128-bit polynomials, as `(high, low)` halves, from four
/// 64 × 64 products.
#[inline(always)]
fn clmul128(a: u128, b: u128, clmul: impl Fn(u64, u64) -> u128) -> (u128, u128) {
    let (a0, a1) = (a as u64, (a >> 64) as u64);
    let (b0, b1) = (b as u64, (b >> 64) as u64);
    let lo = clmul(a0, b0);
    let hi = clmul(a1, b1);
    let mid = clmul(a0, b1) ^ clmul(a1, b0);
    (hi ^ (mid >> 64), lo ^ (mid << 64))
}

/// Reduces a 256-bit polynomial `high · x^128 + low` modulo `x^128 + x^7 + x^2 + x + 1`.
#[inline(always)]
fn reduce(hi: u128, lo: u128) -> u128 {
    // high · (x^7 + x^2 + x + 1) has degree at most 134: its low 128 bits, and the seven bits
    // that overflow, which are folded once more.
    let folded = hi ^ (hi << 1) ^ (hi << 2) ^ (hi << 7);
    let overflow = (hi >> 127) ^ (hi >> 126) ^ (hi >> 121);
    lo ^ folded ^ overflow ^ (overflow << 1) ^ (overflow << 2) ^ (overflow << 7)
}

/// The carry-less product of two 64-bit polynomials, one bit of `b` at a time.
fn clmul64_portable(a: u64, b: u64) -> u128 {
    let a = a as u128;
    let mut acc = 0u128;
    let mut b = b;
    while b != 0 {
        let i = b.trailing_zeros();
        acc ^= a << i;
        b &= b - 1;
    }
    acc
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn clmul64(a: u64, b: u64) -> u128 {
    #[target_feature(enable = "neon,aes")]
    unsafe fn pmull(a: u64, b: u64) -> u128 {
        core::arch::aarch64::vmull_p64(a, b)
    }
    if cfg!(target_feature = "aes") || std::arch::is_aarch64_feature_detected!("aes") {
        // SAFETY: the `aes` feature, which provides `PMULL`, is present on this CPU.
        unsafe { pmull(a, b) }
    } else {
        clmul64_portable(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn clmul64(a: u64, b: u64) -> u128 {
    #[target_feature(enable = "pclmulqdq,sse2")]
    unsafe fn pclmul(a: u64, b: u64) -> u128 {
        use core::arch::x86_64::{_mm_clmulepi64_si128, _mm_set_epi64x};
        let product =
            _mm_clmulepi64_si128(_mm_set_epi64x(0, a as i64), _mm_set_epi64x(0, b as i64), 0);
        core::mem::transmute::<_, u128>(product)
    }
    if cfg!(target_feature = "pclmulqdq") || std::arch::is_x86_feature_detected!("pclmulqdq") {
        // SAFETY: `PCLMULQDQ` is present on this CPU; SSE2 is part of x86-64.
        unsafe { pclmul(a, b) }
    } else {
        clmul64_portable(a, b)
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline(always)]
fn clmul64(a: u64, b: u64) -> u128 {
    clmul64_portable(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, RngCore, SeedableRng};

    /// `x^128` reduced: `x^7 + x^2 + x + 1`.
    const REDUCTION: u128 = 0x87;

    /// Multiplication from the definition: add `a · x^i` for every set bit `i` of `b`, where
    /// multiplying by `x` shifts and, on overflow, adds `x^7 + x^2 + x + 1`.
    fn mul_reference(a: Gf128, b: Gf128) -> Gf128 {
        let (mut a, b) = (a.0, b.0);
        let mut acc = 0u128;
        for i in 0..128 {
            if (b >> i) & 1 == 1 {
                acc ^= a;
            }
            let carry = a >> 127;
            a <<= 1;
            if carry == 1 {
                a ^= REDUCTION;
            }
        }
        Gf128(acc)
    }

    fn random(rng: &mut StdRng) -> Gf128 {
        Gf128(((rng.next_u64() as u128) << 64) | rng.next_u64() as u128)
    }

    fn samples() -> Vec<Gf128> {
        let mut rng = StdRng::seed_from_u64(0x6f128);
        let mut values = vec![
            Gf128::ZERO,
            Gf128::ONE,
            Gf128(2),
            Gf128(REDUCTION),
            Gf128(1 << 127),
            Gf128(u128::MAX),
            Gf128(u64::MAX as u128),
            Gf128((u64::MAX as u128) << 64),
        ];
        values.extend((0..64).map(|_| random(&mut rng)));
        values
    }

    #[test]
    fn multiplication_matches_the_definition() {
        let values = samples();
        for &a in &values {
            for &b in &values {
                let expected = mul_reference(a, b);
                assert_eq!(a * b, expected, "{a:?} * {b:?}");
                assert_eq!(a.mul_portable(b), expected, "portable {a:?} * {b:?}");
            }
        }
    }

    #[test]
    fn field_laws_hold() {
        let values = samples();
        for &a in &values {
            assert_eq!(a * Gf128::ONE, a);
            assert_eq!(a + a, Gf128::ZERO);
            assert_eq!(a.square(), mul_reference(a, a));
            match a.inverse() {
                Some(inv) => assert_eq!(mul_reference(a, inv), Gf128::ONE),
                None => assert!(a.is_zero()),
            }
        }
        for w in values.windows(3) {
            let (a, b, c) = (w[0], w[1], w[2]);
            assert_eq!(a * b, b * a);
            assert_eq!((a * b) * c, a * (b * c));
            assert_eq!(a * (b + c), a * b + a * c);
        }
    }

    #[test]
    fn generator_has_full_order() {
        let order = u128::MAX;
        assert_eq!(GROUP_ORDER_FACTORS.iter().product::<u128>(), order);
        let has_full_order = |g: Gf128| {
            g.pow(order) == Gf128::ONE
                && GROUP_ORDER_FACTORS
                    .iter()
                    .all(|p| g.pow(order / p) != Gf128::ONE)
        };
        assert!(has_full_order(GENERATOR));
        // The pinned generator is the smallest such element.
        assert!((2..GENERATOR_BITS).all(|g| !has_full_order(Gf128(g))));
    }

    #[test]
    fn encoding_round_trips() {
        for a in samples() {
            let bytes = atlas_algebra::serialize::to_bytes(&a).unwrap();
            assert_eq!(bytes.len(), 16);
            assert_eq!(
                atlas_algebra::serialize::from_bytes::<Gf128>(&bytes).unwrap(),
                a
            );
        }
    }
}
