//! The binary field `F_{2^128} = F_2[x] / (x^128 + x^7 + x^2 + x + 1)`.
//!
//! An element is its coefficient vector: bit `i` of the `u128` is the coefficient of `x^i`.
//! Addition is XOR. Multiplication is a carry-less 128 × 128 product, from three 64 × 64
//! products, followed by reduction with `x^128 = x^7 + x^2 + x + 1`; the 64 × 64 products use
//! `PMULL` on aarch64 and `PCLMULQDQ` on x86-64 when the CPU has them, and a bitwise loop
//! otherwise.
//!
//! [`GENERATOR`] generates the multiplicative group, of order `2^128 − 1`: BitZ proves integer
//! claims in its exponent.

use atlas_algebra::serialize::{Canonical, Error};
use std::io::{Read, Write};
use std::ops::{Add, AddAssign, Mul, MulAssign};

/// An element of `F_{2^128}`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
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

    /// A vector of `len` zeros. The memory comes zeroed from the allocator, so nothing is
    /// written until an element is, on whichever thread writes it.
    pub fn zeros(len: usize) -> Vec<Self> {
        let mut words = std::mem::ManuallyDrop::new(vec![0u128; len]);
        // SAFETY: `Gf128` is `repr(transparent)` over `u128`, so the two vectors have the same
        // element layout, and the pointer, length and capacity are those of a live `Vec` that
        // is not dropped.
        unsafe {
            Vec::from_raw_parts(
                words.as_mut_ptr().cast::<Self>(),
                words.len(),
                words.capacity(),
            )
        }
    }

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
        let (hi, lo) = mul_wide_portable(self.0, rhs.0);
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
        Self(mul(self.0, rhs.0))
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

/// Combines the three 64 × 64 products of Karatsuba's method, `a0·b0`, `a1·b1` and
/// `(a0 + a1)·(b0 + b1)`, into the 256-bit product as `(high, low)` halves.
#[inline(always)]
fn karatsuba(low: u128, high: u128, cross: u128) -> (u128, u128) {
    let mid = cross ^ low ^ high;
    (high ^ (mid >> 64), low ^ (mid << 64))
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

fn mul_wide_portable(a: u128, b: u128) -> (u128, u128) {
    let (a0, a1) = (a as u64, (a >> 64) as u64);
    let (b0, b1) = (b as u64, (b >> 64) as u64);
    karatsuba(
        clmul64_portable(a0, b0),
        clmul64_portable(a1, b1),
        clmul64_portable(a0 ^ a1, b0 ^ b1),
    )
}

/// The unreduced product, in one call whether or not the build enables the CPU's carry-less multiply:
/// with the feature enabled at compile time the accelerated body is inlined, and otherwise
/// it is reached through one runtime-detected call for the whole multiplication.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn mul_wide(a: u128, b: u128) -> (u128, u128) {
    #[inline]
    #[target_feature(enable = "neon,aes")]
    unsafe fn accelerated(a: u128, b: u128) -> (u128, u128) {
        use core::arch::aarch64::vmull_p64;
        let (a0, a1) = (a as u64, (a >> 64) as u64);
        let (b0, b1) = (b as u64, (b >> 64) as u64);
        karatsuba(
            vmull_p64(a0, b0),
            vmull_p64(a1, b1),
            vmull_p64(a0 ^ a1, b0 ^ b1),
        )
    }
    if cfg!(target_feature = "aes") || std::arch::is_aarch64_feature_detected!("aes") {
        // SAFETY: the `aes` feature, which provides `PMULL`, is present on this CPU.
        unsafe { accelerated(a, b) }
    } else {
        mul_wide_portable(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn mul_wide(a: u128, b: u128) -> (u128, u128) {
    #[inline]
    #[target_feature(enable = "pclmulqdq,sse2")]
    unsafe fn accelerated(a: u128, b: u128) -> (u128, u128) {
        use core::arch::x86_64::{__m128i, _mm_clmulepi64_si128, _mm_set_epi64x};
        let clmul = |x: u64, y: u64| -> u128 {
            let product: __m128i =
                _mm_clmulepi64_si128(_mm_set_epi64x(0, x as i64), _mm_set_epi64x(0, y as i64), 0);
            core::mem::transmute::<__m128i, u128>(product)
        };
        let (a0, a1) = (a as u64, (a >> 64) as u64);
        let (b0, b1) = (b as u64, (b >> 64) as u64);
        karatsuba(clmul(a0, b0), clmul(a1, b1), clmul(a0 ^ a1, b0 ^ b1))
    }
    if cfg!(target_feature = "pclmulqdq") || std::arch::is_x86_feature_detected!("pclmulqdq") {
        // SAFETY: `PCLMULQDQ` is present on this CPU; SSE2 is part of x86-64.
        unsafe { accelerated(a, b) }
    } else {
        mul_wide_portable(a, b)
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline(always)]
fn mul_wide(a: u128, b: u128) -> (u128, u128) {
    mul_wide_portable(a, b)
}

/// The product of two elements.
#[inline(always)]
fn mul(a: u128, b: u128) -> u128 {
    let (hi, lo) = mul_wide(a, b);
    reduce(hi, lo)
}

/// A sum of products, kept unreduced: reduction is `F_2`-linear, so the 256-bit carry-less
/// products are added as they are and reduced once.
#[derive(Clone, Copy, Debug, Default)]
pub struct WideSum {
    hi: u128,
    lo: u128,
}

impl WideSum {
    pub const ZERO: Self = Self { hi: 0, lo: 0 };

    /// Adds `a · b`.
    #[inline(always)]
    pub fn add_product(&mut self, a: Gf128, b: Gf128) {
        let (hi, lo) = mul_wide(a.0, b.0);
        self.hi ^= hi;
        self.lo ^= lo;
    }

    /// The sum of two sums.
    #[inline]
    pub fn plus(self, other: Self) -> Self {
        Self {
            hi: self.hi ^ other.hi,
            lo: self.lo ^ other.lo,
        }
    }

    /// The sum as a field element.
    #[inline]
    pub fn reduce(self) -> Gf128 {
        Gf128(reduce(self.hi, self.lo))
    }
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
    fn wide_sums_reduce_to_the_sum_of_products() {
        let values = samples();
        let mut wide = WideSum::ZERO;
        let mut expected = Gf128::ZERO;
        for pair in values.chunks_exact(2) {
            wide.add_product(pair[0], pair[1]);
            expected += mul_reference(pair[0], pair[1]);
        }
        assert_eq!(wide.reduce(), expected);
        assert_eq!(wide.plus(wide).reduce(), Gf128::ZERO);
        assert_eq!(WideSum::ZERO.reduce(), Gf128::ZERO);
    }

    #[test]
    fn zero_vectors_are_zero_and_writable() {
        let mut zeros = Gf128::zeros(1000);
        assert_eq!(zeros.len(), 1000);
        assert!(zeros.iter().all(|z| z.is_zero()));
        zeros[999] = Gf128::ONE;
        zeros.push(Gf128(7));
        assert_eq!((zeros[999], zeros[1000]), (Gf128::ONE, Gf128(7)));
        assert!(Gf128::zeros(0).is_empty());
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
