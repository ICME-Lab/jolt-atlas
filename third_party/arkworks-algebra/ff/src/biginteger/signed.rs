use crate::biginteger::{BigInt, BigInteger};

#[cfg(feature = "default")]
use allocative::Allocative;

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, Read, SerializationError, Valid, Validate,
    Write,
};
use ark_std::Zero;
use core::cmp::Ordering;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A signed big integer using arkworks BigInt for magnitude and a sign bit.
///
/// Notes:
/// - Zero is not canonicalized: a zero magnitude can be paired with either sign.
///   Structural equality distinguishes `+0` and `-0` (since the sign bit differs).
/// - Ordering treats `+0` and `-0` as equal: comparisons return `Ordering::Equal` when
///   both magnitudes are zero regardless of sign.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "default", derive(Allocative))]
pub struct SignedBigInt<const N: usize> {
    pub magnitude: BigInt<N>,
    pub is_positive: bool,
}

impl<const N: usize> Default for SignedBigInt<N> {
    #[inline]
    fn default() -> Self {
        Self::zero()
    }
}

impl<const N: usize> Zero for SignedBigInt<N> {
    #[inline]
    fn zero() -> Self {
        Self::zero()
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.magnitude.is_zero()
    }
}

pub type S64 = SignedBigInt<1>;
pub type S128 = SignedBigInt<2>;
pub type S192 = SignedBigInt<3>;
pub type S256 = SignedBigInt<4>;

impl<const N: usize> SignedBigInt<N> {
    #[inline]
    fn cmp_magnitude_mixed<const M: usize>(&self, rhs: &SignedBigInt<M>) -> Ordering {
        let max_limbs = if N > M { N } else { M };
        let mut i = max_limbs;
        while i > 0 {
            let idx = i - 1;
            let a = if idx < N { self.magnitude.0[idx] } else { 0u64 };
            let b = if idx < M { rhs.magnitude.0[idx] } else { 0u64 };
            if a > b {
                return Ordering::Greater;
            }
            if a < b {
                return Ordering::Less;
            }
            i -= 1;
        }
        Ordering::Equal
    }
    /// Construct from limbs and sign; limbs are little-endian.
    #[inline]
    pub fn new(limbs: [u64; N], is_positive: bool) -> Self {
        Self {
            magnitude: BigInt::new(limbs),
            is_positive,
        }
    }

    /// Construct from an existing BigInt magnitude and sign.
    #[inline]
    pub fn from_bigint(magnitude: BigInt<N>, is_positive: bool) -> Self {
        Self {
            magnitude,
            is_positive,
        }
    }

    /// Zero value with a positive sign (negative zero allowed elsewhere).
    #[inline]
    pub fn zero() -> Self {
        Self {
            magnitude: BigInt::from(0u64),
            is_positive: true,
        }
    }

    /// One with a positive sign.
    #[inline]
    pub fn one() -> Self {
        Self {
            magnitude: BigInt::from(1u64),
            is_positive: true,
        }
    }

    /// Borrow the magnitude (absolute value).
    #[inline]
    pub fn as_magnitude(&self) -> &BigInt<N> {
        &self.magnitude
    }

    /// Return the magnitude limbs by value (copy).
    #[inline]
    pub fn magnitude_limbs(&self) -> [u64; N] {
        self.magnitude.0
    }

    /// Borrow the magnitude limbs as a slice (avoids copying the array).
    #[inline]
    pub fn magnitude_slice(&self) -> &[u64] {
        self.magnitude.as_ref()
    }

    /// Return true iff the value is non-negative.
    #[inline]
    pub fn sign(&self) -> bool {
        self.is_positive
    }

    /// Compute self + other modulo 2^(64*N); carry beyond N limbs is dropped.
    #[inline]
    pub fn add(mut self, other: Self) -> Self {
        self += other;
        self
    }

    /// Compute self - other modulo 2^(64*N); borrow beyond N limbs is dropped.
    #[inline]
    pub fn sub(mut self, other: Self) -> Self {
        self -= other;
        self
    }

    /// Compute self * other and keep only the low N limbs; high limbs are discarded.
    #[inline]
    pub fn mul(mut self, other: Self) -> Self {
        self *= other;
        self
    }

    /// Flip the sign; zero is not canonicalized (negative zero may occur).
    #[inline]
    pub fn neg(self) -> Self {
        Self::from_bigint(self.magnitude, !self.is_positive)
    }

    // ===== in-place helpers =====
    /// In-place addition with sign handling; drops overflow beyond N limbs.
    #[inline(always)]
    fn add_assign_in_place(&mut self, rhs: &Self) {
        if self.is_positive == rhs.is_positive {
            let _carry = self.magnitude.add_with_carry(&rhs.magnitude);
            // overflow ignored by design
        } else {
            match self.magnitude.cmp(&rhs.magnitude) {
                Ordering::Greater | Ordering::Equal => {
                    let _borrow = self.magnitude.sub_with_borrow(&rhs.magnitude);
                },
                Ordering::Less => {
                    // Minimize copies: move rhs magnitude into place and subtract old self
                    let old = core::mem::replace(&mut self.magnitude, rhs.magnitude);
                    let _borrow = self.magnitude.sub_with_borrow(&old);
                    self.is_positive = rhs.is_positive;
                },
            }
        }
    }

    /// In-place subtraction with sign handling; drops borrow beyond N limbs.
    #[inline(always)]
    fn sub_assign_in_place(&mut self, rhs: &Self) {
        // Implement directly to avoid temporary construction
        if self.is_positive != rhs.is_positive {
            // Signs differ -> add magnitudes; sign remains self.is_positive
            let _carry = self.magnitude.add_with_carry(&rhs.magnitude);
        } else {
            match self.magnitude.cmp(&rhs.magnitude) {
                Ordering::Greater | Ordering::Equal => {
                    let _borrow = self.magnitude.sub_with_borrow(&rhs.magnitude);
                    // sign stays the same
                },
                Ordering::Less => {
                    // Result takes rhs magnitude minus self magnitude, sign flips
                    let old = core::mem::replace(&mut self.magnitude, rhs.magnitude);
                    let _borrow = self.magnitude.sub_with_borrow(&old);
                    self.is_positive = !self.is_positive;
                },
            }
        }
    }

    /// In-place multiply using low-limb product only; updates sign, discards high limbs.
    #[inline(always)]
    fn mul_assign_in_place(&mut self, rhs: &Self) {
        let low = self.magnitude.mul_low(&rhs.magnitude);
        self.magnitude = low;
        self.is_positive = self.is_positive == rhs.is_positive;
    }

    /// Zero-extend a smaller-width signed big integer into N limbs (little-endian).
    /// Preserves the sign bit; only the magnitude is widened by zero-extension.
    /// Debug-asserts that M <= N.
    #[inline]
    pub fn zero_extend_from<const M: usize>(smaller: &SignedBigInt<M>) -> SignedBigInt<N> {
        debug_assert!(
            M <= N,
            "cannot zero-extend: source has more limbs than destination"
        );
        let widened_mag = BigInt::<N>::zero_extend_from::<M>(&smaller.magnitude);
        SignedBigInt::from_bigint(widened_mag, smaller.is_positive)
    }
}

impl<const N: usize> SignedBigInt<N> {
    // ===== truncated-width operations =====

    /// Truncated add: compute (self + rhs) and fit into M limbs; overflow is ignored.
    #[inline]
    pub fn add_trunc<const M: usize>(&self, rhs: &SignedBigInt<N>) -> SignedBigInt<M> {
        if self.is_positive == rhs.is_positive {
            let mag = self.magnitude.add_trunc::<N, M>(&rhs.magnitude);
            return SignedBigInt::<M> {
                magnitude: mag,
                is_positive: self.is_positive,
            };
        }
        match self.magnitude.cmp(&rhs.magnitude) {
            Ordering::Greater | Ordering::Equal => {
                let mag = self.magnitude.sub_trunc::<N, M>(&rhs.magnitude);
                SignedBigInt::<M> {
                    magnitude: mag,
                    is_positive: self.is_positive,
                }
            },
            Ordering::Less => {
                let mag = rhs.magnitude.sub_trunc::<N, M>(&self.magnitude);
                SignedBigInt::<M> {
                    magnitude: mag,
                    is_positive: rhs.is_positive,
                }
            },
        }
    }

    /// Truncated sub: compute (self - rhs) and fit into M limbs; overflow is ignored.
    #[inline]
    pub fn sub_trunc<const M: usize>(&self, rhs: &SignedBigInt<N>) -> SignedBigInt<M> {
        if self.is_positive != rhs.is_positive {
            let mag = self.magnitude.add_trunc::<N, M>(&rhs.magnitude);
            return SignedBigInt::<M> {
                magnitude: mag,
                is_positive: self.is_positive,
            };
        }
        match self.magnitude.cmp(&rhs.magnitude) {
            Ordering::Greater | Ordering::Equal => {
                let mag = self.magnitude.sub_trunc::<N, M>(&rhs.magnitude);
                SignedBigInt::<M> {
                    magnitude: mag,
                    is_positive: self.is_positive,
                }
            },
            Ordering::Less => {
                let mag = rhs.magnitude.sub_trunc::<N, M>(&self.magnitude);
                SignedBigInt::<M> {
                    magnitude: mag,
                    is_positive: !self.is_positive,
                }
            },
        }
    }

    /// Truncated mixed-width addition: compute (self + rhs) where rhs can have a
    /// different limb count, and fit into P limbs; overflow is ignored.
    #[inline]
    pub fn add_trunc_mixed<const M: usize, const P: usize>(
        &self,
        rhs: &SignedBigInt<M>,
    ) -> SignedBigInt<P> {
        if self.is_positive == rhs.is_positive {
            let mag = self.magnitude.add_trunc::<M, P>(&rhs.magnitude);
            return SignedBigInt::<P> {
                magnitude: mag,
                is_positive: self.is_positive,
            };
        }
        match self.cmp_magnitude_mixed(rhs) {
            Ordering::Greater | Ordering::Equal => {
                let mag = self.magnitude.sub_trunc::<M, P>(&rhs.magnitude);
                SignedBigInt::<P> {
                    magnitude: mag,
                    is_positive: self.is_positive,
                }
            },
            Ordering::Less => {
                let mag = rhs.magnitude.sub_trunc::<N, P>(&self.magnitude);
                SignedBigInt::<P> {
                    magnitude: mag,
                    is_positive: rhs.is_positive,
                }
            },
        }
    }

    /// Truncated mul: compute self * rhs and fit into P limbs; no assumption on P; overflow ignored.
    #[inline]
    pub fn mul_trunc<const M: usize, const P: usize>(
        &self,
        rhs: &SignedBigInt<M>,
    ) -> SignedBigInt<P> {
        let mag = self.magnitude.mul_trunc::<M, P>(&rhs.magnitude);
        let sign = self.is_positive == rhs.is_positive;
        SignedBigInt::<P> {
            magnitude: mag,
            is_positive: sign,
        }
    }

    /// Fused multiply-add: acc += self * rhs, fitted into P limbs; overflow is ignored.
    #[inline]
    pub fn fmadd_trunc<const M: usize, const P: usize>(
        &self,
        rhs: &SignedBigInt<M>,
        acc: &mut SignedBigInt<P>,
    ) {
        let prod_mag = self.magnitude.mul_trunc::<M, P>(&rhs.magnitude);
        let prod_sign = self.is_positive == rhs.is_positive;
        if acc.is_positive == prod_sign {
            let _ = acc.magnitude.add_with_carry(&prod_mag);
        } else {
            match acc.magnitude.cmp(&prod_mag) {
                Ordering::Greater | Ordering::Equal => {
                    let _ = acc.magnitude.sub_with_borrow(&prod_mag);
                },
                Ordering::Less => {
                    let old = core::mem::replace(&mut acc.magnitude, prod_mag);
                    let _ = acc.magnitude.sub_with_borrow(&old);
                    acc.is_positive = prod_sign;
                },
            }
        }
    }
}

impl<const N: usize> SignedBigInt<N> {
    // ===== generic conversions =====

    /// Construct from u64 with positive sign.
    #[inline]
    pub fn from_u64(value: u64) -> Self {
        Self::from_bigint(BigInt::from(value), true)
    }

    /// Construct from (u64, sign); sign=true is non-negative.
    #[inline]
    pub fn from_u64_with_sign(value: u64, is_positive: bool) -> Self {
        Self::from_bigint(BigInt::from(value), is_positive)
    }

    /// Construct from i64; magnitude is |value|, sign reflects value>=0.
    #[inline]
    pub fn from_i64(value: i64) -> Self {
        if value >= 0 {
            Self::from_bigint(BigInt::from(value as u64), true)
        } else {
            // wrapping_neg handles i64::MIN
            Self::from_bigint(BigInt::from(value.wrapping_neg() as u64), false)
        }
    }

    /// Construct from u128 with positive sign (N must be >= 2 in debug builds).
    #[inline]
    pub fn from_u128(value: u128) -> Self {
        debug_assert!(N >= 2, "from_u128 requires at least 2 limbs");
        Self::from_bigint(BigInt::from(value), true)
    }

    /// Construct from i128; magnitude is |value|, sign reflects value>=0 (N must be >= 2 in debug builds).
    #[inline]
    pub fn from_i128(value: i128) -> Self {
        debug_assert!(N >= 2, "from_i128 requires at least 2 limbs");
        if value >= 0 {
            Self::from_bigint(BigInt::from(value as u128), true)
        } else {
            let mag = (value as i128).unsigned_abs();
            Self::from_bigint(BigInt::from(mag), false)
        }
    }

    /// Truncated mixed-width subtraction: compute (self - rhs) where rhs can have a
    /// different limb count, and fit into P limbs; overflow is ignored.
    #[inline]
    pub fn sub_trunc_mixed<const M: usize, const P: usize>(
        &self,
        rhs: &SignedBigInt<M>,
    ) -> SignedBigInt<P> {
        if self.is_positive != rhs.is_positive {
            let mag = self.magnitude.add_trunc::<M, P>(&rhs.magnitude);
            return SignedBigInt::<P> {
                magnitude: mag,
                is_positive: self.is_positive,
            };
        }
        match self.cmp_magnitude_mixed(rhs) {
            Ordering::Greater | Ordering::Equal => {
                let mag = self.magnitude.sub_trunc::<M, P>(&rhs.magnitude);
                SignedBigInt::<P> {
                    magnitude: mag,
                    is_positive: self.is_positive,
                }
            },
            Ordering::Less => {
                let mag = rhs.magnitude.sub_trunc::<N, P>(&self.magnitude);
                SignedBigInt::<P> {
                    magnitude: mag,
                    is_positive: !self.is_positive,
                }
            },
        }
    }
}

impl<const N: usize> From<u64> for SignedBigInt<N> {
    /// From<u64>: positive sign; higher limbs are zeroed.
    #[inline]
    fn from(value: u64) -> Self {
        Self::from_u64(value)
    }
}

impl<const N: usize> From<i64> for SignedBigInt<N> {
    /// From<i64>: sign from value; magnitude is |value|; higher limbs are zeroed.
    #[inline]
    fn from(value: i64) -> Self {
        Self::from_i64(value)
    }
}

impl<const N: usize> From<(u64, bool)> for SignedBigInt<N> {
    /// From<(u64,bool)>: (magnitude, is_positive); higher limbs are zeroed.
    #[inline]
    fn from(value_and_sign: (u64, bool)) -> Self {
        Self::from_u64_with_sign(value_and_sign.0, value_and_sign.1)
    }
}

impl<const N: usize> From<u128> for SignedBigInt<N> {
    /// From<u128>: positive sign; debug-assert N >= 2; higher limbs are zeroed.
    #[inline]
    fn from(value: u128) -> Self {
        debug_assert!(N >= 2, "From<u128> requires at least 2 limbs");
        Self::from_u128(value)
    }
}

impl<const N: usize> From<i128> for SignedBigInt<N> {
    /// From<i128>: sign from value; debug-assert N >= 2; magnitude is |value|.
    #[inline]
    fn from(value: i128) -> Self {
        debug_assert!(N >= 2, "From<i128> requires at least 2 limbs");
        Self::from_i128(value)
    }
}

// Specializations for common sizes
impl S64 {
    /// Convert to i128; any u64 magnitude fits for both signs.
    #[inline]
    pub fn to_i128(&self) -> i128 {
        let magnitude = self.magnitude.0[0];
        if self.is_positive {
            magnitude as i128
        } else {
            -(magnitude as i128)
        }
    }

    /// Return the magnitude as u64
    #[inline]
    pub fn magnitude_as_u64(&self) -> u64 {
        self.magnitude.0[0]
    }

    /// Construct from the difference of two u64 values: a - b.
    /// Returns (a - b) with the appropriate sign.
    #[inline(always)]
    pub fn from_diff_u64s(a: u64, b: u64) -> Self {
        if a < b {
            Self::new([b - a], false)
        } else {
            Self::new([a - b], true)
        }
    }
}

impl S128 {
    /// Convert to i128 using 2^127 bounds: positive requires mag <= i128::MAX; negative allows mag == 2^127.
    #[inline]
    pub fn to_i128(&self) -> Option<i128> {
        let hi = self.magnitude.0[1];
        let lo = self.magnitude.0[0];
        let hi_top_bit = hi >> 63; // bit 127
        if self.is_positive {
            if hi_top_bit != 0 {
                return None;
            }
            let mag = ((hi as u128) << 64) | (lo as u128);
            Some(mag as i128)
        } else {
            if hi_top_bit == 0 {
                let mag = ((hi as u128) << 64) | (lo as u128);
                Some(-(mag as i128))
            } else if hi == (1u64 << 63) && lo == 0 {
                Some(i128::MIN)
            } else {
                None
            }
        }
    }

    /// Return the magnitude as u128
    #[inline]
    pub fn magnitude_as_u128(&self) -> u128 {
        (self.magnitude.0[1] as u128) << 64 | (self.magnitude.0[0] as u128)
    }

    /// Construct from u128 and sign
    #[inline]
    pub fn from_u128_and_sign(value: u128, is_positive: bool) -> Self {
        Self::new([value as u64, (value >> 64) as u64], is_positive)
    }

    /// Exact product of u64 and i64 into S128 (u64 × s64 -> s128)
    #[inline]
    pub fn from_u64_mul_i64(u: u64, s: i64) -> Self {
        let mag = (u as u128) * (s.unsigned_abs() as u128);
        Self::from_u128_and_sign(mag, s >= 0)
    }

    /// Exact product of i64 and u64 into S128 (s64 × u64 -> s128)
    #[inline]
    pub fn from_i64_mul_u64(s: i64, u: u64) -> Self {
        Self::from_u64_mul_i64(u, s)
    }

    /// Exact product of two u64 into S128 (u64 × u64 -> s128, non-negative)
    #[inline]
    pub fn from_u64_mul_u64(a: u64, b: u64) -> Self {
        let mag = (a as u128) * (b as u128);
        Self::from_u128_and_sign(mag, true)
    }

    /// Exact product of two i64 into S128 (s64 × s64 -> s128)
    #[inline]
    pub fn from_i64_mul_i64(a: i64, b: i64) -> Self {
        let mag = (a.unsigned_abs() as u128) * (b.unsigned_abs() as u128);
        let is_positive = (a >= 0) == (b >= 0);
        Self::from_u128_and_sign(mag, is_positive)
    }
}

/// Helper function for single u64 signed arithmetic
/// Adds two signed u64 values (given as magnitude+sign) modulo 2^64; returns (magnitude, sign).
#[inline]
pub fn add_with_sign_u64(a_mag: u64, a_pos: bool, b_mag: u64, b_pos: bool) -> (u64, bool) {
    let a = SignedBigInt::<1>::from_u64_with_sign(a_mag, a_pos);
    let b = SignedBigInt::<1>::from_u64_with_sign(b_mag, b_pos);
    let result = a + b;
    (result.magnitude.0[0], result.is_positive)
}

// ===============================================
// Standard operator trait implementations
// ===============================================

impl<const N: usize> Add for SignedBigInt<N> {
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: Self) -> Self::Output {
        self.add_assign_in_place(&rhs);
        self
    }
}

impl<const N: usize> Sub for SignedBigInt<N> {
    type Output = Self;

    #[inline]
    fn sub(mut self, rhs: Self) -> Self::Output {
        self.sub_assign_in_place(&rhs);
        self
    }
}

impl<const N: usize> Mul for SignedBigInt<N> {
    type Output = Self;

    #[inline]
    fn mul(mut self, rhs: Self) -> Self::Output {
        self.mul_assign_in_place(&rhs);
        self
    }
}

impl<const N: usize> Neg for SignedBigInt<N> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        SignedBigInt::neg(self)
    }
}

impl<const N: usize> AddAssign for SignedBigInt<N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.add_assign_in_place(&rhs);
    }
}

impl<const N: usize> SubAssign for SignedBigInt<N> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.sub_assign_in_place(&rhs);
    }
}

impl<const N: usize> MulAssign for SignedBigInt<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.mul_assign_in_place(&rhs);
    }
}

// Reference variants for efficiency
impl<const N: usize> Add<&SignedBigInt<N>> for SignedBigInt<N> {
    type Output = SignedBigInt<N>;

    #[inline]
    fn add(mut self, rhs: &SignedBigInt<N>) -> Self::Output {
        self.add_assign_in_place(rhs);
        self
    }
}

impl<const N: usize> Sub<&SignedBigInt<N>> for SignedBigInt<N> {
    type Output = SignedBigInt<N>;

    #[inline]
    fn sub(mut self, rhs: &SignedBigInt<N>) -> Self::Output {
        self.sub_assign_in_place(rhs);
        self
    }
}

impl<const N: usize> Mul<&SignedBigInt<N>> for SignedBigInt<N> {
    type Output = SignedBigInt<N>;

    #[inline]
    fn mul(mut self, rhs: &SignedBigInt<N>) -> Self::Output {
        self.mul_assign_in_place(rhs);
        self
    }
}

impl<const N: usize> AddAssign<&SignedBigInt<N>> for SignedBigInt<N> {
    #[inline]
    fn add_assign(&mut self, rhs: &SignedBigInt<N>) {
        self.add_assign_in_place(rhs);
    }
}

impl<const N: usize> SubAssign<&SignedBigInt<N>> for SignedBigInt<N> {
    #[inline]
    fn sub_assign(&mut self, rhs: &SignedBigInt<N>) {
        self.sub_assign_in_place(rhs);
    }
}

impl<const N: usize> MulAssign<&SignedBigInt<N>> for SignedBigInt<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: &SignedBigInt<N>) {
        self.mul_assign_in_place(rhs);
    }
}

// By-ref binary operator variants to avoid copying both operands
impl<const N: usize> core::ops::Add for &SignedBigInt<N> {
    type Output = SignedBigInt<N>;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let mut out = *self;
        out.add_assign_in_place(rhs);
        out
    }
}

impl<const N: usize> core::ops::Sub for &SignedBigInt<N> {
    type Output = SignedBigInt<N>;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        let mut out = *self;
        out.sub_assign_in_place(rhs);
        out
    }
}

impl<const N: usize> core::ops::Mul for &SignedBigInt<N> {
    type Output = SignedBigInt<N>;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let mut out = *self;
        out.mul_assign_in_place(rhs);
        out
    }
}

// ===============================================
// Ordering and canonical serialization
// ===============================================

impl<const N: usize> core::cmp::PartialOrd for SignedBigInt<N> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<const N: usize> core::cmp::Ord for SignedBigInt<N> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        // Treat +0 and -0 as equal in ordering semantics
        if self.magnitude.is_zero() && other.magnitude.is_zero() {
            return Ordering::Equal;
        }
        match (self.is_positive, other.is_positive) {
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            _ => {
                let ord = self.magnitude.cmp(&other.magnitude);
                if self.is_positive {
                    ord
                } else {
                    ord.reverse()
                }
            },
        }
    }
}

impl<const N: usize> CanonicalSerialize for SignedBigInt<N> {
    #[inline]
    fn serialize_with_mode<W: Write>(
        &self,
        mut w: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        // encode sign as a single byte then magnitude
        (self.is_positive as u8).serialize_with_mode(&mut w, compress)?;
        self.magnitude.serialize_with_mode(w, compress)
    }

    #[inline]
    fn serialized_size(&self, compress: Compress) -> usize {
        (self.is_positive as u8).serialized_size(compress)
            + self.magnitude.serialized_size(compress)
    }
}

impl<const N: usize> CanonicalDeserialize for SignedBigInt<N> {
    #[inline]
    fn deserialize_with_mode<R: Read>(
        mut r: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let sign_u8 = u8::deserialize_with_mode(&mut r, compress, validate)?;
        let mag = BigInt::<N>::deserialize_with_mode(r, compress, validate)?;
        Ok(SignedBigInt {
            magnitude: mag,
            is_positive: sign_u8 != 0,
        })
    }
}

impl<const N: usize> Valid for SignedBigInt<N> {
    #[inline]
    fn check(&self) -> Result<(), SerializationError> {
        self.magnitude.check()
    }
}
