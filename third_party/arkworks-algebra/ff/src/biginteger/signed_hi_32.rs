use crate::biginteger::{BigInt, SignedBigInt, S128, S64};

#[cfg(feature = "default")]
use allocative::Allocative;

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, Read, SerializationError, Valid, Validate,
    Write,
};
use ark_std::cmp::Ordering;
use ark_std::vec::Vec;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Compact signed big-integer parameterized by limb count `N` (total width = `N*64 + 32` bits).
///
/// Representation (sign-magnitude):
/// - `magnitude_lo: [u64; N]` holds the low limbs in little-endian order (index 0 is least significant).
/// - `magnitude_hi: u32` holds the high 32-bit tail of the magnitude.
/// - `is_positive: bool` is the sign flag. The magnitude stores the absolute value.
///
/// Arithmetic semantics:
/// - Addition, subtraction, and multiplication operate on magnitudes modulo `2^(64*N + 32)`
///   and then set the sign via standard sign rules.
/// - Zero is not normalized: a zero magnitude can be paired with either sign. Equality is structural,
///   so `+0 != -0`. Callers that require canonical zero should normalize externally.
///
/// Notes:
/// - Zero is not normalized: a zero magnitude can be positive or negative. Structural equality
///   distinguishes `+0` and `-0`, but ordering treats them as equal.
/// - Specialized fast paths exist for `N ∈ {0,1,2}`; larger `N` uses a generic path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "default", derive(Allocative))]
pub struct SignedBigIntHi32<const N: usize> {
    /// Little-endian low limbs: limb 0 = low 64 bits, limb 1 = next 64 bits, and so on
    magnitude_lo: [u64; N],
    /// Top 32 bits
    magnitude_hi: u32,
    /// Whether the value is non-negative
    is_positive: bool,
}

pub type S96 = SignedBigIntHi32<1>;
pub type S160 = SignedBigIntHi32<2>;
pub type S224 = SignedBigIntHi32<3>;

// ------------------------------------------------------------------------------------------------
// Implementation
// ------------------------------------------------------------------------------------------------

impl<const N: usize> SignedBigIntHi32<N> {
    /// Creates a new `SignedBigIntHi32`.
    ///
    /// The sign is not normalized: a zero magnitude can be positive or negative.
    pub const fn new(magnitude_lo: [u64; N], magnitude_hi: u32, is_positive: bool) -> Self {
        Self {
            magnitude_lo,
            magnitude_hi,
            is_positive,
        }
    }

    /// Returns the value `0`.
    pub const fn zero() -> Self {
        Self {
            magnitude_lo: [0; N],
            magnitude_hi: 0,
            is_positive: true,
        }
    }

    /// Returns the value `1`.
    pub fn one() -> Self {
        let mut magnitude_lo = [0; N];
        let magnitude_hi;

        if N == 0 {
            magnitude_hi = 1;
        } else {
            magnitude_lo[0] = 1;
            magnitude_hi = 0;
        }

        Self {
            magnitude_lo,
            magnitude_hi,
            is_positive: true,
        }
    }

    // ------------------------------------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------------------------------------

    /// Returns the low limbs of the magnitude.
    pub const fn magnitude_lo(&self) -> &[u64; N] {
        &self.magnitude_lo
    }

    /// Returns the high 32 bits of the magnitude.
    pub const fn magnitude_hi(&self) -> u32 {
        self.magnitude_hi
    }

    /// Returns the sign flag (`true` for a positive sign).
    /// Note: zero is not canonicalized; a zero magnitude can have either sign.
    pub const fn is_positive(&self) -> bool {
        self.is_positive
    }

    /// Returns `true` if the number is zero.
    pub const fn is_zero(&self) -> bool {
        let mut lo_is_zero = true;
        let mut i = 0;
        while i < N {
            if self.magnitude_lo[i] != 0 {
                lo_is_zero = false;
                break;
            }
            i += 1;
        }
        self.magnitude_hi == 0 && lo_is_zero
    }

    // ------------------------------------------------------------------------------------------------
    // Private arithmetic helpers
    // ------------------------------------------------------------------------------------------------

    fn compare_magnitudes(&self, other: &Self) -> Ordering {
        if self.magnitude_hi != other.magnitude_hi {
            return self.magnitude_hi.cmp(&other.magnitude_hi);
        }
        for i in (0..N).rev() {
            if self.magnitude_lo[i] != other.magnitude_lo[i] {
                return self.magnitude_lo[i].cmp(&other.magnitude_lo[i]);
            }
        }
        Ordering::Equal
    }

    fn add_assign_in_place(&mut self, rhs: &Self) {
        if self.is_positive == rhs.is_positive {
            let (lo, hi, _carry) = self.add_magnitudes_with_carry(rhs);
            self.magnitude_lo = lo;
            self.magnitude_hi = hi;
        } else {
            match self.compare_magnitudes(rhs) {
                Ordering::Greater | Ordering::Equal => {
                    let (lo, hi, _borrow) = self.sub_magnitudes_with_borrow(rhs);
                    self.magnitude_lo = lo;
                    self.magnitude_hi = hi;
                },
                Ordering::Less => {
                    let (lo, hi, _borrow) = rhs.sub_magnitudes_with_borrow(self);
                    self.magnitude_lo = lo;
                    self.magnitude_hi = hi;
                    self.is_positive = rhs.is_positive;
                },
            }
        }
    }

    fn sub_assign_in_place(&mut self, rhs: &Self) {
        let neg_rhs = -*rhs;
        self.add_assign_in_place(&neg_rhs);
    }

    fn mul_magnitudes(&self, other: &Self) -> ([u64; N], u32) {
        // Fast paths for small N to avoid heap allocation and loops
        if N == 0 {
            let a2 = self.magnitude_hi as u64;
            let b2 = other.magnitude_hi as u64;
            let prod = a2.wrapping_mul(b2);
            let hi = (prod & 0xFFFF_FFFF) as u32;
            let lo: [u64; N] = [0u64; N];
            return (lo, hi);
        }

        if N == 1 {
            let a0 = self.magnitude_lo[0];
            let a1 = self.magnitude_hi as u64; // 32-bit value widened
            let b0 = other.magnitude_lo[0];
            let b1 = other.magnitude_hi as u64; // 32-bit value widened

            let t0 = (a0 as u128) * (b0 as u128);
            let lo0 = t0 as u64;

            let cross = (t0 >> 64) + (a0 as u128) * (b1 as u128) + (a1 as u128) * (b0 as u128);

            let hi = (cross as u64 & 0xFFFF_FFFF) as u32;
            let mut lo = [0u64; N];
            lo[0] = lo0;
            return (lo, hi);
        }

        if N == 2 {
            let a0 = self.magnitude_lo[0];
            let a1 = self.magnitude_lo[1];
            let a2 = self.magnitude_hi as u64; // 32-bit value widened
            let b0 = other.magnitude_lo[0];
            let b1 = other.magnitude_lo[1];
            let b2 = other.magnitude_hi as u64; // 32-bit value widened

            // word 0
            let t0 = (a0 as u128) * (b0 as u128);
            let r0 = t0 as u64;
            let carry0 = t0 >> 64;

            // word 1
            let sum1 = carry0 + (a0 as u128) * (b1 as u128) + (a1 as u128) * (b0 as u128);
            let r1 = sum1 as u64;
            let carry1 = sum1 >> 64;

            // word 2
            let sum2 = carry1
                + (a0 as u128) * (b2 as u128)
                + (a1 as u128) * (b1 as u128)
                + (a2 as u128) * (b0 as u128);
            let r2 = sum2 as u64;
            let _carry2 = (sum2 >> 64) as u64;

            // For a 160-bit result, the head (bits 128..159) is the low 32 bits of word2.
            let hi = (r2 & 0xFFFF_FFFF) as u32;
            let mut lo = [0u64; N];
            lo[0] = r0;
            lo[1] = r1;
            return (lo, hi);
        }

        // General path
        // Product of (N*64 + 32)-bit numbers fits in (2*N*64 + 64) bits.
        // Allocate 2*N + 2 u64 limbs to safely propagate carries; we'll truncate to N u64 + 32 bits.
        let mut prod = vec![0u64; 2 * N + 2];

        let self_limbs: Vec<u64> = self
            .magnitude_lo
            .iter()
            .cloned()
            .chain(core::iter::once(self.magnitude_hi as u64))
            .collect();

        let other_limbs: Vec<u64> = other
            .magnitude_lo
            .iter()
            .cloned()
            .chain(core::iter::once(other.magnitude_hi as u64))
            .collect();

        for i in 0..self_limbs.len() {
            let mut carry: u128 = 0;
            for j in 0..other_limbs.len() {
                let idx = i + j;
                let p = (self_limbs[i] as u128) * (other_limbs[j] as u128)
                    + (prod[idx] as u128)
                    + carry;
                prod[idx] = p as u64;
                carry = p >> 64;
            }
            if carry > 0 {
                let spill = i + other_limbs.len();
                if spill < prod.len() {
                    prod[spill] = prod[spill].wrapping_add(carry as u64);
                }
                // else: spill is beyond the truncated width; ignore (mod 2^(64*N+32)).
            }
        }

        // Truncate and split into lo and hi (keep only the low N u64 limbs and the low 32 bits of limb N)
        let mut magnitude_lo = [0u64; N];
        if N > 0 {
            magnitude_lo.copy_from_slice(&prod[0..N]);
        }
        let magnitude_hi = (prod[N] & 0xFFFF_FFFF) as u32;

        (magnitude_lo, magnitude_hi)
    }

    // Returns final carry bit.
    fn add_magnitudes_with_carry(&self, other: &Self) -> ([u64; N], u32, bool) {
        let mut magnitude_lo = [0; N];
        let mut carry: u128 = 0;

        for i in 0..N {
            let sum = (self.magnitude_lo[i] as u128) + (other.magnitude_lo[i] as u128) + carry;
            magnitude_lo[i] = sum as u64;
            carry = sum >> 64;
        }

        let sum_hi = (self.magnitude_hi as u128) + (other.magnitude_hi as u128) + carry;
        let magnitude_hi = sum_hi as u32;

        let final_carry = (sum_hi >> 32) != 0;
        (magnitude_lo, magnitude_hi, final_carry)
    }

    // Returns final borrow bit.
    fn sub_magnitudes_with_borrow(&self, other: &Self) -> ([u64; N], u32, bool) {
        let mut magnitude_lo = [0u64; N];
        let mut borrow = false;

        for i in 0..N {
            let (d1, b1) = self.magnitude_lo[i].overflowing_sub(other.magnitude_lo[i]);
            let (d2, b2) = d1.overflowing_sub(borrow as u64);
            magnitude_lo[i] = d2;
            borrow = b1 || b2;
        }

        let (hi1, b1) = self.magnitude_hi.overflowing_sub(other.magnitude_hi);
        let (hi2, b2) = hi1.overflowing_sub(borrow as u32);
        let final_borrow = b1 || b2;

        (magnitude_lo, hi2, final_borrow)
    }

    /// Return the unsigned magnitude as a BigInt with N+1 limbs (little-endian),
    /// packing `magnitude_lo` followed by `magnitude_hi` (widened to u64).
    /// This ignores the sign; pair with `is_positive()` if you need a signed value.
    #[inline]
    pub fn magnitude_as_bigint_nplus1<const NPLUS1: usize>(&self) -> BigInt<NPLUS1> {
        debug_assert!(
            NPLUS1 == N + 1,
            "NPLUS1 must be N+1 for SignedBigIntHi32 magnitude pack"
        );
        let mut limbs = [0u64; NPLUS1];
        if N > 0 {
            limbs[..N].copy_from_slice(&self.magnitude_lo);
        }
        limbs[N] = self.magnitude_hi as u64;
        BigInt::<NPLUS1>(limbs)
    }

    /// Zero-extend a smaller-width SignedBigIntHi32<M> into width N (little-endian).
    /// Moves the 32-bit head of the smaller value into the next low 64-bit limb on widen,
    /// and clears the head in the widened representation to preserve the numeric value.
    /// Debug-asserts that M <= N.
    #[inline]
    pub fn zero_extend_from<const M: usize>(smaller: &SignedBigIntHi32<M>) -> SignedBigIntHi32<N> {
        debug_assert!(
            M <= N,
            "cannot zero-extend: source has more limbs than destination"
        );
        if N == M {
            return SignedBigIntHi32::<N>::new(
                // copy to avoid borrowing issues
                {
                    let mut lo = [0u64; N];
                    if N > 0 {
                        lo.copy_from_slice(smaller.magnitude_lo());
                    }
                    lo
                },
                smaller.magnitude_hi(),
                smaller.is_positive(),
            );
        }
        // N > M
        let mut lo = [0u64; N];
        if M > 0 {
            lo[..M].copy_from_slice(smaller.magnitude_lo());
        }
        // Place the 32-bit head into limb M
        lo[M] = smaller.magnitude_hi() as u64;
        SignedBigIntHi32::<N>::new(lo, 0u32, smaller.is_positive())
    }

    /// Convert this hi-32 representation into a standard SignedBigInt with N+1 limbs.
    /// Packs the low limbs verbatim and writes the 32-bit head into the highest limb.
    /// Debug-asserts that NPLUS1 == N + 1.
    #[inline]
    pub fn to_signed_bigint_nplus1<const NPLUS1: usize>(&self) -> SignedBigInt<NPLUS1> {
        debug_assert!(
            NPLUS1 == N + 1,
            "to_signed_bigint_nplus1 requires NPLUS1 = N + 1"
        );
        let mut limbs = [0u64; NPLUS1];
        if N > 0 {
            limbs[..N].copy_from_slice(self.magnitude_lo());
        }
        limbs[N] = self.magnitude_hi() as u64;
        let mag = BigInt::<NPLUS1>(limbs);
        SignedBigInt::from_bigint(mag, self.is_positive())
    }
}

// ------------------------------------------------------------------------------------------------
// Operator traits
// ------------------------------------------------------------------------------------------------

impl<const N: usize> Neg for SignedBigIntHi32<N> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(self.magnitude_lo, self.magnitude_hi, !self.is_positive)
    }
}

impl<const N: usize> Add for SignedBigIntHi32<N> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self.add_assign_in_place(&rhs);
        self
    }
}

impl<const N: usize> AddAssign for SignedBigIntHi32<N> {
    fn add_assign(&mut self, rhs: Self) {
        self.add_assign_in_place(&rhs);
    }
}

impl<const N: usize> Sub for SignedBigIntHi32<N> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self.sub_assign_in_place(&rhs);
        self
    }
}

impl<const N: usize> SubAssign for SignedBigIntHi32<N> {
    fn sub_assign(&mut self, rhs: Self) {
        self.sub_assign_in_place(&rhs);
    }
}

impl<const N: usize> MulAssign for SignedBigIntHi32<N> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul(&rhs);
    }
}

// Reference variants for efficiency
impl<const N: usize> Add<&SignedBigIntHi32<N>> for SignedBigIntHi32<N> {
    type Output = SignedBigIntHi32<N>;

    #[inline]
    fn add(mut self, rhs: &SignedBigIntHi32<N>) -> Self::Output {
        self.add_assign_in_place(rhs);
        self
    }
}

impl<const N: usize> Sub<&SignedBigIntHi32<N>> for SignedBigIntHi32<N> {
    type Output = SignedBigIntHi32<N>;

    #[inline]
    fn sub(mut self, rhs: &SignedBigIntHi32<N>) -> Self::Output {
        self.sub_assign_in_place(rhs);
        self
    }
}

impl<const N: usize> Mul<&SignedBigIntHi32<N>> for SignedBigIntHi32<N> {
    type Output = SignedBigIntHi32<N>;

    #[inline]
    fn mul(self, rhs: &SignedBigIntHi32<N>) -> Self::Output {
        let (lo, hi) = self.mul_magnitudes(rhs);
        let is_positive = !(self.is_positive ^ rhs.is_positive);
        Self::new(lo, hi, is_positive)
    }
}

impl<const N: usize> AddAssign<&SignedBigIntHi32<N>> for SignedBigIntHi32<N> {
    #[inline]
    fn add_assign(&mut self, rhs: &SignedBigIntHi32<N>) {
        self.add_assign_in_place(rhs);
    }
}

impl<const N: usize> SubAssign<&SignedBigIntHi32<N>> for SignedBigIntHi32<N> {
    #[inline]
    fn sub_assign(&mut self, rhs: &SignedBigIntHi32<N>) {
        self.sub_assign_in_place(rhs);
    }
}

impl<const N: usize> MulAssign<&SignedBigIntHi32<N>> for SignedBigIntHi32<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: &SignedBigIntHi32<N>) {
        *self = self.mul(rhs);
    }
}

// By-ref binary operator variants to avoid copying both operands
impl<'a, const N: usize> Add for &'a SignedBigIntHi32<N> {
    type Output = SignedBigIntHi32<N>;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let mut out = *self;
        out.add_assign_in_place(rhs);
        out
    }
}

impl<'a, const N: usize> Sub for &'a SignedBigIntHi32<N> {
    type Output = SignedBigIntHi32<N>;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        let mut out = *self;
        out.sub_assign_in_place(rhs);
        out
    }
}

impl<'a, const N: usize> Mul for &'a SignedBigIntHi32<N> {
    type Output = SignedBigIntHi32<N>;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let (lo, hi) = self.mul_magnitudes(rhs);
        let is_positive = !(self.is_positive ^ rhs.is_positive);
        SignedBigIntHi32::new(lo, hi, is_positive)
    }
}

// ------------------------------------------------------------------------------------------------
// S160-specific inherent constructors (ergonomic helpers)
// ------------------------------------------------------------------------------------------------

impl S160 {
    /// Construct from the signed difference of two u64 values: returns |a - b| with sign a>=b.
    #[inline]
    pub fn from_diff_u64(a: u64, b: u64) -> Self {
        let mag = a.abs_diff(b);
        let is_positive = a >= b;
        S160::new([mag, 0], 0, is_positive)
    }

    /// Construct from a u128 magnitude and an explicit sign.
    #[inline]
    pub fn from_magnitude_u128(mag: u128, is_positive: bool) -> Self {
        let lo = mag as u64;
        let hi = (mag >> 64) as u64;
        S160::new([lo, hi], 0, is_positive)
    }

    /// Construct from the signed difference of two u128 values: returns |u1 - u2| with sign u1>=u2.
    #[inline]
    pub fn from_diff_u128(u1: u128, u2: u128) -> Self {
        if u1 >= u2 {
            S160::from_magnitude_u128(u1 - u2, true)
        } else {
            S160::from_magnitude_u128(u2 - u1, false)
        }
    }

    /// Construct from the sum of two u128 values, preserving carry into the top 32-bit head.
    #[inline]
    pub fn from_sum_u128(u1: u128, u2: u128) -> Self {
        let u1_lo = u1 as u64;
        let u1_hi = (u1 >> 64) as u64;
        let u2_lo = u2 as u64;
        let u2_hi = (u2 >> 64) as u64;
        let (sum_lo, carry0) = u1_lo.overflowing_add(u2_lo);
        let (sum_hi1, carry1) = u1_hi.overflowing_add(u2_hi);
        let (sum_hi, carry2) = sum_hi1.overflowing_add(if carry0 { 1 } else { 0 });
        let carry_out = (carry1 as u8 | carry2 as u8) != 0;
        S160::new([sum_lo, sum_hi], if carry_out { 1 } else { 0 }, true)
    }

    /// Construct from (u128 - i128) with full-width integer semantics.
    #[inline]
    pub fn from_u128_minus_i128(u: u128, i: i128) -> Self {
        if i >= 0 {
            S160::from_diff_u128(u, i as u128)
        } else {
            let abs_i: u128 = i.unsigned_abs();
            S160::from_sum_u128(u, abs_i)
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Ordering and canonical serialization
// ------------------------------------------------------------------------------------------------

impl<const N: usize> core::cmp::PartialOrd for SignedBigIntHi32<N> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<const N: usize> core::cmp::Ord for SignedBigIntHi32<N> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        if self.is_zero() && other.is_zero() {
            return Ordering::Equal;
        }
        match (self.is_positive, other.is_positive) {
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            _ => {
                let ord = self.compare_magnitudes(other);
                if self.is_positive {
                    ord
                } else {
                    ord.reverse()
                }
            },
        }
    }
}

impl<const N: usize> CanonicalSerialize for SignedBigIntHi32<N> {
    #[inline]
    fn serialize_with_mode<W: Write>(
        &self,
        mut w: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        // Encode sign, then (hi, lo)
        (self.is_positive as u8).serialize_with_mode(&mut w, compress)?;
        (self.magnitude_hi as i32).serialize_with_mode(&mut w, compress)?;
        for i in 0..N {
            self.magnitude_lo[i].serialize_with_mode(&mut w, compress)?;
        }
        Ok(())
    }

    #[inline]
    fn serialized_size(&self, compress: Compress) -> usize {
        (self.is_positive as u8).serialized_size(compress)
            + (self.magnitude_hi as i32).serialized_size(compress)
            + (0u64).serialized_size(compress) * N
    }
}

impl<const N: usize> CanonicalDeserialize for SignedBigIntHi32<N> {
    #[inline]
    fn deserialize_with_mode<R: Read>(
        mut r: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let sign_u8 = u8::deserialize_with_mode(&mut r, compress, validate)?;
        let hi = i32::deserialize_with_mode(&mut r, compress, validate)?;
        let mut lo = [0u64; N];
        for i in 0..N {
            lo[i] = u64::deserialize_with_mode(&mut r, compress, validate)?;
        }
        Ok(SignedBigIntHi32::new(lo, hi as u32, sign_u8 != 0))
    }
}

impl<const N: usize> Valid for SignedBigIntHi32<N> {
    #[inline]
    fn check(&self) -> Result<(), SerializationError> {
        // No additional invariants beyond structural fields
        Ok(())
    }
}

// ------------------------------------------------------------------------------------------------
// From traits
// ------------------------------------------------------------------------------------------------

impl From<i64> for S96 {
    #[inline]
    fn from(val: i64) -> Self {
        Self::new([val.unsigned_abs()], 0, val.is_positive())
    }
}

impl From<u64> for S96 {
    #[inline]
    fn from(val: u64) -> Self {
        Self::new([val], 0, true)
    }
}

impl From<S64> for S96 {
    #[inline]
    fn from(val: S64) -> Self {
        Self::new([val.magnitude.0[0]], 0, val.is_positive)
    }
}

impl From<i64> for S160 {
    #[inline]
    fn from(val: i64) -> Self {
        Self::new([val.unsigned_abs(), 0], 0, val.is_positive())
    }
}

impl From<u64> for S160 {
    #[inline]
    fn from(val: u64) -> Self {
        Self::new([val, 0], 0, true)
    }
}

impl From<S64> for S160 {
    #[inline]
    fn from(val: S64) -> Self {
        Self::new([val.magnitude.0[0], 0], 0, val.is_positive)
    }
}

impl From<u128> for S160 {
    #[inline]
    fn from(val: u128) -> Self {
        let lo = val as u64;
        let hi = (val >> 64) as u64;
        Self::new([lo, hi], 0, true)
    }
}

impl From<i128> for S160 {
    #[inline]
    fn from(val: i128) -> Self {
        let is_positive = val.is_positive();
        let mag = val.unsigned_abs();
        let lo = mag as u64;
        let hi = (mag >> 64) as u64;
        Self::new([lo, hi], 0, is_positive)
    }
}

impl From<S128> for S160 {
    #[inline]
    fn from(val: S128) -> Self {
        Self::new([val.magnitude.0[0], val.magnitude.0[1]], 0, val.is_positive)
    }
}

impl<const N: usize> From<S224> for crate::biginteger::BigInt<N> {
    #[inline]
    #[allow(unsafe_code)]
    fn from(val: S224) -> Self {
        if N != 4 {
            panic!("FromS224 for BigInt<N> only supports N=4, got N={N}");
        }
        let lo = val.magnitude_lo();
        let hi = val.magnitude_hi() as u64;
        let bigint4 = crate::biginteger::BigInt::<4>([lo[0], lo[1], lo[2], hi]);

        unsafe {
            let ptr = &bigint4 as *const BigInt<4> as *const BigInt<N>;
            ptr.read()
        }
    }
}
