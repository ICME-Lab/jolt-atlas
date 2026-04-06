use crate::{
    bits::{BitIteratorBE, BitIteratorLE},
    const_for, UniformRand,
};

#[cfg(feature = "default")]
use allocative::Allocative;

#[allow(unused)]
use ark_ff_macros::unroll_for_loops;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use ark_std::{
    borrow::Borrow,
    // convert::TryFrom,
    fmt::{Debug, Display, UpperHex},
    io::{Read, Write},
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl,
        ShlAssign, Shr, ShrAssign, Sub, SubAssign,
    },
    rand::{
        distributions::{Distribution, Standard},
        Rng,
    },
    str::FromStr,
    vec::*,
    Zero,
};
use num_bigint::BigUint;
use zeroize::Zeroize;

#[macro_use]
pub mod arithmetic;

pub mod signed;
pub use signed::{SignedBigInt, S128, S192, S256, S64};

pub mod signed_hi_32;
pub use signed_hi_32::{SignedBigIntHi32, S160, S224, S96};

#[derive(Copy, Clone, PartialEq, Eq, Hash, Zeroize)]
#[cfg_attr(feature = "default", derive(Allocative))]
pub struct BigInt<const N: usize>(pub [u64; N]);

impl<const N: usize> Default for BigInt<N> {
    fn default() -> Self {
        Self([0u64; N])
    }
}

impl<const N: usize> CanonicalSerialize for BigInt<N> {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.0.serialized_size(compress)
    }
}

impl<const N: usize> Valid for BigInt<N> {
    fn check(&self) -> Result<(), SerializationError> {
        self.0.check()
    }
}

impl<const N: usize> CanonicalDeserialize for BigInt<N> {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(BigInt::<N>(<[u64; N]>::deserialize_with_mode(
            reader, compress, validate,
        )?))
    }
}

/// Construct a [`struct@BigInt<N>`] element from a literal string.
///
/// # Panics
///
/// If the integer represented by the string cannot fit in the number
/// of limbs of the `BigInt`, this macro results in a
/// * compile-time error if used in a const context
/// * run-time error otherwise.
///
/// # Usage
/// ```rust
/// # use ark_ff::BigInt;
/// const ONE: BigInt<6> = BigInt!("1");
///
/// fn check_correctness() {
///     assert_eq!(ONE, BigInt::from(1u8));
/// }
/// ```
#[macro_export]
macro_rules! BigInt {
    ($c0:expr) => {{
        let (is_positive, limbs) = $crate::ark_ff_macros::to_sign_and_limbs!($c0);
        assert!(is_positive);
        let mut integer = $crate::BigInt::zero();
        assert!(integer.0.len() >= limbs.len());
        $crate::const_for!((i in 0..(limbs.len())) {
            integer.0[i] = limbs[i];
        });
        integer
    }};
}

#[doc(hidden)]
macro_rules! const_modulo {
    ($a:expr, $divisor:expr) => {{
        // Stupid slow base-2 long division taken from
        // https://en.wikipedia.org/wiki/Division_algorithm
        assert!(!$divisor.const_is_zero());
        let mut remainder = BigInt::<N>::new([0u64; N]);
        let mut i = ($a.num_bits() - 1) as isize;
        let mut carry;
        while i >= 0 {
            (remainder, carry) = remainder.const_mul2_with_carry();
            remainder.0[0] |= $a.get_bit(i as usize) as u64;
            if remainder.const_geq($divisor) || carry {
                let (r, borrow) = remainder.const_sub_with_borrow($divisor);
                remainder = r;
                assert!(borrow == carry);
            }
            i -= 1;
        }
        remainder
    }};
}

#[doc(hidden)]
macro_rules! const_quotient {
    ($a:expr, $divisor:expr) => {{
        // Binary long division computing the quotient
        assert!(!$divisor.const_is_zero());
        let mut remainder = BigInt::<N>::new([0u64; N]);
        let mut quotient = BigInt::<N>::new([0u64; N]); // Initialize quotient
        let mut i = ($a.num_bits() - 1) as isize;
        let mut carry;
        while i >= 0 {
            // Left shift remainder by 1
            (remainder, carry) = remainder.const_mul2_with_carry();
            // Bring down the next bit from dividend $a$ into remainder LSB
            remainder.0[0] |= $a.get_bit(i as usize) as u64;

            // If remainder >= divisor
            if remainder.const_geq($divisor) || carry {
                // Subtract divisor from remainder
                let (r, borrow) = remainder.const_sub_with_borrow($divisor);
                remainder = r;
                assert!(borrow == carry);
                // Manually set the i-th bit of the quotient
                quotient.0[(i as usize) / 64] |= 1u64 << ((i as usize) % 64);
            }
            i -= 1;
        }
        quotient // Return the quotient
    }};
}

impl<const N: usize> BigInt<N> {
    pub const fn new(value: [u64; N]) -> Self {
        Self(value)
    }

    pub const fn zero() -> Self {
        Self([0u64; N])
    }

    pub const fn one() -> Self {
        let mut one = Self::zero();
        one.0[0] = 1;
        one
    }

    #[doc(hidden)]
    pub const fn const_is_even(&self) -> bool {
        self.0[0] % 2 == 0
    }

    #[doc(hidden)]
    pub const fn const_is_odd(&self) -> bool {
        self.0[0] % 2 == 1
    }

    #[doc(hidden)]
    pub const fn mod_4(&self) -> u8 {
        // To compute n % 4, we need to simply look at the
        // 2 least significant bits of n, and check their value mod 4.
        (((self.0[0] << 62) >> 62) % 4) as u8
    }

    /// Compute a right shift of `self`
    /// This is equivalent to a (saturating) division by 2.
    #[doc(hidden)]
    pub const fn const_shr(&self) -> Self {
        let mut result = *self;
        let mut t = 0;
        crate::const_for!((i in 0..N) {
            let a = result.0[N - i - 1];
            let t2 = a << 63;
            result.0[N - i - 1] >>= 1;
            result.0[N - i - 1] |= t;
            t = t2;
        });
        result
    }

    pub(crate) const fn const_geq(&self, other: &Self) -> bool {
        const_for!((i in 0..N) {
            let a = self.0[N - i - 1];
            let b = other.0[N - i - 1];
            if a < b {
                return false;
            } else if a > b {
                return true;
            }
        });
        true
    }

    /// Compute the largest integer `s` such that `self = 2**s * t + 1` for odd `t`.
    #[doc(hidden)]
    pub const fn two_adic_valuation(mut self) -> u32 {
        assert!(self.const_is_odd());
        let mut two_adicity = 0;
        // Since `self` is odd, we can always subtract one
        // without a borrow
        self.0[0] -= 1;
        while self.const_is_even() {
            self = self.const_shr();
            two_adicity += 1;
        }
        two_adicity
    }

    /// Compute the smallest odd integer `t` such that `self = 2**s * t + 1` for some
    /// integer `s = self.two_adic_valuation()`.
    #[doc(hidden)]
    pub const fn two_adic_coefficient(mut self) -> Self {
        assert!(self.const_is_odd());
        // Since `self` is odd, we can always subtract one
        // without a borrow
        self.0[0] -= 1;
        while self.const_is_even() {
            self = self.const_shr();
        }
        assert!(self.const_is_odd());
        self
    }

    /// Divide `self` by 2, rounding down if necessary.
    /// That is, if `self.is_odd()`, compute `(self - 1)/2`.
    /// Else, compute `self/2`.
    #[doc(hidden)]
    pub const fn divide_by_2_round_down(mut self) -> Self {
        if self.const_is_odd() {
            self.0[0] -= 1;
        }
        self.const_shr()
    }

    /// Find the number of bits in the binary decomposition of `self`.
    /// Assume that `self` fills out all `N-1` low limbs
    #[doc(hidden)]
    pub const fn const_num_bits(self) -> u32 {
        ((N - 1) * 64) as u32 + (64 - self.0[N - 1].leading_zeros())
    }

    /// Compute `2^((N-1) * 64) * 2^exp`
    /// Assume that `exp < 64`
    #[doc(hidden)]
    pub const fn pow_2(exp: u32) -> Self {
        assert!(exp < 64);
        let mut res = Self::zero();
        res.0[N - 1] = 1;
        let mut i = 0;
        while i < exp {
            res.0[N - 1] = res.0[N - 1] << 1;
            i += 1;
        }
        res
    }

    /// Compute the number of spare (i.e. leading zero) bits in the big integer.
    /// Assumes that `self` fills out all `N-1` low limbs.
    /// This means the number of spare bits is determined by the
    /// leading zeros in the most significant limb.
    #[doc(hidden)]
    pub const fn num_spare_bits(self) -> u32 {
        // Fast path: directly use the intrinsic on the most significant limb
        self.0[N - 1].leading_zeros()
    }

    /// Truncated-width addition: compute self + other into P limbs.
    ///
    /// - Semantics: returns the low P limbs of the sum; higher limbs are discarded.
    /// - Precondition (debug-only): right operand width M must be <= P.
    /// - Debug contract: panics in debug if an addition carry would spill beyond P limbs.
    #[inline]
    pub fn add_trunc<const M: usize, const P: usize>(&self, other: &BigInt<M>) -> BigInt<P> {
        debug_assert!(M <= P, "add_trunc: right operand wider than result width P");
        let mut acc = BigInt::<P>::zero();
        let copy_len = core::cmp::min(P, N);
        acc.0[..copy_len].copy_from_slice(&self.0[..copy_len]);
        acc.add_assign_trunc::<M>(other);
        acc
    }

    /// Truncated-width subtraction: compute self - other into P limbs.
    ///
    /// - Semantics: returns the low P limbs of the difference; higher borrow is discarded.
    /// - Precondition (debug-only): right operand width M must be <= P.
    /// - Debug contract: panics in debug if a borrow would spill beyond P limbs.
    #[inline]
    pub fn sub_trunc<const M: usize, const P: usize>(&self, other: &BigInt<M>) -> BigInt<P> {
        debug_assert!(M <= P, "sub_trunc: right operand wider than result width P");
        let mut acc = BigInt::<P>::zero();
        let copy_len = core::cmp::min(P, N);
        acc.0[..copy_len].copy_from_slice(&self.0[..copy_len]);
        acc.sub_assign_trunc::<M>(other);
        acc
    }

    /// Truncated-width multiplication: compute self * other and fit into P limbs; overflow is ignored.
    #[inline]
    pub fn mul_trunc<const M: usize, const P: usize>(&self, other: &BigInt<M>) -> BigInt<P> {
        let mut res = BigInt::<P>::zero();
        // Use core fused multiply engine specialized on M for unrolling
        self.fm_limbs_into::<M, P>(&other.0, &mut res, false);
        res
    }

    /// Truncated-width addition that mutates self: self += other, keeping N limbs (self's width).
    ///
    /// - Semantics: computes (self + other) mod 2^(64*N).
    /// - Precondition (debug-only): right operand width M must be <= N.
    /// - Debug contract: panics in debug if a carry would spill beyond N limbs.
    #[inline]
    #[unroll_for_loops(9)]
    pub fn add_assign_trunc<const M: usize>(&mut self, other: &BigInt<M>) {
        debug_assert!(
            M <= N,
            "add_assign_trunc: right operand wider than self width N"
        );
        let mut carry = 0u64;
        for i in 0..N {
            let rhs = if i < M { other.0[i] } else { 0 };
            self.0[i] = adc!(self.0[i], rhs, &mut carry);
        }
        debug_assert!(
            carry == 0,
            "add_assign_trunc overflow: carry beyond N limbs"
        );
    }

    /// Truncated-width subtraction that mutates self: self -= other, keeping N limbs (self's width).
    ///
    /// Semantics: computes (self - other) mod 2^(64*N).
    /// Precondition (debug-only): right operand width M must be <= N.
    /// Debug contract: panics in debug if a borrow would spill beyond N limbs.
    #[inline]
    #[unroll_for_loops(9)]
    pub fn sub_assign_trunc<const M: usize>(&mut self, other: &BigInt<M>) {
        debug_assert!(
            M <= N,
            "sub_assign_trunc: right operand wider than self width N"
        );
        let mut borrow = 0u64;
        for i in 0..N {
            let rhs = if i < M { other.0[i] } else { 0 };
            self.0[i] = sbb!(self.0[i], rhs, &mut borrow);
        }
        debug_assert!(
            borrow == 0,
            "sub_assign_trunc underflow: borrow beyond N limbs"
        );
    }

    /// Truncated-width multiplication that mutates self: self = (self * other) mod 2^(64*N).
    /// Keeps exactly N limbs (self's width). Overflow beyond N limbs is ignored.
    #[inline]
    pub fn mul_assign_trunc<const M: usize>(&mut self, other: &BigInt<M>) {
        // Fast paths
        if self.is_zero() || other.is_zero() {
            for i in 0..N {
                self.0[i] = 0;
            }
            return;
        }
        let left = *self; // snapshot original multiplicand
                          // zero self to use as accumulator buffer
        for i in 0..N {
            self.0[i] = 0;
        }
        // Accumulate left * other directly into self within width N; propagate carries within N
        left.fm_limbs_into::<M, N>(&other.0, self, true);
    }

    /// Fused multiply-add with truncation: acc += self * other, fitting into P limbs; overflow is ignored.
    /// This is a generic version for arbitrary limb widths of `self` and `other`.
    #[inline]
    pub fn fmadd_trunc<const M: usize, const P: usize>(
        &self,
        other: &BigInt<M>,
        acc: &mut BigInt<P>,
    ) {
        let i_limit = core::cmp::min(N, P);
        for i in 0..i_limit {
            let mut carry = 0u64;
            let j_limit = core::cmp::min(M, P - i);
            for j in 0..j_limit {
                acc.0[i + j] = mac_with_carry!(acc.0[i + j], self.0[i], other.0[j], &mut carry);
            }
            if i + j_limit < P {
                let (new_val, _of) = acc.0[i + j_limit].overflowing_add(carry);
                acc.0[i + j_limit] = new_val;
            }
        }
    }

    /// Accumulate with a compile-time-known count of multiplier limbs M to enable unrolling.
    #[inline]
    #[unroll_for_loops(10)]
    pub(crate) fn fm_limbs_into<const M: usize, const P: usize>(
        &self,
        other_limbs: &[u64; M],
        acc: &mut BigInt<P>,
        carry_propagate: bool,
    ) {
        for j in 0..M {
            let mul_limb = other_limbs[j];
            if mul_limb == 0 {
                // Skip zero multiplier limb
                // (cannot use `continue` here due to unroll macro limitations)
            } else {
                let base = j;
                let mut carry = 0u64;
                for i in 0..N {
                    let idx = base + i;
                    if idx < P {
                        acc.0[idx] = mac_with_carry!(acc.0[idx], self.0[i], mul_limb, &mut carry);
                    }
                }
                let next = base + N;
                if next < P {
                    let (v, mut of) = acc.0[next].overflowing_add(carry);
                    acc.0[next] = v;
                    if carry_propagate && of {
                        let mut k = next + 1;
                        while of && k < P {
                            let (nv, nof) = acc.0[k].overflowing_add(1);
                            acc.0[k] = nv;
                            of = nof;
                            k += 1;
                        }
                    }
                }
            }
        }
    }

    #[inline]
    pub(crate) const fn const_sub_with_borrow(mut self, other: &Self) -> (Self, bool) {
        let mut borrow = 0;

        const_for!((i in 0..N) {
            self.0[i] = sbb!(self.0[i], other.0[i], &mut borrow);
        });

        (self, borrow != 0)
    }

    #[inline]
    pub(crate) const fn const_add_with_carry(mut self, other: &Self) -> (Self, bool) {
        let mut carry = 0;

        crate::const_for!((i in 0..N) {
            self.0[i] = adc!(self.0[i], other.0[i], &mut carry);
        });

        (self, carry != 0)
    }

    pub(crate) const fn const_mul2_with_carry(mut self) -> (Self, bool) {
        let mut last = 0;
        crate::const_for!((i in 0..N) {
            let a = self.0[i];
            let tmp = a >> 63;
            self.0[i] <<= 1;
            self.0[i] |= last;
            last = tmp;
        });
        (self, last != 0)
    }

    pub(crate) const fn const_is_zero(&self) -> bool {
        let mut is_zero = true;
        crate::const_for!((i in 0..N) {
            is_zero &= self.0[i] == 0;
        });
        is_zero
    }

    /// Computes the Montgomery R constant modulo `self`.
    #[doc(hidden)]
    pub const fn montgomery_r(&self) -> Self {
        let two_pow_n_times_64 = crate::const_helpers::RBuffer::<N>([0u64; N], 1);
        const_modulo!(two_pow_n_times_64, self)
    }

    /// Computes the Montgomery R2 constant modulo `self`.
    #[doc(hidden)]
    pub const fn montgomery_r2(&self) -> Self {
        let two_pow_n_times_64_square =
            crate::const_helpers::R2Buffer::<N>([0u64; N], [0u64; N], 1);
        const_modulo!(two_pow_n_times_64_square, self)
    }

    /// Zero-extend a smaller BigInt<M> into BigInt<N> (little-endian limbs).
    /// Debug-asserts that M <= N.
    #[inline]
    pub fn zero_extend_from<const M: usize>(smaller: &BigInt<M>) -> BigInt<N> {
        debug_assert!(
            M <= N,
            "cannot zero-extend: source has more limbs than destination"
        );
        let mut limbs = [0u64; N];
        let copy_len = if M < N { M } else { N };
        limbs[..copy_len].copy_from_slice(&smaller.0[..copy_len]);
        BigInt::<N>(limbs)
    }
}

impl<const N: usize> Zero for BigInt<N> {
    #[inline]
    fn zero() -> Self {
        Self::zero()
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0.iter().all(|&limb| limb == 0)
    }
}

impl<const N: usize> From<[u64; N]> for BigInt<N> {
    fn from(limbs: [u64; N]) -> Self {
        BigInt(limbs)
    }
}

impl<const N: usize> BigInteger for BigInt<N> {
    const NUM_LIMBS: usize = N;

    #[unroll_for_loops(6)]
    #[inline]
    fn add_with_carry(&mut self, other: &Self) -> bool {
        let mut carry = 0;

        for i in 0..N {
            carry = arithmetic::adc_for_add_with_carry(&mut self.0[i], other.0[i], carry);
        }

        carry != 0
    }

    #[unroll_for_loops(6)]
    #[inline]
    fn sub_with_borrow(&mut self, other: &Self) -> bool {
        let mut borrow = 0;

        for i in 0..N {
            borrow = arithmetic::sbb_for_sub_with_borrow(&mut self.0[i], other.0[i], borrow);
        }

        borrow != 0
    }

    #[inline]
    #[allow(unused)]
    fn mul2(&mut self) -> bool {
        #[cfg(all(target_arch = "x86_64", feature = "asm"))]
        #[allow(unsafe_code)]
        {
            let mut carry = 0;

            for i in 0..N {
                unsafe {
                    use core::arch::x86_64::_addcarry_u64;
                    carry = _addcarry_u64(carry, self.0[i], self.0[i], &mut self.0[i])
                };
            }

            carry != 0
        }

        #[cfg(not(all(target_arch = "x86_64", feature = "asm")))]
        {
            let mut last = 0;
            for i in 0..N {
                let a = &mut self.0[i];
                let tmp = *a >> 63;
                *a <<= 1;
                *a |= last;
                last = tmp;
            }
            last != 0
        }
    }

    #[inline]
    fn muln(&mut self, mut n: u32) {
        if n >= (64 * N) as u32 {
            *self = Self::from(0u64);
            return;
        }

        while n >= 64 {
            let mut t = 0;
            for i in 0..N {
                core::mem::swap(&mut t, &mut self.0[i]);
            }
            n -= 64;
        }

        if n > 0 {
            let mut t = 0;
            #[allow(unused)]
            for i in 0..N {
                let a = &mut self.0[i];
                let t2 = *a >> (64 - n);
                *a <<= n;
                *a |= t;
                t = t2;
            }
        }
    }

    #[inline]
    #[unroll_for_loops(8)]
    fn mul_u64_in_place(&mut self, other: u64) {
        // special cases for 0 and 1
        // if other == 0 || self.is_zero() {
        //     *self = Self::zero();
        //     return;
        // } else if other == 1 {
        //     return;
        // }
        // Use the same low-level multiply-accumulate primitive that already
        // benefits from x86 optimizations in this crate.
        let mut carry = 0u64;
        for i in 0..N {
            self.0[i] = mac_with_carry!(0u64, self.0[i], other, &mut carry);
        }
        // Overflow is ignored by contract; assert in debug to catch misuse.
        debug_assert!(carry == 0, "Overflow in BigInt::mul_u64_in_place");
    }

    #[inline]
    #[unroll_for_loops(8)]
    fn mul_u64_w_carry<const NPLUS1: usize>(&self, other: u64) -> BigInt<NPLUS1> {
        // ensure NPLUS1 is the correct size
        debug_assert!(NPLUS1 == N + 1);
        // special cases for 0 and 1
        // if other == 0 || self.is_zero() {
        //     return BigInt::<NPLUS1>::zero();
        // } else if other == 1 {
        //     let mut res = BigInt::<NPLUS1>::zero();
        //     for i in 0..N {
        //         res.0[i] = self.0[i];
        //     }
        //     return res;
        // }
        // Use the same multiply-accumulate primitive and capture the final carry
        let mut res = BigInt::<NPLUS1>::zero();
        let mut carry = 0u64;
        for i in 0..N {
            res.0[i] = mac_with_carry!(0u64, self.0[i], other, &mut carry);
        }
        res.0[N] = carry;
        res
    }

    #[inline]
    #[unroll_for_loops(8)]
    fn mul_u128_w_carry<const NPLUS1: usize, const NPLUS2: usize>(
        &self,
        other: u128,
    ) -> BigInt<NPLUS2> {
        // NPLUS1 is N + 1, NPLUS2 is N + 2
        debug_assert!(NPLUS1 == N + 1);
        debug_assert!(NPLUS2 == N + 2);
        // special cases for 0 and 1
        if other == 0 || self.is_zero() {
            return BigInt::<NPLUS2>::zero();
        } else if other == 1 {
            let mut res = BigInt::<NPLUS2>::zero();
            for i in 0..N {
                res.0[i] = self.0[i];
            }
            return res;
        }
        // Split other into two u64s and accumulate directly into the result buffer.
        let other_lo = other as u64;
        let other_hi = (other >> 64) as u64;

        let mut res = BigInt::<NPLUS2>::zero();

        // First pass: res[i] += self[i] * other_lo
        let mut carry = 0u64;
        for i in 0..N {
            res.0[i] = mac_with_carry!(res.0[i], self.0[i], other_lo, &mut carry);
        }
        res.0[N] = carry;

        // Second pass: res[i+1] += self[i] * other_hi
        let mut carry2 = 0u64;
        for i in 0..N {
            res.0[i + 1] = mac_with_carry!(res.0[i + 1], self.0[i], other_hi, &mut carry2);
        }
        res.0[N + 1] = carry2;

        res
    }

    #[inline]
    fn mul(&self, other: &Self) -> (Self, Self) {
        if self.is_zero() || other.is_zero() {
            let zero = Self::zero();
            return (zero, zero);
        }

        let mut r = crate::const_helpers::MulBuffer::<N>::zeroed();

        let mut carry = 0;

        for i in 0..N {
            for j in 0..N {
                r[i + j] = mac_with_carry!(r[i + j], self.0[i], other.0[j], &mut carry);
            }
            r.b1[i] = carry;
            carry = 0;
        }

        (Self(r.b0), Self(r.b1))
    }

    #[inline]
    fn mul_low(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }

        let mut res = Self::zero();
        let mut carry = 0;

        for i in 0..N {
            for j in 0..(N - i) {
                res.0[i + j] = mac_with_carry!(res.0[i + j], self.0[i], other.0[j], &mut carry);
            }
            carry = 0;
        }

        res
    }

    #[inline]
    fn mul_high(&self, other: &Self) -> Self {
        self.mul(other).1
    }

    #[inline]
    fn div2(&mut self) {
        let mut t = 0;
        for a in self.0.iter_mut().rev() {
            let t2 = *a << 63;
            *a >>= 1;
            *a |= t;
            t = t2;
        }
    }

    #[inline]
    fn divn(&mut self, mut n: u32) {
        if n >= (64 * N) as u32 {
            *self = Self::from(0u64);
            return;
        }

        while n >= 64 {
            let mut t = 0;
            for i in 0..N {
                core::mem::swap(&mut t, &mut self.0[N - i - 1]);
            }
            n -= 64;
        }

        if n > 0 {
            let mut t = 0;
            #[allow(unused)]
            for i in 0..N {
                let a = &mut self.0[N - i - 1];
                let t2 = *a << (64 - n);
                *a >>= n;
                *a |= t;
                t = t2;
            }
        }
    }

    #[inline]
    fn is_odd(&self) -> bool {
        self.0[0] & 1 == 1
    }

    #[inline]
    fn is_even(&self) -> bool {
        !self.is_odd()
    }

    #[inline]
    fn num_bits(&self) -> u32 {
        let mut ret = N as u32 * 64;
        for i in self.0.iter().rev() {
            let leading = i.leading_zeros();
            ret -= leading;
            if leading != 64 {
                break;
            }
        }

        ret
    }

    #[inline]
    fn get_bit(&self, i: usize) -> bool {
        if i >= 64 * N {
            false
        } else {
            let limb = i / 64;
            let bit = i - (64 * limb);
            (self.0[limb] & (1 << bit)) != 0
        }
    }

    #[inline]
    fn from_bits_be(bits: &[bool]) -> Self {
        let mut bits = bits.to_vec();
        bits.reverse();
        Self::from_bits_le(&bits)
    }

    fn from_bits_le(bits: &[bool]) -> Self {
        let mut res = Self::zero();
        for (bits64, res_i) in bits.chunks(64).zip(&mut res.0) {
            for (i, bit) in bits64.iter().enumerate() {
                *res_i |= (*bit as u64) << i;
            }
        }
        res
    }

    #[inline]
    fn to_bytes_be(&self) -> Vec<u8> {
        let mut le_bytes = self.to_bytes_le();
        le_bytes.reverse();
        le_bytes
    }

    #[inline]
    fn to_bytes_le(&self) -> Vec<u8> {
        let array_map = self.0.iter().map(|limb| limb.to_le_bytes());
        let mut res = Vec::with_capacity(N * 8);
        for limb in array_map {
            res.extend_from_slice(&limb);
        }
        res
    }
}

impl<const N: usize> UpperHex for BigInt<N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:016X}", BigUint::from(*self))
    }
}

impl<const N: usize> Debug for BigInt<N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", BigUint::from(*self))
    }
}

impl<const N: usize> Display for BigInt<N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", BigUint::from(*self))
    }
}

impl<const N: usize> Ord for BigInt<N> {
    #[inline]
    #[cfg_attr(target_arch = "x86_64", unroll_for_loops(12))]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        use core::cmp::Ordering;
        #[cfg(target_arch = "x86_64")]
        for i in 0..N {
            let a = &self.0[N - i - 1];
            let b = &other.0[N - i - 1];
            match a.cmp(b) {
                Ordering::Equal => {},
                order => return order,
            };
        }
        #[cfg(not(target_arch = "x86_64"))]
        for (a, b) in self.0.iter().rev().zip(other.0.iter().rev()) {
            if a < b {
                return Ordering::Less;
            } else if a > b {
                return Ordering::Greater;
            }
        }
        Ordering::Equal
    }
}

impl<const N: usize> PartialOrd for BigInt<N> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<const N: usize> Distribution<BigInt<N>> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> BigInt<N> {
        let mut res = [0u64; N];
        for item in res.iter_mut() {
            *item = rng.gen();
        }
        BigInt::<N>(res)
    }
}

impl<const N: usize> AsMut<[u64]> for BigInt<N> {
    #[inline]
    fn as_mut(&mut self) -> &mut [u64] {
        &mut self.0
    }
}

impl<const N: usize> AsRef<[u64]> for BigInt<N> {
    #[inline]
    fn as_ref(&self) -> &[u64] {
        &self.0
    }
}

impl<const N: usize> From<u128> for BigInt<N> {
    #[inline]
    fn from(val: u128) -> BigInt<N> {
        let mut repr = Self::default();
        repr.0[0] = val as u64;
        if N > 1 {
            repr.0[1] = (val >> 64) as u64;
        }
        repr
    }
}

impl<const N: usize> From<u64> for BigInt<N> {
    #[inline]
    fn from(val: u64) -> BigInt<N> {
        let mut repr = Self::default();
        repr.0[0] = val;
        repr
    }
}

impl<const N: usize> From<u32> for BigInt<N> {
    #[inline]
    fn from(val: u32) -> BigInt<N> {
        let mut repr = Self::default();
        repr.0[0] = u64::from(val);
        repr
    }
}

impl<const N: usize> From<u16> for BigInt<N> {
    #[inline]
    fn from(val: u16) -> BigInt<N> {
        let mut repr = Self::default();
        repr.0[0] = u64::from(val);
        repr
    }
}

impl<const N: usize> From<u8> for BigInt<N> {
    #[inline]
    fn from(val: u8) -> BigInt<N> {
        let mut repr = Self::default();
        repr.0[0] = u64::from(val);
        repr
    }
}

impl<const N: usize> TryFrom<BigUint> for BigInt<N> {
    type Error = ();

    /// Returns `Err(())` if the bit size of `val` is more than `N * 64`.
    #[inline]
    fn try_from(val: num_bigint::BigUint) -> Result<BigInt<N>, Self::Error> {
        let bytes = val.to_bytes_le();

        if bytes.len() > N * 8 {
            Err(())
        } else {
            let mut limbs = [0u64; N];

            bytes
                .chunks(8)
                .into_iter()
                .enumerate()
                .for_each(|(i, chunk)| {
                    let mut chunk_padded = [0u8; 8];
                    chunk_padded[..chunk.len()].copy_from_slice(chunk);
                    limbs[i] = u64::from_le_bytes(chunk_padded)
                });

            Ok(Self(limbs))
        }
    }
}

impl<const N: usize> FromStr for BigInt<N> {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let biguint = BigUint::from_str(s).map_err(|_| ())?;
        Self::try_from(biguint)
    }
}

impl<const N: usize> From<BigInt<N>> for BigUint {
    #[inline]
    fn from(val: BigInt<N>) -> num_bigint::BigUint {
        BigUint::from_bytes_le(&val.to_bytes_le())
    }
}

impl<const N: usize> From<BigInt<N>> for num_bigint::BigInt {
    #[inline]
    fn from(val: BigInt<N>) -> num_bigint::BigInt {
        use num_bigint::Sign;
        let sign = if val.is_zero() {
            Sign::NoSign
        } else {
            Sign::Plus
        };
        num_bigint::BigInt::from_bytes_le(sign, &val.to_bytes_le())
    }
}

impl<B: Borrow<Self>, const N: usize> BitXorAssign<B> for BigInt<N> {
    fn bitxor_assign(&mut self, rhs: B) {
        (0..N).for_each(|i| self.0[i] ^= rhs.borrow().0[i])
    }
}

impl<B: Borrow<Self>, const N: usize> BitXor<B> for BigInt<N> {
    type Output = Self;

    fn bitxor(mut self, rhs: B) -> Self::Output {
        self ^= rhs;
        self
    }
}

impl<B: Borrow<Self>, const N: usize> BitAndAssign<B> for BigInt<N> {
    fn bitand_assign(&mut self, rhs: B) {
        (0..N).for_each(|i| self.0[i] &= rhs.borrow().0[i])
    }
}

impl<B: Borrow<Self>, const N: usize> BitAnd<B> for BigInt<N> {
    type Output = Self;

    fn bitand(mut self, rhs: B) -> Self::Output {
        self &= rhs;
        self
    }
}

impl<B: Borrow<Self>, const N: usize> BitOrAssign<B> for BigInt<N> {
    fn bitor_assign(&mut self, rhs: B) {
        (0..N).for_each(|i| self.0[i] |= rhs.borrow().0[i])
    }
}

impl<B: Borrow<Self>, const N: usize> BitOr<B> for BigInt<N> {
    type Output = Self;

    fn bitor(mut self, rhs: B) -> Self::Output {
        self |= rhs;
        self
    }
}

impl<const N: usize> ShrAssign<u32> for BigInt<N> {
    /// Computes the bitwise shift right operation in place.
    ///
    /// Differently from the built-in numeric types (u8, u32, u64, etc.) this
    /// operation does *not* return an underflow error if the number of bits
    /// shifted is larger than N * 64. Instead the result will be saturated to
    /// zero.
    fn shr_assign(&mut self, mut rhs: u32) {
        if rhs >= (64 * N) as u32 {
            *self = Self::from(0u64);
            return;
        }

        while rhs >= 64 {
            let mut t = 0;
            for limb in self.0.iter_mut().rev() {
                core::mem::swap(&mut t, limb);
            }
            rhs -= 64;
        }

        if rhs > 0 {
            let mut t = 0;
            for a in self.0.iter_mut().rev() {
                let t2 = *a << (64 - rhs);
                *a >>= rhs;
                *a |= t;
                t = t2;
            }
        }
    }
}

impl<const N: usize> Shr<u32> for BigInt<N> {
    type Output = Self;

    /// Computes bitwise shift right operation.
    ///
    /// Differently from the built-in numeric types (u8, u32, u64, etc.) this
    /// operation does *not* return an underflow error if the number of bits
    /// shifted is larger than N * 64. Instead the result will be saturated to
    /// zero.
    fn shr(mut self, rhs: u32) -> Self::Output {
        self >>= rhs;
        self
    }
}

impl<const N: usize> ShlAssign<u32> for BigInt<N> {
    /// Computes the bitwise shift left operation in place.
    ///
    /// Differently from the built-in numeric types (u8, u32, u64, etc.) this
    /// operation does *not* return an overflow error if the number of bits
    /// shifted is larger than N * 64. Instead, the overflow will be chopped
    /// off.
    fn shl_assign(&mut self, mut rhs: u32) {
        if rhs >= (64 * N) as u32 {
            *self = Self::from(0u64);
            return;
        }

        while rhs >= 64 {
            let mut t = 0;
            for i in 0..N {
                core::mem::swap(&mut t, &mut self.0[i]);
            }
            rhs -= 64;
        }

        if rhs > 0 {
            let mut t = 0;
            #[allow(unused)]
            for i in 0..N {
                let a = &mut self.0[i];
                let t2 = *a >> (64 - rhs);
                *a <<= rhs;
                *a |= t;
                t = t2;
            }
        }
    }
}

impl<const N: usize> Shl<u32> for BigInt<N> {
    type Output = Self;

    /// Computes the bitwise shift left operation in place.
    ///
    /// Differently from the built-in numeric types (u8, u32, u64, etc.) this
    /// operation does *not* return an overflow error if the number of bits
    /// shifted is larger than N * 64. Instead, the overflow will be chopped
    /// off.
    fn shl(mut self, rhs: u32) -> Self::Output {
        self <<= rhs;
        self
    }
}

impl<const N: usize> Not for BigInt<N> {
    type Output = Self;

    fn not(self) -> Self::Output {
        let mut result = Self::zero();
        for i in 0..N {
            result.0[i] = !self.0[i];
        }
        result
    }
}

// Arithmetic with truncating semantics for BigInt of different widths
// Note: we cannot let the output have arbitrary width due to Rust's type limitation
// So we set the output width to be the same as the width of the left operand
impl<const N: usize, const M: usize> Add<BigInt<M>> for BigInt<N> {
    type Output = BigInt<N>;

    fn add(self, rhs: BigInt<M>) -> Self::Output {
        debug_assert!(N >= M, "right operand cannot be wider than left operand");
        self.add_trunc::<M, N>(&rhs)
    }
}

impl<const N: usize, const M: usize> Add<&BigInt<M>> for BigInt<N> {
    type Output = BigInt<N>;
    fn add(self, rhs: &BigInt<M>) -> Self::Output {
        debug_assert!(N >= M, "right operand cannot be wider than left operand");
        self.add_trunc::<M, N>(rhs)
    }
}

impl<const N: usize, const M: usize> Sub<BigInt<M>> for BigInt<N> {
    type Output = BigInt<N>;

    fn sub(self, rhs: BigInt<M>) -> Self::Output {
        debug_assert!(N >= M, "right operand cannot be wider than left operand");
        self.sub_trunc::<M, N>(&rhs)
    }
}

impl<const N: usize, const M: usize> Sub<&BigInt<M>> for BigInt<N> {
    type Output = BigInt<N>;
    fn sub(self, rhs: &BigInt<M>) -> Self::Output {
        debug_assert!(N >= M, "right operand cannot be wider than left operand");
        self.sub_trunc::<M, N>(rhs)
    }
}

impl<const N: usize, const M: usize> AddAssign<BigInt<M>> for BigInt<N> {
    fn add_assign(&mut self, rhs: BigInt<M>) {
        debug_assert!(N >= M, "right operand cannot be wider than left operand");
        self.add_assign_trunc::<M>(&rhs);
    }
}

impl<const N: usize, const M: usize> AddAssign<&BigInt<M>> for BigInt<N> {
    fn add_assign(&mut self, rhs: &BigInt<M>) {
        debug_assert!(N >= M, "right operand cannot be wider than left operand");
        self.add_assign_trunc::<M>(rhs);
    }
}

impl<const N: usize, const M: usize> SubAssign<BigInt<M>> for BigInt<N> {
    fn sub_assign(&mut self, rhs: BigInt<M>) {
        debug_assert!(N >= M, "right operand cannot be wider than left operand");
        self.sub_assign_trunc::<M>(&rhs);
    }
}

impl<const N: usize, const M: usize> SubAssign<&BigInt<M>> for BigInt<N> {
    fn sub_assign(&mut self, rhs: &BigInt<M>) {
        debug_assert!(N >= M, "right operand cannot be wider than left operand");
        self.sub_assign_trunc::<M>(rhs);
    }
}

/// Compute the signed modulo operation on a u64 representation, returning the result.
/// If n % modulus > modulus / 2, return modulus - n
/// # Example
/// ```
/// use ark_ff::signed_mod_reduction;
/// let res = signed_mod_reduction(6u64, 8u64);
/// assert_eq!(res, -2i64);
/// ```
pub fn signed_mod_reduction(n: u64, modulus: u64) -> i64 {
    let t = (n % modulus) as i64;
    if t as u64 >= (modulus / 2) {
        t - (modulus as i64)
    } else {
        t
    }
}

pub type BigInteger64 = BigInt<1>;
pub type BigInteger128 = BigInt<2>;
pub type BigInteger256 = BigInt<4>;
pub type BigInteger320 = BigInt<5>;
pub type BigInteger384 = BigInt<6>;
pub type BigInteger448 = BigInt<7>;
pub type BigInteger768 = BigInt<12>;
pub type BigInteger832 = BigInt<13>;

#[cfg(test)]
mod tests;

/// This defines a `BigInteger`, a smart wrapper around a
/// sequence of `u64` limbs, least-significant limb first.
// TODO: get rid of this trait once we can use associated constants in const generics.
pub trait BigInteger:
    CanonicalSerialize
    + CanonicalDeserialize
    + Copy
    + Clone
    + Debug
    + Default
    + Display
    + Eq
    + Ord
    + Send
    + Sized
    + Sync
    + 'static
    + UniformRand
    + Zeroize
    + Zero
    + AsMut<[u64]>
    + AsRef<[u64]>
    + From<u128>
    + From<u64>
    + From<u32>
    + From<u16>
    + From<u8>
    + TryFrom<BigUint, Error = ()>
    + FromStr
    + Into<BigUint>
    + BitXorAssign<Self>
    + for<'a> BitXorAssign<&'a Self>
    + BitXor<Self, Output = Self>
    + for<'a> BitXor<&'a Self, Output = Self>
    + BitAndAssign<Self>
    + for<'a> BitAndAssign<&'a Self>
    + BitAnd<Self, Output = Self>
    + for<'a> BitAnd<&'a Self, Output = Self>
    + BitOrAssign<Self>
    + for<'a> BitOrAssign<&'a Self>
    + BitOr<Self, Output = Self>
    + for<'a> BitOr<&'a Self, Output = Self>
    + Shr<u32, Output = Self>
    + ShrAssign<u32>
    + Shl<u32, Output = Self>
    + ShlAssign<u32>
{
    /// Number of 64-bit limbs representing `Self`.
    const NUM_LIMBS: usize;

    /// Add another [`BigInteger`] to `self`. This method stores the result in `self`,
    /// and returns a carry bit.
    ///
    /// # Example
    ///
    /// ```
    /// use ark_ff::{biginteger::BigInteger64 as B, BigInteger as _};
    ///
    /// // Basic
    /// let (mut one, mut x) = (B::from(1u64), B::from(2u64));
    /// let carry = x.add_with_carry(&one);
    /// assert_eq!(x, B::from(3u64));
    /// assert_eq!(carry, false);
    ///
    /// // Edge-Case
    /// let mut x = B::from(u64::MAX);
    /// let carry = x.add_with_carry(&one);
    /// assert_eq!(x, B::from(0u64));
    /// assert_eq!(carry, true)
    /// ```
    fn add_with_carry(&mut self, other: &Self) -> bool;

    /// Subtract another [`BigInteger`] from this one. This method stores the result in
    /// `self`, and returns a borrow.
    ///
    /// # Example
    ///
    /// ```
    /// use ark_ff::{biginteger::BigInteger64 as B, BigInteger as _};
    ///
    /// // Basic
    /// let (mut one_sub, two, mut three_sub) = (B::from(1u64), B::from(2u64), B::from(3u64));
    /// let borrow = three_sub.sub_with_borrow(&two);
    /// assert_eq!(three_sub, one_sub);
    /// assert_eq!(borrow, false);
    ///
    /// // Edge-Case
    /// let borrow = one_sub.sub_with_borrow(&two);
    /// assert_eq!(one_sub, B::from(u64::MAX));
    /// assert_eq!(borrow, true);
    /// ```
    fn sub_with_borrow(&mut self, other: &Self) -> bool;

    /// Performs a leftwise bitshift of this number, effectively multiplying
    /// it by 2. Overflow is ignored.
    /// # Example
    ///
    /// ```
    /// use ark_ff::{biginteger::BigInteger64 as B, BigInteger as _};
    ///
    /// // Basic
    /// let mut two_mul = B::from(2u64);
    /// two_mul.mul2();
    /// assert_eq!(two_mul, B::from(4u64));
    ///
    /// // Edge-Cases
    /// let mut zero = B::from(0u64);
    /// zero.mul2();
    /// assert_eq!(zero, B::from(0u64));
    ///
    /// let mut arr: [bool; 64] = [false; 64];
    /// arr[0] = true;
    /// let mut mul = B::from_bits_be(&arr);
    /// mul.mul2();
    /// assert_eq!(mul, B::from(0u64));
    /// ```
    fn mul2(&mut self) -> bool;

    /// Performs a leftwise bitshift of this number by n bits, effectively multiplying
    /// it by 2^n. Overflow is ignored.
    /// # Example
    ///
    /// ```
    /// use ark_ff::{biginteger::BigInteger64 as B, BigInteger as _};
    ///
    /// // Basic
    /// let mut one_mul = B::from(1u64);
    /// one_mul.muln(5);
    /// assert_eq!(one_mul, B::from(32u64));
    ///
    /// // Edge-Case
    /// let mut zero = B::from(0u64);
    /// zero.muln(5);
    /// assert_eq!(zero, B::from(0u64));
    ///
    /// let mut arr: [bool; 64] = [false; 64];
    /// arr[4] = true;
    /// let mut mul = B::from_bits_be(&arr);
    /// mul.muln(5);
    /// assert_eq!(mul, B::from(0u64));
    /// ```
    #[deprecated(since = "0.4.2", note = "please use the operator `<<` instead")]
    fn muln(&mut self, amt: u32);

    /// NEW! Multiplies self by a u64 in place. Overflow is ignored.
    fn mul_u64_in_place(&mut self, other: u64);

    /// NEW! Multiplies self by a u64, returning a bigint with one extra limb to hold overflow.
    fn mul_u64_w_carry<const NPLUS1: usize>(&self, other: u64) -> BigInt<NPLUS1>;

    /// NEW! Multiplies self by a u128, returning a bigint with two extra limbs to hold overflow.
    fn mul_u128_w_carry<const NPLUS1: usize, const NPLUS2: usize>(
        &self,
        other: u128,
    ) -> BigInt<NPLUS2>;

    /// Multiplies this [`BigInteger`] by another `BigInteger`, storing the result in `self`.
    /// Overflow is ignored.
    ///
    /// # Example
    ///
    /// ```
    /// use ark_ff::{biginteger::BigInteger64 as B, BigInteger as _};
    ///
    /// // Basic
    /// let mut a = B::from(42u64);
    /// let b = B::from(3u64);
    /// assert_eq!(a.mul_low(&b), B::from(126u64));
    ///
    /// // Edge-Case
    /// let mut zero = B::from(0u64);
    /// assert_eq!(zero.mul_low(&B::from(5u64)), B::from(0u64));
    /// ```
    fn mul_low(&self, other: &Self) -> Self;

    /// Multiplies this [`BigInteger`] by another `BigInteger`, returning the high bits of the result.
    ///
    /// # Example
    ///
    /// ```
    /// use ark_ff::{biginteger::BigInteger64 as B, BigInteger as _};
    ///
    /// // Basic
    /// let (one, x) = (B::from(1u64), B::from(2u64));
    /// let r = x.mul_high(&one);
    /// assert_eq!(r, B::from(0u64));
    ///
    /// // Edge-Case
    /// let mut x = B::from(u64::MAX);
    /// let r = x.mul_high(&B::from(2u64));
    /// assert_eq!(r, B::from(1u64))
    /// ```
    fn mul_high(&self, other: &Self) -> Self;

    /// Multiplies this [`BigInteger`] by another `BigInteger`, returning both low and high bits of the result.
    ///
    /// # Example
    ///
    /// ```
    /// use ark_ff::{biginteger::BigInteger64 as B, BigInteger as _};
    ///
    /// // Basic
    /// let mut a = B::from(42u64);
    /// let b = B::from(3u64);
    /// let (low_bits, high_bits) = a.mul(&b);
    /// assert_eq!(low_bits, B::from(126u64));
    /// assert_eq!(high_bits, B::from(0u64));
    ///
    /// // Edge-Case
    /// let mut x = B::from(u64::MAX);
    /// let mut max_plus_max = x;
    /// max_plus_max.add_with_carry(&x);
    /// let (low_bits, high_bits) = x.mul(&B::from(2u64));
    /// assert_eq!(low_bits, max_plus_max);
    /// assert_eq!(high_bits, B::from(1u64));
    /// ```
    fn mul(&self, other: &Self) -> (Self, Self);

    /// Performs a rightwise bitshift of this number, effectively dividing
    /// it by 2.
    /// # Example
    ///
    /// ```
    /// use ark_ff::{biginteger::BigInteger64 as B, BigInteger as _};
    ///
    /// // Basic
    /// let (mut two, mut four_div) = (B::from(2u64), B::from(4u64));
    /// four_div.div2();
    /// assert_eq!(two, four_div);
    ///
    /// // Edge-Case
    /// let mut zero = B::from(0u64);
    /// zero.div2();
    /// assert_eq!(zero, B::from(0u64));
    ///
    /// let mut one = B::from(1u64);
    /// one.div2();
    /// assert_eq!(one, B::from(0u64));
    /// ```
    fn div2(&mut self);

    /// Performs a rightwise bitshift of this number by some amount.
    /// # Example
    ///
    /// ```
    /// use ark_ff::{biginteger::BigInteger64 as B, BigInteger as _};
    ///
    /// // Basic
    /// let (mut one, mut thirty_two_div) = (B::from(1u64), B::from(32u64));
    /// thirty_two_div.divn(5);
    /// assert_eq!(one, thirty_two_div);
    ///
    /// // Edge-Case
    /// let mut arr: [bool; 64] = [false; 64];
    /// arr[4] = true;
    /// let mut div = B::from_bits_le(&arr);
    /// div.divn(5);
    /// assert_eq!(div, B::from(0u64));
    /// ```
    #[deprecated(since = "0.4.2", note = "please use the operator `>>` instead")]
    fn divn(&mut self, amt: u32);

    /// Returns true iff this number is odd.
    /// # Example
    ///
    /// ```
    /// use ark_ff::{biginteger::BigInteger64 as B, BigInteger as _};
    ///
    /// let mut one = B::from(1u64);
    /// assert!(one.is_odd());
    /// ```
    fn is_odd(&self) -> bool;

    /// Returns true iff this number is even.
    /// # Example
    ///
    /// ```
    /// use ark_ff::{biginteger::BigInteger64 as B, BigInteger as _};
    ///
    /// let mut two = B::from(2u64);
    /// assert!(two.is_even());
    /// ```
    fn is_even(&self) -> bool;

    /// Compute the minimum number of bits needed to encode this number.
    /// # Example
    /// ```
    /// use ark_ff::{biginteger::BigInteger64 as B, BigInteger as _};
    ///
    /// let zero = B::from(0u64);
    /// assert_eq!(zero.num_bits(), 0);
    /// let one = B::from(1u64);
    /// assert_eq!(one.num_bits(), 1);
    /// let max = B::from(u64::MAX);
    /// assert_eq!(max.num_bits(), 64);
    /// let u32_max = B::from(u32::MAX as u64);
    /// assert_eq!(u32_max.num_bits(), 32);
    /// ```
    fn num_bits(&self) -> u32;

    /// Compute the `i`-th bit of `self`.
    /// # Example
    ///
    /// ```
    /// use ark_ff::{biginteger::BigInteger64 as B, BigInteger as _};
    ///
    /// let mut one = B::from(1u64);
    /// assert!(one.get_bit(0));
    /// assert!(!one.get_bit(1));
    /// ```
    fn get_bit(&self, i: usize) -> bool;

    /// Returns the big integer representation of a given big endian boolean
    /// array.
    /// # Example
    ///
    /// ```
    /// use ark_ff::{biginteger::BigInteger64 as B, BigInteger as _};
    ///
    /// let mut arr: [bool; 64] = [false; 64];
    /// arr[63] = true;
    /// let mut one = B::from(1u64);
    /// assert_eq!(B::from_bits_be(&arr), one);
    /// ```
    fn from_bits_be(bits: &[bool]) -> Self;

    /// Returns the big integer representation of a given little endian boolean
    /// array.
    /// # Example
    ///
    /// ```
    /// use ark_ff::{biginteger::BigInteger64 as B, BigInteger as _};
    ///
    /// let mut arr: [bool; 64] = [false; 64];
    /// arr[0] = true;
    /// let mut one = B::from(1u64);
    /// assert_eq!(B::from_bits_le(&arr), one);
    /// ```
    fn from_bits_le(bits: &[bool]) -> Self;

    /// Returns the bit representation in a big endian boolean array,
    /// with leading zeroes.
    /// # Example
    ///
    /// ```
    /// use ark_ff::{biginteger::BigInteger64 as B, BigInteger as _};
    ///
    /// let one = B::from(1u64);
    /// let arr = one.to_bits_be();
    /// let mut vec = vec![false; 64];
    /// vec[63] = true;
    /// assert_eq!(arr, vec);
    /// ```
    fn to_bits_be(&self) -> Vec<bool> {
        BitIteratorBE::new(self).collect::<Vec<_>>()
    }

    /// Returns the bit representation in a little endian boolean array,
    /// with trailing zeroes.
    /// # Example
    ///
    /// ```
    /// use ark_ff::{biginteger::BigInteger64 as B, BigInteger as _};
    ///
    /// let one = B::from(1u64);
    /// let arr = one.to_bits_le();
    /// let mut vec = vec![false; 64];
    /// vec[0] = true;
    /// assert_eq!(arr, vec);
    /// ```
    fn to_bits_le(&self) -> Vec<bool> {
        BitIteratorLE::new(self).collect::<Vec<_>>()
    }

    /// Returns the byte representation in a big endian byte array,
    /// with leading zeros.
    /// # Example
    ///
    /// ```
    /// use ark_ff::{biginteger::BigInteger64 as B, BigInteger as _};
    ///
    /// let one = B::from(1u64);
    /// let arr = one.to_bytes_be();
    /// let mut vec = vec![0; 8];
    /// vec[7] = 1;
    /// assert_eq!(arr, vec);
    /// ```
    fn to_bytes_be(&self) -> Vec<u8>;

    /// Returns the byte representation in a little endian byte array,
    /// with trailing zeros.
    /// # Example
    ///
    /// ```
    /// use ark_ff::{biginteger::BigInteger64 as B, BigInteger as _};
    ///
    /// let one = B::from(1u64);
    /// let arr = one.to_bytes_le();
    /// let mut vec = vec![0; 8];
    /// vec[0] = 1;
    /// assert_eq!(arr, vec);
    /// ```
    fn to_bytes_le(&self) -> Vec<u8>;

    /// Returns the windowed non-adjacent form of `self`, for a window of size `w`.
    fn find_wnaf(&self, w: usize) -> Option<Vec<i64>> {
        // w > 2 due to definition of wNAF, and w < 64 to make sure that `i64`
        // can fit each signed digit
        if (2..64).contains(&w) {
            let mut res = vec![];
            let mut e = *self;

            while !e.is_zero() {
                let z: i64;
                if e.is_odd() {
                    z = signed_mod_reduction(e.as_ref()[0], 1 << w);
                    if z >= 0 {
                        e.sub_with_borrow(&Self::from(z as u64));
                    } else {
                        e.add_with_carry(&Self::from((-z) as u64));
                    }
                } else {
                    z = 0;
                }
                res.push(z);
                e.div2();
            }

            Some(res)
        } else {
            None
        }
    }
}
