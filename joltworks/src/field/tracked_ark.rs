//! [`TrackedFr`]: the BN254 scalar field with every arithmetic operation counted
//! (`crate::utils::counters`), for measuring how many field operations a protocol performs.
//! Values, encodings and challenges are exactly `jolt_field::Fr`'s.

use super::bn254::MulHiBigintU128;
use super::{FieldOps, JoltField};
#[cfg(feature = "challenge-254-bit")]
use crate::field::challenge::Mont254BitChallenge;
#[cfg(not(feature = "challenge-254-bit"))]
use crate::field::challenge::MontU128Challenge;

use crate::utils::counters::{
    ADD_COUNT, BARRETT_REDUCE_COUNT, FROM_BOOL_COUNT, FROM_BYTES_COUNT, FROM_I128_COUNT,
    FROM_I32_COUNT, FROM_I64_COUNT, FROM_U128_COUNT, FROM_U16_COUNT, FROM_U32_COUNT,
    FROM_U64_COUNT, FROM_U8_COUNT, INVERSE_COUNT, MONT_REDUCE_COUNT, MULT_COUNT, MUL_I128_COUNT,
    MUL_I64_COUNT, MUL_U128_COUNT, MUL_U128_UNRED_COUNT, MUL_U64_COUNT, MUL_U64_UNRED_COUNT,
    MUL_UNRED_COUNT, SQUARE_COUNT, SUB_COUNT,
};
use allocative::Allocative;
use ark_ff::UniformRand;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::Rng;
use jolt_field::{
    AdditiveGroup, CanonicalBytes, CanonicalEncoding, Field, Fr, NaiveAccumulator, Ring,
    WithAccumulator,
};
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};
use std::sync::atomic::{AtomicUsize, Ordering};

#[inline(always)]
fn count(counter: &AtomicUsize) {
    counter.fetch_add(1, Ordering::Relaxed);
}

#[derive(
    Clone,
    Default,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Debug,
    CanonicalSerialize,
    CanonicalDeserialize,
    Serialize,
    Deserialize,
)]
#[serde(transparent)]
pub struct TrackedFr(pub Fr);

impl Allocative for TrackedFr {
    fn visit<'a, 'b: 'a>(&self, _visitor: &'a mut allocative::Visitor<'b>) {}
}

impl UniformRand for TrackedFr {
    fn rand<R: Rng + ?Sized>(rng: &mut R) -> Self {
        TrackedFr(Fr::rand(rng))
    }
}

impl fmt::Display for TrackedFr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl PartialEq<Fr> for TrackedFr {
    fn eq(&self, other: &Fr) -> bool {
        self.0 == *other
    }
}

macro_rules! counted_binop {
    ($trait:ident, $method:ident, $op:tt, $counter:ident) => {
        impl $trait for TrackedFr {
            type Output = TrackedFr;
            fn $method(self, rhs: Self) -> Self {
                count(&$counter);
                TrackedFr(self.0 $op rhs.0)
            }
        }
        impl<'a> $trait<&'a TrackedFr> for TrackedFr {
            type Output = TrackedFr;
            fn $method(self, rhs: &'a TrackedFr) -> Self {
                count(&$counter);
                TrackedFr(self.0 $op rhs.0)
            }
        }
        impl $trait<TrackedFr> for &TrackedFr {
            type Output = TrackedFr;
            fn $method(self, rhs: TrackedFr) -> TrackedFr {
                count(&$counter);
                TrackedFr(self.0 $op rhs.0)
            }
        }
        impl<'a, 'b> $trait<&'b TrackedFr> for &'a TrackedFr {
            type Output = TrackedFr;
            fn $method(self, rhs: &'b TrackedFr) -> TrackedFr {
                count(&$counter);
                TrackedFr(self.0 $op rhs.0)
            }
        }
    };
}

counted_binop!(Add, add, +, ADD_COUNT);
counted_binop!(Sub, sub, -, SUB_COUNT);
counted_binop!(Mul, mul, *, MULT_COUNT);

macro_rules! counted_div {
    ($lhs:ty, $rhs:ty) => {
        #[allow(clippy::suspicious_arithmetic_impl)]
        impl Div<$rhs> for $lhs {
            type Output = TrackedFr;
            fn div(self, rhs: $rhs) -> TrackedFr {
                count(&MULT_COUNT);
                count(&INVERSE_COUNT);
                TrackedFr(self.0 * rhs.0.inverse().expect("division by zero"))
            }
        }
    };
}

counted_div!(TrackedFr, TrackedFr);
counted_div!(TrackedFr, &TrackedFr);
counted_div!(&TrackedFr, TrackedFr);
counted_div!(&TrackedFr, &TrackedFr);

impl Neg for TrackedFr {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl AddAssign for TrackedFr {
    fn add_assign(&mut self, other: Self) {
        count(&ADD_COUNT);
        self.0 += other.0;
    }
}

impl SubAssign for TrackedFr {
    fn sub_assign(&mut self, other: Self) {
        count(&SUB_COUNT);
        self.0 -= other.0;
    }
}

impl MulAssign for TrackedFr {
    fn mul_assign(&mut self, other: Self) {
        count(&MULT_COUNT);
        self.0 *= other.0;
    }
}

impl Zero for TrackedFr {
    fn zero() -> Self {
        Self(Fr::zero())
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl One for TrackedFr {
    fn one() -> Self {
        Self(Fr::one())
    }
    fn is_one(&self) -> bool {
        self.0.is_one()
    }
}

impl Sum for TrackedFr {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |a, b| a + b)
    }
}

impl<'a> Sum<&'a Self> for TrackedFr {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |a, b| a + b)
    }
}

impl Product for TrackedFr {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |a, b| a * b)
    }
}

impl<'a> Product<&'a Self> for TrackedFr {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |a, b| a * b)
    }
}

impl AdditiveGroup for TrackedFr {}

impl Ring for TrackedFr {
    fn from_u64(v: u64) -> Self {
        count(&FROM_U64_COUNT);
        TrackedFr(Fr::from_u64(v))
    }

    fn from_i64(v: i64) -> Self {
        count(&FROM_I64_COUNT);
        TrackedFr(Fr::from_i64(v))
    }

    fn from_u128(v: u128) -> Self {
        count(&FROM_U128_COUNT);
        TrackedFr(Fr::from_u128(v))
    }

    fn from_i128(v: i128) -> Self {
        count(&FROM_I128_COUNT);
        TrackedFr(Fr::from_i128(v))
    }

    fn from_bool(v: bool) -> Self {
        count(&FROM_BOOL_COUNT);
        TrackedFr(Fr::from_bool(v))
    }

    fn from_u8(v: u8) -> Self {
        count(&FROM_U8_COUNT);
        TrackedFr(Fr::from_u8(v))
    }

    fn from_u16(v: u16) -> Self {
        count(&FROM_U16_COUNT);
        TrackedFr(Fr::from_u16(v))
    }

    fn from_u32(v: u32) -> Self {
        count(&FROM_U32_COUNT);
        TrackedFr(Fr::from_u32(v))
    }

    fn from_i32(v: i32) -> Self {
        count(&FROM_I32_COUNT);
        TrackedFr(Fr::from_i32(v))
    }

    fn square(&self) -> Self {
        count(&SQUARE_COUNT);
        TrackedFr(self.0.square())
    }

    fn mul_u64(&self, n: u64) -> Self {
        count(&MUL_U64_COUNT);
        TrackedFr(self.0.mul_u64(n))
    }

    fn mul_i64(&self, n: i64) -> Self {
        count(&MUL_I64_COUNT);
        TrackedFr(self.0.mul_i64(n))
    }

    fn mul_u128(&self, n: u128) -> Self {
        count(&MUL_U128_COUNT);
        TrackedFr(self.0.mul_u128(n))
    }

    fn mul_i128(&self, n: i128) -> Self {
        count(&MUL_I128_COUNT);
        TrackedFr(self.0.mul_i128(n))
    }
}

impl Field for TrackedFr {
    fn inverse(&self) -> Option<Self> {
        count(&INVERSE_COUNT);
        self.0.inverse().map(TrackedFr)
    }

    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
        TrackedFr(Fr::random(rng))
    }
}

impl CanonicalBytes for TrackedFr {
    const NUM_BYTES: usize = <Fr as CanonicalBytes>::NUM_BYTES;

    fn to_bytes_le(&self, out: &mut [u8]) {
        self.0.to_bytes_le(out)
    }
}

impl CanonicalEncoding for TrackedFr {
    const MODULUS_BITS: u32 = <Fr as CanonicalEncoding>::MODULUS_BITS;

    fn from_bytes_le_reduced(bytes: &[u8]) -> Self {
        TrackedFr(Fr::from_bytes_le_reduced(bytes))
    }

    fn from_bytes_le_checked(bytes: &[u8]) -> Option<Self> {
        Fr::from_bytes_le_checked(bytes).map(TrackedFr)
    }

    fn to_u128_checked(&self) -> Option<u128> {
        self.0.to_u128_checked()
    }

    fn from_u128_checked(v: u128) -> Option<Self> {
        Fr::from_u128_checked(v).map(TrackedFr)
    }

    fn from_u128_reduced(v: u128) -> Self {
        TrackedFr(Fr::from_u128_reduced(v))
    }

    fn num_bits(&self) -> u32 {
        self.0.num_bits()
    }

    fn from_scalar_challenge_bytes(bytes: &[u8]) -> Self {
        TrackedFr(Fr::from_scalar_challenge_bytes(bytes))
    }
}

impl WithAccumulator for TrackedFr {
    type Accumulator = NaiveAccumulator<TrackedFr>;
    type SmallScalarAccumulator = NaiveAccumulator<TrackedFr>;
    type SignedProductAccumulator = NaiveAccumulator<TrackedFr>;
}

impl FieldOps for TrackedFr {}
impl FieldOps<&TrackedFr, TrackedFr> for TrackedFr {}

impl JoltField for TrackedFr {
    const NUM_BYTES: usize = <Fr as JoltField>::NUM_BYTES;
    const NUM_LIMBS: usize = <Fr as JoltField>::NUM_LIMBS;

    const MONTGOMERY_R: Self = TrackedFr(<Fr as JoltField>::MONTGOMERY_R);
    const MONTGOMERY_R_SQUARE: Self = TrackedFr(<Fr as JoltField>::MONTGOMERY_R_SQUARE);

    type UnreducedElem = <Fr as JoltField>::UnreducedElem;
    type UnreducedMulU64 = <Fr as JoltField>::UnreducedMulU64;
    type UnreducedMulU128 = <Fr as JoltField>::UnreducedMulU128;
    type UnreducedMulU128Accum = <Fr as JoltField>::UnreducedMulU128Accum;
    type UnreducedProduct = <Fr as JoltField>::UnreducedProduct;
    type UnreducedProductAccum = <Fr as JoltField>::UnreducedProductAccum;

    type SmallValueLookupTables = [Vec<Self>; 2];

    type ChallengeCodecField = Fr;

    fn from_challenge_codec(x: Fr) -> Self {
        TrackedFr(x)
    }

    #[cfg(not(feature = "challenge-254-bit"))]
    fn challenge_from_field(x: Self) -> Self::Challenge {
        let [_, _, low, high] = x.0.inner_limbs();
        MontU128Challenge::new(u128::from(low) | (u128::from(high) << 64))
    }

    #[cfg(feature = "challenge-254-bit")]
    fn challenge_from_field(x: Self) -> Self::Challenge {
        Mont254BitChallenge::new(x)
    }

    fn challenge_set_size() -> jolt_transcript::ChallengeSetSize {
        <Fr as JoltField>::challenge_set_size()
    }

    // Default: Use optimized 125-bit MontChallenge
    #[cfg(not(feature = "challenge-254-bit"))]
    type Challenge = MontU128Challenge<TrackedFr>;

    // Optional: Use full 254-bit field elements
    #[cfg(feature = "challenge-254-bit")]
    type Challenge = Mont254BitChallenge<TrackedFr>;

    fn compute_lookup_tables() -> Self::SmallValueLookupTables {
        <Fr as JoltField>::compute_lookup_tables().map(|t| t.into_iter().map(TrackedFr).collect())
    }

    fn to_u64(&self) -> Option<u64> {
        self.0.to_u64()
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        count(&FROM_BYTES_COUNT);
        TrackedFr(<Fr as JoltField>::from_bytes(bytes))
    }

    fn to_unreduced(&self) -> Self::UnreducedElem {
        self.0.to_unreduced()
    }

    fn mul_u64_unreduced(self, other: u64) -> Self::UnreducedMulU64 {
        count(&MUL_U64_UNRED_COUNT);
        self.0.mul_u64_unreduced(other)
    }

    fn mul_u128_unreduced(self, other: u128) -> Self::UnreducedMulU128 {
        count(&MUL_U128_UNRED_COUNT);
        self.0.mul_u128_unreduced(other)
    }

    fn mul_to_product(self, other: Self) -> Self::UnreducedProduct {
        count(&MUL_UNRED_COUNT);
        self.0.mul_to_product(other.0)
    }

    fn mul_to_product_accum(self, other: Self) -> Self::UnreducedProductAccum {
        count(&MUL_UNRED_COUNT);
        self.0.mul_to_product_accum(other.0)
    }

    fn unreduced_mul_u64(a: &Self::UnreducedElem, b: u64) -> Self::UnreducedMulU64 {
        <Fr as JoltField>::unreduced_mul_u64(a, b)
    }

    fn unreduced_mul_to_product_accum(
        a: &Self::UnreducedElem,
        b: &Self::UnreducedElem,
    ) -> Self::UnreducedProductAccum {
        <Fr as JoltField>::unreduced_mul_to_product_accum(a, b)
    }

    #[inline]
    fn mul_to_accum_mag<const M: usize>(
        &self,
        mag: &ark_ff::BigInt<M>,
    ) -> Self::UnreducedMulU128Accum {
        self.0.mul_to_accum_mag(mag)
    }

    #[inline]
    fn mul_to_product_mag<const M: usize>(
        &self,
        mag: &ark_ff::BigInt<M>,
    ) -> Self::UnreducedProduct {
        self.0.mul_to_product_mag(mag)
    }

    fn reduce_mul_u64(x: Self::UnreducedMulU64) -> Self {
        count(&BARRETT_REDUCE_COUNT);
        TrackedFr(<Fr as JoltField>::reduce_mul_u64(x))
    }

    fn reduce_mul_u128(x: Self::UnreducedMulU128) -> Self {
        count(&BARRETT_REDUCE_COUNT);
        TrackedFr(<Fr as JoltField>::reduce_mul_u128(x))
    }

    fn reduce_mul_u128_accum(x: Self::UnreducedMulU128Accum) -> Self {
        count(&BARRETT_REDUCE_COUNT);
        TrackedFr(<Fr as JoltField>::reduce_mul_u128_accum(x))
    }

    fn reduce_product(x: Self::UnreducedProduct) -> Self {
        count(&MONT_REDUCE_COUNT);
        TrackedFr(<Fr as JoltField>::reduce_product(x))
    }

    fn reduce_product_accum(x: Self::UnreducedProductAccum) -> Self {
        count(&MONT_REDUCE_COUNT);
        TrackedFr(<Fr as JoltField>::reduce_product_accum(x))
    }
}

impl TrackedFr {
    #[inline(always)]
    pub fn mul_hi_bigint_u128(&self, n: [u64; 4]) -> Self {
        count(&MULT_COUNT);
        TrackedFr(self.0.mul_hi_bigint_u128(n))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::op_ref)]
    use crate::field::tracked_ark::TrackedFr as Fr;
    use crate::field::OptimizedMul;
    use crate::utils::counters::{
        get_inverse_count, get_mult_count, reset_inverse_count, reset_mult_count,
    };
    use jolt_field::{Field, Ring};
    use std::ops::MulAssign;

    #[test]
    fn test_if_trackers_are_working() {
        reset_mult_count();
        let a = Fr::from_u8(12);
        let b = Fr::from_u8(12);
        let _ = a.mul_0_optimized(b);
        let num_mults = get_mult_count();
        assert_eq!(num_mults, 1);

        reset_mult_count();
        let a = Fr::from_u8(12);
        let b = Fr::from_u8(12);
        let _ = a * &b;
        let num_mults = get_mult_count();
        assert_eq!(num_mults, 1);

        reset_mult_count();
        let a = Fr::from_u8(12);
        let b = Fr::from_u8(12);
        let _ = &a * b;
        let num_mults = get_mult_count();
        assert_eq!(num_mults, 1);

        reset_mult_count();
        let a = Fr::from_u8(12);
        let b = Fr::from_u8(12);
        let _ = &a * &b;
        let num_mults = get_mult_count();
        assert_eq!(num_mults, 1);

        reset_mult_count();
        let a = Fr::from_u8(12);
        let b = Fr::from_u8(12);
        let _ = a * b;
        let num_mults = get_mult_count();
        assert_eq!(num_mults, 1);

        reset_inverse_count();
        let mut a = Fr::from_u8(12);
        let _b = Fr::from_u8(12);
        let c = a.inverse().unwrap();
        let num_mults = get_inverse_count();
        assert_eq!(num_mults, 1);

        reset_mult_count();
        a.mul_assign(b);
        let num_mults = get_mult_count();
        assert_eq!(num_mults, 1);

        reset_mult_count();
        let vals = [&a, &b, &c];
        let _: Fr = vals.into_iter().product();
        let num_mults = get_mult_count();
        assert_eq!(num_mults, 3);
    }
}
