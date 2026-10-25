//! [`JoltField`] for upstream `jolt_field::Fr`, the BN254 scalar field.
//!
//! `jolt_field::Fr` is `#[repr(transparent)]` over `ark_bn254::Fr`, its arithmetic delegates to
//! arkworks, and its `CanonicalSerialize` is arkworks' byte for byte. This impl forwards every
//! method to the `ark_bn254::Fr` impl in [`super::ark`], which stays the arithmetic kernel, so the
//! two types produce identical values, transcripts and proofs.

use super::{ArkScalar, FieldOps, JoltField};
#[cfg(feature = "challenge-254-bit")]
use crate::field::challenge::Mont254BitChallenge;
#[cfg(not(feature = "challenge-254-bit"))]
use crate::field::challenge::MontU128Challenge;
use ark_ff::BigInt;

pub use jolt_field::Fr;

type Ark = ark_bn254::Fr;

/// The arkworks element underneath.
#[inline(always)]
pub fn to_ark(x: Fr) -> Ark {
    x.into()
}

/// Wrap an arkworks element.
#[inline(always)]
pub fn from_ark(x: Ark) -> Fr {
    x.into()
}

/// Borrow the arkworks element underneath.
#[inline(always)]
pub fn as_ark(x: &Fr) -> &Ark {
    // SAFETY: `jolt_field::Fr` is `#[repr(transparent)]` over `ark_bn254::Fr`.
    unsafe { &*(x as *const Fr as *const Ark) }
}

/// Reinterpret a slice of elements as arkworks elements, without copying.
#[inline(always)]
pub fn slice_as_ark(xs: &[Fr]) -> &[Ark] {
    // SAFETY: `jolt_field::Fr` is `#[repr(transparent)]` over `ark_bn254::Fr`, so the two slices
    // have the same layout.
    unsafe { std::slice::from_raw_parts(xs.as_ptr() as *const Ark, xs.len()) }
}

/// Reinterpret a vector of arkworks elements as elements, without copying.
#[inline(always)]
pub fn vec_from_ark(xs: Vec<Ark>) -> Vec<Fr> {
    let mut xs = std::mem::ManuallyDrop::new(xs);
    // SAFETY: `jolt_field::Fr` is `#[repr(transparent)]` over `ark_bn254::Fr`, so pointer,
    // length and capacity describe a valid `Vec<Fr>` allocation.
    unsafe { Vec::from_raw_parts(xs.as_mut_ptr() as *mut Fr, xs.len(), xs.capacity()) }
}

/// Challenge-by-element multiplication on the arkworks kernel (the fork's
/// `mul_hi_bigint_u128`), exposed as a method so the challenge operator macros apply.
pub(crate) trait MulHiBigintU128 {
    fn mul_hi_bigint_u128(&self, n: [u64; 4]) -> Self;
}

impl MulHiBigintU128 for Fr {
    #[inline(always)]
    fn mul_hi_bigint_u128(&self, n: [u64; 4]) -> Self {
        from_ark(as_ark(self).mul_hi_bigint_u128(n))
    }
}

impl ArkScalar for Fr {
    type Ark = Ark;

    #[inline(always)]
    fn to_ark(self) -> Ark {
        to_ark(self)
    }

    #[inline(always)]
    fn slice_as_ark(xs: &[Self]) -> &[Ark] {
        slice_as_ark(xs)
    }

    #[cfg(not(feature = "challenge-254-bit"))]
    #[inline(always)]
    fn challenge_to_ark(c: Self::Challenge) -> MontU128Challenge<Ark> {
        let [_, _, low, high] = c.value();
        MontU128Challenge::new(u128::from(low) | (u128::from(high) << 64))
    }

    #[cfg(feature = "challenge-254-bit")]
    #[inline(always)]
    fn challenge_to_ark(c: Self::Challenge) -> Mont254BitChallenge<Ark> {
        Mont254BitChallenge::new(to_ark(c.value()))
    }
}

impl FieldOps for Fr {}
impl FieldOps<&Fr, Fr> for Fr {}

impl JoltField for Fr {
    const NUM_BYTES: usize = <Ark as JoltField>::NUM_BYTES;
    const NUM_LIMBS: usize = <Ark as JoltField>::NUM_LIMBS;

    // SAFETY: `jolt_field::Fr` is `#[repr(transparent)]` over `ark_bn254::Fr`.
    const MONTGOMERY_R: Self =
        unsafe { std::mem::transmute::<Ark, Fr>(<Ark as JoltField>::MONTGOMERY_R) };
    // SAFETY: as above.
    const MONTGOMERY_R_SQUARE: Self =
        unsafe { std::mem::transmute::<Ark, Fr>(<Ark as JoltField>::MONTGOMERY_R_SQUARE) };

    type UnreducedElem = <Ark as JoltField>::UnreducedElem;
    type UnreducedMulU64 = <Ark as JoltField>::UnreducedMulU64;
    type UnreducedMulU128 = <Ark as JoltField>::UnreducedMulU128;
    type UnreducedMulU128Accum = <Ark as JoltField>::UnreducedMulU128Accum;
    type UnreducedProduct = <Ark as JoltField>::UnreducedProduct;
    type UnreducedProductAccum = <Ark as JoltField>::UnreducedProductAccum;

    type SmallValueLookupTables = [Vec<Self>; 2];

    type ChallengeCodecField = Fr;

    #[inline]
    fn from_challenge_codec(x: Fr) -> Self {
        x
    }

    fn challenge_set_size() -> jolt_transcript::ChallengeSetSize {
        <Ark as JoltField>::challenge_set_size()
    }

    #[cfg(not(feature = "challenge-254-bit"))]
    type Challenge = MontU128Challenge<Fr>;
    #[cfg(feature = "challenge-254-bit")]
    type Challenge = Mont254BitChallenge<Fr>;

    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
        from_ark(<Ark as JoltField>::random(rng))
    }

    fn compute_lookup_tables() -> Self::SmallValueLookupTables {
        <Ark as JoltField>::compute_lookup_tables().map(vec_from_ark)
    }

    #[inline]
    fn from_bool(val: bool) -> Self {
        from_ark(<Ark as JoltField>::from_bool(val))
    }

    #[inline]
    fn from_u8(n: u8) -> Self {
        from_ark(<Ark as JoltField>::from_u8(n))
    }

    #[inline]
    fn from_u16(n: u16) -> Self {
        from_ark(<Ark as JoltField>::from_u16(n))
    }

    #[inline]
    fn from_u32(n: u32) -> Self {
        from_ark(<Ark as JoltField>::from_u32(n))
    }

    #[inline]
    fn from_u64(n: u64) -> Self {
        from_ark(<Ark as JoltField>::from_u64(n))
    }

    #[cfg(test)]
    fn from_i8(val: i8) -> Self {
        from_ark(<Ark as JoltField>::from_i8(val))
    }

    #[cfg(test)]
    fn from_i16(val: i16) -> Self {
        from_ark(<Ark as JoltField>::from_i16(val))
    }

    #[inline]
    fn from_i32(val: i32) -> Self {
        from_ark(<Ark as JoltField>::from_i32(val))
    }

    #[inline]
    fn from_i64(val: i64) -> Self {
        from_ark(<Ark as JoltField>::from_i64(val))
    }

    #[inline]
    fn from_i128(val: i128) -> Self {
        from_ark(<Ark as JoltField>::from_i128(val))
    }

    #[inline]
    fn from_u128(val: u128) -> Self {
        from_ark(<Ark as JoltField>::from_u128(val))
    }

    #[inline]
    fn to_u64(&self) -> Option<u64> {
        <Ark as JoltField>::to_u64(as_ark(self))
    }

    #[inline]
    fn square(&self) -> Self {
        from_ark(<Ark as JoltField>::square(as_ark(self)))
    }

    #[inline]
    fn inverse(&self) -> Option<Self> {
        <Ark as JoltField>::inverse(as_ark(self)).map(from_ark)
    }

    #[inline]
    fn from_bytes(bytes: &[u8]) -> Self {
        from_ark(<Ark as JoltField>::from_bytes(bytes))
    }

    #[inline]
    fn num_bits(&self) -> u32 {
        <Ark as JoltField>::num_bits(as_ark(self))
    }

    #[inline]
    fn to_unreduced(&self) -> Self::UnreducedElem {
        <Ark as JoltField>::to_unreduced(as_ark(self))
    }

    #[inline]
    fn mul_u64(&self, n: u64) -> Self {
        from_ark(<Ark as JoltField>::mul_u64(as_ark(self), n))
    }

    #[inline]
    fn mul_i64(&self, n: i64) -> Self {
        from_ark(<Ark as JoltField>::mul_i64(as_ark(self), n))
    }

    #[inline]
    fn mul_u128(&self, n: u128) -> Self {
        from_ark(<Ark as JoltField>::mul_u128(as_ark(self), n))
    }

    #[inline]
    fn mul_i128(&self, n: i128) -> Self {
        from_ark(<Ark as JoltField>::mul_i128(as_ark(self), n))
    }

    #[inline]
    fn mul_u64_unreduced(self, other: u64) -> Self::UnreducedMulU64 {
        <Ark as JoltField>::mul_u64_unreduced(to_ark(self), other)
    }

    #[inline]
    fn mul_u128_unreduced(self, other: u128) -> Self::UnreducedMulU128 {
        <Ark as JoltField>::mul_u128_unreduced(to_ark(self), other)
    }

    #[inline]
    fn mul_to_product(self, other: Self) -> Self::UnreducedProduct {
        <Ark as JoltField>::mul_to_product(to_ark(self), to_ark(other))
    }

    #[inline]
    fn mul_to_product_accum(self, other: Self) -> Self::UnreducedProductAccum {
        <Ark as JoltField>::mul_to_product_accum(to_ark(self), to_ark(other))
    }

    #[inline]
    fn unreduced_mul_u64(a: &Self::UnreducedElem, b: u64) -> Self::UnreducedMulU64 {
        <Ark as JoltField>::unreduced_mul_u64(a, b)
    }

    #[inline]
    fn unreduced_mul_to_product_accum(
        a: &Self::UnreducedElem,
        b: &Self::UnreducedElem,
    ) -> Self::UnreducedProductAccum {
        <Ark as JoltField>::unreduced_mul_to_product_accum(a, b)
    }

    #[inline]
    fn mul_to_accum_mag<const M: usize>(&self, mag: &BigInt<M>) -> Self::UnreducedMulU128Accum {
        <Ark as JoltField>::mul_to_accum_mag(as_ark(self), mag)
    }

    #[inline]
    fn mul_to_product_mag<const M: usize>(&self, mag: &BigInt<M>) -> Self::UnreducedProduct {
        <Ark as JoltField>::mul_to_product_mag(as_ark(self), mag)
    }

    #[inline]
    fn reduce_mul_u64(x: Self::UnreducedMulU64) -> Self {
        from_ark(<Ark as JoltField>::reduce_mul_u64(x))
    }

    #[inline]
    fn reduce_mul_u128(x: Self::UnreducedMulU128) -> Self {
        from_ark(<Ark as JoltField>::reduce_mul_u128(x))
    }

    #[inline]
    fn reduce_mul_u128_accum(x: Self::UnreducedMulU128Accum) -> Self {
        from_ark(<Ark as JoltField>::reduce_mul_u128_accum(x))
    }

    #[inline]
    fn reduce_product(x: Self::UnreducedProduct) -> Self {
        from_ark(<Ark as JoltField>::reduce_product(x))
    }

    #[inline]
    fn reduce_product_accum(x: Self::UnreducedProductAccum) -> Self {
        from_ark(<Ark as JoltField>::reduce_product_accum(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_serialize::CanonicalSerialize;
    use ark_std::{test_rng, UniformRand};

    /// The optimized challenge product (arkworks' `mul_hi_bigint_u128` through the bridge)
    /// equals the plain field product of the challenge's field element.
    #[test]
    fn challenge_product_matches_field_product() {
        let mut rng = test_rng();
        for _ in 0..256 {
            let x = <Fr as JoltField>::random(&mut rng);
            let c = <Fr as JoltField>::Challenge::rand(&mut rng);
            let c_field: Fr = c.into();
            assert_eq!(x * c, x * c_field);
            assert_eq!(c * x, c_field * x);
        }
    }

    /// Elements serialize to arkworks' bytes, so proofs and transcripts are unchanged by the
    /// representation.
    #[test]
    fn serialization_matches_arkworks() {
        let mut rng = test_rng();
        for _ in 0..64 {
            let x = <Fr as JoltField>::random(&mut rng);
            let (mut ours, mut ark) = (Vec::new(), Vec::new());
            x.serialize_compressed(&mut ours).unwrap();
            to_ark(x).serialize_compressed(&mut ark).unwrap();
            assert_eq!(ours, ark);
        }
    }
}
