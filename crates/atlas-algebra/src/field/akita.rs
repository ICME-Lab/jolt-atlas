//! Atlas field traits for Jolt's native Akita field.

use super::{
    challenge::ChallengeCodec,
    folded_accum::{Folded128MulU64, Folded128Product},
    Accumulator, CanonicalBytes, CanonicalEncoding, Field, FieldOps, JoltField, Limbs, Ring,
    WithAccumulator,
};
use crate::serialize::{Canonical, Error};
use allocative::Allocative;
use jolt_field::solinas::Prime128Offset6A23801F as Native;
use jolt_field::{
    Accumulator as NativeAccumulator, CanonicalBytes as NativeBytes,
    CanonicalEncoding as NativeEncoding, Field as NativeField, Ring as NativeRing,
    WithAccumulator as NativeWithAccumulator,
};
use std::{
    fmt::{Display, Formatter},
    io::{Read, Write},
    ops::Div,
};

/// Atlas's field wrapper around Jolt's native `Prime128Offset6A23801F`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Allocative)]
pub struct AkitaField(#[allocative(skip)] Native);

impl AkitaField {
    /// View this value as the native Jolt field element.
    pub fn into_native(self) -> Native {
        self.0
    }

    /// Wrap a field element produced by Jolt or Akita.
    pub fn from_native(value: Native) -> Self {
        Self(value)
    }
}

impl Display for AkitaField {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, f)
    }
}

crate::impl_ring_ops!(impl[] AkitaField {
    add(a, b): AkitaField(a.0 + b.0),
    sub(a, b): AkitaField(a.0 - b.0),
    mul(a, b): AkitaField(a.0 * b.0),
    neg(a): AkitaField(-a.0),
    zero: AkitaField(Native::from_u64(0)),
    one: AkitaField(Native::from_u64(1)),
});

impl Div for AkitaField {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        std::ops::Mul::mul(self, rhs.inverse().expect("division by zero"))
    }
}

impl Div<&Self> for AkitaField {
    type Output = Self;

    fn div(self, rhs: &Self) -> Self {
        self / *rhs
    }
}

impl FieldOps for AkitaField {}
impl FieldOps<&AkitaField, AkitaField> for AkitaField {}

impl From<u128> for AkitaField {
    fn from(value: u128) -> Self {
        Self(Native::from_u128(value))
    }
}

impl Ring for AkitaField {
    fn from_u64(value: u64) -> Self {
        Self(Native::from_u64(value))
    }

    fn from_i64(value: i64) -> Self {
        Self(Native::from_i64(value))
    }

    fn from_u128(value: u128) -> Self {
        Self::from(value)
    }

    fn from_i128(value: i128) -> Self {
        Self(Native::from_i128(value))
    }

    fn square(&self) -> Self {
        Self(self.0.square())
    }
}

impl Field for AkitaField {
    fn inverse(&self) -> Option<Self> {
        self.0.inverse().map(Self)
    }

    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
        Self(Native::random(rng))
    }

    fn mul_add(self, rhs: Self, addend: Self) -> Self {
        Self(self.0.mul_add(rhs.0, addend.0))
    }

    fn half(self) -> Self {
        Self(self.0.half())
    }

    fn two_inv() -> Self {
        Self(Native::two_inv())
    }
}

impl CanonicalBytes for AkitaField {
    const NUM_BYTES: usize = Native::NUM_BYTES;

    fn to_bytes_le(&self, out: &mut [u8]) {
        self.0.to_bytes_le(out);
    }
}

impl CanonicalEncoding for AkitaField {
    const MODULUS_BITS: u32 = Native::MODULUS_BITS;

    fn from_bytes_le_reduced(bytes: &[u8]) -> Self {
        Self(Native::from_bytes_le_reduced(bytes))
    }

    fn from_bytes_le_checked(bytes: &[u8]) -> Option<Self> {
        Native::from_bytes_le_checked(bytes).map(Self)
    }

    fn to_u128_checked(&self) -> Option<u128> {
        self.0.to_u128_checked()
    }

    fn from_u128_checked(value: u128) -> Option<Self> {
        Native::from_u128_checked(value).map(Self)
    }

    fn from_u128_reduced(value: u128) -> Self {
        Self(Native::from_u128_reduced(value))
    }

    fn num_bits(&self) -> u32 {
        self.0.num_bits()
    }
}

crate::impl_serde_bytes!(impl[] AkitaField, 16);

impl Canonical for AkitaField {
    fn encode<W: Write>(&self, writer: W) -> Result<(), Error> {
        self.0.to_canonical_u128().encode(writer)
    }

    fn encoded_len(&self) -> usize {
        16
    }

    fn decode<R: Read>(reader: R) -> Result<Self, Error> {
        Self::from_u128_checked(u128::decode(reader)?).ok_or(Error::InvalidData)
    }
}

/// Deferred product accumulation using Jolt's native field accumulator.
#[derive(Clone, Copy, Default)]
pub struct AkitaAccumulator(<Native as NativeWithAccumulator>::Accumulator);

impl Accumulator for AkitaAccumulator {
    type Element = AkitaField;

    fn add(&mut self, value: AkitaField) {
        self.0.add(value.0);
    }

    fn merge(&mut self, other: Self) {
        self.0.merge(other.0);
    }

    fn reduce(self) -> AkitaField {
        AkitaField(self.0.reduce())
    }

    fn fmadd(&mut self, a: AkitaField, b: AkitaField) {
        self.0.fmadd(a.0, b.0);
    }

    fn fmadd_u64(&mut self, a: AkitaField, b: u64) {
        self.0.fmadd_u64(a.0, b);
    }

    fn fmadd_i64(&mut self, a: AkitaField, b: i64) {
        self.0.fmadd_i64(a.0, b);
    }

    fn fmadd_i128(&mut self, a: AkitaField, b: i128) {
        self.0.fmadd_i128(a.0, b);
    }
}

impl WithAccumulator for AkitaField {
    type Accumulator = AkitaAccumulator;
}

// Reuse Atlas's Fiat–Shamir challenge encoding with the native Akita modulus.
type CodecField = super::solinas::Fp128<{ 0u128.wrapping_sub(Native::C) }>;

impl JoltField for AkitaField {
    const NUM_BYTES: usize = 16;
    // SAFETY: 1 is a canonical field representative and the kernel stores it as such.
    const MONTGOMERY_R: Self = Self(unsafe { Native::from_canonical_u128(1) });
    const MONTGOMERY_R_SQUARE: Self = Self::MONTGOMERY_R;

    type UnreducedElem = Limbs<2>;
    type UnreducedMulU64 = Folded128MulU64;
    type UnreducedMulU128 = Folded128Product;
    type UnreducedMulU128Accum = Folded128Product;
    type UnreducedProduct = Folded128Product;
    type UnreducedProductAccum = Folded128Product;

    type SmallValueLookupTables = [Vec<Self>; 2];

    type Challenge = Self;
    type ChallengeCodecField = CodecField;

    fn from_challenge_codec(value: CodecField) -> Self {
        Self::from(value.to_canonical_u128())
    }

    fn challenge_from_field(value: Self) -> Self {
        value
    }

    fn challenge_set_size() -> super::challenge::ChallengeSetSize {
        CodecField::CHALLENGE_SET_SIZE
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        Self::from_bytes_le_reduced(bytes)
    }

    fn to_u64(&self) -> Option<u64> {
        self.to_u64_checked()
    }

    fn to_unreduced(&self) -> Limbs<2> {
        Limbs::new(self.0.to_limbs())
    }

    fn mul_u64_unreduced(self, other: u64) -> Folded128MulU64 {
        Folded128MulU64::from_bigint(Limbs::new(self.0.mul_wide_u64(other)))
    }

    fn mul_u128_unreduced(self, other: u128) -> Folded128Product {
        Folded128Product::from_bigint(Limbs::new(self.0.mul_wide_u128(other)))
    }

    fn mul_to_product(self, other: Self) -> Folded128Product {
        Folded128Product::from_bigint(Limbs::new(self.0.mul_wide(other.0)))
    }

    fn mul_to_product_accum(self, other: Self) -> Folded128Product {
        self.mul_to_product(other)
    }

    fn unreduced_mul_u64(value: &Limbs<2>, other: u64) -> Folded128MulU64 {
        Self(Native::solinas_reduce(&value.0)).mul_u64_unreduced(other)
    }

    fn unreduced_mul_to_product_accum(a: &Limbs<2>, b: &Limbs<2>) -> Folded128Product {
        Self(Native::solinas_reduce(&a.0)).mul_to_product(Self(Native::solinas_reduce(&b.0)))
    }

    fn mul_to_accum_mag<const M: usize>(&self, mag: &Limbs<M>) -> Folded128Product {
        self.mul_to_product_mag(mag)
    }

    fn mul_to_product_mag<const M: usize>(&self, mag: &Limbs<M>) -> Folded128Product {
        let value = self.0 * Native::solinas_reduce(&mag.0);
        Folded128Product::from_bigint(Limbs::new(value.to_limbs()))
    }

    fn reduce_mul_u64(value: Folded128MulU64) -> Self {
        Self(Native::solinas_reduce(&value.normalize().0))
    }

    fn reduce_mul_u128(value: Folded128Product) -> Self {
        Self::reduce_product(value)
    }

    fn reduce_mul_u128_accum(value: Folded128Product) -> Self {
        Self::reduce_product(value)
    }

    fn reduce_product(value: Folded128Product) -> Self {
        Self(Native::solinas_reduce(&value.normalize().0))
    }

    fn reduce_product_accum(value: Folded128Product) -> Self {
        Self::reduce_product(value)
    }
}
