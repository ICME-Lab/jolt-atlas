//! The arkworks BN254 scalar-field kernel behind [`jolt_field::Fr`]'s [`JoltField`] impl
//! (`super::bn254`): the unreduced-product ladder and the conversions upstream does not provide,
//! on `ark_bn254::Fr` limbs. `ark_bn254::Fr` itself is not a `JoltField`.

use super::UnreducedInteger;
use crate::field::folded_accum::{
    Folded256MulU128, Folded256MulU128Accum, Folded256MulU64, Folded256Product,
    Folded256ProductAccum,
};
use crate::utils::thread::unsafe_allocate_zero_vec;
use ark_ff::{BigInt, BigInteger, PrimeField};
use common::parallel::par_enabled;
use rayon::prelude::*;

impl<const N: usize> UnreducedInteger for BigInt<N> {}

/// The kernel operations [`super::bn254`] forwards to, implemented on arkworks limbs.
pub(crate) trait Bn254Kernel: Sized {
    const MONTGOMERY_R: Self;
    const MONTGOMERY_R_SQUARE: Self;

    type UnreducedElem;
    type UnreducedMulU64;
    type UnreducedMulU128;
    type UnreducedMulU128Accum;
    type UnreducedProduct;
    type UnreducedProductAccum;

    fn compute_lookup_tables() -> [Vec<Self>; 2];
    fn to_u64(&self) -> Option<u64>;
    fn from_bytes(bytes: &[u8]) -> Self;
    fn to_unreduced(&self) -> Self::UnreducedElem;
    fn mul_u64_unreduced(self, other: u64) -> Self::UnreducedMulU64;
    fn mul_u128_unreduced(self, other: u128) -> Self::UnreducedMulU128;
    fn mul_to_product(self, other: Self) -> Self::UnreducedProduct;
    fn mul_to_product_accum(self, other: Self) -> Self::UnreducedProductAccum;
    fn unreduced_mul_u64(a: &Self::UnreducedElem, b: u64) -> Self::UnreducedMulU64;
    fn unreduced_mul_to_product_accum(
        a: &Self::UnreducedElem,
        b: &Self::UnreducedElem,
    ) -> Self::UnreducedProductAccum;
    fn mul_to_accum_mag<const M: usize>(&self, mag: &BigInt<M>) -> Self::UnreducedMulU128Accum;
    fn mul_to_product_mag<const M: usize>(&self, mag: &BigInt<M>) -> Self::UnreducedProduct;
    fn reduce_mul_u64(x: Self::UnreducedMulU64) -> Self;
    fn reduce_mul_u128(x: Self::UnreducedMulU128) -> Self;
    fn reduce_mul_u128_accum(x: Self::UnreducedMulU128Accum) -> Self;
    fn reduce_product(x: Self::UnreducedProduct) -> Self;
    fn reduce_product_accum(x: Self::UnreducedProductAccum) -> Self;
}

impl Bn254Kernel for ark_bn254::Fr {
    // SAFETY: Transmuting from the Montgomery R constants from arkworks,
    // which are guaranteed to be valid field elements in Montgomery form.
    const MONTGOMERY_R: Self = unsafe {
        use ark_ff::MontConfig;
        std::mem::transmute(<ark_bn254::FrConfig as MontConfig<4>>::R)
    };
    const MONTGOMERY_R_SQUARE: Self = unsafe {
        use ark_ff::MontConfig;
        std::mem::transmute(<ark_bn254::FrConfig as MontConfig<4>>::R2)
    };

    type UnreducedElem = BigInt<4>;
    type UnreducedMulU64 = Folded256MulU64;
    type UnreducedMulU128 = Folded256MulU128;
    type UnreducedMulU128Accum = Folded256MulU128Accum;
    type UnreducedProduct = Folded256Product;
    type UnreducedProductAccum = Folded256ProductAccum;

    fn compute_lookup_tables() -> [Vec<Self>; 2] {
        let mut lookup_tables = [
            unsafe_allocate_zero_vec(1 << 16),
            unsafe_allocate_zero_vec(1 << 16),
        ];

        for i in 0..2 {
            let bitshift = 16 * i;
            let unit = ark_bn254::Fr::from(1u64 << bitshift);
            lookup_tables[i] = (0usize..(1 << 16))
                .into_par_iter()
                .with_min_len(par_enabled())
                .map(|j| unit * ark_bn254::Fr::from(j as u64))
                .collect();
        }

        lookup_tables
    }

    #[inline]
    fn to_u64(&self) -> Option<u64> {
        let bigint = <Self as PrimeField>::into_bigint(*self);
        let limbs: &[u64] = bigint.as_ref();
        let result = limbs[0];

        if ark_bn254::Fr::from(result) != *self {
            None
        } else {
            Some(result)
        }
    }

    #[inline]
    fn from_bytes(bytes: &[u8]) -> Self {
        ark_bn254::Fr::from_le_bytes_mod_order(bytes)
    }

    #[inline(always)]
    fn to_unreduced(&self) -> Self::UnreducedElem {
        self.0
    }

    #[inline]
    fn mul_u64_unreduced(self, other: u64) -> Folded256MulU64 {
        Folded256MulU64::from_bigint(self.0.mul_trunc::<1, 5>(&BigInt::new([other])))
    }

    #[inline]
    fn mul_u128_unreduced(self, other: u128) -> Folded256MulU128 {
        Folded256MulU128::from_bigint(
            self.0
                .mul_trunc::<2, 6>(&BigInt::new([other as u64, (other >> 64) as u64])),
        )
    }

    #[inline]
    fn mul_to_product(self, other: Self) -> Folded256Product {
        Folded256Product::from_mul(self.0, other.0)
    }

    #[inline]
    fn mul_to_product_accum(self, other: Self) -> Folded256ProductAccum {
        Folded256ProductAccum::from_mul(self.0, other.0)
    }

    #[inline]
    fn unreduced_mul_u64(a: &BigInt<4>, b: u64) -> Folded256MulU64 {
        Folded256MulU64::from_bigint(a.mul_u64_w_carry::<5>(b))
    }

    #[inline]
    fn unreduced_mul_to_product_accum(a: &BigInt<4>, b: &BigInt<4>) -> Folded256ProductAccum {
        Folded256ProductAccum::from_mul(*a, *b)
    }

    #[inline]
    fn mul_to_accum_mag<const M: usize>(&self, mag: &BigInt<M>) -> Folded256MulU128Accum {
        Folded256MulU128Accum::from_bigint(self.0.mul_trunc::<M, 7>(mag))
    }

    #[inline]
    fn mul_to_product_mag<const M: usize>(&self, mag: &BigInt<M>) -> Folded256Product {
        Folded256Product::from_bigint(self.0.mul_trunc::<M, 8>(mag))
    }

    #[inline]
    fn reduce_mul_u64(x: Folded256MulU64) -> Self {
        ark_bn254::Fr::from_barrett_reduce::<5, 5>(x.normalize())
    }

    #[inline]
    fn reduce_mul_u128(x: Folded256MulU128) -> Self {
        ark_bn254::Fr::from_barrett_reduce::<6, 5>(x.normalize())
    }

    #[inline]
    fn reduce_mul_u128_accum(x: Folded256MulU128Accum) -> Self {
        ark_bn254::Fr::from_barrett_reduce::<7, 5>(x.normalize())
    }

    #[inline]
    fn reduce_product(x: Folded256Product) -> Self {
        ark_bn254::Fr::from_montgomery_reduce::<8, 5>(x.normalize())
    }

    #[inline]
    fn reduce_product_accum(x: Folded256ProductAccum) -> Self {
        ark_bn254::Fr::from_montgomery_reduce::<9, 5>(x.normalize())
    }
}
