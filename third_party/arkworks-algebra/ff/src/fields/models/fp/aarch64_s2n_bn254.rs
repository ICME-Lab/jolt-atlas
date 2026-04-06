//! Experimental AArch64 backend hook for BN254 Montgomery arithmetic.
//!
//! This provides an opt-in path for routing BN254 Montgomery multiplication and
//! squaring through `s2n-bignum` on Apple Silicon / other AArch64 targets.

#![allow(unsafe_code)]

#[cfg(all(feature = "s2n-bignum-aarch64-mul", target_arch = "aarch64"))]
unsafe extern "C" {
    fn bignum_montmul(k: u64, z: *mut u64, x: *const u64, y: *const u64, m: *const u64);
}

#[cfg(all(feature = "s2n-bignum-aarch64-square", target_arch = "aarch64"))]
unsafe extern "C" {
    fn bignum_montsqr(k: u64, z: *mut u64, x: *const u64, m: *const u64);
}

#[cfg(all(feature = "s2n-bignum-aarch64-mul", target_arch = "aarch64"))]
#[inline(always)]
pub(super) fn try_mul_assign<const N: usize>(a: &mut [u64; N], b: &[u64; N], m: &[u64; N]) -> bool {
    unsafe {
        let x = *a;
        bignum_montmul(N as u64, a.as_mut_ptr(), x.as_ptr(), b.as_ptr(), m.as_ptr());
    }
    true
}

#[cfg(not(all(feature = "s2n-bignum-aarch64-mul", target_arch = "aarch64")))]
#[allow(dead_code)]
#[inline(always)]
pub(super) fn try_mul_assign<const N: usize>(a: &mut [u64; N], b: &[u64; N], m: &[u64; N]) -> bool {
    let _ = (a, b, m);
    false
}

#[cfg(all(feature = "s2n-bignum-aarch64-square", target_arch = "aarch64"))]
#[inline(always)]
pub(super) fn try_square_in_place<const N: usize>(a: &mut [u64; N], m: &[u64; N]) -> bool {
    unsafe {
        let x = *a;
        bignum_montsqr(N as u64, a.as_mut_ptr(), x.as_ptr(), m.as_ptr());
    }
    true
}

#[cfg(not(all(feature = "s2n-bignum-aarch64-square", target_arch = "aarch64")))]
#[allow(dead_code)]
#[inline(always)]
pub(super) fn try_square_in_place<const N: usize>(a: &mut [u64; N], m: &[u64; N]) -> bool {
    let _ = (a, m);
    false
}
