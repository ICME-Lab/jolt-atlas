//! Tiny fixed-point primitives shared by every kernel: the fused-rescale
//! split, the checked `i32` narrowing, floor division, and the trig reduction constants.
//!
//! These functions are part of the proof specification (see the module docs
//! of [`crate::semantics`]).

use crate::id::Scale;

use super::coeffs::PI_Q64;

/// Narrow an `i64` to `i32`: `None` outside the `i32` range. An arithmetic result that does not
/// fit is an execution error, never a clamped value.
pub fn narrow_i32(v: i64) -> Option<i32> {
    i32::try_from(v).ok()
}

/// Split a raw `i64` accumulation by `divisor` (Euclidean): returns
/// `(quotient, remainder)` with `acc = quotient·divisor + remainder` and
/// `remainder ∈ [0, divisor)`.
///
/// This is the single definition of the fused-rescale contract used by
/// einsum, mul, square, cube and mean-of-squares. The op's output is
/// the quotient, which must lie in `i32`; the remainder is captured in the trace for the
/// prover.
pub fn rebase(acc: i64, divisor: i64) -> (i64, i32) {
    debug_assert!(divisor > 0);
    // For a power of two the Euclidean split is the arithmetic shift and the low bits.
    if divisor & (divisor - 1) == 0 {
        let shift = divisor.trailing_zeros();
        return (acc >> shift, (acc & (divisor - 1)) as i32);
    }
    (acc.div_euclid(divisor), acc.rem_euclid(divisor) as i32)
}

/// Floor division (toward negative infinity) for any divisor sign.
///
/// Matches the old crate's `nonlinearities::div`/`const_div` ("zkVM division
/// semantics"). The caller guarantees `b != 0`.
pub fn floor_div(a: i32, b: i32) -> i32 {
    debug_assert!(b != 0);
    let q = a.wrapping_div(b);
    let r = a.wrapping_rem(b);
    if r != 0 && ((r < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}

/// Floor modulus: remainder with the sign of the divisor (non-negative for
/// positive divisors). Matches the old crate's `const_rem`.
pub fn rem_floor(a: i32, m: i32) -> i32 {
    debug_assert!(m != 0);
    let r = a % m;
    if r != 0 && ((r < 0) != (m < 0)) {
        r + m
    } else {
        r
    }
}

/// Reciprocal square root of a quantized value: `⌊√⌊S³ / x⌋⌋`, with
/// non-positive inputs mapping to `0`. Bit-for-bit port of the old kernel.
pub fn rsqrt(x: i32, scale: Scale) -> i32 {
    if x <= 0 {
        return 0;
    }
    // 3·scale ≤ 45 bits, so S³ fits i64 comfortably.
    let s_cubed = 1i64 << (3 * scale.log2());
    (s_cubed / x as i64).isqrt() as i32
}

/// Upper bound (in vars) on any custom-sized lookup table, to keep
/// prefix-suffix proving cost bounded. (Ported from `common::consts`.)
pub const MAX_CUSTOM_TABLE_VARS: u32 = 16;

/// Absolute-error tolerance for [`trig_period_modulus`]'s search, in Q64.64
/// (`round(0.01 · 2^64)`).
const TRIG_PERIOD_ERROR_TOLERANCE_Q64: u128 = 184_467_440_737_095_516;

/// Search bound for [`trig_period_modulus`]; comfortably above what scales
/// 1..=15 need.
const TRIG_PERIOD_SEARCH_MAX_K: u128 = 1 << 16;

/// Fixed-point modulus `round(k · 2π · 2^scale)` for the smallest `k` whose
/// rounding error is within tolerance. Used to reduce sin/cos inputs before
/// lookup and to size their tables. Bit-for-bit port of
/// `common::consts::trig_period_modulus`.
pub const fn trig_period_modulus(scale: u32) -> u32 {
    let mut k: u128 = 1;
    loop {
        // val_q64 = k · 2π · 2^scale, in Q64.64.
        let val_q64 = 2 * k * PI_Q64 * (1u128 << scale);
        let m = (val_q64 + (1u128 << 63)) >> 64;
        let approx_q64 = m << 64;
        let err_q64 = val_q64.abs_diff(approx_q64);
        if err_q64 <= TRIG_PERIOD_ERROR_TOLERANCE_Q64 {
            assert!(m <= u32::MAX as u128, "trig period modulus overflowed u32");
            return m as u32;
        }
        assert!(
            k < TRIG_PERIOD_SEARCH_MAX_K,
            "no 2π multiple found within search bound"
        );
        k += 1;
    }
}

/// Bits to right-shift a trig remainder by before the sin/cos table lookup,
/// so the table never exceeds [`MAX_CUSTOM_TABLE_VARS`] vars. Bit-for-bit
/// port of `common::consts::trig_downscale_bits`.
pub const fn trig_downscale_bits(scale: u32) -> u32 {
    let modulus = trig_period_modulus(scale);
    let full_log_size = modulus.next_power_of_two().ilog2();
    full_log_size.saturating_sub(MAX_CUSTOM_TABLE_VARS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebase_identity() {
        for &(acc, d) in &[(100i64, 16i64), (-100, 16), (0, 16), (-1, 1 << 14)] {
            let (q, r) = rebase(acc, d);
            assert_eq!(q * d + r as i64, acc);
            assert!((0..d).contains(&(r as i64)));
        }
    }

    /// The split by a power of two is the Euclidean one, at the ends of `i64` too.
    #[test]
    fn rebase_by_a_power_of_two_is_euclidean() {
        let values = [
            0,
            1,
            -1,
            15,
            16,
            17,
            -15,
            -16,
            -17,
            i64::MAX,
            i64::MIN,
            i64::MIN + 1,
        ];
        for shift in [0u32, 1, 4, 14, 30] {
            let d = 1i64 << shift;
            for &acc in &values {
                let want = (acc.div_euclid(d), acc.rem_euclid(d) as i32);
                assert_eq!(rebase(acc, d), want, "{acc} by {d}");
            }
        }
    }

    #[test]
    fn floor_div_matches_reference() {
        // Floor division for all sign combinations.
        assert_eq!(floor_div(7, 2), 3);
        assert_eq!(floor_div(-7, 2), -4);
        assert_eq!(floor_div(7, -2), -4);
        assert_eq!(floor_div(-7, -2), 3);
        assert_eq!(floor_div(6, 2), 3);
        assert_eq!(floor_div(-6, 2), -3);
        // i32::MIN / -1 would overflow a plain div; floor_div(a, 1) edge:
        assert_eq!(floor_div(i32::MIN, 1), i32::MIN);
    }

    #[test]
    fn rem_floor_matches_reference() {
        assert_eq!(rem_floor(7, 5), 2);
        assert_eq!(rem_floor(-7, 5), 3);
        assert_eq!(rem_floor(7, -5), -3);
        assert_eq!(rem_floor(-7, -5), -2);
    }

    #[test]
    fn rsqrt_matches_old_doc_example() {
        // From the old crate's rsqrt doctest at scale 7.
        let scale = Scale::new(7).unwrap();
        let inputs = [32, 128, 512, 2048, 8, 1];
        let expected = [256, 128, 64, 32, 512, 1448];
        for (x, e) in inputs.into_iter().zip(expected) {
            assert_eq!(rsqrt(x, scale), e);
        }
        assert_eq!(rsqrt(0, scale), 0);
        assert_eq!(rsqrt(-5, scale), 0);
    }

    #[test]
    fn trig_modulus_matches_old_crate() {
        // Pinned in the old crate's tests (common/src/consts/trig.rs).
        assert_eq!(trig_period_modulus(8), 3217);
    }

    #[test]
    fn narrow_edges() {
        assert_eq!(narrow_i32(i32::MAX as i64), Some(i32::MAX));
        assert_eq!(narrow_i32(i32::MIN as i64), Some(i32::MIN));
        assert_eq!(narrow_i32(i32::MAX as i64 + 1), None);
        assert_eq!(narrow_i32(i32::MIN as i64 - 1), None);
        assert_eq!(narrow_i32(42), Some(42));
    }
}
