use super::general::MODEL_SCALE;

/// Fixed-point modulus `round(k * 2π * 2^MODEL_SCALE)` for the smallest `k` that keeps rounding
/// error low (see [`trig_period_search`]), used to reduce Cos/Sin inputs before lookup, and to
/// size their lookup tables.
pub const TRIG_PERIOD_MODULUS: u32 = trig_period_modulus(MODEL_SCALE as u32);

/// Guards the `u32`→`i32` cast at `TRIG_PERIOD_MODULUS`'s many tensor-arithmetic call sites:
/// fails the build if `MODEL_SCALE` is ever pushed high enough for that cast to overflow.
const _: () = assert!(
    TRIG_PERIOD_MODULUS <= i32::MAX as u32,
    "TRIG_PERIOD_MODULUS overflows i32 at this MODEL_SCALE"
);

/// Compile-time derivation of [`TRIG_PERIOD_MODULUS`] for an arbitrary `scale`.
pub const fn trig_period_modulus(scale: u32) -> u32 {
    let m = trig_period_search(scale);
    assert!(m <= u32::MAX as u128, "trig period modulus overflowed u32");
    m as u32
}

/// Upper bound (in vars) on any custom-sized lookup table this codebase builds, to keep
/// prefix-suffix proving cost bounded.
pub const MAX_CUSTOM_TABLE_VARS: usize = 16;

/// Bits to right-shift a trig remainder by before the Cos/Sin table lookup, so the table
/// never exceeds [`MAX_CUSTOM_TABLE_VARS`] vars. `0` if the natural table already fits.
pub const TRIG_DOWNSCALE_BITS: u32 = trig_downscale_bits(MODEL_SCALE as u32);

/// Compile-time derivation of [`TRIG_DOWNSCALE_BITS`] for an arbitrary `scale`.
pub const fn trig_downscale_bits(scale: u32) -> u32 {
    let modulus = trig_period_modulus(scale);
    let full_log_size = modulus.next_power_of_two().ilog2();
    full_log_size.saturating_sub(MAX_CUSTOM_TABLE_VARS as u32)
}

/// `round(π * 2^64)`: π in Q64.64 fixed point.
const PI_Q64: u128 = 57_952_155_664_616_982_739;

/// Absolute-error tolerance for [`trig_period_search`], in Q64.64 fixed point (`round(0.01 *
/// 2^64)`).
const TRIG_PERIOD_ERROR_TOLERANCE_Q64: u128 = 184_467_440_737_095_516;

/// Search bound for [`trig_period_search`]; comfortably above what scales 4..=20 need.
const TRIG_PERIOD_SEARCH_MAX_K: u128 = 1 << 16;

/// Searches multiples `k = 1, 2, 3, ...` of `2π` for the smallest whose fixed-point modulus
/// `round(k * 2π * 2^scale)` is within [`TRIG_PERIOD_ERROR_TOLERANCE_Q64`] of the true value —
/// smaller `k` means a smaller lookup table, so it's preferred when accurate enough. Returns the
/// modulus.
const fn trig_period_search(scale: u32) -> u128 {
    let mut k: u128 = 1;
    loop {
        // val_q64 = k * 2π * 2^scale, in Q64.64 fixed point.
        let val_q64 = 2 * k * PI_Q64 * (1u128 << scale);
        let m = (val_q64 + (1u128 << 63)) >> 64;
        let approx_q64 = m << 64;
        let err_q64 = val_q64.abs_diff(approx_q64);
        if err_q64 <= TRIG_PERIOD_ERROR_TOLERANCE_Q64 {
            return m;
        }
        assert!(
            k < TRIG_PERIOD_SEARCH_MAX_K,
            "no 2π multiple found within search bound"
        );
        k += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trig_period_modulus_at_scale_8() {
        assert_eq!(trig_period_modulus(8), 3217);
    }
}
