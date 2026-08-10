pub const XLEN: usize = 32;
pub const LOG_K_CHUNK: usize = 4;
pub const K_CHUNK: usize = 1 << LOG_K_CHUNK;

/// Logarithm of K = XLEN * 2, the total number of address bits for prefix-suffix
/// lookup tables used in read-raf checking.
pub const LOG_K: usize = XLEN * 2;

/// Global model scale (log2) the fixed-shape lookup tables below are tuned for (matches
/// `atlas-onnx-tracer`'s `DEFAULT_SCALE`). Several lookup-table shapes in this codebase
/// (activation clamping, softmax's saturating clamp) are const-generic on a bound derived from
/// this value rather than the model's runtime scale, so changing it requires recompiling.
pub const MODEL_SCALE: usize = 12;

/// Clamps Erf/Sigmoid/Tanh input to `[-8, 8)` at model scale [`MODEL_SCALE`], before the small
/// activation-table lookup.
pub const ACTIVATION_BOUND: usize = MODEL_SCALE + 3;

/// One more than [`ACTIVATION_BOUND`]; the small activation table's log2 size.
pub const ACTIVATION_TABLE_VARS: usize = ACTIVATION_BOUND + 1;

/// log2 of softmax's saturating-clamp table size at [`MODEL_SCALE`]: the padded
/// `hi_size * base` from `generate_exp_lut_decomposed` (`atlas-onnx-tracer::ops::softmax`) at
/// that scale, rounded up to the next power of two. Derived at compile time by
/// [`softmax_clamp_bound`].
pub const SOFTMAX_CLAMP_BOUND: usize = softmax_clamp_bound(MODEL_SCALE as u32);

/// Fixed-point modulus `round(k * 2π * 2^MODEL_SCALE)` for the smallest `k` that keeps rounding
/// error low (see [`trig_period_search`]), used to reduce Cos/Sin inputs before lookup, and to
/// size their lookup tables.
pub const TRIG_PERIOD_MODULUS: u32 = trig_period_modulus(MODEL_SCALE as u32);

/// Compile-time derivation of [`TRIG_PERIOD_MODULUS`] for an arbitrary `scale`.
pub const fn trig_period_modulus(scale: u32) -> u32 {
    let m = trig_period_search(scale);
    assert!(m <= u32::MAX as u128, "trig period modulus overflowed u32");
    m as u32
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

/// Compile-time derivation of [`SOFTMAX_CLAMP_BOUND`] for an arbitrary `scale`, mirroring
/// `atlas_onnx_tracer::ops::softmax::generate_exp_lut_decomposed`'s arithmetic in fixed-point
/// integers (that function needs non-const `f64::ln`, so can't run in `const` context; since its
/// multiplier `S = 2^scale` is always a power of two, `ln(2S) = (scale + 1) * ln(2)` avoids
/// needing a real logarithm at all).
pub const fn softmax_clamp_bound(scale: u32) -> usize {
    /// `round(ln(2) * 2^32)`.
    const LN2_Q32: u128 = 2_977_044_472;
    const Q32: u128 = 1 << 32;

    let s = 1u128 << scale; // S = 2^scale

    // needed = ceil(S * ln(2S)) + 2, where ln(2S) = (scale + 1) * ln(2).
    let numerator = s * (scale as u128 + 1) * LN2_Q32;
    let needed = ceil_div(numerator, Q32) + 2;

    // base = 2 ^ ceil(log2(needed) / 2): power-of-two closest to sqrt(needed).
    let log2_needed = ceil_ilog2(needed);
    let log2_base = ceil_div(log2_needed as u128, 2) as u32;
    let base = 1u128 << log2_base;

    // Table size is hi_size*base = needed floored to a multiple of base, plus a 2*base margin;
    // needed + 2*base is a safe upper bound without reconstructing hi_size.
    ceil_ilog2(needed + 2 * base) as usize
}

/// `ceil(a / b)` for `b > 0`.
const fn ceil_div(a: u128, b: u128) -> u128 {
    a.div_ceil(b)
}

/// `ceil(log2(x))` for `x >= 1`.
const fn ceil_ilog2(x: u128) -> u32 {
    if x <= 1 { 0 } else { (x - 1).ilog2() + 1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_known_values() {
        assert_eq!(softmax_clamp_bound(8), 11);
        assert_eq!(softmax_clamp_bound(12), 16);
    }

    #[test]
    fn trig_period_modulus_at_scale_8() {
        assert_eq!(trig_period_modulus(8), 3217);
    }
}
