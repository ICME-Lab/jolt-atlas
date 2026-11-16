use super::general::MODEL_SCALE;

/// log2 of the softmax tables' length at [`MODEL_SCALE`], and so the bound of softmax's clamp
/// table: the address width `atlas_trace::semantics::softmax::softmax_tables` derives, which the
/// verifier checks against this constant. Derived at compile time by [`softmax_clamp_bound`].
pub const SOFTMAX_CLAMP_BOUND: usize = softmax_clamp_bound(MODEL_SCALE as u32);

/// Compile-time derivation of [`SOFTMAX_CLAMP_BOUND`] for an arbitrary `scale`: the power of two
/// covering `⌈S·ln(2S)⌉` with a margin. At [`MODEL_SCALE`] it equals the softmax tables' own
/// width; the verifier rejects a mismatch, so a scale where the two differ is not provable.
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
}
