//! Float→fixed-point weight quantization, ported from the old crate with
//! its pinned sentinel tests.
//!
//! Quantization happens once, at lowering, after all rewrites — so pattern
//! matchers see exact float constants (eps values, mask sentinels,
//! exponents), never quantized ones.
//!
//! Determinism note: `round(x_f32 as f64 · 2^s)` is exact — an f32 widens
//! exactly to f64 and the product has ≤ 24+15 significant bits — so this is
//! bit-reproducible everywhere despite using floating point.

use atlas_trace::Scale;

use crate::{
    error::ImportError,
    graph::{ConstData, ConstTensor},
};

/// Finite constants below this magnitude are treated as −∞ stand-ins
/// (attention-mask sentinels like GPT-2's -3.4e38), never as values to
/// represent.
const NEG_INFINITY_PROXY: f64 = -1e6;

/// Quantize one float: `round(x · 2^scale)`, with the old crate's pinned
/// special cases. Exactly one of four things happens:
///
/// 1. `round(x·2^s)` fits i32 → that value (sub-resolution rounds to 0);
/// 2. unrepresentable and `x < -1e6` (e.g. the -3.4e38 attention mask in
///    GPT-2) → the scale-indexed mask sentinel;
/// 3. anything else — NaN, +∞ stand-ins, or a finite value too large for
///    i32 fixed point at this scale — → a typed error.
pub fn quantize_float(x: f64, scale: Scale, tensor_name: &str) -> Result<i32, ImportError> {
    let mult = scale.denom_i64() as f64;
    let scaled = (x * mult).round();

    // Representable (NaN fails both comparisons and falls through). Values
    // below the fixed-point resolution round to 0 — the old crate flushed
    // them to ±1, inflating sub-resolution weights by orders of magnitude.
    // The one constant that genuinely needed the flush, the eps feeding
    // rsqrt in normalization, is floored explicitly by the importer's
    // `rsqrt_eps_floor` pass instead.
    if (i32::MIN as f64..=i32::MAX as f64).contains(&scaled) {
        return Ok(scaled as i32);
    }

    // −∞ stand-in: the minimal magnitude whose softmax weight is exactly
    // zero (see `mask_sentinel_magnitude`) — the fixed-point meaning of an
    // additive attention mask. Retires when the masked-softmax selector
    // lands and mask-add patterns lower to `SoftmaxLastAxis.active_len`.
    // (There is deliberately no +∞ counterpart: nothing in the zoo produces
    // one, and a huge positive constant has no principled stand-in — a
    // model that needs e.g. +inf Clip bounds should get a Clip-elision
    // pass, not a magic value.)
    if x < NEG_INFINITY_PROXY {
        return Ok((-mask_sentinel_magnitude(scale) * mult) as i32);
    }

    Err(ImportError::QuantRange {
        tensor: tensor_name.to_string(),
        value: x as f32,
    })
}

/// Quantize a whole constant tensor by its dtype: floats scale by `2^s`,
/// integers and bools stay raw.
pub fn quantize_const(t: &ConstTensor, scale: Scale, name: &str) -> Result<Vec<i32>, ImportError> {
    match &t.data {
        ConstData::F32(v) => v
            .iter()
            .map(|&x| quantize_float(x as f64, scale, name))
            .collect(),
        ConstData::I64(v) => v
            .iter()
            .map(|&x| {
                i32::try_from(x).map_err(|_| ImportError::QuantRange {
                    tensor: name.to_string(),
                    value: x as f32,
                })
            })
            .collect(),
        ConstData::Bool(v) => Ok(v.iter().map(|&b| b as i32).collect()),
    }
}

/// `C(scale) = ⌈(scale+1)·ln2⌉ + 1`: magnitude (in units of `2^scale`) of
/// the sentinel that extreme-negative mask constants clamp to. It stays
/// above the softmax exp cutoff, so masked positions still saturate to
/// weight zero bit-exactly.
pub fn mask_sentinel_magnitude(scale: Scale) -> f64 {
    ((scale.log2() as f64 + 1.0) * std::f64::consts::LN_2).ceil() + 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: u8) -> Scale {
        Scale::new(v).unwrap()
    }

    #[test]
    fn basic_rounding() {
        assert_eq!(quantize_float(3.5, s(10), "t").unwrap(), 3584);
        assert_eq!(quantize_float(-1.25, s(8), "t").unwrap(), -320);
    }

    /// Pinned from the old crate: sentinel magnitudes and the clamp value.
    #[test]
    fn sentinel_matches_old_crate() {
        assert_eq!(mask_sentinel_magnitude(s(8)), 8.0);
        assert_eq!(mask_sentinel_magnitude(s(12)), 11.0);
        assert_eq!(quantize_float(-3.4e38, s(8), "t").unwrap(), -2048); // -(8 << 8)
        assert_eq!(quantize_float(-3.4e38, s(12), "t").unwrap(), -45056); // -(11 << 12)
    }

    #[test]
    fn positive_saturation_and_errors() {
        // Unrepresentable positive values error — there is no +∞ stand-in.
        assert!(quantize_float(3.4e38, s(14), "t").is_err());
        assert!(quantize_float(500_000.0, s(14), "t").is_err());
        assert!(quantize_float(-500_000.0, s(14), "t").is_err());
    }

    /// Sub-resolution constants round to zero (the old crate flushed them to
    /// ±1; LayerNorm eps no longer needs that — see `rsqrt`'s zero handling).
    #[test]
    fn sub_resolution_rounds_to_zero() {
        assert_eq!(quantize_float(1e-5, s(14), "t").unwrap(), 0); // LayerNorm eps
        assert_eq!(quantize_float(-1e-9, s(14), "t").unwrap(), 0);
        assert_eq!(quantize_float(0.0, s(14), "t").unwrap(), 0);
        assert_eq!(quantize_float(4e-5, s(14), "t").unwrap(), 1); // ≥ half ulp
    }

    /// NaN used to slip through the range comparisons, saturate to 0 in the
    /// cast, and leave through the flush-to-±1 branch as -1. It must error.
    #[test]
    fn nan_is_rejected() {
        assert!(quantize_float(f64::NAN, s(14), "t").is_err());
    }

    /// -∞ takes the mask-sentinel path like -3.4e38; +∞ errors like any
    /// unrepresentable positive value.
    #[test]
    fn infinities_take_sentinel_paths() {
        assert_eq!(
            quantize_float(f64::NEG_INFINITY, s(14), "t").unwrap(),
            quantize_float(-3.4e38, s(14), "t").unwrap()
        );
        assert!(quantize_float(f64::INFINITY, s(14), "t").is_err());
    }

    /// The representability gate is on the exact scaled value: the largest
    /// in-range f64 quantizes to i32::MAX, and the first value whose rounding
    /// crosses 2^31 errors instead of silently saturating in the `as` cast.
    #[test]
    fn exact_boundary() {
        assert_eq!(
            quantize_float((i32::MAX as f64) / 16384.0, s(14), "t").unwrap(),
            i32::MAX
        );
        assert!(quantize_float(131_072.0, s(14), "t").is_err()); // rounds to 2^31
        assert_eq!(
            quantize_float(-131_072.0, s(14), "t").unwrap(), // -2^31 fits
            i32::MIN
        );
    }
}
