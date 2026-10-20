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

/// `C(scale) = ⌈(scale+1)·ln2⌉ + 1`: magnitude (in units of `2^scale`) of
/// the sentinel that extreme-negative mask constants clamp to. It stays
/// above the softmax exp cutoff, so masked positions still saturate to
/// weight zero bit-exactly.
pub fn mask_sentinel_magnitude(scale: Scale) -> f64 {
    ((scale.log2() as f64 + 1.0) * std::f64::consts::LN_2).ceil() + 1.0
}

/// Quantize one float: `round(x · 2^scale)` with the old crate's special
/// cases, pinned by tests below.
pub fn quantize_float(x: f64, scale: Scale, tensor_name: &str) -> Result<i32, ImportError> {
    let mult = scale.denom_i64() as f64;
    let max_value = ((i32::MAX as f64) / mult).round();

    let clamped = if x < -max_value {
        if x < -1e6 {
            // Extreme values (e.g. the -3.4e38 attention mask in GPT-2)
            // clamp to the scale-indexed sentinel.
            -mask_sentinel_magnitude(scale)
        } else {
            return Err(ImportError::QuantRange {
                tensor: tensor_name.to_string(),
                value: x as f32,
            });
        }
    } else if x > max_value {
        if x > 1e6 {
            max_value / 2.0
        } else {
            return Err(ImportError::QuantRange {
                tensor: tensor_name.to_string(),
                value: x as f32,
            });
        }
    } else {
        x
    };
    let scaled = (clamped * mult).round() as i32;

    // Flush-to-±1: constants that would quantize to zero keep their sign
    // (load-bearing for LayerNorm eps, which rounds to 0 at scale 14
    // otherwise). Constants only; a warning-worthy wart inherited from the
    // old crate, to be retired when eps folds into Rsqrt.
    Ok(if scaled == 0 && x != 0.0 {
        if x > 0.0 { 1 } else { -1 }
    } else {
        scaled
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
        // > 1e6 saturates to max_value/2.
        let v = quantize_float(3.4e38, s(14), "t").unwrap();
        assert_eq!(
            v,
            ((i32::MAX as f64 / 16384.0).round() / 2.0 * 16384.0) as i32
        );
        // In (max_value, 1e6]: hard error.
        assert!(quantize_float(500_000.0, s(14), "t").is_err());
        assert!(quantize_float(-500_000.0, s(14), "t").is_err());
    }

    #[test]
    fn flush_to_one() {
        assert_eq!(quantize_float(1e-5, s(14), "t").unwrap(), 1); // LayerNorm eps
        assert_eq!(quantize_float(-1e-9, s(14), "t").unwrap(), -1);
        assert_eq!(quantize_float(0.0, s(14), "t").unwrap(), 0);
    }
}
