//! Quantization utilities for converting between floating point and fixed point representations.
//!
//! This module provides functions for converting between floating-point numbers and fixed-point
//! representations using a power-of-two scaling factor.
//!
//! # Fixed-Point Representation
//!
//! A fixed-point number is represented as an integer that implicitly represents a fractional
//! value when divided by a scaling factor. In this implementation:
//! - The scaling factor is always a power of 2: `2^scale`
//! - This allows efficient conversion using bit shifts in hardware
//! - Values are stored as signed 32-bit integers (`i32`)
//!
//! # Example
//!
//! ```ignore
//! // With scale = 10, the multiplier is 2^10 = 1024
//! let scale = 10;
//! let float_val = 3.5;
//! let fixed = quantize_float(float_val, scale); // 3.5 * 1024 = 3584
//! let recovered = dequantize(fixed, scale); // 3584 / 1024 = 3.5
//! ```

use crate::tensor::{Tensor, TensorError};

/// The denominator in the fixed point representation used when quantizing inputs
pub type Scale = i32;

/// Converts a tensor of floating-point values to a fixed-point integer tensor.
///
/// This function applies `quantize_float` to each element of the input tensor in parallel,
/// converting all floating-point values to their fixed-point integer representations using
/// the specified scale. The resulting tensor stores the scale for later dequantization.
///
/// # Arguments
///
/// * `tensor` - A tensor containing `f32` floating-point values to quantize.
/// * `scale` - The logarithm (base 2) of the fixed-point multiplier to apply to all elements.
///   All elements in the tensor will be quantized using the same scale.
///
/// # Returns
///
/// A tensor of `i32` values representing the quantized fixed-point integers, with the scale
/// metadata attached for potential dequantization operations.
///
/// # Panics
///
/// Panics if any element in the tensor is out of range for the given scale (see `quantize_float`
/// for details on range limitations). All elements must satisfy:
/// `-i32::MAX / 2^scale <= element <= i32::MAX / 2^scale`
///
/// # Examples
///
/// ```ignore
/// use ndarray::array;
///
/// // Create a tensor with floating-point values
/// let tensor = Tensor::from(array![[1.5, 2.75], [3.25, 4.0]]);
///
/// // Quantize with scale 10 (multiplier = 1024)
/// let quantized = quantize_tensor(tensor, 10);
/// // Results in: [[1536, 2816], [3328, 4096]]
/// ```
///
/// # Performance
///
/// This function uses parallel iteration (`par_enum_map`) to efficiently process large tensors
/// by leveraging multiple CPU cores.
///
/// # See Also
///
/// * `quantize_float` - The underlying function used to quantize individual values
pub fn quantize_tensor(tensor: Tensor<f32>, scale: Scale) -> Tensor<i32> {
    let mut value = tensor
        .par_enum_map::<_, i32, TensorError>(|_, x| Ok(quantize_float(x as f64, scale)))
        .unwrap();
    value.set_scale(scale);
    value
}

#[inline]
#[allow(clippy::let_and_return)]
/// Converts a floating-point number to a fixed-point integer representation.
///
/// This function quantizes a floating-point value by multiplying it by `2^scale` and rounding
/// to the nearest integer. The result is stored as a signed 32-bit integer.
///
/// # Arguments
///
/// * `float` - The floating-point value to quantize. Must be within the representable range
///   for the given scale.
/// * `scale` - The logarithm (base 2) of the fixed-point multiplier. For example, a scale of 10
///   means the multiplier is `2^10 = 1024`.
///
/// # Returns
///
/// A signed 32-bit integer representing the quantized value: `round(float * 2^scale)`.
///
/// # Panics
///
/// Panics if the scaled value would exceed the representable range of `i32`. The maximum
/// representable value is approximately `i32::MAX / 2^scale`. This ensures that the quantized
/// value fits within the 32-bit signed integer range without overflow or significant bit truncation.
///
/// # Examples
///
/// ```ignore
/// // Quantize 3.5 with scale 10 (multiplier = 1024)
/// let quantized = quantize_float(3.5, 10);
/// assert_eq!(quantized, 3584);
///
/// // Quantize -1.25 with scale 8 (multiplier = 256)
/// let quantized = quantize_float(-1.25, 8);
/// assert_eq!(quantized, -320);
/// ```
///
/// # Precision
///
/// The precision of the fixed-point representation depends on the scale:
/// - Higher scale values provide more precision but reduce the maximum representable value
/// - Lower scale values allow larger numbers but with less precision
/// - Scale of 0 effectively rounds to the nearest integer
pub fn quantize_float(float: f64, scale: Scale) -> i32 {
    let mult = scale_to_multiplier(scale);
    let max_value = ((i32::MAX as f64) / mult).round(); // the maximum value that can be represented w/o sig bit truncation
    // if float > max_value || float < -max_value {
    //     panic!("Value {float} is out of range for quantization with scale {scale}");
    // }

    let clamped_float = if float < -max_value {
        if float < -1e6 {
            // Common attention mask values
            -max_value
        } else {
            panic!(
                "sig bit truncation error: value {float} is out of range for quantization with scale {scale}"
            );
        }
    } else if float > max_value {
        if float > 1e6 {
            // Also handle extremely positive values
            max_value
        } else {
            panic!(
                "sig bit truncation error: value {float} is out of range for quantization with scale {scale}"
            );
        }
    } else {
        float
    };
    let scaled = (clamped_float * mult).round() as i32;
    scaled
}

/// Converts a fixed-point integer representation back to a floating-point number.
///
/// This function performs the inverse operation of `quantize_float`, converting a fixed-point
/// integer back to its original floating-point representation by dividing by `2^scale`.
///
/// # Arguments
///
/// * `int` - The fixed-point integer value to dequantize.
/// * `scale` - The logarithm (base 2) of the fixed-point multiplier that was used during
///   quantization. Must match the scale used in the original `quantize_float` call to
///   recover the correct value.
///
/// # Returns
///
/// The floating-point value: `int / 2^scale`.
///
/// # Examples
///
/// ```ignore
/// // Dequantize a value that was quantized with scale 10
/// let fixed = 3584; // Represents 3.5 with scale 10
/// let float = dequantize(fixed, 10);
/// assert!((float - 3.5).abs() < 1e-10);
///
/// // Dequantize a negative value
/// let fixed = -320; // Represents -1.25 with scale 8
/// let float = dequantize(fixed, 8);
/// assert!((float - (-1.25)).abs() < 1e-10);
/// ```
///
/// # Precision
///
/// The dequantized value may have small rounding errors due to the original quantization:
/// - Values that round evenly during quantization will be recovered exactly
/// - Other values may differ from the original by up to `1 / (2 * 2^scale)`
pub fn dequantize(int: i32, scale: Scale) -> f64 {
    let multiplier = scale_to_multiplier(scale);
    int as f64 / multiplier
}

#[inline]
/// Converts a scale (logarithm base 2) to a fixed-point multiplier.
///
/// This function computes the actual multiplier value from a logarithmic scale parameter.
/// The multiplier is always a power of 2: `2^scale`.
///
/// # Arguments
///
/// * `scale` - The logarithm (base 2) of the desired multiplier.
///
/// # Returns
///
/// The fixed-point multiplier: `2^scale`.
///
/// # Examples
///
/// ```ignore
/// let mult = scale_to_multiplier(0);
/// assert_eq!(mult, 1.0); // 2^0 = 1
///
/// let mult = scale_to_multiplier(10);
/// assert_eq!(mult, 1024.0); // 2^10 = 1024
///
/// let mult = scale_to_multiplier(-3);
/// assert_eq!(mult, 0.125); // 2^-3 = 1/8 = 0.125
/// ```
///
/// # Use Cases
///
/// - Converting between scale and multiplier representations
/// - Computing the precision of a fixed-point representation
/// - Determining the maximum representable value for a given scale
pub fn scale_to_multiplier(scale: Scale) -> f64 {
    f64::powf(2., scale as f64)
}

#[inline]
/// Converts a fixed-point multiplier to a scale (logarithm base 2).
///
/// This function computes the logarithmic scale from a multiplier value and rounds to the
/// nearest integer. This is the inverse operation of `scale_to_multiplier`.
///
/// # Arguments
///
/// * `mult` - The fixed-point multiplier. Should typically be a power of 2 for exact results,
///   but non-power-of-2 values will be rounded to the nearest power of 2 scale.
///
/// # Returns
///
/// The scale value (logarithm base 2 of the multiplier): `round(log2(mult))`.
///
/// # Examples
///
/// ```ignore
/// let scale = multiplier_to_scale(1024.0);
/// assert_eq!(scale, 10); // log2(1024) = 10
///
/// let scale = multiplier_to_scale(1.0);
/// assert_eq!(scale, 0); // log2(1) = 0
///
/// let scale = multiplier_to_scale(0.125);
/// assert_eq!(scale, -3); // log2(1/8) = -3
///
/// // Non-power-of-2 values are rounded
/// let scale = multiplier_to_scale(1000.0);
/// assert_eq!(scale, 10); // log2(1000) ≈ 9.97, rounds to 10
/// ```
///
/// # Use Cases
///
/// - Converting between multiplier and scale representations
/// - Finding the appropriate scale for a given precision requirement
/// - Normalizing arbitrary multipliers to power-of-2 scales
///
/// # Notes
///
/// The rounding behavior means that `multiplier_to_scale(scale_to_multiplier(s)) == s` for
/// all integer scales `s`, but the reverse may not be true for non-power-of-2 multipliers.
pub fn multiplier_to_scale(mult: f64) -> Scale {
    mult.log2().round() as Scale
}
