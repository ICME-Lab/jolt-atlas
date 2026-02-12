//! Neural teleportation lookup tables for activation functions.
//!
//! This module contains lookup tables for activation functions (tanh, erf, etc.)
//! that are proven using small table lookups. Unlike the tables in lookup_tables/,
//! these don't use prefix-suffix decomposition and are designed for smaller
//! table sizes typical of activation functions.

pub mod division;
pub mod tanh;

/// Fixed-point scale factor: maps [-1, 1] to [-128, 128]
pub const SCALE: f64 = 128.0;

/// Converts an n-bit index to a signed integer using two's complement.
///
/// For an n-bit representation, indices in [0, 2^(n-1)) remain positive,
/// while indices in [2^(n-1), 2^n) are mapped to negative values in [-2^(n-1), 0).
///
/// # Arguments
/// * `i` - The unsigned index to convert (must be in range [0, 2^n))
/// * `n` - The bit width for the signed representation
///
/// # Returns
/// A signed i32 in the range [-2^(n-1), 2^(n-1))
///
/// # Examples
/// ```
/// use jolt_atlas_core::onnx_proof::neural_teleport::usize_to_n_bits;
///
/// // 4-bit signed integers
/// assert_eq!(usize_to_n_bits(0, 4), 0);    // 0000 -> 0
/// assert_eq!(usize_to_n_bits(7, 4), 7);    // 0111 -> 7
/// assert_eq!(usize_to_n_bits(8, 4), -8);   // 1000 -> -8
/// assert_eq!(usize_to_n_bits(15, 4), -1);  // 1111 -> -1
/// ```
pub fn usize_to_n_bits(i: usize, n: usize) -> i32 {
    if i >= 1 << (n - 1) {
        i as i32 - (1 << n)
    } else {
        i as i32
    }
}

/// Converts an n-bit signed integer to its usize index representation.
///
/// This is the inverse of `usize_to_n_bits`. It converts a signed n-bit integer
/// in two's complement representation to its corresponding unsigned index in [0, 2^n).
/// Negative values are mapped to indices in [2^(n-1), 2^n), and positive values
/// to indices in [0, 2^(n-1)).
///
/// # Arguments
/// * `i` - The signed integer to convert (must be in range [-2^(n-1), 2^(n-1)))
/// * `n` - The bit width for the representation
///
/// # Returns
/// A usize index in the range [0, 2^n)
///
/// # Examples
/// ```
/// use jolt_atlas_core::onnx_proof::neural_teleport::n_bits_to_usize;
///
/// // 4-bit signed integers
/// assert_eq!(n_bits_to_usize(0, 4), 0);    // 0 -> 0000
/// assert_eq!(n_bits_to_usize(7, 4), 7);    // 7 -> 0111
/// assert_eq!(n_bits_to_usize(-8, 4), 8);   // -8 -> 1000
/// assert_eq!(n_bits_to_usize(-1, 4), 15);  // -1 -> 1111
/// ```
pub fn n_bits_to_usize(i: i32, n: usize) -> usize {
    if i < 0 {
        (i + (1 << n)) as usize
    } else {
        i as usize
    }
}
