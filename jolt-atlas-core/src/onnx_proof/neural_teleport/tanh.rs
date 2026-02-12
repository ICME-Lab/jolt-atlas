//! Tanh (Hyperbolic Tangent) activation lookup table.
//!
//! The tanh function is a common activation function that maps inputs to [-1, 1]:
//! tanh(x) = (e^x - e^-x) / (e^x + e^-x)

use super::{usize_to_n_bits, SCALE};
use atlas_onnx_tracer::tensor::Tensor;

/// Hyperbolic tangent lookup table implementation.
///
/// The tanh function maps inputs to the range [-1, 1]:
/// - tanh(0) = 0
/// - tanh(∞) = 1
/// - tanh(-∞) = -1
/// - tanh(-x) = -tanh(x) (odd function)
///
/// Tanh is similar to erf but has different curvature:
/// - tanh saturates faster than erf
/// - tanh(x) ≈ erf(x * sqrt(π) / 2) for small x
#[derive(Debug, Clone, Copy, Default)]
pub struct TanhTable {
    log_table_size: usize,
}

impl TanhTable {
    /// Create a new TanhTable with the specified bit width
    pub fn new(log_table_size: usize) -> Self {
        Self { log_table_size }
    }

    /// Returns the size of the table (2^log_table_size)
    pub fn table_size(&self) -> usize {
        1 << self.log_table_size
    }

    /// Returns the log2 of the table size
    pub fn log_table_size(&self) -> usize {
        self.log_table_size
    }

    /// Materialize the lookup table with tanh values.
    ///
    /// Creates a lookup table where indices represent signed integers in two's complement,
    /// and values are the result of tanh applied to those integers, scaled by SCALE.
    pub fn materialize(&self) -> Vec<i32> {
        let table_size = self.table_size();
        let indices: Vec<i32> = (0..table_size)
            .map(|i| usize_to_n_bits(i, self.log_table_size))
            .collect();
        let indices_tensor = Tensor::new(Some(&indices), &[1, table_size]).unwrap();
        let result = atlas_onnx_tracer::tensor::ops::nonlinearities::tanh(&indices_tensor, SCALE);
        result.data().to_vec()
    }
}
