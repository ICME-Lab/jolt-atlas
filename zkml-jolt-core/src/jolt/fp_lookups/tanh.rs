//! Tanh (Hyperbolic Tangent) activation table implementation.
//!
//! The tanh function is a common activation function that maps inputs to [-1, 1]:
//! tanh(x) = (e^x - e^-x) / (e^x + e^-x)

use super::{ActivationTable, ActivationType, usize_to_n_bits};
use onnx_tracer::tensor::Tensor;
use serde::{Deserialize, Serialize};

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
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct TanhTable;

impl TanhTable {
    /// Default scale factor for tanh output (maps [-1, 1] to [-128, 128])
    pub const DEFAULT_SCALE: f64 = 128.0;
}

impl ActivationTable for TanhTable {
    fn name(&self) -> &'static str {
        "tanh"
    }

    fn activation_type(&self) -> ActivationType {
        ActivationType::Tanh
    }

    fn default_scale(&self) -> f64 {
        Self::DEFAULT_SCALE
    }

    fn materialize(&self, log_table_size: usize, scale: f64) -> Vec<i32> {
        let table_size = 1 << log_table_size;
        let indices: Vec<i32> = (0..table_size)
            .map(|i| usize_to_n_bits(i, log_table_size))
            .collect();
        let indices_tensor = Tensor::new(Some(&indices), &[1, table_size]).unwrap();
        let result = onnx_tracer::tensor::ops::nonlinearities::tanh(&indices_tensor, scale);
        result.data().to_vec()
    }
}
