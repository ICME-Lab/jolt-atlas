//! Erf (Error Function) activation table implementation.
//!
//! The error function is commonly used in GELU activations:
//! GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))

use super::{ActivationTable, ActivationType, usize_to_n_bits};
use onnx_tracer::tensor::Tensor;

/// Error function lookup table implementation.
///
/// The erf function maps inputs to the range [-1, 1]:
/// - erf(0) = 0
/// - erf(∞) = 1
/// - erf(-∞) = -1
/// - erf(-x) = -erf(x) (odd function)
#[derive(Debug, Clone, Copy, Default)]
pub struct ErfTable;

impl ErfTable {
    /// Default scale factor for erf output (maps [-1, 1] to [-128, 128])
    pub const DEFAULT_SCALE: f64 = 128.0;
}

impl ActivationTable for ErfTable {
    fn name(&self) -> &'static str {
        "erf"
    }

    fn activation_type(&self) -> ActivationType {
        ActivationType::Erf
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
        let result = onnx_tracer::tensor::ops::nonlinearities::erffunc(&indices_tensor, scale);
        result.data().to_vec()
    }
}
