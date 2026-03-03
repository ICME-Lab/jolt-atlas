//! Erf (Error Function) activation lookup table.
//!
//! The error function maps inputs to [-1, 1]:
//! erf(x) = (2 / sqrt(pi)) * integral(exp(-t^2), t=0..x)

use super::{usize_to_n_bits, SCALE};
use onnx_tracer::tensor::Tensor;

/// Error function lookup table implementation.
///
/// The erf function maps inputs to the range [-1, 1]:
/// - erf(0) = 0
/// - erf(∞) = 1
/// - erf(-∞) = -1
/// - erf(-x) = -erf(x) (odd function)
#[derive(Debug, Clone, Copy, Default)]
pub struct ErfTable {
    log_table_size: usize,
}

impl ErfTable {
    /// Create a new ErfTable with the specified bit width.
    pub fn new(log_table_size: usize) -> Self {
        Self { log_table_size }
    }

    /// Returns the size of the table (2^log_table_size).
    pub fn table_size(&self) -> usize {
        1 << self.log_table_size
    }

    /// Returns the log2 of the table size.
    pub fn log_table_size(&self) -> usize {
        self.log_table_size
    }

    /// Materialize the lookup table with erf values.
    ///
    /// Creates a lookup table where indices represent signed integers in two's complement,
    /// and values are the result of erf applied to those integers, scaled by SCALE.
    pub fn materialize(&self) -> Vec<i32> {
        let table_size = self.table_size();
        let indices: Vec<i32> = (0..table_size)
            .map(|i| usize_to_n_bits(i, self.log_table_size))
            .collect();
        let indices_tensor = Tensor::new(Some(&indices), &[1, table_size])
            .expect("failed to build erf LUT input tensor");
        let result =
            onnx_tracer::tensor::ops::nonlinearities::erffunc(&indices_tensor, SCALE);
        result.data().to_vec()
    }
}
