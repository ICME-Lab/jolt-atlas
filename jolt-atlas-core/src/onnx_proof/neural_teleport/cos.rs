//! Cosine activation lookup table for neural teleportation.
//!
//! This table stores cos(x) values for x in [0, 4π), scaled by `SCALE`.

use super::SCALE;
use atlas_onnx_tracer::{model::consts::FOUR_PI_APPROX, tensor::Tensor};

/// Fixed lookup table bit width for cosine teleportation.
///
/// The table size is the next power of two above the `4π` period approximation,
/// so all valid remainders in [0, FOUR_PI_APPROX) map to valid table indices.
pub const COS_LOG_TABLE_SIZE: usize =
    (FOUR_PI_APPROX as usize).next_power_of_two().ilog2() as usize;

/// Cosine lookup table implementation for neural teleportation.
#[derive(Debug, Clone, Copy, Default)]
pub struct CosTable;

impl CosTable {
    /// Returns the size of the table (2^COS_LOG_TABLE_SIZE).
    pub fn table_size() -> usize {
        1 << COS_LOG_TABLE_SIZE
    }

    /// Materialize the lookup table with cosine values.
    ///
    /// Creates a lookup table where indices represent teleported remainders
    /// in [0, 8π), and values are `cos(index)` scaled by `SCALE`.
    pub fn materialize() -> Vec<i32> {
        let table_size = Self::table_size();
        let indices: Vec<i32> = (0..table_size).map(|i| i as i32).collect();
        let indices_tensor = Tensor::new(Some(&indices), &[1, table_size])
            .expect("failed to build cos LUT input tensor");
        let result = atlas_onnx_tracer::tensor::ops::nonlinearities::cos(&indices_tensor, SCALE);
        result.data().to_vec()
    }
}
