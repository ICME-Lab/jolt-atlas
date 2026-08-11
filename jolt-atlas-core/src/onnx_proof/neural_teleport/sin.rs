//! Sine lookup table for neural teleportation, sized to the downscaled remainder
//! (see [`super::trig_downscale`]); rescaled back up to `2^MODEL_SCALE` precision.

use atlas_onnx_tracer::{tensor::Tensor, utils::quantize::scale_to_multiplier};
use common::consts::{MODEL_SCALE, TRIG_DOWNSCALE_BITS, TRIG_PERIOD_MODULUS};

/// Lookup table bit width for sine teleportation, after downscaling.
///
/// The un-downscaled table size is the next power of two above the trig period
/// modulus, so all valid remainders in `[0, TRIG_PERIOD_MODULUS)` map to valid table
/// indices; downscaling then shrinks that by `TRIG_DOWNSCALE_BITS`.
pub const SIN_TABLE_VARS: usize = (TRIG_PERIOD_MODULUS as usize).next_power_of_two().ilog2()
    as usize
    - TRIG_DOWNSCALE_BITS as usize;

/// Sine lookup table implementation for neural teleportation.
#[derive(Debug, Clone, Copy, Default)]
pub struct SinTable;

impl SinTable {
    /// Returns the size of the table (2^SIN_LOG_TABLE_SIZE).
    pub fn table_size() -> usize {
        1 << SIN_TABLE_VARS
    }

    /// Materialize the lookup table: `sin(index)` at the reduced scale, rescaled back
    /// up to `2^MODEL_SCALE` precision — matches `eval_trig`'s round-trip.
    pub fn materialize() -> Vec<i32> {
        let table_size = Self::table_size();
        let indices: Vec<i32> = (0..table_size).map(|i| i as i32).collect();
        let indices_tensor = Tensor::new(Some(&indices), &[1, table_size])
            .expect("failed to build sin LUT input tensor");
        let reduced_multiplier =
            scale_to_multiplier(MODEL_SCALE as i32 - TRIG_DOWNSCALE_BITS as i32);
        let result = atlas_onnx_tracer::tensor::ops::nonlinearities::sin(
            &indices_tensor,
            reduced_multiplier,
        );
        let rescale_factor = 1i32 << TRIG_DOWNSCALE_BITS;
        result.data().iter().map(|&v| v * rescale_factor).collect()
    }
}
