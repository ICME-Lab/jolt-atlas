//! Sine lookup table for neural teleportation, sized to the downscaled remainder
//! (see [`super::trig_downscale`]); rescaled back up to `2^MODEL_SCALE` precision.

use atlas_algebra::utils::errors::ProofVerifyError;
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

    /// Materialize the lookup table from the deterministic integer trig
    /// table in [`atlas_trace::semantics`] (entries already carry the
    /// `2^TRIG_DOWNSCALE_BITS` rescale), extended by the same rule up to the
    /// power-of-two table size. `Err` if the semantics define no such table.
    pub fn materialize() -> Result<Vec<i32>, ProofVerifyError> {
        let table = atlas_trace::semantics::tables::table(
            atlas_trace::semantics::tables::TableKind::Sin,
            super::utils::table_scale(MODEL_SCALE as u8)?,
            atlas_trace::SemVersion::V1,
        );
        table
            .entries
            .get(..Self::table_size())
            .map(<[i32]>::to_vec)
            .ok_or_else(|| {
                ProofVerifyError::Malformed(format!(
                    "the sin table has {} entries, fewer than its {} addresses",
                    table.entries.len(),
                    Self::table_size()
                ))
            })
    }
}
