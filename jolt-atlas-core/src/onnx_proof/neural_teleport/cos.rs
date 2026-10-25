//! Cosine lookup table for neural teleportation, sized to the downscaled remainder
//! (see [`super::trig_downscale`]); rescaled back up to `2^MODEL_SCALE` precision.

use common::consts::{MODEL_SCALE, TRIG_DOWNSCALE_BITS, TRIG_PERIOD_MODULUS};

/// Lookup table bit width for cosine teleportation, after downscaling.
///
/// The un-downscaled table size is the next power of two above the trig period
/// modulus, so all valid remainders in `[0, TRIG_PERIOD_MODULUS)` map to valid table
/// indices; downscaling then shrinks that by `TRIG_DOWNSCALE_BITS`.
pub const COS_TABLE_VARS: usize = (TRIG_PERIOD_MODULUS as usize).next_power_of_two().ilog2()
    as usize
    - TRIG_DOWNSCALE_BITS as usize;

/// Cosine lookup table implementation for neural teleportation.
#[derive(Debug, Clone, Copy, Default)]
pub struct CosTable;

impl CosTable {
    /// Returns the size of the table (2^COS_LOG_TABLE_SIZE).
    pub fn table_size() -> usize {
        1 << COS_TABLE_VARS
    }

    /// Materialize the lookup table from the deterministic integer trig
    /// table in [`atlas_trace::semantics`] (entries already carry the
    /// `2^TRIG_DOWNSCALE_BITS` rescale), extended by the same rule up to the
    /// power-of-two table size.
    pub fn materialize() -> Vec<i32> {
        let table = atlas_trace::semantics::tables::table(
            atlas_trace::semantics::tables::TableKind::Cos,
            atlas_trace::Scale::new(MODEL_SCALE as u8).expect("scale in range"),
            atlas_trace::SemVersion::V1,
        );
        assert!(table.entries.len() >= Self::table_size());
        table.entries[..Self::table_size()].to_vec()
    }
}
