//! Shared utilities for neural teleportation operations.
//!
//! This module centralizes helper functions used by activation ops, and periodic functions.

use super::usize_to_n_bits;

/// Build a signed two's-complement lookup table for a unary activation.
///
/// The table domain is `[-2^(n-1), 2^(n-1))` encoded in two's complement, and
/// outputs are quantized at the model scale `2^scale`. Entries come from the
/// deterministic integer tables in [`atlas_trace::semantics`] — the same
/// functions the tracer's kernels look up, so prover tables and trace values
/// agree bit for bit by construction.
pub fn materialize_signed_activation_table(
    log_table_size: usize,
    tau: i32,
    scale: u8,
    kind: atlas_trace::semantics::tables::TableKind,
) -> Vec<i32> {
    assert_eq!(tau, 1, "scaled activation tables are materialized per-op");
    let table = atlas_trace::semantics::tables::table(
        kind,
        atlas_trace::Scale::new(scale).expect("scale in range"),
        atlas_trace::SemVersion::V1,
    );
    let table_size = 1usize << log_table_size;
    (0..table_size)
        .map(|i| table.lookup(usize_to_n_bits(i, log_table_size)))
        .collect()
}
