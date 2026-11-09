//! Shared utilities for neural teleportation operations.
//!
//! This module centralizes helper functions used by activation ops, and periodic functions.

use super::usize_to_n_bits;
use atlas_algebra::utils::errors::ProofVerifyError;

/// The fixed-point scale `2^scale` of a semantics table; `Err` outside the range the semantics
/// define.
pub fn table_scale(scale: u8) -> Result<atlas_trace::Scale, ProofVerifyError> {
    atlas_trace::Scale::new(scale)
        .map_err(|e| ProofVerifyError::Malformed(format!("model: table scale: {e}")))
}

/// Build a signed two's-complement lookup table for a unary activation.
///
/// The table domain is `[-2^(n-1), 2^(n-1))` encoded in two's complement, and
/// outputs are quantized at the model scale `2^scale`. Entries come from the
/// deterministic integer tables in [`atlas_trace::semantics`] — the same
/// functions the tracer's kernels look up, so prover tables and trace values
/// agree bit for bit by construction. `Err` if `scale` is outside the semantics' range.
pub fn materialize_signed_activation_table(
    log_table_size: usize,
    tau: i32,
    scale: u8,
    kind: atlas_trace::semantics::tables::TableKind,
) -> Result<Vec<i32>, ProofVerifyError> {
    if tau != 1 {
        return Err(ProofVerifyError::Malformed(format!(
            "scaled activation tables are materialized per-op, got tau = {tau}"
        )));
    }
    let table = atlas_trace::semantics::tables::table(
        kind,
        table_scale(scale)?,
        atlas_trace::SemVersion::V1,
    );
    let table_size = 1usize << log_table_size;
    Ok((0..table_size)
        .map(|i| table.lookup(usize_to_n_bits(i, log_table_size)))
        .collect())
}
