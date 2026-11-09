//! Exp-LUT materialization and small helpers for the softmax sumchecks; the
//! witness itself is the trace-captured [`atlas_trace::SoftmaxWitness`].

use atlas_algebra::utils::errors::ProofVerifyError;
pub use atlas_trace::semantics::tables::ExpLutDecomposed;
use atlas_trace::{Scale, SemVersion};

/// Softmax exp sub-tables for a fixed-point `multiplier` (`2^scale`) —
/// deterministic integer generation via `atlas_trace::semantics`. `Err` unless `multiplier` is
/// `2^s` with `s` in the semantics' scale range.
pub fn generate_exp_lut_decomposed(multiplier: i32) -> Result<ExpLutDecomposed, ProofVerifyError> {
    let unsupported = || {
        ProofVerifyError::Malformed(format!(
            "model: softmax multiplier must be 2^s with s in 1..=15, got {multiplier}"
        ))
    };
    let scale = Scale::new(multiplier.trailing_zeros() as u8).map_err(|_| unsupported())?;
    if !(multiplier.is_positive() && multiplier.count_ones() == 1) {
        return Err(unsupported());
    }
    Ok(atlas_trace::semantics::tables::exp_lut_decomposed(
        scale,
        SemVersion::V1,
    ))
}

/// `z[k,j] = max_k[k] - x[k,j]` over flat rows of width `last_dim`, one row per entry of
/// `max_k`.
pub fn softmax_z(x: &[i32], max_k: &[i32], last_dim: usize) -> Vec<i32> {
    x.chunks(last_dim)
        .zip(max_k)
        .flat_map(|(row, &max)| row.iter().map(move |&xi| max - xi))
        .collect()
}
