//! Exp-LUT materialization and small helpers for the softmax sumchecks; the
//! witness itself is the trace-captured [`atlas_trace::SoftmaxWitness`].

pub use atlas_trace::semantics::tables::ExpLutDecomposed;
use atlas_trace::{Scale, SemVersion};

/// Softmax exp sub-tables for a fixed-point `multiplier` (`2^scale`) —
/// deterministic integer generation via `atlas_trace::semantics`.
pub fn generate_exp_lut_decomposed(multiplier: i32) -> ExpLutDecomposed {
    let scale = Scale::new(multiplier.trailing_zeros() as u8)
        .expect("softmax multiplier must be 2^s with s in 1..=15");
    assert!(
        multiplier.is_positive() && multiplier.count_ones() == 1,
        "softmax multiplier must be a power of two, got {multiplier}"
    );
    atlas_trace::semantics::tables::exp_lut_decomposed(scale, SemVersion::V1)
}

/// `z[k,j] = max_k[k] - x[k,j]` over flat rows of width `last_dim`.
pub fn softmax_z(x: &[i32], max_k: &[i32], last_dim: usize) -> Vec<i32> {
    x.iter()
        .enumerate()
        .map(|(idx, &xi)| max_k[idx / last_dim] - xi)
        .collect()
}
