//! Softmax witness view consumed by the softmax sumchecks, assembled from the
//! trace-captured [`atlas_trace::SoftmaxWitness`] — nothing is re-executed.

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

/// Decomposed-exp witness fields.
#[derive(Debug, Clone)]
pub struct DecomposedExpWitness {
    /// The exp sub-tables.
    pub lut: ExpLutDecomposed,
    /// High digits of the saturated centered logits.
    pub z_hi: Vec<i32>,
    /// Low digits.
    pub z_lo: Vec<i32>,
    /// `LUT_hi[z_hi]`.
    pub exp_hi: Vec<i32>,
    /// `LUT_lo[z_lo]`.
    pub exp_lo: Vec<i32>,
    /// `exp_hi·exp_lo − exp_q·S ∈ [0, S)`.
    pub r_exp: Vec<i32>,
}

/// Full softmax witness as the sumchecks consume it (flat fields plus the
/// materialized exp sub-tables).
#[derive(Debug, Clone)]
#[allow(non_snake_case)]
pub struct SoftmaxLastAxisTrace {
    /// Multiplier `S = 2^scale`.
    pub scale: i64,
    /// Input logits, flat.
    pub x: Vec<i32>,
    /// Per-row max.
    pub max_k: Vec<i32>,
    /// Per-row argmax.
    pub argmax_k: Vec<usize>,
    /// LUT exponentials.
    pub exp_q: Vec<i32>,
    /// Per-row exp sums.
    pub exp_sum_q: Vec<i32>,
    /// `⌊S²/exp_sum_q⌋` per row.
    pub inv_sum: Vec<i32>,
    /// `exp_q·inv_sum − softmax_q·S ∈ [0, S)`.
    pub R: Vec<i32>,
    /// Decomposed exp sub-witness.
    pub decomposed_exp: DecomposedExpWitness,
}

impl SoftmaxLastAxisTrace {
    /// Build the sumcheck view from a trace-captured witness.
    pub fn from_witness(w: &atlas_trace::SoftmaxWitness) -> Self {
        SoftmaxLastAxisTrace {
            scale: w.scale,
            x: w.x.clone(),
            max_k: w.max_k.clone(),
            argmax_k: w.argmax_k.clone(),
            exp_q: w.exp_q.clone(),
            exp_sum_q: w.exp_sum_q.clone(),
            inv_sum: w.inv_sum.clone(),
            R: w.r.clone(),
            decomposed_exp: DecomposedExpWitness {
                lut: generate_exp_lut_decomposed(w.scale as i32),
                z_hi: w.z_hi.clone(),
                z_lo: w.z_lo.clone(),
                exp_hi: w.exp_hi.clone(),
                exp_lo: w.exp_lo.clone(),
                r_exp: w.r_exp.clone(),
            },
        }
    }
}

/// `z[k,j] = max_k[k] - x[k,j]` over flat rows of width `last_dim`.
pub fn softmax_z(x: &[i32], max_k: &[i32], last_dim: usize) -> Vec<i32> {
    x.iter()
        .enumerate()
        .map(|(idx, &xi)| max_k[idx / last_dim] - xi)
        .collect()
}
