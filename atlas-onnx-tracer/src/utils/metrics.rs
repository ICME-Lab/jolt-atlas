//! Error metrics for quantization analysis.
//!
//! All functions take `&[f64]` slices and return `f64`.

/// Cosine similarity between two vectors: dot(a,b) / (||a|| * ||b||).
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut dot, mut na, mut nb) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        if na == 0.0 && nb == 0.0 { 1.0 } else { 0.0 }
    } else {
        dot / denom
    }
}

/// Pearson correlation coefficient (mean-centered cosine similarity).
///
/// Unlike raw cosine similarity, this removes the mean from each vector
/// before computing the angle. This is the correct "shape agreement" metric
/// for logit vectors that may have large constant offsets.
///
/// Returns a value in \[-1, 1\], where 1 means identical relative pattern.
pub fn pearson_correlation(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    if n == 0.0 {
        return 1.0;
    }
    let mean_a: f64 = a.iter().sum::<f64>() / n;
    let mean_b: f64 = b.iter().sum::<f64>() / n;
    let (mut dot, mut na, mut nb) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let dx = x - mean_a;
        let dy = y - mean_b;
        dot += dx * dy;
        na += dx * dx;
        nb += dy * dy;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 { 1.0 } else { dot / denom }
}

/// Mean squared error.
pub fn mse(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return 0.0;
    }
    let sum: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum();
    sum / a.len() as f64
}

/// Root mean squared error.
pub fn rmse(a: &[f64], b: &[f64]) -> f64 {
    mse(a, b).sqrt()
}

/// Maximum absolute error.
pub fn max_abs_error(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Mean absolute error.
pub fn mean_abs_error(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return 0.0;
    }
    let sum: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).sum();
    sum / a.len() as f64
}

/// Relative MSE: MSE(a,b) / Var(reference).
/// `reference` is the first argument, `predicted` is the second.
pub fn relative_mse(reference: &[f64], predicted: &[f64]) -> f64 {
    let m = mse(reference, predicted);
    let mean: f64 = reference.iter().sum::<f64>() / reference.len() as f64;
    let var: f64 =
        reference.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / reference.len() as f64;
    if var == 0.0 {
        if m == 0.0 { 0.0 } else { f64::INFINITY }
    } else {
        m / var
    }
}

/// KL divergence from logits: KL(softmax(a) || softmax(b)).
pub fn kl_divergence_from_logits(a_logits: &[f64], b_logits: &[f64]) -> f64 {
    assert_eq!(a_logits.len(), b_logits.len());
    let p = softmax(a_logits);
    let q = softmax(b_logits);
    let eps = 1e-30;
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            if pi > eps {
                pi * ((pi + eps).ln() - (qi + eps).ln())
            } else {
                0.0
            }
        })
        .sum()
}

/// Top-k agreement: fraction of top-k elements (by value) from `a` that also
/// appear in the top-k of `b`.
pub fn top_k_agreement(a: &[f64], b: &[f64], k: usize) -> f64 {
    assert_eq!(a.len(), b.len());
    let top_a = top_k_indices(a, k);
    let top_b = top_k_indices(b, k);
    let overlap = top_a.iter().filter(|idx| top_b.contains(idx)).count();
    overlap as f64 / k as f64
}

/// Spearman rank correlation.
pub fn spearman_rank_correlation(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    if n < 2 {
        return 1.0;
    }
    let ranks_a = ranks(a);
    let ranks_b = ranks(b);
    // Pearson correlation of ranks
    let mean_a: f64 = ranks_a.iter().sum::<f64>() / n as f64;
    let mean_b: f64 = ranks_b.iter().sum::<f64>() / n as f64;
    let (mut cov, mut va, mut vb) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&ra, &rb) in ranks_a.iter().zip(ranks_b.iter()) {
        let da = ra - mean_a;
        let db = rb - mean_b;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    let denom = (va * vb).sqrt();
    if denom == 0.0 { 1.0 } else { cov / denom }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

fn top_k_indices(v: &[f64], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f64)> = v.iter().enumerate().map(|(i, &x)| (i, x)).collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().take(k).map(|(i, _)| i).collect()
}

fn ranks(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut indexed: Vec<(usize, f64)> = v.iter().enumerate().map(|(i, &x)| (i, x)).collect();
    indexed.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0_f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && indexed[j].1 == indexed[i].1 {
            j += 1;
        }
        // Average rank for ties
        let avg_rank = (i + j + 1) as f64 / 2.0; // 1-based
        for item in indexed.iter().take(j).skip(i) {
            r[item.0] = avg_rank;
        }
        i = j;
    }
    r
}
