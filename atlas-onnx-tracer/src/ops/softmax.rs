use crate::{
    ops::{Op, SoftmaxAxes, SoftmaxLastAxis},
    tensor::Tensor,
};

impl Op for SoftmaxAxes {
    #[tracing::instrument(name = "SoftmaxAxes::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        crate::tensor::ops::nonlinearities::softmax_axes(inputs[0], self.scale.into(), &[self.axes])
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}

impl Op for SoftmaxLastAxis {
    #[tracing::instrument(name = "SoftmaxLastAxis::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        softmax_last_axis_decomposed(inputs[0], self.scale).0
        // softmax_last_axis(inputs[0], self.scale).0
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}

/// Full witness trace produced by [`softmax_last_axis`].
///
/// Stores per-slice and per-element intermediate values used by the proof
/// pipeline (argmax indicator, centered logits, LUT exponentials, reciprocal
/// normalization terms, and final quantized softmax outputs).
#[derive(Debug, Clone)]
pub struct SoftmaxLastAxisTrace {
    /// Scale S (= 2^scale)
    pub scale: i64,
    /// Number of feature vectors (product of all dims except last)
    pub num_slices: usize,
    /// Size of each feature vector (last dim)
    pub last_dim: usize,
    /// X[k,j] — input logits, flat [F*N]
    pub x: Vec<i32>,
    /// max_k — max of feature vector k, [F]
    pub max_k: Vec<i32>,
    /// e_k[j] — one-hot indicator at argmax_k, flat [F*N] (1 at argmax, 0 elsewhere)
    pub e_k: Vec<i32>,
    /// z[k,j] — max_k − X[k,j] (≥ 0), flat [F*N]
    pub z: Vec<i32>,
    /// exp_q[k,j] — LUT[z[k,j]], flat [F*N]
    pub exp_q: Vec<i32>,
    /// exp_sum_q[k] — Σ_j exp_q[k,j], [F]
    pub exp_sum_q: Vec<i32>,
    /// inv_sum[k] — ⌊S² / exp_sum_q[k]⌋, [F]
    pub inv_sum: Vec<i32>,
    /// r_inv[k] — S² − inv_sum[k]·exp_sum_q[k], [F]  ∈ [0, exp_sum_q[k])
    pub r_inv: Vec<i32>,
    /// softmax_q[k,j] — ⌊exp_q[k,j] · inv_sum[k] / S⌋, flat [F*N]
    pub softmax_q: Vec<i32>,
    /// R[k,j] — exp_q[k,j]·inv_sum[k] − softmax_q[k,j]·S, flat [F*N]  ∈ [0, S)
    pub R: Vec<i32>,
    /// The exp LUT used (for the prover's Shout stage)
    pub exp_lut: Vec<i32>,
    /// Decomposed exp witness (filled when using sub-table decomposition)
    pub decomposed_exp: DecomposedExpWitness,
}

/// Per-element witness data from the decomposed exp lookup.
///
/// For the proof pipeline: two Shout lookups (tiny tables) +
/// multiplication relation (exp_hi · exp_lo = exp_q · S + r_exp) +
/// range checks (z_lo ∈ [0,B), r_exp ∈ [0,S)) +
/// digit reconstruction (z = z_hi · B + z_lo).
#[derive(Debug, Clone)]
pub struct DecomposedExpWitness {
    /// The sub-tables used
    pub lut: ExpLutDecomposed,
    /// z_hi[k,j] = z[k,j] >> log2(B) — high digit of centered logit
    pub z_hi: Vec<i32>,
    /// z_lo[k,j] = z[k,j] & (B-1) — low digit of centered logit
    pub z_lo: Vec<i32>,
    /// exp_hi[k,j] = LUT_hi[z_hi[k,j]]
    pub exp_hi: Vec<i32>,
    /// exp_lo[k,j] = LUT_lo[z_lo[k,j]]
    pub exp_lo: Vec<i32>,
    /// r_exp[k,j] = exp_hi·exp_lo − exp_q·S  ∈ [0, S)
    pub r_exp: Vec<i32>,
}

/// Softmax with decomposed exp sub-tables.
///
///
/// This introduces at most ±2 per-entry error vs the flat LUT (from sub-table
/// rounding), which is negligible relative to overall quantization noise.
#[allow(clippy::needless_range_loop)]
#[tracing::instrument(name = "softmax_last_axis_decomposed", skip_all)]
pub fn softmax_last_axis_decomposed(
    a: &Tensor<i32>,
    scale: i32,
) -> (Tensor<i32>, SoftmaxLastAxisTrace) {
    let dims = a.dims();
    let last_dim = *dims.last().unwrap();
    let num_slices: usize = dims.iter().product::<usize>() / last_dim;
    let data = a.data();
    debug_assert!(
        scale.trailing_zeros() <= 15,
        "scale={scale} exceeds 2^15; i32 intermediates would overflow"
    );
    let s = scale;
    let s_sq = s * s;
    let total = num_slices * last_dim;

    let decomp = generate_exp_lut_decomposed(scale);

    // Pre-allocate all witness vectors.
    let mut max_k = Vec::with_capacity(num_slices);
    let mut e_k = vec![0i32; total];
    let mut z = vec![0i32; total];
    let mut exp_q = vec![0i32; total];
    let mut exp_sum_q = Vec::with_capacity(num_slices);
    let mut inv_sum = Vec::with_capacity(num_slices);
    let mut r_inv = Vec::with_capacity(num_slices);
    let mut softmax_q = vec![0i32; total];
    #[allow(non_snake_case)]
    let mut R = vec![0i32; total];

    // Decomposed witness vectors.
    let mut w_z_hi = vec![0i32; total];
    let mut w_z_lo = vec![0i32; total];
    let mut w_exp_hi = vec![0i32; total];
    let mut w_exp_lo = vec![0i32; total];
    let mut w_r_exp = vec![0i32; total];

    for k in 0..num_slices {
        let offset = k * last_dim;
        let slice = &data[offset..offset + last_dim];

        // 1. max_k
        let mv = *slice.iter().max().unwrap();
        max_k.push(mv);

        // 2. e_k: one-hot at first occurrence of max
        let argmax = slice.iter().position(|&x| x == mv).unwrap();
        e_k[offset + argmax] = 1;

        // 3. z and exp_q via DECOMPOSED lookup
        let mut sum_exp: i32 = 0;
        for j in 0..last_dim {
            let idx = offset + j;
            z[idx] = mv - data[idx]; // ≥ 0

            // Decomposed digit split
            let zu = z[idx] as usize;
            let z_hi = (zu >> decomp.log2_base) as i32;
            let z_lo = (zu & (decomp.base - 1)) as i32;
            w_z_hi[idx] = z_hi;
            w_z_lo[idx] = z_lo;

            // Sub-table lookups
            let hi_val = if (z_hi as usize) < decomp.lut_hi.len() {
                decomp.lut_hi[z_hi as usize]
            } else {
                0
            };
            let lo_val = decomp.lut_lo[z_lo as usize];
            w_exp_hi[idx] = hi_val;
            w_exp_lo[idx] = lo_val;

            // Combine: exp_q = ⌊hi·lo / S⌋
            if hi_val == 0 {
                exp_q[idx] = 0;
                w_r_exp[idx] = 0;
            } else {
                let product = hi_val as i64 * lo_val as i64;
                exp_q[idx] = (product / s as i64) as i32;
                w_r_exp[idx] = (product - exp_q[idx] as i64 * s as i64) as i32;
                debug_assert!(
                    w_r_exp[idx] >= 0 && w_r_exp[idx] < s,
                    "r_exp out of range: {}, S={s}",
                    w_r_exp[idx]
                );
            }

            sum_exp += exp_q[idx];
        }

        // 4. exp_sum_q
        exp_sum_q.push(sum_exp);

        // 5. inv_sum and r_inv
        let is = s_sq / sum_exp;
        let ri = s_sq - is * sum_exp;
        inv_sum.push(is);
        r_inv.push(ri);
        debug_assert!(
            ri >= 0 && ri < sum_exp,
            "r_inv out of range: {ri}, sum={sum_exp}"
        );

        // 6-7. softmax_q and R
        for j in 0..last_dim {
            let idx = offset + j;
            let product = exp_q[idx] * is; // exp_q[j] · inv_sum
            let sq = product / s; // ⌊product / S⌋
            let rem = product - sq * s; // product − sq·S
            softmax_q[idx] = sq;
            R[idx] = rem;
            debug_assert!(rem >= 0 && rem < s, "R out of range: {rem}, S={s}");
        }
    }

    let mut result = Tensor::new(Some(&softmax_q), &[total]).unwrap();
    result.reshape(dims).unwrap();

    let trace = SoftmaxLastAxisTrace {
        scale: s as i64,
        num_slices,
        last_dim,
        x: data.to_vec(),
        max_k,
        e_k,
        z,
        exp_q,
        exp_sum_q,
        inv_sum,
        r_inv,
        softmax_q: softmax_q.clone(),
        R,
        exp_lut: vec![], // unused in decomposed path; sub-tables are in decomposed_exp
        decomposed_exp: DecomposedExpWitness {
            lut: decomp,
            z_hi: w_z_hi,
            z_lo: w_z_lo,
            exp_hi: w_exp_hi,
            exp_lo: w_exp_lo,
            r_exp: w_r_exp,
        },
    };

    (result, trace)
}

/// Decomposed exp lookup tables exploiting e^{a+b} = e^a · e^b.
///
/// Splits the index z into high and low digits: z = z_hi · B + z_lo,
/// then uses two small sub-tables instead of one large flat table:
///   exp_q ≈ ⌊ LUT_hi[z_hi] · LUT_lo[z_lo] / S ⌋
///
/// For S=4096 (scale=12), B=256: ~145 + 256 = 401 entries vs 65K flat.
#[derive(Debug, Clone)]
pub struct ExpLutDecomposed {
    /// LUT_hi[h] = round(exp(-h·B / S) · S),  h ∈ [0, hi_size)
    pub lut_hi: Vec<i32>,
    /// LUT_lo[l] = round(exp(-l / S) · S),  l ∈ [0, B)
    pub lut_lo: Vec<i32>,
    /// Digit base B (power of two): z_hi = z >> log2_base, z_lo = z & (B-1)
    pub base: usize,
    /// log2(base) — for bit-shift decomposition
    pub log2_base: u32,
}

/// Generate decomposed exp sub-tables for the given scale.
///
/// The base B is chosen as the power-of-two closest to √(active_range)
/// to minimize total sub-table entries.
pub fn generate_exp_lut_decomposed(scale: i32) -> ExpLutDecomposed {
    let sf = scale as f64;
    // Same cutoff as flat LUT: exp(-i/S)*S < 0.5
    let needed = (sf * (2.0 * sf).ln()).ceil() as usize + 2;

    // Pick B ≈ √needed, rounded up to next power-of-two
    let log2_b = ((needed as f64).log2() / 2.0).ceil() as u32;
    let base = 1usize << log2_b;

    // LUT_hi: indexed by z_hi = z / B
    let hi_size = needed / base + 2;
    let mut lut_hi = Vec::with_capacity(hi_size);
    for h in 0..hi_size {
        let val = (sf * (-(h as f64 * base as f64) / sf).exp()).round();
        lut_hi.push(val.max(0.0) as i32);
    }

    // LUT_lo: indexed by z_lo = z % B, l ∈ [0, B)
    let mut lut_lo = Vec::with_capacity(base);
    for l in 0..base {
        let val = (sf * (-(l as f64) / sf).exp()).round();
        lut_lo.push(val.max(0.0) as i32);
    }

    ExpLutDecomposed {
        lut_hi,
        lut_lo,
        base,
        log2_base: log2_b,
    }
}

/// Decomposed exp lookup: exp_q = ⌊ LUT_hi[z >> log2B] · LUT_lo[z & (B-1)] / S ⌋.
///
/// `z` must be ≥ 0 (the centered value: max − x).
#[inline]
pub fn exp_lut_lookup_decomposed(z: i32, decomp: &ExpLutDecomposed, scale: i32) -> i32 {
    debug_assert!(z >= 0, "exp_lut_lookup_decomposed requires z >= 0, got {z}");
    let zu = z as usize;
    let z_hi = zu >> decomp.log2_base;
    let z_lo = zu & (decomp.base - 1);

    let hi_val = if z_hi < decomp.lut_hi.len() {
        decomp.lut_hi[z_hi]
    } else {
        0
    };
    if hi_val == 0 {
        return 0;
    }
    let lo_val = decomp.lut_lo[z_lo];

    (hi_val as i64 * lo_val as i64 / scale as i64) as i32
}

#[allow(clippy::needless_range_loop)]
/// Generate an **exp** lookup table for an arbitrary fixed-point scale `S`.
///
/// Entry `i` = `round(exp(-i / S) * S)` for `i ∈ [0, table_size)`.
/// Beyond the table, `exp(-i/S)*S` rounds to 0 so we clamp to 0.
///
/// The table is used by `softmax_last_axis`: after centering (subtracting
/// max), every element is ≤ 0, so the index is `−centered ≥ 0`.
pub fn generate_exp_lut(scale: i32) -> Vec<i32> {
    // exp(-i/S)*S < 0.5 when i > S * ln(2S).
    // Add a generous margin so the caller never needs to bounds-check.
    let sf = scale as f64;
    let needed = (sf * (2.0 * sf).ln()).ceil() as usize + 2;
    // Round up to next power-of-two for tidy table sizes.
    let table_size = needed.next_power_of_two();

    let mut lut = vec![0i32; table_size];
    for i in 0..table_size {
        let val = (sf * (-(i as f64) / sf).exp()).round();
        lut[i] = val.max(0.0) as i32;
    }
    lut
}

/// Pure-integer exp lookup: `exp(z_q / S) * S` for `z_q ≤ 0`.
#[inline]
pub fn exp_lut_lookup(z_q: i32, lut: &[i32]) -> i32 {
    debug_assert!(z_q <= 0, "exp_lut_lookup requires z_q <= 0, got {z_q}");
    let idx = (-z_q) as usize;
    if idx < lut.len() { lut[idx] } else { 0 }
}

// /// Softmax along the last axis using i32 intermediate precision,
// /// collecting all witness quantities needed by the ZK proof pipeline.
// ///
// /// For each 1-D slice along the last dimension:
// ///   1. Find max (exact integer, no precision loss)
// ///   2. Center: z[j] = max − X[j]  (≥ 0)
// ///   3. Exp via LUT: exp_q[j] = LUT[z[j]]
// ///   4. Sum: exp_sum_q = Σ_j exp_q[j]
// ///   5. Reciprocal: inv_sum = ⌊S² / exp_sum_q⌋
// ///   6. Normalize: softmax_q[j] = ⌊exp_q[j] · inv_sum / S⌋
// ///   7. R: R[j] = exp_q[j]·inv_sum − softmax_q[j]·S
// #[tracing::instrument(name = "tensor::ops::nonlinearities::softmax_last_axis", skip_all)]
// pub fn softmax_last_axis(a: &Tensor<i32>, scale: i32) -> (Tensor<i32>, SoftmaxLastAxisTrace) {
//     let dims = a.dims();
//     let last_dim = *dims.last().unwrap();
//     let num_slices: usize = dims.iter().product::<usize>() / last_dim;
//     let data = a.data();
//     debug_assert!(
//         scale.trailing_zeros() <= 15,
//         "scale={scale} exceeds 2^15; i32 intermediates would overflow (S²>i32::MAX)"
//     );
//     let s = scale;
//     let s_sq = s * s;
//     let total = num_slices * last_dim;

//     let exp_lut = generate_exp_lut(scale);

//     // Pre-allocate all witness vectors.
//     let mut max_k = Vec::with_capacity(num_slices);
//     let mut e_k = vec![0i32; total];
//     let mut z = vec![0i32; total];
//     let mut exp_q = vec![0i32; total];
//     let mut exp_sum_q = Vec::with_capacity(num_slices);
//     let mut inv_sum = Vec::with_capacity(num_slices);
//     let mut r_inv = Vec::with_capacity(num_slices);
//     let mut softmax_q = vec![0i32; total];
//     let mut R = vec![0i32; total];

//     for k in 0..num_slices {
//         let offset = k * last_dim;
//         let slice = &data[offset..offset + last_dim];

//         // 1. max_k
//         let mv = *slice.iter().max().unwrap();
//         max_k.push(mv);

//         // 2. e_k: one-hot at first occurrence of max
//         let argmax = slice.iter().position(|&x| x == mv).unwrap();
//         e_k[offset + argmax] = 1;

//         // 3. z and exp_q
//         let mut sum_exp: i32 = 0;
//         for j in 0..last_dim {
//             let idx = offset + j;
//             z[idx] = mv - data[idx]; // ≥ 0
//             exp_q[idx] = exp_lut_lookup(-(z[idx]), &exp_lut);
//             sum_exp += exp_q[idx];
//         }

//         // 4. exp_sum_q
//         exp_sum_q.push(sum_exp);

//         // 5. inv_sum and r_inv
//         let is = s_sq / sum_exp;
//         let ri = s_sq - is * sum_exp;
//         inv_sum.push(is);
//         r_inv.push(ri);
//         debug_assert!(
//             ri >= 0 && ri < sum_exp,
//             "r_inv out of range: {ri}, sum={sum_exp}"
//         );

//         // 6-7. softmax_q and R
//         for j in 0..last_dim {
//             let idx = offset + j;
//             let product = exp_q[idx] * is; // exp_q[j] · inv_sum
//             let sq = product / s; // ⌊product / S⌋
//             let rem = product - sq * s; // product − sq·S
//             softmax_q[idx] = sq;
//             R[idx] = rem;
//             debug_assert!(rem >= 0 && rem < s, "R out of range: {rem}, S={s}");
//         }
//     }

//     let mut result = Tensor::new(Some(&softmax_q), &[total]).unwrap();
//     result.reshape(dims).unwrap();

//     let trace = SoftmaxLastAxisTrace {
//         scale: s as i64,
//         num_slices,
//         last_dim,
//         x: data.to_vec(),
//         max_k,
//         e_k,
//         z,
//         exp_q,
//         exp_sum_q,
//         inv_sum,
//         r_inv,
//         softmax_q: softmax_q.clone(),
//         R,
//         exp_lut,
//         decomposed_exp: None,
//     };

//     (result, trace)
// }
