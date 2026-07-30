use crate::{
    ops::{Op, SoftmaxLastAxis, SoftmaxLastAxisFlatExp, SoftmaxLastAxisSatClamp},
    tensor::Tensor,
    utils::quantize::scale_to_multiplier,
};

impl Op for SoftmaxLastAxis {
    #[tracing::instrument(name = "SoftmaxLastAxis::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        softmax_last_axis_decomposed(inputs[0], scale_to_multiplier(self.scale) as i32).0
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}

/// Benchmark-only sibling of [`SoftmaxLastAxis`]: identical execution/tracing semantics (same
/// decomposed trace), proven via a different proof path
/// (`jolt_atlas_core::onnx_proof::ops::softmax_last_axis_satclamp`) that replaces the
/// `sat_diff` complementary-slackness sumcheck with a `ClampBoundedTable` lookup. A distinct
/// operator type is needed only because Rust forbids implementing `OperatorProofTrait` for
/// `SoftmaxLastAxis` twice. Reachable only via `ModelBuilder`, not real ONNX op-name routing.
impl Op for SoftmaxLastAxisSatClamp {
    #[tracing::instrument(name = "SoftmaxLastAxisSatClamp::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        softmax_last_axis_decomposed_padded_bound(inputs[0], scale_to_multiplier(self.scale) as i32)
            .0
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}

/// Benchmark-only sibling of [`SoftmaxLastAxis`] using a single flat `exp_z` lookup table
/// (no digit split, no `exp_hi*exp_lo` multiplication relation, no `r_exp` remainder) instead of
/// the two-sub-table decomposition — see
/// `jolt_atlas_core::onnx_proof::ops::softmax_last_axis_flatexp`. Compares proving cost of "one
/// big table" against "two small tables + combine".
impl Op for SoftmaxLastAxisFlatExp {
    #[tracing::instrument(name = "SoftmaxLastAxisFlatExp::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        softmax_last_axis_flat(inputs[0], scale_to_multiplier(self.scale) as i32).0
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
    /// X[k,j] = input logits, flat [F*N]
    pub x: Vec<i32>,
    /// max_k = max of feature vector k, [F]
    pub max_k: Vec<i32>,
    /// argmax_k[k] = position of the max in each feature vector, [F]
    pub argmax_k: Vec<usize>,
    /// exp_q[k,j] = LUT[z[k,j]], flat [F*N]
    pub exp_q: Vec<i32>,
    /// exp_sum_q[k] = Σ_j exp_q[k,j], [F]
    pub exp_sum_q: Vec<i32>,
    /// inv_sum[k] = ⌊S² / exp_sum_q[k]⌋, [F]
    pub inv_sum: Vec<i32>,
    /// R[k,j] = exp_q[k,j]·inv_sum[k] − softmax_q[k,j]·S, flat [F*N]  ∈ [0, S)
    pub R: Vec<i32>,
    /// Decomposed exp witness (sub-table lookups, digit splits, saturation)
    pub decomposed_exp: DecomposedExpWitness,
}

/// Per-element witness data from the decomposed exp lookup.
///
/// For the proof pipeline: two Shout lookups (tiny tables) +
/// multiplication relation (exp_hi · exp_lo = exp_q · S + r_exp) +
/// range checks (z_lo ∈ [0,B), r_exp ∈ [0,S)) +
/// digit reconstruction (z_c = z_hi · B + z_lo).
#[derive(Debug, Clone)]
pub struct DecomposedExpWitness {
    /// The sub-tables used
    pub lut: ExpLutDecomposed,
    /// z_hi[k,j] = z_c[k,j] >> log2(B), high digit of centered logit
    pub z_hi: Vec<i32>,
    /// z_lo[k,j] = z_c[k,j] & (B-1), low digit of centered logit
    pub z_lo: Vec<i32>,
    /// exp_hi[k,j] = LUT_hi[z_hi[k,j]]
    pub exp_hi: Vec<i32>,
    /// exp_lo[k,j] = LUT_lo[z_lo[k,j]]
    pub exp_lo: Vec<i32>,
    /// r_exp[k,j] = exp_hi·exp_lo − exp_q·S  ∈ [0, S)
    pub r_exp: Vec<i32>,
    /// sat_diff[k,j] = z[k,j] − z_c[k,j]  (≥ 0, saturation overflow),
    /// where z_c[k,j] = min(z[k,j], z_bound − 1) is z clamped to the sub-table range.
    pub sat_diff: Vec<i32>,
}

/// Softmax with decomposed exp sub-tables.
///
/// This introduces at most ±2 per-entry error vs the flat LUT (from sub-table
/// rounding), which is negligible relative to overall quantization noise.
pub fn softmax_last_axis_decomposed(
    a: &Tensor<i32>,
    scale: i32,
) -> (Tensor<i32>, SoftmaxLastAxisTrace) {
    softmax_last_axis_decomposed_impl(a, scale, false)
}

/// Same as [`softmax_last_axis_decomposed`], but saturates `z` to the **padded** table-size
/// cutoff (`hi_size.next_power_of_two() * base - 1`) instead of the tight unpadded cutoff
/// (`hi_size * base - 1`).
///
/// Used by `SoftmaxLastAxisSatClamp`, whose saturating-clamp lookup
/// (`joltworks::lookup_tables::clamp::SoftmaxSatClampTable`, a `ClampBoundedTable`) can only
/// represent power-of-two-minus-one ceilings — the tight cutoff generally isn't of that form
/// (`hi_size` isn't a power of two). Numerically equivalent to
/// [`softmax_last_axis_decomposed`]'s `exp_q`/final softmax output: the extra range between the
/// two cutoffs is already the zero-padded tail of `lut_hi` (both regions round `exp` to 0) —
/// this only changes the internal `z_hi`/`z_lo`/`sat_diff` witness for saturated rows.
pub fn softmax_last_axis_decomposed_padded_bound(
    a: &Tensor<i32>,
    scale: i32,
) -> (Tensor<i32>, SoftmaxLastAxisTrace) {
    softmax_last_axis_decomposed_impl(a, scale, true)
}

#[allow(clippy::needless_range_loop)]
#[tracing::instrument(name = "softmax_last_axis_decomposed", skip_all)]
fn softmax_last_axis_decomposed_impl(
    a: &Tensor<i32>,
    scale: i32,
    pad_z_bound: bool,
) -> (Tensor<i32>, SoftmaxLastAxisTrace) {
    let dims = a.dims();
    let last_dim = *dims.last().unwrap();
    let num_slices: usize = dims.iter().product::<usize>() / last_dim;
    let data = a.data();
    debug_assert!(
        scale <= (1 << 15),
        "scale={scale} must be at most 2^15; i32 intermediates would overflow"
    );
    let s = scale;
    let s_sq = s * s;
    let total = num_slices * last_dim;

    let mut decomp = generate_exp_lut_decomposed(scale);
    if pad_z_bound {
        // Zero-pad lut_hi to the next power of two so z_hi can range over the full padded
        // domain without an out-of-bounds lookup (mirrors the proof side's `pad_to_power_of_two`
        // on `table_hi`).
        decomp
            .lut_hi
            .resize(decomp.lut_hi.len().next_power_of_two(), 0);
    }
    let z_bound = (decomp.lut_hi.len() * decomp.base) as i32;

    // Pre-allocate all witness vectors.
    let mut max_k = Vec::with_capacity(num_slices);
    let mut argmax_k = Vec::with_capacity(num_slices);
    let mut z = vec![0i32; total];
    let mut exp_q = vec![0i32; total];
    let mut exp_sum_q = Vec::with_capacity(num_slices);
    let mut inv_sum = Vec::with_capacity(num_slices);
    let mut softmax_q = vec![0i32; total];
    let mut R = vec![0i32; total];

    // Decomposed witness vectors.
    let mut w_z_hi = vec![0i32; total];
    let mut w_z_lo = vec![0i32; total];
    let mut w_exp_hi = vec![0i32; total];
    let mut w_exp_lo = vec![0i32; total];
    let mut w_r_exp = vec![0i32; total];
    let mut w_sat_diff = vec![0i32; total];

    for k in 0..num_slices {
        let offset = k * last_dim;
        let slice = &data[offset..offset + last_dim];

        // 1. max_k
        let mv = *slice.iter().max().unwrap();
        max_k.push(mv);

        // 2. argmax
        let argmax = slice.iter().position(|&x| x == mv).unwrap();
        argmax_k.push(argmax);

        // 3. z and exp_q via DECOMPOSED lookup
        let mut sum_exp: i32 = 0;
        for j in 0..last_dim {
            let idx = offset + j;
            z[idx] = mv - data[idx]; // ≥ 0

            // Saturate to sub-table range: z_c = min(z, z_bound - 1)
            // where z_bound = K_hi * B.  For values beyond the table,
            // exp decays to 0 anyway; clamping keeps Shout indices in range.
            let z_c = z[idx].min(z_bound - 1);
            w_sat_diff[idx] = z[idx] - z_c;

            // Decomposed digit split (on clamped value)
            let zu = z_c as usize;
            let z_hi = (zu >> decomp.log2_base) as i32;
            let z_lo = (zu & (decomp.base - 1)) as i32;
            w_z_hi[idx] = z_hi;
            w_z_lo[idx] = z_lo;

            // Sub-table lookups
            let hi_val = decomp.lut_hi[z_hi as usize];
            let lo_val = decomp.lut_lo[z_lo as usize];
            w_exp_hi[idx] = hi_val;
            w_exp_lo[idx] = lo_val;

            // Combine: exp_q = ⌊hi·lo / S⌋
            let product = hi_val as i64 * lo_val as i64;
            exp_q[idx] = (product / s as i64) as i32;
            w_r_exp[idx] = (product - exp_q[idx] as i64 * s as i64) as i32;
            debug_assert!(
                w_r_exp[idx] >= 0 && w_r_exp[idx] < s,
                "r_exp out of range: {}, S={s}",
                w_r_exp[idx]
            );

            sum_exp += exp_q[idx];
        }

        // 4. exp_sum_q
        exp_sum_q.push(sum_exp);

        // 5. inv_sum
        let is = s_sq / sum_exp;
        inv_sum.push(is);
        debug_assert!(
            {
                let ri = s_sq - is * sum_exp;
                ri >= 0 && ri < sum_exp
            },
            "r_inv out of range, sum={sum_exp}"
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
        x: data.to_vec(),
        max_k,
        argmax_k,
        exp_q,
        exp_sum_q,
        inv_sum,
        R,
        decomposed_exp: DecomposedExpWitness {
            lut: decomp,
            z_hi: w_z_hi,
            z_lo: w_z_lo,
            exp_hi: w_exp_hi,
            exp_lo: w_exp_lo,
            r_exp: w_r_exp,
            sat_diff: w_sat_diff,
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

/// Witness data for the single-flat-table exp lookup (no digit split).
#[derive(Debug, Clone)]
pub struct FlatExpWitness {
    /// `table[z_c] = round(exp(-z_c/S) * S)`, zero-padded to the next power of two.
    pub table: Vec<i32>,
    /// `z_bound - 1`, the saturation ceiling (padded table length minus one).
    pub z_bound_minus_1: i32,
}

/// Full witness trace produced by [`softmax_last_axis_flat`].
#[derive(Debug, Clone)]
pub struct SoftmaxLastAxisFlatTrace {
    /// Scale S (= 2^scale)
    pub scale: i64,
    /// X[k,j] = input logits, flat [F*N]
    pub x: Vec<i32>,
    /// max_k = max of feature vector k, [F]
    pub max_k: Vec<i32>,
    /// argmax_k[k] = position of the max in each feature vector, [F]
    pub argmax_k: Vec<usize>,
    /// z[k,j] = max_k[k] - x[k,j], flat [F*N] (≥ 0, unclamped)
    pub z: Vec<i32>,
    /// exp_q[k,j] = table[min(z[k,j], z_bound-1)], flat [F*N] — exact, no remainder
    pub exp_q: Vec<i32>,
    /// exp_sum_q[k] = Σ_j exp_q[k,j], [F]
    pub exp_sum_q: Vec<i32>,
    /// inv_sum[k] = ⌊S² / exp_sum_q[k]⌋, [F]
    pub inv_sum: Vec<i32>,
    /// R[k,j] = exp_q[k,j]·inv_sum[k] − softmax_q[k,j]·S, flat [F*N]  ∈ [0, S)
    pub R: Vec<i32>,
    /// The flat exp table + its saturation ceiling.
    pub flat_exp: FlatExpWitness,
}

/// Generates a single flat exp lookup table (no digit split): `table[z] = round(exp(-z/S)*S)`
/// for `z` in `[0, needed)`, using the same cutoff (`exp(-z/S)*S < 0.5`) as
/// [`generate_exp_lut_decomposed`]. Unpadded — callers pad to the next power of two themselves
/// (see [`softmax_last_axis_flat`]).
pub fn generate_exp_lut_flat(scale: i32) -> Vec<i32> {
    let sf = scale as f64;
    let needed = (sf * (2.0 * sf).ln()).ceil() as usize + 2;
    (0..needed)
        .map(|z| (sf * (-(z as f64) / sf).exp()).round().max(0.0) as i32)
        .collect()
}

/// Softmax via a single flat exp lookup table: `exp_q[k,j] = table[min(z[k,j], z_bound-1)]`,
/// exact (no remainder/multiplication relation needed, unlike [`softmax_last_axis_decomposed`]'s
/// digit-split `exp_hi*exp_lo/S` combination).
#[tracing::instrument(name = "softmax_last_axis_flat", skip_all)]
pub fn softmax_last_axis_flat(
    a: &Tensor<i32>,
    scale: i32,
) -> (Tensor<i32>, SoftmaxLastAxisFlatTrace) {
    let dims = a.dims();
    let last_dim = *dims.last().unwrap();
    let num_slices: usize = dims.iter().product::<usize>() / last_dim;
    let data = a.data();
    debug_assert!(
        scale <= (1 << 15),
        "scale={scale} must be at most 2^15; i32 intermediates would overflow"
    );
    let s = scale;
    let s_sq = s * s;
    let total = num_slices * last_dim;

    let mut table = generate_exp_lut_flat(scale);
    table.resize(table.len().next_power_of_two(), 0);
    let z_bound_minus_1 = (table.len() - 1) as i32;

    let mut max_k = Vec::with_capacity(num_slices);
    let mut argmax_k = Vec::with_capacity(num_slices);
    let mut z = vec![0i32; total];
    let mut exp_q = vec![0i32; total];
    let mut exp_sum_q = Vec::with_capacity(num_slices);
    let mut inv_sum = Vec::with_capacity(num_slices);
    let mut softmax_q = vec![0i32; total];
    let mut R = vec![0i32; total];

    for k in 0..num_slices {
        let offset = k * last_dim;
        let slice = &data[offset..offset + last_dim];

        let mv = *slice.iter().max().unwrap();
        max_k.push(mv);

        let argmax = slice.iter().position(|&x| x == mv).unwrap();
        argmax_k.push(argmax);

        let mut sum_exp: i32 = 0;
        for j in 0..last_dim {
            let idx = offset + j;
            z[idx] = mv - data[idx]; // ≥ 0

            let z_c = z[idx].min(z_bound_minus_1);
            exp_q[idx] = table[z_c as usize];

            sum_exp += exp_q[idx];
        }

        exp_sum_q.push(sum_exp);

        let is = s_sq / sum_exp;
        inv_sum.push(is);
        debug_assert!(
            {
                let ri = s_sq - is * sum_exp;
                ri >= 0 && ri < sum_exp
            },
            "r_inv out of range, sum={sum_exp}"
        );

        for j in 0..last_dim {
            let idx = offset + j;
            let product = exp_q[idx] * is;
            let sq = product / s;
            let rem = product - sq * s;
            softmax_q[idx] = sq;
            R[idx] = rem;
            debug_assert!(rem >= 0 && rem < s, "R out of range: {rem}, S={s}");
        }
    }

    let mut result = Tensor::new(Some(&softmax_q), &[total]).unwrap();
    result.reshape(dims).unwrap();

    let trace = SoftmaxLastAxisFlatTrace {
        scale: s as i64,
        x: data.to_vec(),
        max_k,
        argmax_k,
        z,
        exp_q,
        exp_sum_q,
        inv_sum,
        R,
        flat_exp: FlatExpWitness {
            table,
            z_bound_minus_1,
        },
    };

    (result, trace)
}

/// Computes `z[k,j] = max_k[k] - x[k,j]` for every `(k,j)`, given the flat per-row max and
/// the flat `[F*N]` input. Used by softmax's saturating-clamp lookup (the `sat_clamp`
/// replacement for `sat_diff`'s complementary-slackness sumcheck) to re-derive the pre-clamp
/// witness without re-running the full decomposed trace.
pub fn softmax_z(x: &[i32], max_k: &[i32], last_dim: usize) -> Vec<i32> {
    x.iter()
        .enumerate()
        .map(|(idx, &xi)| max_k[idx / last_dim] - xi)
        .collect()
}
