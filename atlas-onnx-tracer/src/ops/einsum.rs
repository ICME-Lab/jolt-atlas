use crate::{
    ops::FusedIntermediates,
    ops::{Einsum, Op},
    tensor::{Tensor, TensorError},
};
use common::parallel::par_enabled;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use tract_onnx::prelude::tract_itertools::Itertools;

impl Op for Einsum {
    #[tracing::instrument(name = "Einsum::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        // Fused: i64 accumulate, floor-rescale by `1 << scale`, saturating clamp
        // to i32. Replaces the einsum + its ScalarConstDiv rebase node .
        einsum_i32_with_i64_rebase(&self.equation, &inputs, self.scale).unwrap()
    }

    fn f_with_intermediates(
        &self,
        inputs: Vec<&Tensor<i32>>,
    ) -> (Tensor<i32>, Option<FusedIntermediates>) {
        let acc = einsum_acc_i64(&self.equation, &inputs).unwrap();
        let (out, ints) = FusedIntermediates::from_acc(&acc, 1i64 << self.scale);
        (out, Some(ints))
    }

    fn rebase_scale_factor(&self) -> Option<usize> {
        Some(1) // Einsum involves multiplication, needs div by (1 << scale)
    }
}

/// Re-execute an [`Einsum`] node's fused accumulate+rebase, returning the
/// pre-clamp `i64` intermediate (the saturating-clamp lookup index).
///
/// The proof system uses this to recover the intermediate without storing it
/// in the trace — analogous to `sat_binop_intermediate` for Add/Sub.
pub fn einsum_intermediate(op: &Einsum, inputs: &[&Tensor<i32>]) -> Tensor<i64> {
    einsum_accumulate_i64(&op.equation, inputs, op.scale)
        .unwrap_or_else(|e| panic!("einsum_intermediate: {e:?}"))
}

/// Accumulate the einsum in `i64` (multiply + sum over the contraction axes),
/// returning the **raw** pre-rebase accumulation `acc = Σ_k left·right`.
///
/// Rebasing (floor-division by `1 << scale`) and the remainder `R = acc mod 2^S`
/// are derived from this by the thin wrappers below, so both share one pass of
/// the contraction kernel.
#[tracing::instrument(name = "tensor::ops::einsum_acc_i64", skip_all)]
fn einsum_acc_i64(equation: &str, inputs: &[&Tensor<i32>]) -> Result<Tensor<i64>, TensorError> {
    if let Some(res) = gemm::einsum_acc_i64_gemm(equation, inputs) {
        return res;
    }
    einsum_acc_i64_generic(equation, inputs)
}

/// Reference kernel: one gather-based inner loop per output element. Used
/// when the contraction does not fit [`gemm`]'s batched-GEMM shape (three or
/// more operands, repeated letters, a summed letter private to one operand).
fn einsum_acc_i64_generic(
    equation: &str,
    inputs: &[&Tensor<i32>],
) -> Result<Tensor<i64>, TensorError> {
    // Parse equation (identical logic to generic einsum)
    let mut equation_parts = equation.split("->");
    let inputs_eq_str = equation_parts.next().unwrap();
    let output_eq = equation_parts.next().unwrap();
    let inputs_eq: Vec<&str> = inputs_eq_str.split(',').collect();

    if inputs.len() != inputs_eq.len() {
        return Err(TensorError::DimMismatch("einsum_i64".to_string()));
    }

    let mut indices_to_size = HashMap::new();
    for (i, input) in inputs.iter().enumerate() {
        for (j, c) in inputs_eq[i].chars().enumerate() {
            if let std::collections::hash_map::Entry::Vacant(e) = indices_to_size.entry(c) {
                e.insert(input.dims()[j]);
            } else if indices_to_size[&c] != input.dims()[j] {
                return Err(TensorError::DimMismatch("einsum_i64".to_string()));
            }
        }
    }

    for c in output_eq.chars() {
        indices_to_size.entry(c).or_insert(1);
    }

    let mut output_shape: Vec<usize> = output_eq
        .chars()
        .map(|c| *indices_to_size.get(&c).unwrap())
        .collect();
    if output_shape.is_empty() {
        output_shape.push(1);
    }

    let output_chars: HashSet<char> = output_eq.chars().collect();
    let mut seen = HashSet::new();
    let mut sum_indices: Vec<char> = Vec::new();
    for inp_eq in &inputs_eq {
        for c in inp_eq.chars() {
            if seen.insert(c) && !output_chars.contains(&c) {
                sum_indices.push(c);
            }
        }
    }
    let sum_sizes: Vec<usize> = sum_indices.iter().map(|c| indices_to_size[c]).collect();
    let sum_total: usize = sum_sizes.iter().product::<usize>().max(1);

    let input_strides: Vec<Vec<usize>> = inputs
        .iter()
        .map(|inp| {
            let dims = inp.dims();
            let ndim = dims.len();
            if ndim == 0 {
                return vec![];
            }
            let mut strides = vec![1usize; ndim];
            for d in (0..ndim - 1).rev() {
                strides[d] = strides[d + 1] * dims[d + 1];
            }
            strides
        })
        .collect();

    let sum_ndim = sum_sizes.len();
    let sum_strides: Vec<usize> = {
        let mut s = vec![1usize; sum_ndim];
        for d in (0..sum_ndim.saturating_sub(1)).rev() {
            s[d] = s[d + 1] * sum_sizes[d + 1];
        }
        s
    };

    let input_dim_maps: Vec<Vec<(bool, usize, usize)>> = inputs_eq
        .iter()
        .enumerate()
        .map(|(inp_idx, &eq)| {
            eq.chars()
                .enumerate()
                .map(|(dim_idx, c)| {
                    let stride = input_strides[inp_idx][dim_idx];
                    if let Some(out_pos) = output_eq.find(c) {
                        (true, out_pos, stride)
                    } else {
                        let sum_pos = sum_indices.iter().position(|&x| x == c).unwrap();
                        (false, sum_pos, stride)
                    }
                })
                .collect()
        })
        .collect();

    let sum_coords: Vec<Vec<usize>> = if sum_ndim > 0 {
        (0..sum_total)
            .map(|s_flat| {
                let mut coord = vec![0usize; sum_ndim];
                let mut remaining = s_flat;
                for d in 0..sum_ndim {
                    coord[d] = remaining / sum_strides[d];
                    remaining %= sum_strides[d];
                }
                coord
            })
            .collect()
    } else {
        vec![vec![]]
    };

    let sum_partials: Vec<Vec<usize>> = (0..inputs.len())
        .map(|inp_idx| {
            sum_coords
                .iter()
                .map(|s_coord| {
                    let mut partial = 0usize;
                    for &(is_output, coord_pos, stride) in &input_dim_maps[inp_idx] {
                        if !is_output {
                            partial += s_coord[coord_pos] * stride;
                        }
                    }
                    partial
                })
                .collect()
        })
        .collect();

    let cartesian_coord = output_shape
        .iter()
        .map(|d| 0..*d)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    let output: Vec<i64> = cartesian_coord
        .par_iter()
        .with_min_len(par_enabled())
        .map(|out_coord| {
            let out_partials: Vec<usize> = (0..inputs.len())
                .map(|inp_idx| {
                    let mut partial = 0usize;
                    for &(is_output, coord_pos, stride) in &input_dim_maps[inp_idx] {
                        if is_output {
                            partial += out_coord[coord_pos] * stride;
                        }
                    }
                    partial
                })
                .collect();

            // Accumulate in i64 for full precision (raw, pre-rebase).
            let mut sum: i64 = 0;
            for s_idx in 0..sum_total {
                let mut product: i64 = 1;
                for (inp_idx, input) in inputs.iter().enumerate() {
                    let flat_idx = out_partials[inp_idx] + sum_partials[inp_idx][s_idx];
                    product *= input.inner[flat_idx] as i64;
                }
                sum += product;
            }

            sum
        })
        .collect();

    let mut output = Tensor::<i64>::new(Some(&output), &output_shape)?;
    output.reshape(&output_shape)?;

    Ok(output)
}

/// Accumulate and **rebase** by floor-dividing (Euclidean) the raw `i64`
/// accumulation by `1 << scale`, returning the pre-cast `i64` rescaled value.
///
/// Floor (rather than truncating) division is deliberate: it makes the rebase a
/// pure arithmetic right shift, so the proof-side remainder `R = acc mod 2^S`
/// (see [`einsum_remainder`]) is always in `[0, 2^S)` even when `acc` is
/// negative — directly range-checkable .
fn einsum_accumulate_i64(
    equation: &str,
    inputs: &[&Tensor<i32>],
    scale: i32,
) -> Result<Tensor<i64>, TensorError> {
    let acc = einsum_acc_i64(equation, inputs)?;
    let divisor = 1i64 << scale;
    let data: Vec<i64> = acc.data().iter().map(|&v| v.div_euclid(divisor)).collect();
    Tensor::<i64>::new(Some(&data), acc.dims())
}

/// Re-execute the einsum's rescaling **remainder** `R = acc mod 2^S` (Euclidean,
/// so `R ∈ [0, 2^S)`), where `acc = Σ_k left·right` and `output = acc >> S`.
///
/// The proof system uses this to recover the per-element remainder without
/// storing it in the trace — the einsum sumcheck binds `output·2^S + R = acc`
/// and range-checks `R` . `R` fits `i32` for any `scale < 31`.
pub fn einsum_remainder(op: &Einsum, inputs: &[&Tensor<i32>]) -> Tensor<i32> {
    let acc =
        einsum_acc_i64(&op.equation, inputs).unwrap_or_else(|e| panic!("einsum_remainder: {e:?}"));
    let divisor = 1i64 << op.scale;
    let data: Vec<i32> = acc
        .data()
        .iter()
        .map(|&v| v.rem_euclid(divisor) as i32)
        .collect();
    Tensor::<i32>::new(Some(&data), acc.dims())
        .unwrap_or_else(|e| panic!("einsum_remainder: {e:?}"))
}

/// Quotient and remainder of the fused rescale — `rescaled = acc >> S` and
/// `R = acc mod 2^S` — from **one** contraction pass. The proof system prefers
/// this over separate [`einsum_intermediate`] + [`einsum_remainder`] calls,
/// which each re-run the (expensive) contraction kernel.
pub fn einsum_intermediate_and_remainder(
    op: &Einsum,
    inputs: &[&Tensor<i32>],
) -> (Tensor<i64>, Tensor<i32>) {
    let acc = einsum_acc_i64(&op.equation, inputs)
        .unwrap_or_else(|e| panic!("einsum_intermediate_and_remainder: {e:?}"));
    (
        super::floor_rebase_i64(&acc, op.scale),
        super::rebase_remainder_i32(&acc, op.scale),
    )
}

/// Like `einsum_accumulate_i64`, but saturates the i64 result to i32.
///
/// Replaces both the Einsum and its subsequent ScalarConstDiv rebase node,
/// avoiding the lossy wrapping-then-dividing path.
#[tracing::instrument(name = "tensor::ops::einsum_i64_rebase", skip_all)]
pub fn einsum_i32_with_i64_rebase(
    equation: &str,
    inputs: &[&Tensor<i32>],
    scale: i32,
) -> Result<Tensor<i32>, TensorError> {
    let acc = einsum_accumulate_i64(equation, inputs, scale)?;
    Ok(super::clamp_to_i32(&acc))
}

/// Batched-GEMM fast path for two-operand einsums.
///
/// Letters are classified as batch (both operands and the output), `M` (left
/// and output), `N` (right and output) or `K` (both operands, not the output).
/// Both operands are copied into contiguous `[batch, M, K]` / `[batch, K, N]`
/// buffers, multiplied with a cache-blocked `i64` accumulate (each `B` row is
/// read once; the accumulator tile stays in cache), and the `[batch, M, N]`
/// result is permuted into the output's letter order.
mod gemm {
    use crate::tensor::{Tensor, TensorError};
    use common::parallel::par_enabled;
    use rayon::prelude::*;
    use std::collections::HashMap;

    /// Copy `src` (dims `dims`) into a contiguous buffer whose axis order is
    /// `axes` (a permutation of `0..dims.len()`).
    fn permute_copy<T: Copy + Send + Sync>(src: &[T], dims: &[usize], axes: &[usize]) -> Vec<T> {
        let n = dims.len();
        if axes.iter().enumerate().all(|(i, &a)| i == a) {
            return src.to_vec();
        }
        let mut strides = vec![1usize; n];
        for d in (0..n.saturating_sub(1)).rev() {
            strides[d] = strides[d + 1] * dims[d + 1];
        }
        let dst_dims: Vec<usize> = axes.iter().map(|&a| dims[a]).collect();
        let dst_strides_in_src: Vec<usize> = axes.iter().map(|&a| strides[a]).collect();
        let total: usize = dst_dims.iter().product();
        // Innermost destination axis: contiguous runs of `inner` elements.
        let inner = *dst_dims.last().unwrap_or(&1);
        let inner_stride = *dst_strides_in_src.last().unwrap_or(&1);
        let mut dst = Vec::new();
        dst.resize(total, src[0]);
        dst.par_chunks_mut(inner.max(1))
            .with_min_len(par_enabled().min(64))
            .enumerate()
            .for_each(|(o, run)| {
                let mut rem = o;
                let mut base = 0usize;
                for d in (0..n.saturating_sub(1)).rev() {
                    let c = rem % dst_dims[d];
                    rem /= dst_dims[d];
                    base += c * dst_strides_in_src[d];
                }
                for (i, x) in run.iter_mut().enumerate() {
                    *x = src[base + i * inner_stride];
                }
            });
        dst
    }

    /// `C[b, i, j] = Σ_k A[b, i, k] · B[b, k, j]` in i64, `A: [b, m, k]`, `B: [b, k, n]`.
    fn gemm_i64(a: &[i32], b: &[i32], batch: usize, m: usize, k: usize, n: usize) -> Vec<i64> {
        const N_BLOCK: usize = 2048;
        const M_BLOCK: usize = 16;
        let mut out = vec![0i64; batch * m * n];
        let n_blocks = n.div_ceil(N_BLOCK);
        let m_blocks = m.div_ceil(M_BLOCK);
        // Parallel over (batch, m-block, n-block) output tiles; each tile
        // streams its slice of `B` exactly once.
        out.par_chunks_mut(m * n)
            .enumerate()
            .flat_map(|(bi, out_b)| {
                // Split the batch's output into disjoint m-block row bands.
                out_b
                    .par_chunks_mut(M_BLOCK * n)
                    .enumerate()
                    .map(move |(mb, band)| (bi, mb, band))
            })
            .for_each(|(bi, mb, band)| {
                let a_b = &a[bi * m * k..(bi + 1) * m * k];
                let b_b = &b[bi * k * n..(bi + 1) * k * n];
                let m0 = mb * M_BLOCK;
                let m_len = (m - m0).min(M_BLOCK);
                // Column blocks in parallel within the band.
                let cols: Vec<(usize, Vec<i64>)> = (0..n_blocks)
                    .into_par_iter()
                    .with_min_len(par_enabled())
                    .map(|nb| {
                        let j0 = nb * N_BLOCK;
                        let j_len = (n - j0).min(N_BLOCK);
                        let mut acc = vec![0i64; m_len * j_len];
                        for kk in 0..k {
                            let b_row = &b_b[kk * n + j0..kk * n + j0 + j_len];
                            for i in 0..m_len {
                                let a_ik = a_b[(m0 + i) * k + kk] as i64;
                                if a_ik == 0 {
                                    continue;
                                }
                                let acc_row = &mut acc[i * j_len..(i + 1) * j_len];
                                for (c, &bv) in acc_row.iter_mut().zip(b_row) {
                                    *c += a_ik * bv as i64;
                                }
                            }
                        }
                        (nb, acc)
                    })
                    .collect();
                for (nb, acc) in cols {
                    let j0 = nb * N_BLOCK;
                    let j_len = (n - j0).min(N_BLOCK);
                    for i in 0..m_len {
                        band[i * n + j0..i * n + j0 + j_len]
                            .copy_from_slice(&acc[i * j_len..(i + 1) * j_len]);
                    }
                }
            });
        let _ = m_blocks;
        out
    }

    /// `Some(result)` when the contraction fits the batched-GEMM shape.
    pub(super) fn einsum_acc_i64_gemm(
        equation: &str,
        inputs: &[&Tensor<i32>],
    ) -> Option<Result<Tensor<i64>, TensorError>> {
        let mut parts = equation.split("->");
        let ins = parts.next()?;
        let out_eq: Vec<char> = parts.next()?.chars().collect();
        let ins: Vec<Vec<char>> = ins.split(',').map(|s| s.chars().collect()).collect();
        if ins.len() != 2 || inputs.len() != 2 {
            return None;
        }
        let (l, r) = (&ins[0], &ins[1]);
        // No repeated letters within an operand or the output.
        for s in [l, r, &out_eq] {
            let mut seen = std::collections::HashSet::new();
            if !s.iter().all(|c| seen.insert(*c)) {
                return None;
            }
        }
        if l.len() != inputs[0].dims().len() || r.len() != inputs[1].dims().len() {
            return None;
        }
        let mut size: HashMap<char, usize> = HashMap::new();
        for (letters, t) in [(l, inputs[0]), (r, inputs[1])] {
            for (c, d) in letters.iter().zip(t.dims()) {
                if let Some(&prev) = size.get(c) {
                    if prev != *d {
                        return Some(Err(TensorError::DimMismatch("einsum_gemm".into())));
                    }
                }
                size.insert(*c, *d);
            }
        }
        let in_l = |c: &char| l.contains(c);
        let in_r = |c: &char| r.contains(c);
        let in_o = |c: &char| out_eq.contains(c);
        // Classify (keeping the left operand's letter order for K so both
        // operands agree on the contraction layout).
        let batch: Vec<char> = l.iter().copied().filter(|c| in_r(c) && in_o(c)).collect();
        let m_l: Vec<char> = l.iter().copied().filter(|c| !in_r(c) && in_o(c)).collect();
        let n_l: Vec<char> = r.iter().copied().filter(|c| !in_l(c) && in_o(c)).collect();
        let k_l: Vec<char> = l.iter().copied().filter(|c| in_r(c) && !in_o(c)).collect();
        // A summed letter private to one operand needs a pre-reduction; only
        // trivially sized ones (dim 1) are accepted here.
        for c in l.iter().filter(|c| !in_r(c) && !in_o(c)) {
            if size[c] != 1 {
                return None;
            }
        }
        for c in r.iter().filter(|c| !in_l(c) && !in_o(c)) {
            if size[c] != 1 {
                return None;
            }
        }
        let prod = |ls: &[char]| ls.iter().map(|c| size[c]).product::<usize>().max(1);
        let (nb, nm, nn, nk) = (prod(&batch), prod(&m_l), prod(&n_l), prod(&k_l));

        // Left → [batch, M, K, (private size-1 letters anywhere)].
        let axes_l: Vec<usize> = batch
            .iter()
            .chain(m_l.iter())
            .chain(k_l.iter())
            .chain(l.iter().filter(|c| !in_r(c) && !in_o(c)))
            .map(|c| l.iter().position(|x| x == c).unwrap())
            .collect();
        let axes_r: Vec<usize> = batch
            .iter()
            .chain(k_l.iter())
            .chain(n_l.iter())
            .chain(r.iter().filter(|c| !in_l(c) && !in_o(c)))
            .map(|c| r.iter().position(|x| x == c).unwrap())
            .collect();
        let a = permute_copy(inputs[0].data(), inputs[0].dims(), &axes_l);
        let b = permute_copy(inputs[1].data(), inputs[1].dims(), &axes_r);
        let c = gemm_i64(&a, &b, nb, nm, nk, nn);
        drop((a, b));

        // [batch, M, N] → output letter order. Output letters absent from both
        // operands have size 1 and are inserted as unit axes.
        let mut c_letters: Vec<char> = batch
            .iter()
            .chain(m_l.iter())
            .chain(n_l.iter())
            .copied()
            .collect();
        let mut c_dims: Vec<usize> = c_letters.iter().map(|c| size[c]).collect();
        for c in out_eq.iter() {
            if !c_letters.contains(c) {
                c_letters.push(*c);
                c_dims.push(1);
            }
        }
        let axes_o: Vec<usize> = out_eq
            .iter()
            .map(|c| c_letters.iter().position(|x| x == c).unwrap())
            .collect();
        // Letters of `c_letters` not in the output all have size 1 (batch/M/N
        // letters are in the output by construction); append them so the
        // permutation is complete.
        let mut axes_full = axes_o.clone();
        for i in 0..c_letters.len() {
            if !axes_full.contains(&i) {
                axes_full.push(i);
            }
        }
        let out = permute_copy(&c, &c_dims, &axes_full);
        let mut out_shape: Vec<usize> = out_eq
            .iter()
            .map(|c| size.get(c).copied().unwrap_or(1))
            .collect();
        if out_shape.is_empty() {
            out_shape.push(1);
        }
        Some(Tensor::<i64>::new(Some(&out), &out_shape))
    }

    #[cfg(test)]
    mod tests {
        use super::super::einsum_acc_i64_generic;
        use super::*;
        use rand::{Rng, SeedableRng, rngs::StdRng};

        fn rand_tensor(rng: &mut StdRng, dims: &[usize]) -> Tensor<i32> {
            let n: usize = dims.iter().product();
            let v: Vec<i32> = (0..n).map(|_| rng.gen_range(-50_000..50_000)).collect();
            Tensor::new(Some(&v), dims).unwrap()
        }

        #[test]
        fn gemm_path_matches_generic_kernel() {
            let cases: &[(&str, &[usize], &[usize])] = &[
                ("mk,kn->mn", &[5, 7], &[7, 3]),
                ("amk,kn->amn", &[1, 4, 6], &[6, 5]),
                ("amk,kn->mn", &[1, 4, 6], &[6, 5]),
                ("abmk,abnk->abmn", &[2, 3, 4, 5], &[2, 3, 6, 5]),
                ("acbmk,kcn->cbmn", &[1, 2, 3, 4, 4], &[4, 2, 5]),
                ("cbmk,cbkn->amn", &[2, 3, 4, 5], &[2, 3, 5, 6]),
                ("m,an->abnm", &[3], &[1, 4]),
                ("ij,jk->ik", &[3, 2], &[2, 4]),
                ("k,k->", &[9], &[9]),
                ("amk,kn->amn", &[1, 3, 40], &[40, 4100]),
            ];
            let mut rng = StdRng::seed_from_u64(7);
            for (eq, dl, dr) in cases {
                let a = rand_tensor(&mut rng, dl);
                let b = rand_tensor(&mut rng, dr);
                let fast = einsum_acc_i64_gemm(eq, &[&a, &b])
                    .unwrap_or_else(|| panic!("{eq}: gemm path should apply"))
                    .unwrap();
                let slow = einsum_acc_i64_generic(eq, &[&a, &b]).unwrap();
                assert_eq!(fast.dims(), slow.dims(), "{eq} dims");
                assert_eq!(fast.data(), slow.data(), "{eq} values");
            }
        }
    }
}
