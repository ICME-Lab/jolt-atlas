use crate::{
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
