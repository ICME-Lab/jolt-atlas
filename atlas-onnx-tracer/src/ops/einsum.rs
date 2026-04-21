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
        #[cfg(feature = "fused-ops")]
        {
            einsum_i32_with_i64_rebase(&self.equation, &inputs, self.scale).unwrap()
        }
        #[cfg(not(feature = "fused-ops"))]
        {
            crate::tensor::ops::einsum(&self.equation, &inputs).unwrap()
        }
    }

    fn rebase_scale_factor(&self) -> Option<usize> {
        Some(1) // Einsum involves multiplication, needs div by (1 << scale)
    }
}

/// Like `einsum`, but uses i64 intermediate precision and fuses the rebase.
///
/// Upcasts i32 inputs to i64 for multiply+accumulate, then floor-divides by
/// `1 << scale` while still in i64 (preserving full precision), and truncates
/// to i32. This replaces both the Einsum **and** its subsequent ScalarConstDiv
/// rebase node, avoiding the lossy wrapping-then-dividing path.
#[tracing::instrument(name = "tensor::ops::einsum_i64_rebase", skip_all)]
pub fn einsum_i32_with_i64_rebase(
    equation: &str,
    inputs: &[&Tensor<i32>],
    scale: i32,
) -> Result<Tensor<i32>, TensorError> {
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

    let rebase_divisor: i64 = 1i64 << scale;

    let output: Vec<i32> = cartesian_coord
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

            // Accumulate in i64 for full precision
            let mut sum: i64 = 0;
            for s_idx in 0..sum_total {
                let mut product: i64 = 1;
                for (inp_idx, input) in inputs.iter().enumerate() {
                    let flat_idx = out_partials[inp_idx] + sum_partials[inp_idx][s_idx];
                    product *= input.inner[flat_idx] as i64;
                }
                sum += product;
            }

            let result = sum / rebase_divisor;

            // Saturate instead of wrapping — prevents sign-flip corruption
            result.clamp(i32::MIN as i64, i32::MAX as i64) as i32
        })
        .collect();

    let mut output: Tensor<i32> = output.into_iter().into();
    output.reshape(&output_shape)?;

    Ok(output)
}
