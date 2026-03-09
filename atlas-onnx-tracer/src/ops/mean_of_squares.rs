use crate::{
    ops::{MeanOfSquares, Op},
    tensor::{Tensor, TensorError},
};
use tract_onnx::prelude::tract_itertools::Itertools;

impl Op for MeanOfSquares {
    #[tracing::instrument(name = "MeanOfSquares::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        mean_of_squares_axes_fused_i64(inputs[0], &self.axes, self.scale).unwrap()
    }
}

/// Fused mean-of-squares reduction: squares each element in i64, accumulates
/// x²/S in i64, then divides by the reduction count.
///
/// This avoids an i32 intermediate for the squared values. The per-element
/// Square in the unfused path can saturate at i32::MAX (~2.1B) when
/// |centered_activation| > 724 at scale=12 (S=4096), causing LayerNorm to
/// underestimate variance and amplify subsequent activations by ~3x per layer.
///
/// Here each x²/S term lives in i64 (max value ~ 2^50 for i32 inputs,
/// well within i64::MAX). The sum of up to 768 such terms fits in i64 with
/// ample headroom: 768 × 2^50 ≈ 8.6×10^17 < 9.2×10^18.
#[tracing::instrument(name = "tensor::ops::mean_of_squares_axes_fused_i64", skip_all)]
pub fn mean_of_squares_axes_fused_i64(
    a: &Tensor<i32>,
    axes: &[usize],
    scale: i32,
) -> Result<Tensor<i32>, TensorError> {
    let mut new_dims = vec![];
    for i in 0..a.dims().len() {
        if !axes.contains(&i) {
            new_dims.push(a.dims()[i]);
        } else {
            new_dims.push(1);
        }
    }

    // Calculate the divisor: S (scale rebase) × count (mean)
    let count: usize = axes.iter().map(|&ax| a.dims()[ax]).product();
    let s: i64 = 1i64 << scale;
    let divisor: i64 = s * count as i64;

    let res = Tensor::new(None, &new_dims)?;

    let cartesian_coord = new_dims
        .iter()
        .map(|x| 0..*x)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    let res = res.par_enum_map(|i, _: i32| {
        let coord = cartesian_coord[i].clone();
        let mut prod_dims = vec![];
        for (j, c) in coord.iter().enumerate() {
            if axes.contains(&j) {
                prod_dims.push(0..a.dims()[j]);
            } else {
                prod_dims.push(*c..*c + 1);
            }
        }

        let slice = a.get_slice(&prod_dims)?;

        // Accumulate x²/S in i64 — each term fits i64 (i32² / 2^12 ≤ 2^50)
        // and the sum of ≤768 terms ≤ 768 × 2^50 ≈ 8.6×10^17 < i64::MAX.
        let mut acc: i64 = 0;
        let _ = slice.map(|v| {
            let v64 = v as i64;
            acc += v64 * v64;
        });

        let result = acc / divisor;

        // Saturate to i32
        Ok(result.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
    })?;

    Ok(res)
}
