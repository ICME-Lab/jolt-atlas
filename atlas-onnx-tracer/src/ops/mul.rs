use common::parallel::par_enabled;
use rayon::prelude::*;

use crate::{
    ops::{Mul, Op},
    tensor::{Tensor, TensorError},
};

impl Op for Mul {
    #[tracing::instrument(name = "Mul::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        #[cfg(feature = "fused-ops")]
        {
            mult_i32_with_i64_rebase(&inputs, self.scale).unwrap()
        }
        #[cfg(not(feature = "fused-ops"))]
        {
            crate::tensor::ops::mult(&inputs).unwrap()
        }
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }

    #[cfg(not(feature = "fused-ops"))]
    fn rebase_scale_factor(&self) -> Option<usize> {
        Some(1) // Mul: x * y produces result at scale^2, needs div by (1 << scale)
    }
}

/// Element-wise multiply two i32 tensors using i64 intermediate precision,
/// with fused floor-division rebase by `1 << scale`.
///
/// Replaces Mul + ScalarConstDiv(1<<scale) in one step: `(a_i * b_i) >> scale`.
#[tracing::instrument(name = "tensor::ops::mult_i64_rebase", skip_all)]
pub fn mult_i32_with_i64_rebase(
    t: &[&Tensor<i32>],
    scale: i32,
) -> Result<Tensor<i32>, TensorError> {
    let rebase_divisor: i64 = 1i64 << scale;
    let mut output = t[0].clone();
    for &rhs in &t[1..] {
        let rhs_expanded = rhs.expand(output.dims()).unwrap_or_else(|_| rhs.clone());
        output
            .par_iter_mut()
            .zip(rhs_expanded.data().par_iter())
            .with_min_len(par_enabled())
            .for_each(|(o, r)| {
                let prod: i64 = (*o as i64) * (*r as i64);

                let result = prod / rebase_divisor;

                // Saturate instead of wrapping — prevents sign-flip corruption
                *o = result.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
            });
    }
    Ok(output)
}
