use crate::{
    ops::{Cube, Op},
    tensor::{Tensor, TensorError},
};

impl Op for Cube {
    #[tracing::instrument(name = "Cube::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        #[cfg(feature = "fused-ops")]
        {
            cube_i32_with_i64_rebase(inputs[0], self.scale).unwrap()
        }
        #[cfg(not(feature = "fused-ops"))]
        {
            inputs[0].pow(3).unwrap()
        }
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }

    fn rebase_scale_factor(&self) -> Option<usize> {
        Some(2) // Cube: x^3 produces result at scale^3, needs div by (1 << (scale * 2))
    }
}

/// Element-wise cube an i32 tensor using i64 intermediate precision,
/// with fused rebase by `1 << (scale * 2)` and saturating cast.
///
/// Handles i64 overflow in x³ (possible for |x| > 2^21) via checked_mul,
/// and saturates i32 output for rare outliers.
#[tracing::instrument(name = "tensor::ops::cube_i64_rebase", skip_all)]
pub fn cube_i32_with_i64_rebase(a: &Tensor<i32>, scale: i32) -> Result<Tensor<i32>, TensorError> {
    let rebase_divisor: i64 = 1i64 << (scale * 2);
    let output: Tensor<i32> = a
        .par_enum_map(|_, x| {
            let x64 = x as i64;
            let sq = x64 * x64; // always fits i64 for i32 inputs (max 2^62)
            // x³ can overflow i64 for |x| > 2^21 ≈ 2M — saturate on overflow
            let cb = sq * x64;

            let result = cb / rebase_divisor;

            // Saturate instead of wrapping — prevents sign-flip corruption
            Ok::<_, TensorError>(result.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
        })
        .unwrap();
    Ok(output)
}
