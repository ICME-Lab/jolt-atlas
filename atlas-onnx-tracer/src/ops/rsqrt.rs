use crate::{
    ops::{Op, Rsqrt},
    tensor::{Tensor, TensorError},
};

impl Op for Rsqrt {
    #[tracing::instrument(name = "Rsqrt::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        rsqrt(inputs[0], self.scale)
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}

/// Elementwise reciprocal square root of a quantized tensor.
///
/// For a fixed-point input `x̂` with scale `S = 2^scale`, the exact reciprocal
/// square root at the same scale is `S / √(x̂ / S) = √(S³ / x̂)`. We fuse the
/// division and the square root into a single floor operation:
///
/// ```text
/// output = ⌊√⌊S³ / x̂⌋⌋
/// ```
///
/// Non-positive inputs map to `0`.
///
/// # Examples
/// ```
/// use atlas_onnx_tracer::tensor::Tensor;
/// use atlas_onnx_tracer::ops::rsqrt::rsqrt;
/// let x = Tensor::<i32>::new(Some(&[32, 128, 512, 2048, 8, 1]), &[2, 3]).unwrap();
/// let result = rsqrt(&x, 7);
/// // output[i] = isqrt((1 << 21) / x[i])
/// let expected = Tensor::<i32>::new(Some(&[256, 128, 64, 32, 512, 1448]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
#[tracing::instrument(name = "rsqrt", skip_all)]
pub fn rsqrt(a: &Tensor<i32>, scale: i32) -> Tensor<i32> {
    // `S³` in i64 to avoid overflow at higher scales.
    let s_cubed = 1i64 << (3 * scale);
    a.par_enum_map(|_, a_i| {
        if a_i <= 0 {
            return Ok::<_, TensorError>(0i32);
        }
        let output = (s_cubed / (a_i as i64)).isqrt();
        Ok(output as i32)
    })
    .unwrap()
}
