use crate::{
    ops::{Op, Rsqrt},
    tensor::{Tensor, TensorError},
};

impl Op for Rsqrt {
    #[tracing::instrument(name = "Rsqrt::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        #[cfg(feature = "fused-ops")]
        {
            rsqrt_2(inputs[0], self.scale.into())
        }
        #[cfg(not(feature = "fused-ops"))]
        {
            crate::tensor::ops::nonlinearities::rsqrt(inputs[0], self.scale.into())
        }
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}

/// Elementwise applies reciprocal square root to a tensor of integers.
/// # Arguments
///
/// * `a` - Tensor
/// * `scale_input` - Single value
/// * `scale_output` - Single value
/// # Examples
/// ```
/// use atlas_onnx_tracer::tensor::Tensor;
/// use atlas_onnx_tracer::tensor::ops::nonlinearities::rsqrt;
/// let x = Tensor::<i32>::new(
///     Some(&[32, 128, 512, 2048, 8, 1]),
///     &[2, 3],
/// ).unwrap();
/// let result = rsqrt(&x, 7.0);
/// let expected = Tensor::<i32>::new(Some(&[256, 128, 64, 32, 512, 1448]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
#[tracing::instrument(name = "rsqrt", skip_all)]
pub fn rsqrt_2(a: &Tensor<i32>, scale_input: f64) -> Tensor<i32> {
    let sf_log = scale_input as i32;
    // Use i64 to avoid overflow at higher scales.
    // rsqrt(x) = S / sqrt(x/S) = sqrt(S³ / x)
    let sf = 1i64 << sf_log;
    let s_cubed = sf * sf * sf;
    a.par_enum_map(|_, a_i| {
        if a_i <= 0 {
            return Ok::<_, TensorError>(0i32);
        }
        let rsqrt = (s_cubed / (a_i as i64)).isqrt();
        Ok(rsqrt as i32)
    })
    .unwrap()
}
