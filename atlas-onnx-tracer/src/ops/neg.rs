use crate::{
    ops::{Op, Neg},
    tensor::{self, Tensor},
};

impl Op for Neg {
    #[tracing::instrument(name = "Neg::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        tensor::ops::neg(inputs[0]).unwrap()
    }
    
  // NOTE:
  // Neg is a unary sign-flip operation, so this intentionally rely on the Op trait defaults:
  // - requires_shape_equality() = false
  //   (no pairwise input shape matching is needed for a single input)
  // - rebase_scale_factor() = None
  //   (negation does not increase fixed-point scale, so no rebase is required)
}
