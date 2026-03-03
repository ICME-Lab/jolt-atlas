use crate::{
    ops::{Mul, Op},
    tensor::{self, Tensor},
};

impl Op for Mul {
    #[tracing::instrument(name = "Mul::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        tensor::ops::mult(&inputs).unwrap()
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }

    fn rebase_scale_factor(&self) -> Option<usize> {
        Some(1) // Mul: x * y produces result at scale^2, needs div by (1 << scale)
    }
}
