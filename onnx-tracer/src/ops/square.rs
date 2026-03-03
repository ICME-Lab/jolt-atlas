use crate::{
    ops::{Op, Square},
    tensor::Tensor,
};

impl Op for Square {
    #[tracing::instrument(name = "Square::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        inputs[0].pow(2).unwrap()
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }

    fn rebase_scale_factor(&self) -> Option<usize> {
        Some(1) // Square: x^2 produces result at scale^2, needs div by (1 << scale)
    }
}
