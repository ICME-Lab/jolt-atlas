use crate::{
    ops::{Cube, Op},
    tensor::Tensor,
};

impl Op for Cube {
    #[tracing::instrument(name = "Cube::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        inputs[0].pow(3).unwrap()
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }

    fn rebase_scale_factor(&self) -> Option<usize> {
        Some(2) // Cube: x^3 produces result at scale^3, needs div by (1 << (scale * 2))
    }
}
