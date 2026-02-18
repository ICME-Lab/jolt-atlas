use super::{Add, Op};
use crate::tensor::{self, Tensor};

impl Op for Add {
    #[tracing::instrument(name = "Add::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        tensor::ops::add(&inputs).unwrap()
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}
