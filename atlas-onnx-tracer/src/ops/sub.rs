use crate::{
    ops::{Op, Sub},
    tensor::{self, Tensor},
};

impl Op for Sub {
    #[tracing::instrument(name = "Sub::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        tensor::ops::sub(&inputs).unwrap()
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}
