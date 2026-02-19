use crate::{
    ops::{Iff, Op},
    tensor::{self, Tensor},
};

impl Op for Iff {
    #[tracing::instrument(name = "Iff::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        tensor::ops::iff(inputs[0], inputs[1], inputs[2]).unwrap()
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}
