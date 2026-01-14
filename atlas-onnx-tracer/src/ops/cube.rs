use crate::{
    ops::{Cube, Op},
    tensor::Tensor,
};

impl Op for Cube {
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        inputs[0].pow(3).unwrap()
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}
