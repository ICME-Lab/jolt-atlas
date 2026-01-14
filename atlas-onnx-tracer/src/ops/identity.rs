use crate::{
    ops::{Identity, Op},
    tensor::Tensor,
};

impl Op for Identity {
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        inputs[0].clone()
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}
