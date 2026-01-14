use super::{And, Op};
use crate::tensor::{self, Tensor};

impl Op for And {
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        tensor::ops::and(inputs[0], inputs[1]).unwrap()
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}
