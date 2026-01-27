use super::{Op, ULessThan};
use crate::tensor::{self, Tensor};

impl Op for ULessThan {
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        tensor::ops::less(inputs[0], inputs[1]).unwrap().0
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}
