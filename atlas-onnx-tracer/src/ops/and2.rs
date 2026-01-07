use super::{And2, Op};
use crate::tensor::{self, Tensor};

impl Op for And2 {
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        tensor::ops::and2(inputs[0], inputs[1]).unwrap()
    }
}
