use super::{Add, Op};
use crate::tensor::{self, Tensor};

impl Op for Add {
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        tensor::ops::add(&inputs).unwrap()
    }
}
