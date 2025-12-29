use crate::{
    ops::{Mul, Op},
    tensor::{self, Tensor},
};

impl Op for Mul {
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        tensor::ops::mult(&inputs).unwrap()
    }
}
