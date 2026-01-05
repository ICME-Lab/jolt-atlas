use crate::{
    ops::{Op, Sub},
    tensor::{self, Tensor},
};

impl Op for Sub {
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        tensor::ops::sub(&inputs).unwrap()
    }
}
