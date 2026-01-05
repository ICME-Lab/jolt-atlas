use crate::{
    ops::{Op, Square},
    tensor::Tensor,
};

impl Op for Square {
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        inputs[0].pow(2).unwrap()
    }
}
