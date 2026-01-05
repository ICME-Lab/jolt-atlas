use crate::{
    ops::{Op, Tanh},
    tensor::{self, Tensor},
};

impl Op for Tanh {
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        tensor::ops::nonlinearities::tanh(inputs[0], self.scale.into())
    }
}
