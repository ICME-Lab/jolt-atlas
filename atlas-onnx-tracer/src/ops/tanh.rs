use crate::{
    ops::{Op, Tanh},
    tensor::{self, Tensor},
};

impl Op for Tanh {
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        let input = tensor::ops::nonlinearities::const_div(inputs[0], self.tau as f64);
        tensor::ops::nonlinearities::tanh(&input, self.scale.into())
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}
