// TODO: Remove this file when we have the Softmax opcode decomposition in parser

use crate::{
    ops::{Op, Softmax},
    tensor::{self, Tensor},
};

impl Op for Softmax {
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        tensor::ops::nonlinearities::softmax_axes(inputs[0], self.scale.into(), &[self.axes])
    }
}

// TODO: Test for quantized softmax: sum_i softmax_i = 1
