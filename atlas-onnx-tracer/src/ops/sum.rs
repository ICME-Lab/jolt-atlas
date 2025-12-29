use crate::{
    ops::{Op, Sum},
    tensor::{self, Tensor},
};

impl Op for Sum {
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        tensor::ops::sum_axes(inputs[0], &self.axes).unwrap()
    }
}
