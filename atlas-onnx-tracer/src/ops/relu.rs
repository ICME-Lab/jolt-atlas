use crate::{
    ops::{Op, ReLU},
    tensor::{self, Tensor},
};

impl Op for ReLU {
    #[tracing::instrument(name = "ReLU::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        tensor::ops::nonlinearities::leakyrelu(inputs[0], 0_f64)
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}
