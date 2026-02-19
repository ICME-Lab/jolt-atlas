use crate::{
    ops::{Op, Rsqrt},
    tensor::{self, Tensor},
};

impl Op for Rsqrt {
    #[tracing::instrument(name = "Rsqrt::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        tensor::ops::nonlinearities::rsqrt(inputs[0], self.scale.into())
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}
