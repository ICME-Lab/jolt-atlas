use crate::{
    ops::{Clamp, Op},
    tensor::{self, Tensor},
};

impl Op for Clamp {
    #[tracing::instrument(name = "Clamp::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        tensor::ops::nonlinearities::clamp_axes(inputs[0], self.axes, self.max_spread)
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}
