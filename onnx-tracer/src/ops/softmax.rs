use crate::{
    ops::{Op, SoftmaxAxes},
    tensor::{self, Tensor},
};

impl Op for SoftmaxAxes {
    #[tracing::instrument(name = "SoftmaxAxes::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        tensor::ops::nonlinearities::softmax_axes(inputs[0], self.scale.into(), &[self.axes])
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}
