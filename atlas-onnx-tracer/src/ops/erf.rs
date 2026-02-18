use crate::{
    ops::{Erf, Op},
    tensor::{self, Tensor},
};

impl Op for Erf {
    #[tracing::instrument(name = "Erf::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        tensor::ops::nonlinearities::erffunc(inputs[0], self.scale.into())
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}
