use crate::{
    ops::{Op, Sum},
    tensor::{self, Tensor},
};

impl Op for Sum {
    #[tracing::instrument(name = "Sum::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        tensor::ops::sum_axes(inputs[0], &self.axes).unwrap()
    }
}
