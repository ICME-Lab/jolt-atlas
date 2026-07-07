use crate::{
    ops::{Op, Sub},
    tensor::Tensor,
};

impl Op for Sub {
    #[tracing::instrument(name = "Sub::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        super::sat_binop(inputs, "Sub", |a, b| a - b)
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}
