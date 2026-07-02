use super::{Add, Op};
use crate::tensor::Tensor;

impl Op for Add {
    #[tracing::instrument(name = "Add::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        super::sat_binop(inputs, "Add", |a, b| a + b)
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}
