use super::{Constant, Op};
use crate::tensor::Tensor;

impl Op for Constant {
    #[tracing::instrument(name = "Constant::f", skip_all)]
    fn f(&self, _inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        self.0.clone()
    }
}
