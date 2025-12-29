use super::{Constant, Op};
use crate::tensor::Tensor;

impl Op for Constant {
    fn f(&self, _inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        self.0.clone()
    }
}
