use crate::{
    ops::{Noop, Op},
    tensor::Tensor,
};

impl Op for Noop {
    fn f(&self, _inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        // Noop
        Tensor::new(None, &[0]).unwrap()
    }
}
