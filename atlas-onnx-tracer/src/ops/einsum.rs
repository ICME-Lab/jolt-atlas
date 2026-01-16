use crate::{
    ops::{Einsum, Op},
    tensor::{self, Tensor},
};

impl Op for Einsum {
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        tensor::ops::einsum(&self.equation, &inputs).unwrap()
    }

    fn rebase_scale_factor(&self) -> Option<usize> {
        Some(1) // Einsum involves multiplication, needs div by (1 << scale)
    }
}
