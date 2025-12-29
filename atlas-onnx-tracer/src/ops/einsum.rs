use crate::{
    ops::{Einsum, Op},
    tensor::{self, Tensor},
};

impl Op for Einsum {
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        tensor::ops::einsum(&self.equation, &inputs).unwrap()
    }
}
