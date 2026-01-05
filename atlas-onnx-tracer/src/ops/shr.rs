use crate::{
    ops::{Op, Shr},
    tensor::{self, Tensor},
};

impl Op for Shr {
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        tensor::ops::sra(inputs[0], inputs[1])
    }
}
