use crate::{
    ops::{Broadcast, Op},
    tensor::Tensor,
};

impl Op for Broadcast {
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        inputs[0].expand(&self.shape).unwrap()
    }
}
