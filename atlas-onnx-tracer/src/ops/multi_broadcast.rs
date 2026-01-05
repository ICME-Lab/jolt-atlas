use crate::{
    ops::{MultiBroadcast, Op},
    tensor::Tensor,
};

impl Op for MultiBroadcast {
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        inputs[0].expand(&self.shape).unwrap()
    }
}
