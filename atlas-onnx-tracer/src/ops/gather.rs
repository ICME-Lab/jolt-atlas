use crate::{
    ops::{Gather, Op},
    tensor::{self, Tensor},
};

impl Op for Gather {
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        let [x, y] = inputs[..] else {
            panic!("Expected exactly two inputs")
        };
        tensor::ops::gather(x, &y.map(|v| v as usize), self.dim).unwrap()
    }
}
