use crate::{
    ops::{GatherLarge, GatherSmall, Op},
    tensor::{self, Tensor},
};

fn eval_gather(axis: usize, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
    assert_eq!(axis, 0, "Only axis 0 is currently supported for Gather");
    let [x, y] = inputs[..] else {
        panic!("Expected exactly two inputs")
    };
    tensor::ops::gather(x, &y.map(|v| v as usize), axis).unwrap()
}

impl Op for GatherSmall {
    #[tracing::instrument(name = "GatherSmall::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        eval_gather(self.axis, inputs)
    }
}

impl Op for GatherLarge {
    #[tracing::instrument(name = "GatherLarge::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        eval_gather(self.axis, inputs)
    }
}
