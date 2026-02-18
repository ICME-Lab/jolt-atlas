use crate::{
    ops::{Input, Op},
    tensor::Tensor,
};

impl Op for Input {
    #[tracing::instrument(name = "Input::f", skip_all)]
    fn f(&self, _inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        panic!("Input operation does not perform computation")
    }
}
