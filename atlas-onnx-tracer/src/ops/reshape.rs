use crate::{
    ops::{Op, Reshape},
    tensor::Tensor,
};

impl Op for Reshape {
    #[tracing::instrument(name = "Reshape::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        let mut t = inputs[0].clone();
        t.reshape(&self.shape).unwrap();
        t
    }
}
