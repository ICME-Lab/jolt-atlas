use crate::{
    ops::{Identity, Op},
    tensor::Tensor,
};

impl Op for Identity {
    #[tracing::instrument(name = "Identity::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        inputs[0].clone()
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}
