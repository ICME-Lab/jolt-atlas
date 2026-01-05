use crate::{
    ops::{MoveAxis, Op},
    tensor::Tensor,
};

impl Op for MoveAxis {
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        inputs[0]
            .clone()
            .move_axis(self.source, self.destination)
            .unwrap()
    }
}
