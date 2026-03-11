use crate::{
    ops::{Concat, Op},
    tensor::{self, Tensor},
};

impl Op for Concat {
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        assert!(!inputs.is_empty(), "Concat requires at least one input");
        let rank = inputs.first().map(|t| t.dims().len()).unwrap_or(0) as isize;
        assert!(
            self.axis >= -rank && self.axis < rank,
            "Axis out of bounds for Concat"
        );
        let axis = if self.axis < 0 {
            (self.axis + rank) as usize
        } else {
            self.axis as usize
        };
        tensor::ops::concat(&inputs, axis).unwrap()
    }

    fn requires_shape_equality(&self) -> bool {
        false
    }
}
