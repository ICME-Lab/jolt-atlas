use crate::{
    ops::{IsNan, Op},
    tensor::Tensor,
};

impl Op for IsNan {
    fn f(&self, _inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        Tensor::new(
            Some(
                &(0..self.out_dims.iter().product())
                    .map(|_| 0i32)
                    .collect::<Vec<i32>>(),
            ),
            &self.out_dims,
        )
        .unwrap()
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}
