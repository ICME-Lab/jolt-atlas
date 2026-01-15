use super::{Div, Op};
use crate::tensor::Tensor;

impl Op for Div {
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        let a = inputs[0];
        let b = inputs[1];
        let data: Vec<i32> = a
            .data()
            .iter()
            .zip(b.data().iter())
            .map(|(&x, &y)| {
                let denom = y;
                let mut d_inv_x = x / (denom);
                let remainder = x % denom;
                if (remainder < 0 && denom > 0) || (remainder > 0 && denom < 0) {
                    d_inv_x -= 1;
                }
                d_inv_x
            })
            .collect();
        Tensor::new(Some(&data), a.dims()).unwrap()
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}
