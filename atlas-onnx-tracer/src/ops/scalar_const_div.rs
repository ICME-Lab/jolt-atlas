use super::Op;
use crate::{ops::ScalarConstDiv, tensor::Tensor};

impl Op for ScalarConstDiv {
    #[tracing::instrument(name = "ScalarConstDiv::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        let a = inputs[0];
        let b = self.divisor;
        let data: Vec<i32> = a
            .data()
            .iter()
            .map(|&x| {
                let mut d_inv_x = x / (b);
                let remainder = x % b;
                if (remainder < 0 && b > 0) || (remainder > 0 && b < 0) {
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
