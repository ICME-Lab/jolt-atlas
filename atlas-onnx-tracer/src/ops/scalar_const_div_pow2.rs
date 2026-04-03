use super::Op;
use crate::{ops::ScalarConstDivPow2, tensor::Tensor};

impl Op for ScalarConstDivPow2 {
    #[tracing::instrument(name = "ScalarConstDivPow2::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        let a = inputs[0];
        let b = self.divisor;
        assert!(
            b > 0 && (b as u32).is_power_of_two(),
            "ScalarConstDivPow2 requires a positive power-of-two divisor, got {b}"
        );
        let data: Vec<i32> = a
            .data()
            .iter()
            .map(|&x| {
                let mut d_inv_x = x / b;
                let remainder = x % b;
                if remainder < 0 {
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
