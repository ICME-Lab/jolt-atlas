use super::{Add, Op};
use crate::tensor::{Tensor, get_broadcasted_shape};
use common::parallel::par_enabled;
use rayon::prelude::*;

impl Op for Add {
    #[tracing::instrument(name = "Add::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        let mut output = inputs[0].clone();
        for &rhs in &inputs[1..] {
            let shape = get_broadcasted_shape(output.dims(), rhs.dims())
                .expect("Add: incompatible broadcast shapes");
            let lhs_exp = output.expand(&shape).expect("Add: expand lhs");
            let rhs_exp = rhs.expand(&shape).expect("Add: expand rhs");
            let data: Vec<i32> = lhs_exp
                .data()
                .par_iter()
                .zip(rhs_exp.data().par_iter())
                .with_min_len(par_enabled())
                .map(|(&a, &b)| {
                    let sum: i64 = a as i64 + b as i64;
                    sum.clamp(i32::MIN as i64, i32::MAX as i64) as i32
                })
                .collect();
            output = Tensor::new(Some(&data), &shape).expect("Add: Tensor::new");
        }
        output
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}
