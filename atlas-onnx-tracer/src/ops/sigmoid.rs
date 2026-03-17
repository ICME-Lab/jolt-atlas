use std::ops::Mul;

use crate::{
    ops::{Op, Sigmoid},
    tensor::{self, Tensor},
};

impl Op for Sigmoid {
    #[tracing::instrument(name = "Sigmoid::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        let input = tensor::ops::nonlinearities::const_div(inputs[0], self.tau as f64);
        let teleport_recip = input.mul(self.tau).unwrap();

        tensor::ops::nonlinearities::sigmoid(&teleport_recip, self.scale.into())
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::Sigmoid;
    use crate::{
        ops::Op,
        tensor::{Tensor, ops::nonlinearities::sigmoid},
        utils::{f32::F32, precision::assert_quantized_precision},
    };
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn test_sigmoid_precision_stats() {
        const SCALE: f64 = 128.0;
        const TAU: i32 = 2;
        const SAMPLE_SIZE: usize = 1 << 14;
        const MIN_INPUT: i32 = -(1 << 14);
        const MAX_INPUT: i32 = 1 << 14;
        const WORST_ERROR_BOUND_QUANTIZED: i32 = 8;

        let mut rng = StdRng::seed_from_u64(0x889);
        let input = Tensor::random_range(&mut rng, &[SAMPLE_SIZE], MIN_INPUT..MAX_INPUT);

        let op = Sigmoid {
            scale: F32(SCALE as f32),
            tau: TAU,
            log_table: 12,
        };
        let actual = op.f(vec![&input]).data().to_vec();

        let expected: Vec<i32> = sigmoid(&input, SCALE).inner;

        assert_quantized_precision(
            "Sigmoid teleportation",
            &input,
            &actual,
            &expected,
            SCALE,
            (MIN_INPUT, MAX_INPUT),
            WORST_ERROR_BOUND_QUANTIZED,
        );
    }
}
