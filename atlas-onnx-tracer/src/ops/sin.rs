use crate::{
    model::consts::FOUR_PI_APPROX,
    ops::{Op, Sin},
    tensor::{self, Tensor},
};

impl Op for Sin {
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        let remainder = tensor::ops::nonlinearities::const_rem(inputs[0], FOUR_PI_APPROX);
        tensor::ops::nonlinearities::sin(&remainder, self.scale.into())
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::Sin;
    use crate::{
        ops::Op,
        tensor::{Tensor, ops::nonlinearities::sin},
        utils::{f32::F32, precision::assert_quantized_precision},
    };
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn test_sin_precision_stats() {
        const SCALE: f64 = 256.0;
        const SAMPLE_SIZE: usize = 1 << 14;
        const MIN_INPUT: i32 = -(1 << 20);
        const MAX_INPUT: i32 = 1 << 20;
        const WORST_ERROR_BOUND_QUANTIZED: i32 = 8;

        let mut rng = StdRng::seed_from_u64(0x517_517);
        let input = Tensor::random_range(&mut rng, &[SAMPLE_SIZE], MIN_INPUT..MAX_INPUT);

        let op = Sin {
            scale: F32(SCALE as f32),
        };
        let actual = op.f(vec![&input]).data().to_vec();

        let expected: Vec<i32> = sin(&input, SCALE).inner;

        assert_quantized_precision(
            "Sin teleportation",
            &input,
            &actual,
            &expected,
            SCALE,
            (MIN_INPUT, MAX_INPUT),
            WORST_ERROR_BOUND_QUANTIZED,
        );
    }
}
