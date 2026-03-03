use crate::{
    model::consts::EIGHT_PI_APPROX,
    ops::{Cos, Op},
    tensor::{self, Tensor},
};

impl Op for Cos {
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        let remainder = tensor::ops::nonlinearities::const_rem(inputs[0], EIGHT_PI_APPROX);
        tensor::ops::nonlinearities::cos(&remainder, self.scale.into())
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::Cos;
    use crate::{ops::Op, tensor::Tensor, utils::f32::F32};
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn test_cos_teleportation_random_precision_stats() {
        const SCALE: f64 = 128.0;
        const SAMPLE_SIZE: usize = 1 << 14;
        const MIN_INPUT: i32 = -(1 << 20);
        const MAX_INPUT: i32 = 1 << 20;
        const WORST_ERROR_BOUND_QUANTIZED: i32 = 8;

        let mut rng = StdRng::seed_from_u64(0x888);
        let input = Tensor::random_range(&mut rng, &[SAMPLE_SIZE], MIN_INPUT..MAX_INPUT);

        let op = Cos {
            scale: F32(SCALE as f32),
        };
        let actual = op.f(vec![&input]).data().to_vec();

        let expected: Vec<i32> = input
            .iter()
            .map(|&x| {
                let x_real = x as f64 / SCALE;
                (SCALE * x_real.cos()).round() as i32
            })
            .collect();

        let mut total_abs_error_quantized: i64 = 0;
        let mut worst_error_quantized: i32 = 0;
        let mut worst_error_index: usize = 0;

        for (idx, (actual_i, expected_i)) in actual.iter().zip(expected.iter()).enumerate() {
            let err = (actual_i - expected_i).abs();
            total_abs_error_quantized += err as i64;
            if err > worst_error_quantized {
                worst_error_quantized = err;
                worst_error_index = idx;
            }
        }

        let avg_abs_error_quantized = total_abs_error_quantized as f64 / SAMPLE_SIZE as f64;
        let avg_abs_error_real = avg_abs_error_quantized / SCALE;
        let worst_error_real = worst_error_quantized as f64 / SCALE;

        println!(
            "Cos teleportation precision stats (n={SAMPLE_SIZE}, input_range=[{MIN_INPUT}, {MAX_INPUT})):\
            \nAverage error (Quantized)={avg_abs_error_quantized:.4},\
            \nWorst error (Quantized)={worst_error_quantized},\
            \nAverage error (Real)={avg_abs_error_real:.6},\
            \nWorst error (Real)={worst_error_real:.6},\
            \nWorst case=(idx={worst_error_index}, input={}, actual={}, expected={})",
            input[worst_error_index], actual[worst_error_index], expected[worst_error_index]
        );

        assert!(
            worst_error_quantized <= WORST_ERROR_BOUND_QUANTIZED,
            "worst quantized error too high: worst={worst_error_quantized}, \
            \nbound={WORST_ERROR_BOUND_QUANTIZED}, avg={avg_abs_error_quantized:.4}, \
            \ninput={}, actual={}, expected={}",
            input[worst_error_index],
            actual[worst_error_index],
            expected[worst_error_index]
        );
    }
}
