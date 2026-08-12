use crate::{
    ops::{Op, Sigmoid},
    tensor::{self, Tensor},
    utils::quantize::scale_to_multiplier,
};

impl Op for Sigmoid {
    #[tracing::instrument(name = "Sigmoid::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        let activation_bound = self.scale as usize + 3;
        let clamped = tensor::ops::nonlinearities::clamp(inputs[0], activation_bound);
        tensor::ops::nonlinearities::sigmoid(&clamped, scale_to_multiplier(self.scale))
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
        utils::precision::assert_quantized_precision,
    };
    use common::consts::{ACTIVATION_BOUND, MODEL_SCALE};
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn test_sigmoid_precision_stats() {
        let scale: f64 = (1u32 << MODEL_SCALE) as f64;
        const SAMPLE_SIZE: usize = 1 << 14;
        const MIN_INPUT: i32 = -(1 << 17);
        const MAX_INPUT: i32 = 1 << 17;
        const WORST_ERROR_BOUND_QUANTIZED: i32 = 8;

        let mut rng = StdRng::seed_from_u64(0x889);
        let input = Tensor::random_range(&mut rng, &[SAMPLE_SIZE], MIN_INPUT..MAX_INPUT);

        let op = Sigmoid {
            scale: MODEL_SCALE as i32,
        };
        let actual = op.f(vec![&input]).data().to_vec();

        let bound = 1i32 << ACTIVATION_BOUND;
        let clamped = input.map(|v| v.clamp(-bound, bound - 1));
        let expected: Vec<i32> = sigmoid(&clamped, scale).inner;

        assert_quantized_precision(
            "Sigmoid clamped",
            &input,
            &actual,
            &expected,
            scale,
            (MIN_INPUT, MAX_INPUT),
            WORST_ERROR_BOUND_QUANTIZED,
        );
    }
}
