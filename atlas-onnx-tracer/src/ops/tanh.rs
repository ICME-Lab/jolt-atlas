use crate::{
    ops::{Op, Tanh},
    tensor::{self, Tensor},
    utils::quantize::scale_to_multiplier,
};
use common::consts::{ACTIVATION_BOUND, SCALE_8};

impl Op for Tanh {
    #[tracing::instrument(name = "Tanh::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        debug_assert_eq!(
            self.scale, SCALE_8 as i32,
            "Tanh only supports scale={SCALE_8}"
        );
        let clamped = tensor::ops::nonlinearities::clamp(inputs[0], ACTIVATION_BOUND);
        tensor::ops::nonlinearities::tanh(&clamped, scale_to_multiplier(SCALE_8 as i32))
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::Tanh;
    use crate::{
        ops::Op,
        tensor::{Tensor, ops::nonlinearities::tanh},
        utils::precision::assert_quantized_precision,
    };
    use common::consts::{ACTIVATION_BOUND, SCALE_8};
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn test_tanh_precision_stats() {
        let scale: f64 = (1u32 << SCALE_8) as f64;
        const SAMPLE_SIZE: usize = 1 << 14;
        const MIN_INPUT: i32 = -(1 << 17);
        const MAX_INPUT: i32 = 1 << 17;
        const WORST_ERROR_BOUND_QUANTIZED: i32 = 8;

        let mut rng = StdRng::seed_from_u64(0x88A);
        let input = Tensor::random_range(&mut rng, &[SAMPLE_SIZE], MIN_INPUT..MAX_INPUT);

        let op = Tanh {
            scale: SCALE_8 as i32,
        };
        let actual = op.f(vec![&input]).data().to_vec();

        let bound = 1i32 << ACTIVATION_BOUND;
        let clamped = input.map(|v| v.clamp(-bound, bound - 1));
        let expected: Vec<i32> = tanh(&clamped, scale).inner;

        assert_quantized_precision(
            "Tanh clamped",
            &input,
            &actual,
            &expected,
            scale,
            (MIN_INPUT, MAX_INPUT),
            WORST_ERROR_BOUND_QUANTIZED,
        );
    }
}
