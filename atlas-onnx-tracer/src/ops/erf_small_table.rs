use crate::{
    ops::{ErfSmallTable, Op},
    tensor::{self, Tensor},
    utils::quantize::scale_to_multiplier,
};
use common::consts::{ACTIVATION_BOUND, SCALE_12};

impl Op for ErfSmallTable {
    #[tracing::instrument(name = "ErfSmallTable::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        let clamped = tensor::ops::nonlinearities::clamp(inputs[0], ACTIVATION_BOUND);
        tensor::ops::nonlinearities::erffunc(&clamped, scale_to_multiplier(SCALE_12 as i32))
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::ErfSmallTable;
    use crate::{
        ops::Op,
        tensor::{ops::nonlinearities::erffunc, Tensor},
        utils::precision::assert_quantized_precision,
    };
    use common::consts::{ACTIVATION_BOUND, SCALE_12};
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_erf_small_table_precision_stats() {
        let scale: f64 = (1u32 << SCALE_12) as f64;
        const SAMPLE_SIZE: usize = 1 << 14;
        const MIN_INPUT: i32 = -(1 << 17);
        const MAX_INPUT: i32 = 1 << 17;
        const WORST_ERROR_BOUND_QUANTIZED: i32 = 8;

        let mut rng = StdRng::seed_from_u64(0x88C);
        let input = Tensor::random_range(&mut rng, &[SAMPLE_SIZE], MIN_INPUT..MAX_INPUT);

        let op = ErfSmallTable;
        let actual = op.f(vec![&input]).data().to_vec();

        // Reference: clamp to [-8, 8) in real units (matches ACTIVATION_BOUND at
        // scale=SCALE_12, i.e. [-32768, 32768)), then apply erf at full precision,
        // unlike the wider, lossy-`tau`-divided `Erf` op.
        let bound = 1i32 << ACTIVATION_BOUND;
        let clamped = input.map(|v| v.clamp(-bound, bound - 1));
        let expected: Vec<i32> = erffunc(&clamped, scale).inner;

        assert_quantized_precision(
            "ErfSmallTable clamped",
            &input,
            &actual,
            &expected,
            scale,
            (MIN_INPUT, MAX_INPUT),
            WORST_ERROR_BOUND_QUANTIZED,
        );
    }
}
