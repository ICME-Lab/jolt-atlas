use crate::{
    ops::{Op, Tanh},
    tensor::{Tensor, ops::nonlinearities::const_div},
};
use std::ops::Mul;

impl Op for Tanh {
    #[tracing::instrument(name = "Tanh::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        let input = inputs[0];

        // TODO: #192
        // let scale_i = self.scale.0 as i64;
        // let lut = generate_tanh_lut(scale_i);
        // input
        //     .par_enum_map(|_, a_i| {
        //         Ok::<_, TensorError>(tanh_lut_lookup(a_i, scale_i as i32, &lut))
        //     })
        //     .unwrap()

        let lower_bound = -(1 << (self.log_table - 1));
        let upper_bound = (1 << (self.log_table - 1)) - 1;
        let input = const_div(input, self.tau as f64);
        let input = input.mul(self.tau).unwrap();
        let input = clamp_tensor(&input, lower_bound, upper_bound);
        crate::tensor::ops::nonlinearities::tanh(&input, self.scale.into())
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}

/// Generate a **tanh** lookup table for an arbitrary fixed-point scale `S`.
///
/// Entry `i` = `round(tanh(i / S) * S)` for `i ∈ [0, table_size)`.
/// Only the non-negative half is stored; `tanh` is odd so the caller
/// negates the result for negative inputs.
///
/// `tanh(x)` saturates to ±1 for `|x| > ~4`, so the table covers
/// `[0, 4·S]` — beyond that the output is `±S`.
pub fn generate_tanh_lut(scale: i64) -> Vec<i32> {
    let sf = scale as f64;
    // tanh(4) ≈ 0.9993 — close enough to 1.0 for any practical scale.
    let table_size = (4 * scale) as usize + 1;

    let mut lut = vec![0i32; table_size];
    for i in 0..table_size {
        let x = i as f64 / sf;
        lut[i] = (sf * x.tanh()).round() as i32;
    }
    lut
}

/// Pure-integer tanh lookup.  
/// Input `a_i` is a fixed-point value at scale `S`.  
/// Returns `round(tanh(a_i / S) * S)` using odd symmetry + LUT.
#[inline]
pub fn tanh_lut_lookup(a_i: i32, scale: i32, lut: &[i32]) -> i32 {
    let abs_val = a_i.unsigned_abs() as usize;
    let magnitude = if abs_val < lut.len() {
        lut[abs_val]
    } else {
        scale // saturates to ±1
    };
    if a_i >= 0 { magnitude } else { -magnitude }
}

/// Clamp a tensor to a given range.
pub fn clamp_tensor(t: &Tensor<i32>, lower_bound: i32, upper_bound: i32) -> Tensor<i32> {
    use common::parallel::par_enabled;
    use rayon::prelude::*;

    let data: Vec<i32> = t
        .data()
        .par_iter()
        .with_min_len(par_enabled())
        .map(|&v| v.clamp(lower_bound, upper_bound))
        .collect();
    crate::tensor::Tensor::new(Some(&data), t.dims())
        .unwrap_or_else(|_| panic!("clamp_to_i32: Tensor::new"))
}

#[cfg(test)]
mod tests {
    use super::Tanh;
    use crate::{
        ops::Op,
        tensor::{Tensor, ops::nonlinearities::tanh},
        utils::{f32::F32, precision::assert_quantized_precision},
    };
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn test_tanh_precision_stats() {
        const SCALE: f64 = 256.0;
        const TAU: i32 = 2;
        const SAMPLE_SIZE: usize = 1 << 14;
        const MIN_INPUT: i32 = -(1 << 14);
        const MAX_INPUT: i32 = 1 << 14;
        const WORST_ERROR_BOUND_QUANTIZED: i32 = 8;

        let mut rng = StdRng::seed_from_u64(0x88A);
        let input = Tensor::random_range(&mut rng, &[SAMPLE_SIZE], MIN_INPUT..MAX_INPUT);

        let op = Tanh {
            scale: F32(SCALE as f32),
            tau: TAU,
            log_table: 12,
        };
        let actual = op.f(vec![&input]).data().to_vec();

        let expected: Vec<i32> = tanh(&input, SCALE).inner;

        assert_quantized_precision(
            "Tanh teleportation",
            &input,
            &actual,
            &expected,
            SCALE,
            (MIN_INPUT, MAX_INPUT),
            WORST_ERROR_BOUND_QUANTIZED,
        );
    }
}
