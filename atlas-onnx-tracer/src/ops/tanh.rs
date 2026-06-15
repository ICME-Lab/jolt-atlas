#[cfg(feature = "fused-ops")]
use crate::tensor::TensorError;
use crate::{
    ops::{Op, Tanh},
    tensor::{self, Tensor},
};
#[cfg(not(feature = "fused-ops"))]
use std::ops::Mul;

impl Op for Tanh {
    #[tracing::instrument(name = "Tanh::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        let input = tensor::ops::nonlinearities::const_div(inputs[0], self.tau as f64);

        #[cfg(feature = "fused-ops")]
        {
            let scale_i = self.scale.0 as i64;
            let lut = generate_tanh_lut(scale_i, self.tau);
            input
                .par_enum_map(|_, a_i| {
                    Ok::<_, TensorError>(tanh_lut_lookup(a_i, &lut))
                })
                .unwrap()
        }
        #[cfg(not(feature = "fused-ops"))]
        {
            // Multiply by tau to undo the const_div, then compute float tanh.
            let teleport_recip = input.mul(self.tau).unwrap();

            crate::tensor::ops::nonlinearities::tanh(&teleport_recip, self.scale.into())
        }
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}

/// Generate a **tanh** lookup table for neural-teleported fixed-point evaluation.
///
/// After `const_div(z, tau)` produces quotient `q = floor(z / tau)`, this table
/// maps `q → round(tanh(q * tau / S) * S)`.  Because the argument to tanh is
/// `q * tau ≈ z`, the output approximates `tanh(z / S) * S` with at most
/// 1 quantization unit of error (from the floor rounding).
///
/// Only the non-negative half is stored; `tanh` is odd so the caller
/// negates the result for negative inputs.
///
/// The table covers quotients `q ∈ [0, 4·S / tau]`; beyond that tanh is
/// saturated (output clamped to `±S`), giving a table `tau`-times smaller
/// than the un-teleported case.
pub fn generate_tanh_lut(scale: i64, tau: i32) -> Vec<i32> {
    let sf = scale as f64;
    let tau_f = tau as f64;
    // tanh saturates at |q * tau / S| ≥ 4, i.e. |q| ≥ 4 * S / tau.
    let table_size = (4 * scale / tau as i64) as usize + 1;

    let mut lut = vec![0i32; table_size];
    for i in 0..table_size {
        let x = i as f64 * tau_f / sf; // q * tau / scale
        lut[i] = (sf * x.tanh()).round() as i32;
    }
    *lut.last_mut().unwrap() = scale as i32;
    lut
}

/// Pure-integer tanh lookup.  
/// Input `a_i` is a fixed-point value at scale `S`.  
/// Returns `round(tanh(a_i / S) * S)` using odd symmetry + LUT.
#[inline]
pub fn tanh_lut_lookup(a_i: i32, lut: &[i32]) -> i32 {
    let abs_val = (a_i.unsigned_abs() as usize).min(lut.len() - 1);
    let magnitude = lut[abs_val];
    if a_i >= 0 { magnitude } else { -magnitude }
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
