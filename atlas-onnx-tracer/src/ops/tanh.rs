#[cfg(feature = "fused-ops")]
use crate::tensor::TensorError;
use crate::{
    ops::{Op, Tanh},
    tensor::{self, Tensor},
};

impl Op for Tanh {
    #[tracing::instrument(name = "Tanh::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        let input = tensor::ops::nonlinearities::const_div(inputs[0], self.tau as f64);
        #[cfg(feature = "fused-ops")]
        {
            let scale_i = self.scale.0 as i64;
            let lut = generate_tanh_lut(scale_i);
            input
                .par_enum_map(|_, a_i| {
                    Ok::<_, TensorError>(tanh_lut_lookup(a_i, scale_i as i32, &lut))
                })
                .unwrap()
        }
        #[cfg(not(feature = "fused-ops"))]
        {
            crate::tensor::ops::nonlinearities::tanh(&input, self.scale.into())
        }
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
