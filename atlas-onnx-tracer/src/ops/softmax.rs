use crate::{
    ops::{Op, SoftmaxAxes},
    tensor::Tensor,
};

impl Op for SoftmaxAxes {
    #[tracing::instrument(name = "SoftmaxAxes::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        crate::tensor::ops::nonlinearities::softmax_axes(inputs[0], self.scale.into(), &[self.axes])
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}

#[allow(clippy::needless_range_loop)]
/// Generate an **exp** lookup table for an arbitrary fixed-point scale `S`.
///
/// Entry `i` = `round(exp(-i / S) * S)` for `i ∈ [0, table_size)`.
/// Beyond the table, `exp(-i/S)*S` rounds to 0 so we clamp to 0.
///
/// The table is used by `softmax_last_axis`: after centering (subtracting
/// max), every element is ≤ 0, so the index is `−centered ≥ 0`.
pub fn generate_exp_lut(scale: i64) -> Vec<i32> {
    // exp(-i/S)*S < 0.5 when i > S * ln(2S).
    // Add a generous margin so the caller never needs to bounds-check.
    let sf = scale as f64;
    let needed = (sf * (2.0 * sf).ln()).ceil() as usize + 2;
    // Round up to next power-of-two for tidy table sizes.
    let table_size = needed.next_power_of_two();

    let mut lut = vec![0i32; table_size];
    for i in 0..table_size {
        let val = (sf * (-(i as f64) / sf).exp()).round();
        lut[i] = val.max(0.0) as i32;
    }
    lut
}

/// Pure-integer exp lookup: `exp(z_q / S) * S` for `z_q ≤ 0`.
#[inline]
pub fn exp_lut_lookup(z_q: i32, lut: &[i32]) -> i32 {
    debug_assert!(z_q <= 0, "exp_lut_lookup requires z_q <= 0, got {z_q}");
    let idx = (-z_q) as usize;
    if idx < lut.len() { lut[idx] } else { 0 }
}
