use crate::{
    ops::{Op, Square},
    tensor::Tensor,
};
use common::parallel::par_enabled;
use rayon::prelude::*;

impl Op for Square {
    #[tracing::instrument(name = "Square::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        if self.scale == 0 {
            // Raw square (no rebase, no clamp): building block for mean-of-squares
            // and friends, whose downstream scaling handles the rebase.
            return inputs[0].pow(2).unwrap();
        }
        // Fused: i64 accumulate, floor-rescale by `1 << scale`, saturating clamp
        // to i32. Replaces Square + its ScalarConstDiv rebase node .
        super::floor_rebase_clamp_i32(&square_acc_i64(inputs[0]), self.scale)
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}

/// Raw i64 accumulation of an element-wise square, `acc = x²` (always fits i64
/// for i32 inputs). The shared pre-rebase value behind [`square_intermediate`]
/// and [`square_remainder`]; the fused [`Square`] sumcheck binds
/// `acc = rescaled·2^S + R`.
fn square_acc_i64(input: &Tensor<i32>) -> Tensor<i64> {
    let data: Vec<i64> = input
        .data()
        .par_iter()
        .with_min_len(par_enabled())
        .map(|&x| (x as i64) * (x as i64))
        .collect();
    Tensor::<i64>::new(Some(&data), input.dims())
        .unwrap_or_else(|e| panic!("square_acc_i64: {e:?}"))
}

/// Re-execute a fused [`Square`] node's pre-clamp rescaled intermediate
/// `rescaled = x² >> scale` (floor) — the saturating-clamp lookup index.
pub fn square_intermediate(op: &Square, inputs: &[&Tensor<i32>]) -> Tensor<i64> {
    super::floor_rebase_i64(&square_acc_i64(inputs[0]), op.scale)
}

/// Re-execute a fused [`Square`] node's rescaling remainder
/// `R = x² mod 2^scale ∈ [0, 2^scale)` .
pub fn square_remainder(op: &Square, inputs: &[&Tensor<i32>]) -> Tensor<i32> {
    super::rebase_remainder_i32(&square_acc_i64(inputs[0]), op.scale)
}
