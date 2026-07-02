use crate::{
    ops::{Cube, Op},
    tensor::Tensor,
};
use common::parallel::par_enabled;
use rayon::prelude::*;

impl Op for Cube {
    #[tracing::instrument(name = "Cube::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        if self.scale == 0 {
            // Raw cube (no rebase, no clamp): building block path.
            return inputs[0].pow(3).unwrap();
        }
        // Fused: i64 accumulate, floor-rescale by `1 << (scale*2)`, saturating
        // clamp to i32.
        super::floor_rebase_clamp_i32(&cube_acc_i64(inputs[0]), cube_rebase_bits(self))
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}

/// Rebase width (in bits) for a cube: `x³` is at scale `3S` and rebases to `S`,
/// so the divisor is `2^(2·scale)`.
fn cube_rebase_bits(op: &Cube) -> i32 {
    op.scale * 2
}

/// Raw i64 accumulation of an element-wise cube, `acc = x³`. The shared
/// pre-rebase value behind [`cube_intermediate`] and [`cube_remainder`]; the
/// fused [`Cube`] sumcheck binds `acc = rescaled·2^(2S) + R`.
///
/// `x³` overflows i64 for `|x| > 2^21`; such models are out of range for the
/// fused path (see [`Cube::f`]).
fn cube_acc_i64(input: &Tensor<i32>) -> Tensor<i64> {
    let data: Vec<i64> = input
        .data()
        .par_iter()
        .with_min_len(par_enabled())
        .map(|&x| {
            let x = x as i64;
            x * x * x
        })
        .collect();
    Tensor::<i64>::new(Some(&data), input.dims()).unwrap_or_else(|e| panic!("cube_acc_i64: {e:?}"))
}

/// Re-execute a fused [`Cube`] node's pre-clamp rescaled intermediate
/// `rescaled = x³ >> (2·scale)` (floor) — the saturating-clamp lookup index.
pub fn cube_intermediate(op: &Cube, inputs: &[&Tensor<i32>]) -> Tensor<i64> {
    super::floor_rebase_i64(&cube_acc_i64(inputs[0]), cube_rebase_bits(op))
}

/// Re-execute a fused [`Cube`] node's rescaling remainder
/// `R = x³ mod 2^(2·scale) ∈ [0, 2^(2·scale))`.
pub fn cube_remainder(op: &Cube, inputs: &[&Tensor<i32>]) -> Tensor<i32> {
    super::rebase_remainder_i32(&cube_acc_i64(inputs[0]), cube_rebase_bits(op))
}
