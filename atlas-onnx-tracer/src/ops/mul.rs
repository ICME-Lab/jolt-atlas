use crate::{
    ops::{Mul, Op},
    tensor::Tensor,
};

impl Op for Mul {
    #[tracing::instrument(name = "Mul::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        if self.scale == 0 {
            // Raw element-wise product (no rebase, no clamp): the building-block
            // multiply used inside op decompositions, where one operand has been
            // pre-divided so the result is already at the target scale and small
            // enough not to overflow `i32`.
            return crate::tensor::ops::mult(&inputs).unwrap();
        }
        // Fused: i64 accumulate, floor-rescale by `1 << scale`, saturating clamp
        // to i32. Replaces Mul + its ScalarConstDiv rebase node .
        super::floor_rebase_clamp_i32(&mul_acc_i64(&inputs), self.scale)
    }

    fn requires_shape_equality(&self) -> bool {
        true
    }
}

/// Raw i64 accumulation of a (broadcast) binary multiply, `acc = left · right`.
///
/// This is the shared pre-rebase value behind [`mul_intermediate`] and
/// [`mul_remainder`]; the fused [`Mul`] sumcheck binds `acc = rescaled·2^S + R`.
fn mul_acc_i64(inputs: &[&Tensor<i32>]) -> Tensor<i64> {
    let [left, right] = inputs[..] else {
        panic!("Mul (fused) expects two operands, got {}", inputs.len())
    };
    super::sat_accumulate_pair(left, right, "Mul", |a, b| a * b)
}

/// Re-execute a fused [`Mul`] node's pre-clamp rescaled intermediate
/// `rescaled = (left·right) >> scale` (floor) — the saturating-clamp lookup
/// index. Analogous to `einsum_intermediate`; recovered without a trace change.
pub fn mul_intermediate(op: &Mul, inputs: &[&Tensor<i32>]) -> Tensor<i64> {
    super::floor_rebase_i64(&mul_acc_i64(inputs), op.scale)
}

/// Re-execute a fused [`Mul`] node's rescaling remainder
/// `R = (left·right) mod 2^scale ∈ [0, 2^scale)` .
pub fn mul_remainder(op: &Mul, inputs: &[&Tensor<i32>]) -> Tensor<i32> {
    super::rebase_remainder_i32(&mul_acc_i64(inputs), op.scale)
}
