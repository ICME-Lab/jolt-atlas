use crate::{
    ops::{MeanOfSquares, Op},
    tensor::{Tensor, TensorError},
};
use tract_onnx::prelude::tract_itertools::Itertools;

impl Op for MeanOfSquares {
    #[tracing::instrument(name = "MeanOfSquares::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        // Fused: i64 accumulate Σx² over the reduced axes, floor-divide by
        // `N·2^S` (mean rebase), saturating clamp to i32. Replaces the
        // Square → Sum → Div(N) decomposition in one node .
        super::clamp_to_i32(&mos_intermediate(self, &inputs))
    }
}

/// The fused mean-of-squares divisor `D = N · 2^S`, where `N` (= `op.count`) is
/// the product of the reduced-axis sizes (the mean denominator) and `2^S` is the
/// rescale. `count` is stored on the operator so the verifier can recover `D`
/// from the node alone (it has neither the input tensor nor its dims).
pub fn mos_divisor(op: &MeanOfSquares) -> i64 {
    (1i64 << op.scale) * op.count as i64
}

/// Raw i64 accumulation of the mean-of-squares: `acc[h] = Σ_j x[h,j]²` over the
/// reduced `axes` (no division). The shared pre-rebase value behind
/// [`mos_intermediate`] and [`mos_remainder`]; the fused [`MeanOfSquares`]
/// reduction sumcheck binds `acc = rescaled·D + R` with `D = N·2^S`.
///
/// Each `x²` term fits i64 (i32² ≤ 2^62) and the sum of the reduced axis stays
/// well within i64 for supported hidden sizes.
#[tracing::instrument(name = "tensor::ops::mos_acc_i64", skip_all)]
pub fn mos_acc_i64(a: &Tensor<i32>, axes: &[usize]) -> Result<Tensor<i64>, TensorError> {
    let mut new_dims = vec![];
    for i in 0..a.dims().len() {
        if !axes.contains(&i) {
            new_dims.push(a.dims()[i]);
        } else {
            new_dims.push(1);
        }
    }

    let res = Tensor::<i64>::new(None, &new_dims)?;

    let cartesian_coord = new_dims
        .iter()
        .map(|x| 0..*x)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    res.par_enum_map(|i, _: i64| {
        let coord = cartesian_coord[i].clone();
        let mut prod_dims = vec![];
        for (j, c) in coord.iter().enumerate() {
            if axes.contains(&j) {
                prod_dims.push(0..a.dims()[j]);
            } else {
                prod_dims.push(*c..*c + 1);
            }
        }

        let slice = a.get_slice(&prod_dims)?;
        let mut acc: i64 = 0;
        let _ = slice.map(|v| {
            let v64 = v as i64;
            acc += v64 * v64;
        });
        Ok::<_, TensorError>(acc)
    })
}

/// Re-execute a fused [`MeanOfSquares`] node's pre-clamp rescaled intermediate
/// `rescaled = (Σx²) / (N·2^S)` (floor; `Σx² ≥ 0` so floor = truncation) — the
/// saturating-clamp lookup index.
pub fn mos_intermediate(op: &MeanOfSquares, inputs: &[&Tensor<i32>]) -> Tensor<i64> {
    let acc =
        mos_acc_i64(inputs[0], &op.axes).unwrap_or_else(|e| panic!("mos_intermediate: {e:?}"));
    let divisor = mos_divisor(op);
    let data: Vec<i64> = acc.data().iter().map(|&v| v.div_euclid(divisor)).collect();
    Tensor::<i64>::new(Some(&data), acc.dims())
        .unwrap_or_else(|e| panic!("mos_intermediate: {e:?}"))
}

/// Re-execute a fused [`MeanOfSquares`] node's rescaling remainder
/// `R = (Σx²) mod (N·2^S) ∈ [0, N·2^S)` . The fused reduction binds
/// `Σx² = rescaled·D + R`; a `R < D` range check bounds it.
pub fn mos_remainder(op: &MeanOfSquares, inputs: &[&Tensor<i32>]) -> Tensor<i32> {
    let acc = mos_acc_i64(inputs[0], &op.axes).unwrap_or_else(|e| panic!("mos_remainder: {e:?}"));
    let divisor = mos_divisor(op);
    let data: Vec<i32> = acc
        .data()
        .iter()
        .map(|&v| v.rem_euclid(divisor) as i32)
        .collect();
    Tensor::<i32>::new(Some(&data), acc.dims()).unwrap_or_else(|e| panic!("mos_remainder: {e:?}"))
}

/// [`mos_intermediate`] and [`mos_remainder`] from one accumulation pass.
pub fn mos_intermediate_and_remainder(
    op: &MeanOfSquares,
    inputs: &[&Tensor<i32>],
) -> (Tensor<i64>, Tensor<i32>) {
    let acc = mos_acc_i64(inputs[0], &op.axes)
        .unwrap_or_else(|e| panic!("mos_intermediate_and_remainder: {e:?}"));
    let divisor = mos_divisor(op);
    let quotient: Vec<i64> = acc.data().iter().map(|&v| v.div_euclid(divisor)).collect();
    let remainder: Vec<i32> = acc
        .data()
        .iter()
        .map(|&v| v.rem_euclid(divisor) as i32)
        .collect();
    (
        Tensor::<i64>::new(Some(&quotient), acc.dims())
            .unwrap_or_else(|e| panic!("mos_intermediate_and_remainder: {e:?}")),
        Tensor::<i32>::new(Some(&remainder), acc.dims())
            .unwrap_or_else(|e| panic!("mos_intermediate_and_remainder: {e:?}")),
    )
}
