use rayon::prelude::*;
use tract_onnx::prelude::tract_itertools::Itertools;

use crate::{
    ops::{Op, Sum},
    tensor::{Tensor, TensorError},
};

impl Op for Sum {
    #[tracing::instrument(name = "Sum::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        sum_axes_i32(inputs[0], &self.axes).unwrap()
    }
}

/// Sum over specified axes, accumulating in `i64` to avoid overflow (the
/// pre-clamp intermediate). Each output element is the exact integer sum over
/// the reduced axes; summed axes collapse to length 1.
///
/// This is the polynomial the saturating-clamp proof reasons about: the proof
/// system recovers it to bridge `output = SatClamp(acc)`.
#[tracing::instrument(name = "tensor::ops::sum_axes_i64", skip_all)]
pub fn sum_axes_i64(a: &Tensor<i32>, axes: &[usize]) -> Result<Tensor<i64>, TensorError> {
    if axes.is_empty() {
        let data: Vec<i64> = a.data().par_iter().map(|&v| v as i64).collect();
        return Tensor::new(Some(&data), a.dims());
    }

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
        let _ = slice.map(|v| acc += v as i64);
        Ok(acc)
    })
}

/// Saturating sum over specified axes: [`sum_axes_i64`] clamped back to `i32`.
///
/// Saturating (rather than wrapping) prevents sign-flip corruption on overflow.
#[tracing::instrument(name = "tensor::ops::sum_axes_i32", skip_all)]
pub fn sum_axes_i32(a: &Tensor<i32>, axes: &[usize]) -> Result<Tensor<i32>, TensorError> {
    let acc = sum_axes_i64(a, axes)?;
    let data: Vec<i32> = acc
        .data()
        .par_iter()
        .map(|&v| v.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
        .collect();
    Tensor::new(Some(&data), acc.dims())
}
