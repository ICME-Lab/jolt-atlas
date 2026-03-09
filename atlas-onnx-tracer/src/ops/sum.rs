use tract_onnx::prelude::tract_itertools::Itertools;

use crate::{
    ops::{Op, Sum},
    tensor::{Tensor, TensorError},
};

impl Op for Sum {
    #[tracing::instrument(name = "Sum::f", skip_all)]
    fn f(&self, inputs: Vec<&Tensor<i32>>) -> Tensor<i32> {
        #[cfg(feature = "fused-ops")]
        {
            sum_axes_i32(inputs[0], &self.axes).unwrap()
        }
        #[cfg(not(feature = "fused-ops"))]
        {
            crate::tensor::ops::sum_axes(inputs[0], &self.axes).unwrap()
        }
    }
}

/// Sum over specified axes of an i32 tensor, accumulating in i64 to prevent overflow,
#[tracing::instrument(name = "tensor::ops::sum_axes_i32", skip_all)]
pub fn sum_axes_i32(a: &Tensor<i32>, axes: &[usize]) -> Result<Tensor<i32>, TensorError> {
    if axes.is_empty() {
        return Ok(a.clone());
    }

    let mut new_dims = vec![];
    for i in 0..a.dims().len() {
        if !axes.contains(&i) {
            new_dims.push(a.dims()[i]);
        } else {
            new_dims.push(1);
        }
    }

    let res = Tensor::new(None, &new_dims)?;

    let cartesian_coord = new_dims
        .iter()
        .map(|x| 0..*x)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    let res = res.par_enum_map(|i, _: i32| {
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
        // Accumulate in i64
        let mut acc: i64 = 0;
        let _ = slice.map(|v| acc += v as i64);

        // Saturate instead of wrapping — prevents sign-flip corruption
        Ok(acc.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
    })?;

    Ok(res)
}
