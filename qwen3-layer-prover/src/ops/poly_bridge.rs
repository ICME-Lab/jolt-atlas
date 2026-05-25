use joltworks::field::JoltField;

use crate::{
    claim::{Poly, Shape},
    error::Result,
};

pub(crate) fn logical_i32_values<F: JoltField, C>(
    poly: &Poly<F, C>,
    shape: &Shape,
) -> Result<Vec<i32>> {
    let padded_dims = shape.padded_power_of_two().0;
    let strides = row_major_strides(shape.dims());
    let padded_strides = row_major_strides(&padded_dims);
    let mut out = Vec::with_capacity(shape.numel());
    for flat in 0..shape.numel() {
        let mut padded_flat = 0;
        for (dim, (&stride, &padded_stride)) in strides.iter().zip(&padded_strides).enumerate() {
            let coord = (flat / stride) % shape.dims()[dim];
            padded_flat += coord * padded_stride;
        }
        out.push(poly.data.get_coeff_i64(padded_flat) as i32);
    }
    Ok(out)
}

fn row_major_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; dims.len()];
    let mut stride = 1;
    for (idx, &dim) in dims.iter().enumerate().rev() {
        strides[idx] = stride;
        stride *= dim;
    }
    strides
}
