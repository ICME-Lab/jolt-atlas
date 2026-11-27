use super::{projection_shape, Projection, Shape};
use rayon::prelude::*;

type Witness = (Vec<i32>, Vec<u32>);
const LIBRARY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/projection.metallib"));

unsafe extern "C" {
    fn atlas_metal_projection(
        library: *const u8,
        library_size: usize,
        input: *const i32,
        weights: *const i32,
        bias: *const i32,
        rows: u32,
        inner: u32,
        cols: u32,
        bits: u32,
        xd: u32,
        wd: u32,
        output: *mut i32,
        remainder: *mut u32,
    ) -> bool;
}

fn supported(s: Shape, cols: usize, bits: u32) -> bool {
    // Signed 7-bit products sum exactly in FP32 for at most 4096 terms.
    s.cols <= 4096
        && bits <= 30
        && s.cols % 16 == 0
        && cols % 16 == 0
        && s.cells().saturating_mul(cols) >= (1 << 28)
        && [s.cells(), s.cols * cols, s.rows * cols]
            .iter()
            .all(|&n| n <= (u32::MAX / 5) as usize)
}

pub(super) fn project(x: &[i32], s: Shape, bits: u32, p: Projection<'_>) -> Option<Witness> {
    let shape = projection_shape(s, p).ok()?;
    if !supported(s, shape.cols, bits) || x.len() != s.cells() {
        return None;
    }
    let max = |v: &[i32]| {
        v.par_iter()
            .with_min_len(16384)
            .map(|v| v.unsigned_abs())
            .max()
            .unwrap_or(0)
    };
    let (x_max, w_max) = rayon::join(|| max(x), || max(p.weights));
    let bound = |x| {
        let mut carry = u64::from(x);
        let mut digits = 1;
        while carry > 63 {
            carry = (carry + 64) >> 7;
            digits += 1;
        }
        // Bound the sum of absolute digit contributions before cancellation.
        let bound = u128::from(x) + 128 * ((1u128 << (7 * (digits - 1))) - 1) / 127;
        (digits, bound)
    };
    let (xd, xb) = bound(x_max);
    let (wd, wb) = bound(w_max);
    let bias = u128::from(p.bias.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0)) << bits;
    if xb * wb * s.cols as u128 + bias > i64::MAX as u128 {
        return None;
    }
    let (mut output, mut remainder) = (vec![0; shape.cells()], vec![0; shape.cells()]);
    // All input and output buffers have the stated dimensions and live through the synchronous call.
    let ok = unsafe {
        atlas_metal_projection(
            LIBRARY.as_ptr(),
            LIBRARY.len(),
            x.as_ptr(),
            p.weights.as_ptr(),
            p.bias.as_ptr(),
            s.rows as u32,
            s.cols as u32,
            shape.cols as u32,
            bits,
            xd,
            wd,
            output.as_mut_ptr(),
            remainder.as_mut_ptr(),
        )
    };
    ok.then_some((output, remainder))
}

pub(super) fn grouped<'a>(
    x: &[i32],
    s: Shape,
    bits: u32,
    specs: impl Iterator<Item = Projection<'a>> + Clone,
) -> Option<Vec<Witness>> {
    let cols = specs.clone().map(|p| p.bias.len()).sum::<usize>();
    if !supported(s, cols, bits) {
        return None;
    }
    let mut start = 0;
    let mut ranges = Vec::new();
    for p in specs.clone() {
        projection_shape(s, p).ok()?;
        ranges.push(start..start + p.bias.len());
        start += p.bias.len();
    }
    let weights: Vec<_> = specs
        .clone()
        .flat_map(|p| p.weights.iter().copied())
        .collect();
    let bias: Vec<_> = specs.flat_map(|p| p.bias.iter().copied()).collect();
    let (values, remainders) = project(
        x,
        s,
        bits,
        Projection {
            weights: &weights,
            bias: &bias,
        },
    )?;
    Some(
        ranges
            .into_par_iter()
            .map(|range| {
                let output = values
                    .chunks_exact(cols)
                    .flat_map(|row| row[range.clone()].iter().copied())
                    .collect();
                let remainder = remainders
                    .chunks_exact(cols)
                    .flat_map(|row| row[range.clone()].iter().copied())
                    .collect();
                (output, remainder)
            })
            .collect(),
    )
}
