use jolt_field::Ring;
use rayon::prelude::*;

use super::{Projection, Rotary};
use crate::protocol::{
    field::F,
    opening::{Layout, Packed, Request},
    range::{Ranges, Width, I32},
    sumcheck::{Column, Session},
    tensor::{bind_rows, equality, inner, weights, Claim, Shape},
    Result,
};

pub(crate) struct Rounded {
    pub values: Vec<i32>,
    pub(crate) remainder: Vec<i32>,
}

impl Rounded {
    pub(crate) fn make(
        shape: Shape,
        bits: usize,
        compute: impl Fn(usize, usize) -> i128 + Sync,
    ) -> Result<Self> {
        let mut values = vec![0; shape.len()];
        let mut remainder = vec![0; shape.len()];
        values
            .par_chunks_mut(shape.width)
            .zip(remainder.par_chunks_mut(shape.width))
            .enumerate()
            .try_for_each(|(row, (values, remainder))| {
                for (col, (value, rem)) in values.iter_mut().zip(remainder).enumerate() {
                    let raw = compute(row, col);
                    *value = i32::try_from(raw >> bits).map_err(|_| "fixed-point overflow")?;
                    *rem = (raw & ((1 << bits) - 1)) as i32;
                }
                Ok(())
            })?;
        Ok(Self { values, remainder })
    }
}

pub(crate) struct Rounding {
    pub shape: Shape,
    bits: usize,
    ranges: Ranges<2>,
}

impl Rounding {
    pub(crate) fn new(layout: &mut Layout, shape: Shape, bits: usize) -> Self {
        let ranges = Ranges::new(layout, shape, [I32, Width::unsigned(bits)]);
        Self {
            shape,
            bits,
            ranges,
        }
    }

    pub(crate) fn execute(
        &self,
        packed: &mut Packed<'_>,
        compute: impl Fn(usize, usize) -> i128 + Sync,
    ) -> Result<Rounded> {
        let rounded = Rounded::make(self.shape, self.bits, compute)?;
        self.ranges
            .record(packed, [&rounded.values, &rounded.remainder])?;
        Ok(rounded)
    }

    pub(crate) fn recover<'w>(
        &self,
        io: &mut Session<'_, '_>,
        requests: &mut Vec<Request>,
        claim: &Claim,
        trace: impl FnOnce() -> &'w [i32],
    ) -> Result<F> {
        let [remainder] = io.values(|| [self.shape.evaluate(trace(), &claim.point)])?;
        self.ranges
            .open(io, requests, &claim.point, [claim.value, remainder])?;
        Ok(F::from_u64(1 << self.bits) * claim.value + remainder)
    }
}

pub(crate) fn wide_dot(left: &[i32], right: &[i32]) -> i128 {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| i128::from(a) * i128::from(b))
        .sum()
}

pub(crate) fn narrow_dot(left: &[i32], right: &[i32]) -> i128 {
    let mut sums = [0i64; 8];
    let mut lhs = left.chunks_exact(8);
    let mut rhs = right.chunks_exact(8);
    for (a, b) in lhs.by_ref().zip(rhs.by_ref()) {
        for i in 0..8 {
            sums[i] += i64::from(a[i]) * i64::from(b[i]);
        }
    }
    i128::from(sums.into_iter().sum::<i64>()) + wide_dot(lhs.remainder(), rhs.remainder())
}

pub(crate) struct Linear<'a> {
    input: Shape,
    projection: Projection<'a>,
    weights_norm: Vec<u128>,
    rounding: Rounding,
}

impl<'a> Linear<'a> {
    pub(crate) fn new(
        layout: &mut Layout,
        input: Shape,
        projection: Projection<'a>,
        bits: usize,
    ) -> Result<Self> {
        let shape = Shape::new(input.rows, projection.bias.len())?;
        if input.width.checked_mul(shape.width) != Some(projection.weights.len()) || bits > 30 {
            return Err("invalid projection dimensions");
        }
        let weights_norm = projection
            .weights
            .chunks_exact(input.width)
            .map(|w| w.iter().map(|x| u128::from(x.unsigned_abs())).sum())
            .collect();
        Ok(Self {
            input,
            projection,
            weights_norm,
            rounding: Rounding::new(layout, shape, bits),
        })
    }

    pub(crate) fn shape(&self) -> Shape {
        self.rounding.shape
    }

    pub(crate) fn execute(&self, input: &[i32], packed: &mut Packed<'_>) -> Result<Rounded> {
        if input.len() != self.input.len() {
            return Err("invalid projection input");
        }
        let stride = self.input.width;
        let bounds: Vec<_> = input
            .chunks_exact(stride)
            .map(|row| u128::from(row.iter().map(|x| x.unsigned_abs()).max().unwrap_or(0)))
            .collect();
        self.rounding.execute(packed, |row, col| {
            let x = &input[row * stride..][..stride];
            let w = &self.projection.weights[col * stride..][..stride];
            let dot = if bounds[row] * self.weights_norm[col] <= i64::MAX as u128 {
                narrow_dot(x, w)
            } else {
                wide_dot(x, w)
            };
            dot + (i128::from(self.projection.bias[col]) << self.rounding.bits)
        })
    }

    pub(crate) fn reverse<'w>(
        &self,
        io: &mut Session<'_, '_>,
        requests: &mut Vec<Request>,
        claim: Claim,
        witness: impl Fn() -> (&'w Rounded, &'w [i32]),
    ) -> Result<Claim> {
        let raw = self
            .rounding
            .recover(io, requests, &claim, || &witness().0.remainder)?;
        let (row, col) = claim.point.split_at(self.input.row_bits());
        let active: F = weights(row).into_iter().take(self.input.rows).sum();
        let bias = inner(
            weights(col),
            self.projection.bias.iter().copied().map(F::from_i32),
        );
        let public = bind_rows(self.input.width, self.projection.weights, col);
        let raw = raw - F::from_u64(1 << self.rounding.bits) * active * bias;
        let end = io.dot(raw, &public, || {
            bind_rows(self.input.width, witness().1, row)
        })?;
        Ok(Claim::new([row, &end.point].concat(), end.value))
    }
}

pub(crate) struct Hadamard(Rounding);

impl Hadamard {
    pub(crate) fn new(layout: &mut Layout, shape: Shape, bits: usize) -> Self {
        Self(Rounding::new(layout, shape, bits))
    }

    pub(crate) fn execute(
        &self,
        left: &[i32],
        right: &[i32],
        packed: &mut Packed<'_>,
    ) -> Result<Rounded> {
        let shape = self.0.shape;
        if left.len() != shape.len() || right.len() != shape.len() {
            return Err("invalid product dimensions");
        }
        self.0.execute(packed, |row, col| {
            let i = row * shape.width + col;
            i128::from(left[i]) * i128::from(right[i])
        })
    }

    pub(crate) fn reverse<'w>(
        &self,
        io: &mut Session<'_, '_>,
        requests: &mut Vec<Request>,
        claim: Claim,
        witness: impl Fn() -> (&'w Rounded, &'w [i32], &'w [i32]),
    ) -> Result<[Claim; 2]> {
        let raw = self
            .0
            .recover(io, requests, &claim, || &witness().0.remainder)?;
        let shape = self.0.shape;
        let len = 1 << shape.bits();
        let (end, [left, right]) = io.fold(
            raw,
            shape.bits(),
            3,
            || {
                let (_, a, b) = witness();
                vec![
                    Column::source(len, move |i| shape.at(a, i)),
                    Column::source(len, move |i| shape.at(b, i)),
                    Column::Dense(weights(&claim.point)),
                ]
            },
            [0, 1],
            |v| v[0] * v[1] * v[2],
        )?;
        io.expect(&end, || {
            left * right * equality(&claim.point, &end.challenges)
        })?;
        Ok([
            Claim::new(end.challenges.clone(), left),
            Claim::new(end.challenges, right),
        ])
    }
}

pub(crate) struct Rotation<'a> {
    rounding: Rounding,
    table: Rotary<'a>,
}

impl<'a> Rotation<'a> {
    pub(crate) fn new(
        layout: &mut Layout,
        shape: Shape,
        bits: usize,
        table: Rotary<'a>,
    ) -> Result<Self> {
        if !shape.width.is_power_of_two()
            || shape.width < 2
            || table.cos.len() != shape.len() / 2
            || table.sin.len() != table.cos.len()
        {
            return Err("invalid rotary dimensions");
        }
        Ok(Self {
            rounding: Rounding::new(layout, shape, bits),
            table,
        })
    }

    fn coefficients(&self, row: usize, col: usize) -> (i32, i32) {
        let half = self.rounding.shape.width / 2;
        let i = row * half + col % half;
        let sine = self.table.sin[i];
        (self.table.cos[i], if col < half { -sine } else { sine })
    }

    pub(crate) fn execute(&self, input: &[i32], packed: &mut Packed<'_>) -> Result<Rounded> {
        let width = self.rounding.shape.width;
        self.rounding.execute(packed, |row, col| {
            let (cos, sin) = self.coefficients(row, col);
            let x = &input[row * width..][..width];
            i128::from(cos) * i128::from(x[col])
                + i128::from(sin) * i128::from(x[col ^ (width / 2)])
        })
    }

    pub(crate) fn reverse<'w>(
        &self,
        io: &mut Session<'_, '_>,
        requests: &mut Vec<Request>,
        claim: Claim,
        witness: impl Fn() -> (&'w Rounded, &'w [i32]),
    ) -> Result<Claim> {
        let sum = self
            .rounding
            .recover(io, requests, &claim, || &witness().0.remainder)?;
        let shape = self.rounding.shape;
        let mut coefficients = weights(&claim.point);
        for (row, entries) in coefficients
            .chunks_exact_mut(shape.width)
            .take(shape.rows)
            .enumerate()
        {
            let half = shape.width / 2;
            for i in 0..half {
                let (c, s) = self.coefficients(row, i);
                let (c, s) = (F::from_i32(c), F::from_i32(s));
                let (a, b) = (entries[i], entries[i + half]);
                entries[i] = a * c - b * s;
                entries[i + half] = b * c + a * s;
            }
        }
        io.dot(sum, &coefficients, || {
            let input = witness().1;
            (0..coefficients.len())
                .map(|i| shape.at(input, i))
                .collect()
        })
    }
}
