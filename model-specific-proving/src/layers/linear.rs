//! Matrix and pointwise products, with shared Euclidean remainder checks.

use std::borrow::Cow;

use jolt_poly::MultilinearPoly;
use jolt_transcript::Transcript;
use rayon::prelude::*;

use crate::protocol::{
    pcs::{bind_i32, Advice, Layout},
    range::{Range, Ranges, I32},
    session::{instance, Claim, Column, ProofField, Session},
    tensor::{bind_rows, dot, eq, integer_dot, weights, Shape},
    Result,
};

pub(crate) struct Rounded {
    pub(crate) output: Vec<i32>,
    remainder: Vec<i32>,
}

pub(super) fn quotient_remainder(raw: i128, bits: usize) -> Result<(i32, i32)> {
    let output = i32::try_from(raw >> bits).map_err(|_| "fixed-point overflow")?;
    let remainder = (raw & ((1i128 << bits) - 1)) as i32;
    Ok((output, remainder))
}

struct Rounding {
    shape: Shape,
    bits: usize,
    check_output: bool,
    ranges: Ranges,
}

impl Rounding {
    fn new(layout: &mut Layout, shape: Shape, bits: usize, check_output: bool) -> Result<Self> {
        if bits > 30 {
            return Err("invalid fixed-point scale");
        }
        let first = if check_output {
            I32
        } else {
            Range::unsigned(0)
        };
        let ranges = Ranges::new(layout, shape, [first, Range::unsigned(bits)])?;
        Ok(Self {
            shape,
            bits,
            check_output,
            ranges,
        })
    }

    fn execute(
        &self,
        advice: &mut Advice,
        values: impl IntoIterator<Item = i128>,
    ) -> Result<Rounded> {
        let mut output = Vec::with_capacity(self.shape.rows * self.shape.width);
        let mut remainder = Vec::with_capacity(output.capacity());
        for raw in values {
            let (q, r) = quotient_remainder(raw, self.bits)?;
            output.push(q);
            remainder.push(r);
        }
        self.record(advice, Rounded { output, remainder })
    }

    fn record(&self, advice: &mut Advice, rounded: Rounded) -> Result<Rounded> {
        let values: &[i32] = if self.check_output {
            &rounded.output
        } else {
            &[]
        };
        self.ranges.record(advice, &[values, &rounded.remainder])?;
        Ok(rounded)
    }

    fn reverse<'w, F: ProofField>(
        &self,
        q: &Claim<F>,
        witness: impl FnOnce() -> &'w Rounded,
        io: &mut Session<'_, '_, F>,
    ) -> Result<F> {
        let [rem] = io.values(|| [self.shape.eval(&witness().remainder, &q.point)])?;
        let output = if self.check_output {
            q.value
        } else {
            F::zero()
        };
        self.ranges.check(&[(&q.point, [output, rem])], io)?;
        Ok(F::from_u64(1 << self.bits) * q.value + rem)
    }
}

/// A public matrix with weights stored by output channel.
#[derive(Clone, Copy)]
pub struct Projection<'a> {
    pub weights: &'a [i32],
    pub bias: &'a [i32],
}

impl Projection<'_> {
    pub(super) fn output_bound(self, input: usize, bound: u128, scale: u128) -> u128 {
        self.weights
            .chunks_exact(input)
            .zip(self.bias)
            .map(|(row, &bias)| {
                let norm: u128 = row.iter().map(|w| u128::from(w.unsigned_abs())).sum();
                (norm * bound).div_ceil(scale) + u128::from(bias.unsigned_abs())
            })
            .max()
            .unwrap_or(0)
            .min(1 << 31)
    }
}

pub(super) fn narrow_dot(left: &[i32], right: &[i32]) -> i64 {
    let mut lanes = [0i64; 8];
    let mut left = left.chunks_exact(8);
    let mut right = right.chunks_exact(8);
    for (a, b) in left.by_ref().zip(right.by_ref()) {
        for i in 0..8 {
            lanes[i] += i64::from(a[i]) * i64::from(b[i]);
        }
    }
    let tail: i64 = left
        .remainder()
        .iter()
        .zip(right.remainder())
        .map(|(&a, &b)| i64::from(a) * i64::from(b))
        .sum();
    lanes.into_iter().sum::<i64>() + tail
}

pub(crate) struct Linear<'a> {
    weights: Projection<'a>,
    weight_norms: Vec<u128>,
    input: Shape,
    rounding: Rounding,
}

impl<'a> Linear<'a> {
    pub(crate) fn new(
        layout: &mut Layout,
        input: Shape,
        projection: Projection<'a>,
        bits: usize,
        public: bool,
    ) -> Result<Self> {
        let output = Shape::new(input.rows, projection.bias.len())?;
        if input.width.checked_mul(output.width) != Some(projection.weights.len()) {
            return Err("invalid projection dimensions");
        }
        layout.integer_bits = layout
            .integer_bits
            .max(input.width.next_power_of_two().ilog2() + 65);
        Ok(Self {
            weights: projection,
            weight_norms: projection
                .weights
                .chunks_exact(input.width)
                .map(|row| row.iter().map(|w| u128::from(w.unsigned_abs())).sum())
                .collect(),
            input,
            rounding: Rounding::new(layout, output, bits, !public)?,
        })
    }

    pub(crate) fn shape(&self) -> Shape {
        self.rounding.shape
    }

    pub(crate) fn bind(&self, t: &mut impl Transcript) {
        bind_i32(t, self.weights.weights);
        bind_i32(t, self.weights.bias);
    }

    pub(crate) fn execute(&self, x: &[i32], advice: &mut Advice) -> Result<Rounded> {
        self.input.check(x)?;
        let scale = 1i128 << self.rounding.bits;
        let bits = self.rounding.bits;
        let (rows, width) = (self.input.rows, self.rounding.shape.width);
        let mut output = vec![0; rows * width];
        let mut remainder = vec![0; rows * width];
        let input_width = self.input.width;
        output
            .par_chunks_mut(width)
            .zip(remainder.par_chunks_mut(width))
            .zip(x.par_chunks_exact(input_width))
            .try_for_each(|((out, rem), row)| -> Result<()> {
                let max_input = u128::from(row.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0));
                for (col, (out, rem)) in out.iter_mut().zip(rem).enumerate() {
                    let weights = &self.weights.weights[col * input_width..][..input_width];
                    let dot = if max_input * self.weight_norms[col] <= i64::MAX as u128 {
                        i128::from(narrow_dot(row, weights))
                    } else {
                        integer_dot(row, weights)
                    };
                    (*out, *rem) =
                        quotient_remainder(dot + i128::from(self.weights.bias[col]) * scale, bits)?;
                }
                Ok(())
            })?;
        self.rounding.record(advice, Rounded { output, remainder })
    }

    pub(crate) fn reverse<'w, F: ProofField>(
        &self,
        q: Claim<F>,
        trace: impl Fn() -> (&'w [i32], &'w Rounded),
        io: &mut Session<'_, '_, F>,
    ) -> Result<Claim<F>> {
        let raw = self.rounding.reverse(&q, || trace().1, io)?;
        let (row, col) = q.point.split_at(self.input.row_vars());
        let active: F = weights::<F>(row).into_iter().take(self.input.rows).sum();
        let bias = dot(
            self.weights.bias.iter().copied().map(F::from_i32),
            weights(col),
        );
        let sum = raw - F::from_u64(1 << self.rounding.bits) * active * bias;
        let public = bind_rows(self.input.width, self.weights.weights, col);
        let mut claim =
            io.public_dot(sum, &public, || bind_rows(self.input.width, trace().0, row))?;
        claim.point = [row, &claim.point].concat();
        Ok(claim)
    }
}

#[derive(Clone, Copy)]
pub(crate) enum ProductKind {
    Pointwise,
    Matrix {
        inner: usize,
        transpose: bool,
        factor: i32,
    },
}

pub(crate) struct Product {
    kind: ProductKind,
    rounding: Rounding,
}

pub(super) struct ValueColumns {
    values: Vec<i32>,
    max_abs: u128,
}

impl ValueColumns {
    pub(super) fn new(values: &[i32], shape: Shape) -> Self {
        let mut transposed = vec![0; values.len()];
        for (row, values) in values.chunks_exact(shape.width).enumerate() {
            for (col, &value) in values.iter().enumerate() {
                transposed[col * shape.rows + row] = value;
            }
        }
        Self {
            values: transposed,
            max_abs: u128::from(values.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0)),
        }
    }
}

impl Product {
    pub(crate) fn new(
        layout: &mut Layout,
        shape: Shape,
        bits: usize,
        kind: ProductKind,
    ) -> Result<Self> {
        Self::with_output_range(layout, shape, bits, kind, true)
    }

    pub(super) fn with_output_range(
        layout: &mut Layout,
        shape: Shape,
        mut bits: usize,
        mut kind: ProductKind,
        check_output: bool,
    ) -> Result<Self> {
        if bits > 30 {
            return Err("invalid fixed-point scale");
        }
        if let ProductKind::Matrix {
            inner,
            ref mut factor,
            ..
        } = kind
        {
            // Cancel the common power of two before proving Euclidean division.
            let shift = (factor.trailing_zeros() as usize).min(bits);
            *factor >>= shift;
            bits -= shift;
            let integer_bits = inner.next_power_of_two().ilog2()
                + (u32::BITS - factor.unsigned_abs().leading_zeros())
                + 65;
            layout.integer_bits = layout.integer_bits.max(integer_bits);
        }
        Ok(Self {
            kind,
            rounding: Rounding::new(layout, shape, bits, check_output)?,
        })
    }

    pub(crate) fn execute(
        &self,
        left: &[i32],
        right: &[i32],
        advice: &mut Advice,
    ) -> Result<Rounded> {
        let shape = self.rounding.shape;
        match self.kind {
            ProductKind::Pointwise => {
                shape.check(left)?;
                shape.check(right)?;
                self.rounding.execute(
                    advice,
                    left.iter()
                        .zip(right)
                        .map(|(&a, &b)| i128::from(a) * i128::from(b)),
                )
            }
            ProductKind::Matrix { inner, .. } => {
                Shape::new(shape.rows, inner)?.check(left)?;
                self.execute_matrix(left.chunks_exact(inner).map(Cow::Borrowed), right, advice)
            }
        }
    }

    pub(super) fn execute_matrix<'a>(
        &self,
        left: impl Iterator<Item = Cow<'a, [i32]>>,
        right: &[i32],
        advice: &mut Advice,
    ) -> Result<Rounded> {
        let shape = self.rounding.shape;
        let ProductKind::Matrix {
            inner,
            transpose,
            factor,
        } = self.kind
        else {
            return Err("expected a matrix product");
        };
        if transpose {
            Shape::new(shape.width, inner)?.check(right)?;
        } else {
            Shape::new(inner, shape.width)?.check(right)?;
        }
        self.rounding.execute(
            advice,
            left.flat_map(|row| {
                (0..shape.width).map(move |j| {
                    let sum: i128 = row
                        .iter()
                        .enumerate()
                        .map(|(k, &x)| {
                            let index = if transpose {
                                j * inner + k
                            } else {
                                k * shape.width + j
                            };
                            i128::from(x) * i128::from(right[index])
                        })
                        .sum();
                    i128::from(factor) * sum
                })
            }),
        )
    }

    pub(super) fn execute_values(
        &self,
        left: impl IndexedParallelIterator<Item = Vec<i32>>,
        right: &ValueColumns,
        advice: &mut Advice,
    ) -> Result<Rounded> {
        let shape = self.rounding.shape;
        let ProductKind::Matrix {
            inner,
            transpose: false,
            factor,
        } = self.kind
        else {
            return Err("expected an attention value product");
        };
        let mut output = vec![0; shape.rows * shape.width];
        let mut remainder = vec![0; output.len()];
        left.zip(output.par_chunks_mut(shape.width))
            .zip(remainder.par_chunks_mut(shape.width))
            .try_for_each(|((row, output), remainder)| -> Result<()> {
                let norm: u128 = row.iter().map(|v| u128::from(v.unsigned_abs())).sum();
                let narrow = norm * right.max_abs <= i64::MAX as u128;
                for ((out, rem), values) in output
                    .iter_mut()
                    .zip(remainder)
                    .zip(right.values.chunks_exact(inner))
                {
                    let values = &values[..row.len()];
                    let dot = if narrow {
                        i128::from(narrow_dot(&row, values))
                    } else {
                        integer_dot(&row, values)
                    };
                    (*out, *rem) =
                        quotient_remainder(dot * i128::from(factor), self.rounding.bits)?;
                }
                Ok(())
            })?;
        self.rounding.record(advice, Rounded { output, remainder })
    }

    pub(super) fn reverse_values<'w, F: ProofField>(
        products: &[&Self],
        point: &[F],
        claims: &[F],
        trace: impl Fn(usize) -> &'w Rounded,
        values: impl FnOnce() -> &'w [i32],
        probabilities: impl Fn(usize, &[F]) -> Vec<F> + Sync,
        io: &mut Session<'_, '_, F>,
    ) -> Result<(Vec<Claim<F>>, Claim<F>)> {
        let first = products[0];
        let shape = first.rounding.shape;
        let ProductKind::Matrix {
            inner,
            transpose: false,
            factor,
        } = first.kind
        else {
            return Err("expected an attention value product");
        };
        let (row, col) = point.split_at(shape.row_vars());
        let raw: Vec<_> = products
            .iter()
            .zip(claims)
            .enumerate()
            .map(|(i, (product, &value))| {
                product.rounding.reverse(
                    &Claim {
                        point: point.to_vec(),
                        value,
                    },
                    || trace(i),
                    io,
                )
            })
            .collect::<Result<_>>()?;
        let mix = if raw.len() == 1 {
            vec![F::one()]
        } else {
            io.point(raw.len())?
        };
        let sum = dot(raw, mix.iter().copied());
        let scale = F::from_i32(factor);
        let size = inner.next_power_of_two();
        let mut columns = Vec::new();
        let (end, [value]) = io.fold(
            instance(sum, size.ilog2() as usize, 2),
            || {
                columns = (0..products.len())
                    .into_par_iter()
                    .map(|i| probabilities(i, row))
                    .collect();
                let mut mixed = vec![F::zero(); size];
                for (column, &weight) in columns.iter().zip(&mix) {
                    for (out, &entry) in mixed.iter_mut().zip(column) {
                        *out += weight * entry;
                    }
                }
                let w = weights::<F>(col);
                let mut v = vec![F::zero(); size];
                for (out, row) in v.iter_mut().zip(values().chunks_exact(shape.width)) {
                    *out = dot(row.iter().copied().map(F::from_i32), w.iter().copied());
                }
                [Column::owned(v), Column::owned(mixed)]
            },
            [0],
            |v| scale * v[0] * v[1],
        )?;
        let probabilities = (0..products.len())
            .map(|i| {
                let [value] = io.values(|| [columns[i].evaluate(&end.challenges)])?;
                Ok(Claim {
                    point: [row, &end.challenges].concat(),
                    value,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        io.check_value(&end, || {
            scale * value * dot(probabilities.iter().map(|q| q.value), mix)
        })?;
        Ok((
            probabilities,
            Claim {
                point: [&end.challenges, col].concat(),
                value,
            },
        ))
    }

    pub(crate) fn reverse<'w, F: ProofField>(
        &self,
        q: Claim<F>,
        trace: impl Fn() -> (&'w [i32], &'w [i32], &'w Rounded),
        io: &mut Session<'_, '_, F>,
    ) -> Result<[Claim<F>; 2]> {
        let shape = self.rounding.shape;
        match self.kind {
            ProductKind::Pointwise => {
                let sum = self.rounding.reverse(&q, || trace().2, io)?;
                io.pow(shape.vars())?;
                let (point, [left, right]) = io.relation(
                    instance(sum, shape.vars(), 3),
                    || {
                        [
                            Column::integer(trace().0, shape),
                            Column::integer(trace().1, shape),
                            Column::equality(&q.point),
                        ]
                    },
                    |v| v[0] * v[1] * v[2],
                    |p, &[l, r]| eq(&q.point, p) * l * r,
                )?;
                Ok([
                    Claim {
                        point: point.clone(),
                        value: left,
                    },
                    Claim {
                        point,
                        value: right,
                    },
                ])
            }
            ProductKind::Matrix { inner, .. } => self.reverse_matrix(
                q,
                || (trace().1, trace().2),
                |row| bind_rows(inner, trace().0, row),
                io,
            ),
        }
    }

    pub(super) fn reverse_matrix<'w, F: ProofField>(
        &self,
        q: Claim<F>,
        trace: impl Fn() -> (&'w [i32], &'w Rounded),
        bind_left: impl FnOnce(&[F]) -> Vec<F>,
        io: &mut Session<'_, '_, F>,
    ) -> Result<[Claim<F>; 2]> {
        let ProductKind::Matrix {
            inner,
            transpose,
            factor,
        } = self.kind
        else {
            return Err("expected a matrix product");
        };
        let sum = self.rounding.reverse(&q, || trace().1, io)?;
        let shape = self.rounding.shape;
        let (row, col) = q.point.split_at(shape.row_vars());
        let scale = F::from_i32(factor);
        let (point, [left, right]) = io.relation(
            instance(sum, inner.next_power_of_two().ilog2() as usize, 2),
            || {
                let a = bind_left(row);
                let b = trace().0;
                let b = if transpose {
                    bind_rows(inner, b, col)
                } else {
                    let w = weights::<F>(col);
                    let mut out = vec![F::zero(); inner.next_power_of_two()];
                    for (x, row) in out.iter_mut().zip(b.chunks_exact(shape.width)) {
                        *x = dot(row.iter().copied().map(F::from_i32), w.iter().copied());
                    }
                    out
                };
                [Column::owned(a), Column::owned(b)]
            },
            |v| scale * v[0] * v[1],
            |_, &[a, b]| scale * a * b,
        )?;
        let rhs = if transpose {
            [col, &point].concat()
        } else {
            [&point, col].concat()
        };
        Ok([
            Claim {
                point: [row, &point].concat(),
                value: left,
            },
            Claim {
                point: rhs,
                value: right,
            },
        ])
    }
}

#[derive(Clone, Copy)]
pub struct Rotary<'a> {
    pub cos: &'a [i32],
    pub sin: &'a [i32],
}

pub(crate) struct Rotate<'a> {
    coefficients: Rotary<'a>,
    rounding: Rounding,
}

impl<'a> Rotate<'a> {
    pub(crate) fn new(
        layout: &mut Layout,
        shape: Shape,
        bits: usize,
        c: Rotary<'a>,
    ) -> Result<Self> {
        let len = shape.rows * (shape.width / 2);
        if !shape.width.is_multiple_of(2)
            || c.cos.len() != len
            || c.sin.len() != len
            || c.cos
                .iter()
                .chain(c.sin)
                .any(|x| i64::from(*x).abs() > 1 << bits)
        {
            return Err("invalid rotary coefficients");
        }
        Ok(Self {
            coefficients: c,
            rounding: Rounding::new(layout, shape, bits, true)?,
        })
    }

    pub(crate) fn bind(&self, t: &mut impl Transcript) {
        bind_i32(t, self.coefficients.cos);
        bind_i32(t, self.coefficients.sin);
    }

    pub(crate) fn execute(&self, x: &[i32], advice: &mut Advice) -> Result<Rounded> {
        let w = self.rounding.shape.width;
        let half = w / 2;
        self.rounding.execute(
            advice,
            x.chunks_exact(w).enumerate().flat_map(|(r, row)| {
                (0..w).map(move |i| {
                    let j = i % half;
                    let (c, s) = (
                        self.coefficients.cos[r * half + j],
                        self.coefficients.sin[r * half + j],
                    );
                    let cross = if i < half {
                        -i128::from(row[j + half])
                    } else {
                        i128::from(row[j])
                    };
                    i128::from(row[i]) * i128::from(c) + i128::from(s) * cross
                })
            }),
        )
    }

    pub(crate) fn reverse<'w, F: ProofField>(
        &self,
        q: Claim<F>,
        trace: impl Fn() -> (&'w [i32], &'w Rounded),
        io: &mut Session<'_, '_, F>,
    ) -> Result<Claim<F>> {
        let sum = self.rounding.reverse(&q, || trace().1, io)?;
        let shape = self.rounding.shape;
        let half = shape.width / 2;
        let mut w = weights(&q.point);
        for (r, row) in w
            .chunks_exact_mut(shape.width.next_power_of_two())
            .enumerate()
        {
            if r >= shape.rows {
                row.fill(F::zero());
                continue;
            }
            for j in 0..half {
                let (a, b) = (row[j], row[j + half]);
                let c = F::from_i32(self.coefficients.cos[r * half + j]);
                let s = F::from_i32(self.coefficients.sin[r * half + j]);
                row[j] = a * c + b * s;
                row[j + half] = b * c - a * s;
            }
            row[shape.width..].fill(F::zero());
        }
        io.public_dot(sum, &w, || shape.padded(trace().0, F::zero(), F::from_i32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_rounding_cancels_the_public_scale_for_negative_scores() {
        let mut layout = Layout::default();
        let product = Product::new(
            &mut layout,
            Shape::new(1, 2).unwrap(),
            5,
            ProductKind::Matrix {
                inner: 1,
                transpose: true,
                factor: 12,
            },
        )
        .unwrap();
        let trace = product
            .execute(&[1], &[-3, 3], &mut layout.witness())
            .unwrap();
        assert_eq!(product.rounding.bits, 3);
        assert_eq!(trace.output, [-2, 1]);
        assert_eq!(trace.remainder, [7, 1]);
    }
}

#[cfg(test)]
#[test]
fn parallel_projection_preserves_wide_and_narrow_remainders() {
    let mut layout = Layout::default();
    let input = Shape::new(3, 4).unwrap();
    let weights = [i32::MIN, i32::MIN, i32::MAX, i32::MAX, -1, 3, 1, -3];
    let bias = [0, -2];
    let plan = Linear::new(
        &mut layout,
        input,
        Projection {
            weights: &weights,
            bias: &bias,
        },
        30,
        false,
    )
    .unwrap();
    let x = [
        i32::MIN,
        i32::MIN,
        i32::MIN,
        i32::MIN,
        -7,
        -7,
        -7,
        -7,
        0,
        -1,
        0,
        0,
    ];
    let actual = plan.execute(&x, &mut layout.witness()).unwrap();
    let scale = 1i128 << 30;
    let (output, remainder): (Vec<_>, Vec<_>) = x
        .chunks_exact(4)
        .flat_map(|row| {
            weights.chunks_exact(4).zip(bias).map(move |(w, b)| {
                let value = integer_dot(row, w) + i128::from(b) * scale;
                (
                    value.div_euclid(scale) as i32,
                    value.rem_euclid(scale) as i32,
                )
            })
        })
        .unzip();
    assert_eq!(actual.output, output);
    assert_eq!(actual.remainder, remainder);
}
