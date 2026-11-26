use jolt_field::{Ring, Zero};

use super::{NormalizationKind, NormalizationParameters};
use crate::protocol::{
    field::F,
    opening::{Layout, Packed, Request},
    range::{Ranges, Width, I32},
    sumcheck::{Column, Session},
    tensor::{equality, inner, weights, Claim, Shape},
    Result,
};

pub(crate) struct Norm<'a> {
    params: NormalizationParameters<'a>,
    bits: usize,
    rows: Ranges<9>,
    cells: Ranges<3>,
    remainder: Ranges<1>,
    remainder_gap: Ranges<1>,
}

pub(crate) struct NormTrace {
    pub values: Vec<i32>,
    quotients: Vec<i64>,
    stats: Vec<[i128; 9]>,
}

impl<'a> Norm<'a> {
    pub(crate) fn new(
        layout: &mut Layout,
        shape: Shape,
        bits: usize,
        params: NormalizationParameters<'a>,
    ) -> Result<Self> {
        if bits > 15
            || params.epsilon_variance_units <= 0
            || shape.width > 1 << 30
            || params.gain.len() != shape.width
            || params.bias.len() != shape.width
        {
            return Err("invalid normalization parameters");
        }
        let u = Width::unsigned;
        let width = shape.width.next_power_of_two().ilog2() as usize;
        let gap = if shape.width.is_power_of_two() {
            0
        } else {
            width
        };
        let layer = params.kind == NormalizationKind::Layer;
        let row_widths = [
            u(65),
            u(width),
            u(gap),
            u(33),
            u(34),
            u(34),
            if layer { I32 } else { u(0) },
            u(if layer { width } else { 0 }),
            u(if layer { gap } else { 0 }),
        ];
        let z_bits = if layer {
            49
        } else {
            let dev = (params.epsilon_variance_units as u128).isqrt();
            let root = (shape.width as u128).isqrt();
            let ceil_root = root + u128::from(root * root < shape.width as u128);
            ((ceil_root << bits) * (dev + 1)).div_ceil(dev).ilog2() as usize + 2
        };
        let quotient_limit = 1u128 << (z_bits - 1);
        let output_limit = params
            .gain
            .iter()
            .zip(params.bias)
            .map(|(&gain, &bias)| {
                (quotient_limit * u128::from(gain.unsigned_abs())).div_ceil(1 << bits)
                    + u128::from(bias.unsigned_abs())
            })
            .max()
            .unwrap_or(0);
        let output_bits = (output_limit.max(1).ilog2() as usize + 2).min(32);
        Ok(Self {
            bits,
            params,
            rows: Ranges::new(layout, Shape::new(shape.rows, 1)?, row_widths),
            cells: Ranges::new(
                layout,
                shape,
                [Width::signed(z_bits), Width::signed(output_bits), u(bits)],
            ),
            remainder: Ranges::adaptive(layout, shape, 33),
            remainder_gap: Ranges::adaptive(layout, shape, 33),
        })
    }

    pub(crate) fn bounds(&self) -> Vec<u128> {
        if self.params.kind == NormalizationKind::Layer {
            return vec![1 << 31; self.cells.shape.width];
        }
        let dev = (self.params.epsilon_variance_units as u128).isqrt();
        let width = self.cells.shape.width as u128;
        let root = width.isqrt();
        let root = root + u128::from(root * root < width);
        let magnitude = ((root << self.bits) * (dev + 1)).div_ceil(dev);
        self.params
            .gain
            .iter()
            .zip(self.params.bias)
            .map(|(&gain, &bias)| {
                ((magnitude * u128::from(gain.unsigned_abs())).div_ceil(1 << self.bits)
                    + u128::from(bias.unsigned_abs()))
                .min(1 << 31)
            })
            .collect()
    }

    fn row_stats(&self, cells: &[i32]) -> [i128; 9] {
        let width = cells.len() as i128;
        let sum = if self.params.kind == NormalizationKind::Layer {
            cells.iter().copied().map(i128::from).sum()
        } else {
            0
        };
        let mean = sum.div_euclid(width);
        let squares: i128 = cells.iter().map(|&x| (i128::from(x) - mean).pow(2)).sum();
        let variance = squares / width + i128::from(self.params.epsilon_variance_units);
        let dev = variance.isqrt();
        let root = variance - dev * dev;
        let gap = |r| {
            if cells.len().is_power_of_two() {
                0
            } else {
                width - 1 - r
            }
        };
        let var_rem = squares % width;
        let mean_rem = sum.rem_euclid(width);
        [
            variance,
            var_rem,
            gap(var_rem),
            dev,
            root,
            2 * dev - root,
            mean,
            mean_rem,
            if self.params.kind == NormalizationKind::Layer {
                gap(mean_rem)
            } else {
                0
            },
        ]
    }

    fn cell(
        &self,
        input: i32,
        col: usize,
        stats: &[i128; 9],
        quotient: i64,
        output: i32,
    ) -> [i128; 5] {
        let q = i128::from(quotient);
        let scaled = (i128::from(input) - stats[6]) << self.bits;
        let rem = scaled - stats[3] * q;
        let product = q * i128::from(self.params.gain[col]);
        [
            q,
            i128::from(output),
            rem,
            stats[3] - rem - 1,
            product & ((1 << self.bits) - 1),
        ]
    }

    pub(crate) fn execute(&self, input: &[i32], packed: &mut Packed<'_>) -> Result<NormTrace> {
        let shape = self.cells.shape;
        if input.len() != shape.len() {
            return Err("invalid normalization input");
        }
        let mut trace = NormTrace {
            values: Vec::with_capacity(input.len()),
            quotients: Vec::with_capacity(input.len()),
            stats: Vec::with_capacity(shape.rows),
        };
        let mut columns: [Vec<i128>; 5] = std::array::from_fn(|_| Vec::with_capacity(input.len()));
        for row in input.chunks_exact(shape.width) {
            let stats = self.row_stats(row);
            for (col, &input) in row.iter().enumerate() {
                let q = (((i128::from(input) - stats[6]) << self.bits).div_euclid(stats[3])) as i64;
                let affine = i128::from(q) * i128::from(self.params.gain[col]);
                let output =
                    i32::try_from((affine >> self.bits) + i128::from(self.params.bias[col]))
                        .map_err(|_| "normalization overflow")?;
                trace.values.push(output);
                trace.quotients.push(q);
                for (column, value) in columns
                    .iter_mut()
                    .zip(self.cell(input, col, &stats, q, output))
                {
                    column.push(value);
                }
            }
            trace.stats.push(stats);
        }
        let stats: [Vec<i128>; 9] =
            std::array::from_fn(|i| trace.stats.iter().map(|row| row[i]).collect());
        self.rows
            .record(packed, stats.each_ref().map(Vec::as_slice))?;
        self.cells
            .record(packed, [&columns[0], &columns[1], &columns[4]])?;
        self.remainder.record_adaptive(packed, &columns[2])?;
        self.remainder_gap.record_adaptive(packed, &columns[3])?;
        Ok(trace)
    }

    pub(crate) fn reverse<'w>(
        &self,
        io: &mut Session<'_, '_>,
        requests: &mut Vec<Request>,
        claims: &[Claim],
        witness: impl Fn() -> (&'w NormTrace, &'w [i32]),
    ) -> Result<Claim> {
        let shape = self.cells.shape;
        let (vars, row_bits) = (shape.bits(), shape.row_bits());
        let len = 1 << vars;
        let stride = shape.width.next_power_of_two();
        let mix = io.random(
            9 + vars + row_bits + claims.len(),
            9 + vars + row_bits + claims.len(),
        )?;
        let (coeff, random) = mix.split_at(9);
        let (p, random) = random.split_at(vars);
        let (r, batch) = random.split_at(row_bits);
        let scale = F::from_u64(1 << self.bits);
        let width = F::from_u64(shape.width as u64);
        let epsilon = F::from_i64(self.params.epsilon_variance_units);
        let use_mean = self.params.kind == NormalizationKind::Layer;
        let with_gap = !shape.width.is_power_of_two();
        let formula = |v: &[F]| {
            let [x, z, y, rem, gap, arem] = v[..6].try_into().expect("normalization columns");
            let [var, vr, vg, dev, root, rg, mean, mr, mg] =
                v[6..15].try_into().expect("normalization statistics");
            let [gain, bias, lane, active, first, blend, pos, row] =
                v[15..].try_into().expect("normalization weights");
            let x0 = x - mean;
            let row_terms = [
                lane * x0 * x0 - first * (width * (var - epsilon) + vr),
                if with_gap {
                    first * (vr + vg - width + F::from_u64(1))
                } else {
                    F::zero()
                },
                first * (var - dev * dev - root),
                first * (dev + dev - root - rg),
                if use_mean {
                    lane * x - first * (width * mean + mr)
                } else {
                    F::zero()
                },
            ];
            let cell_terms = [
                scale * x0 - dev * z - rem,
                dev - F::from_u64(1) - rem - gap,
                gain * z - scale * (y - bias) - arem,
            ];
            let mean_gap = if use_mean && with_gap {
                first * (mr + mg - width + F::from_u64(1))
            } else {
                F::zero()
            };
            row * (inner(coeff[..5].iter().copied(), row_terms) + coeff[5] * mean_gap)
                + active * (blend * y + pos * inner(coeff[6..].iter().copied(), cell_terms))
        };
        let sum = inner(claims.iter().map(|q| q.value), batch.iter().copied());
        let evals = |v: &[i32]| {
            (0..stride)
                .map(|i| F::from_i32(v.get(i).copied().unwrap_or(0)))
                .collect::<Vec<_>>()
        };
        let gain = evals(self.params.gain);
        let bias = evals(self.params.bias);
        let mut row_values = weights(r);
        row_values
            .iter_mut()
            .skip(shape.rows)
            .for_each(|x| *x = F::zero());
        let blend = |i| {
            inner(
                batch.iter().copied(),
                claims.iter().map(|q| {
                    q.point
                        .iter()
                        .enumerate()
                        .fold(F::from_u64(1), |acc, (b, &x)| {
                            acc * if i >> (vars - b - 1) & 1 != 0 {
                                x
                            } else {
                                F::from_u64(1) - x
                            }
                        })
                }),
            )
        };
        let (end, values) = io.fold(
            sum,
            vars,
            4,
            || {
                let (trace, input) = witness();
                let mut cols = vec![Column::source(len, move |i| shape.at(input, i))];
                for c in 0..5 {
                    cols.push(Column::source(len, move |i| {
                        let (row, col) = (i / stride, i % stride);
                        if row >= shape.rows || col >= shape.width {
                            return F::zero();
                        }
                        let j = row * shape.width + col;
                        F::from_i128(
                            self.cell(
                                input[j],
                                col,
                                &trace.stats[row],
                                trace.quotients[j],
                                trace.values[j],
                            )[c],
                        )
                    }));
                }
                for c in 0..9 {
                    cols.push(Column::source(len, move |i| {
                        F::from_i128(trace.stats.get(i / stride).map_or(0, |row| row[c]))
                    }));
                }
                cols.extend([
                    Column::source(len, |i| gain[i % stride]),
                    Column::source(len, |i| bias[i % stride]),
                    Column::source(len, |i| F::from_bool(i % stride < shape.width)),
                    Column::source(len, |i| {
                        F::from_bool(i / stride < shape.rows && i % stride < shape.width)
                    }),
                    Column::source(len, |i| F::from_bool(i % stride == 0)),
                    Column::source(len, blend),
                    Column::Dense(weights(p)),
                    Column::source(len, |i| row_values[i / stride]),
                ]);
                cols
            },
            [0, 1, 2, 3, 4, 5],
            formula,
        )?;
        let (rp, cp) = end.challenges.split_at(row_bits);
        let fields = io.values(|| {
            std::array::from_fn::<_, 9, _>(|i| {
                inner(
                    weights(rp),
                    witness().0.stats.iter().map(|row| F::from_i128(row[i])),
                )
            })
        })?;
        io.expect(&end, || {
            let cw = weights(cp);
            let rw = weights(rp);
            let lanes: F = cw.iter().take(shape.width).copied().sum();
            let active_rows: F = rw.iter().take(shape.rows).copied().sum();
            let mut endpoints = values.to_vec();
            endpoints.extend(fields);
            endpoints.extend([
                inner(cw.iter().copied(), gain.iter().copied()),
                inner(cw.iter().copied(), bias.iter().copied()),
                lanes,
                lanes * active_rows,
                cw[0],
                inner(
                    batch.iter().copied(),
                    claims.iter().map(|q| equality(&q.point, &end.challenges)),
                ),
                equality(p, &end.challenges),
                inner(rw, row_values),
            ]);
            formula(&endpoints)
        })?;
        self.rows.open(io, requests, rp, fields)?;
        self.cells.open(
            io,
            requests,
            &end.challenges,
            [values[1], values[2], values[5]],
        )?;
        self.remainder
            .open(io, requests, &end.challenges, [values[3]])?;
        self.remainder_gap
            .open(io, requests, &end.challenges, [values[4]])?;
        Ok(Claim::new(end.challenges, values[0]))
    }
}
