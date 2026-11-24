//! Fixed-table activation and normalization constraints.

use jolt_poly::MultilinearPoly;
use jolt_transcript::Transcript;

use crate::protocol::{
    lookup::{Batches, Lookup, Read, Table},
    pcs::{bind_i32, Advice, Kind, Layout, Opening, Part, Slot},
    range::{Range, Ranges, I32},
    session::{instance, query_column, query_weight, Claim, Column, ProofField, Session},
    tensor::{dot, eq, weights, Shape},
    Result,
};

#[derive(Clone, Copy)]
pub struct Activation<'a> {
    pub lower: i32,
    pub correction: &'a [i32],
}

pub(crate) struct Activate<'a> {
    shape: Shape,
    table: Activation<'a>,
    ranges: Ranges,
    sign: Slot,
    lookup: Lookup,
}

pub(crate) struct ActivationTrace<F> {
    pub(crate) output: Vec<i32>,
    columns: [Vec<F>; 5],
    indices: Vec<usize>,
}

impl<'a> Activate<'a> {
    pub(crate) fn new(layout: &mut Layout, shape: Shape, table: Activation<'a>) -> Result<Self> {
        let lookup = Lookup::new(layout, shape.vars(), table.correction.len())?;
        let upper = i64::from(table.lower) + (table.correction.len() - 1) as i64;
        if upper > i64::from(i32::MAX) {
            return Err("invalid activation bounds");
        }
        let ranges = Ranges::new(layout, shape, [Range::unsigned(32); 2])?;
        let sign = layout.add(shape.vars(), Kind::Bits { planes: 1 })?;
        Ok(Self {
            shape,
            table,
            ranges,
            sign,
            lookup,
        })
    }

    pub(crate) fn bind(&self, t: &mut impl Transcript) {
        t.append(&(i64::from(self.table.lower) as u64));
        bind_i32(t, self.table.correction);
    }

    pub(crate) fn execute<F: ProofField>(
        &self,
        x: &[i32],
        a: &mut Advice,
    ) -> Result<ActivationTrace<F>> {
        self.shape.check(x)?;
        let lower = i64::from(self.table.lower);
        let upper = lower + (self.table.correction.len() - 1) as i64;
        let mut output = Vec::new();
        let mut magnitude = Vec::new();
        let mut excess = Vec::new();
        let mut indices = Vec::new();
        let mut columns: [Vec<F>; 5] = std::array::from_fn(|_| Vec::new());
        let mut signs = Vec::with_capacity(1 << self.shape.vars());
        let stride = self.shape.width.next_power_of_two();
        for (i, x) in self.shape.padded(x, 0, i64::from).into_iter().enumerate() {
            let index = x.clamp(lower, upper) - lower;
            let correction = i64::from(self.table.correction[index as usize]);
            let outside = (x - upper).max(0) + (lower - x).max(0);
            indices.push(index as usize);
            let real = i / stride < self.shape.rows && i % stride < self.shape.width;
            signs.push(u8::from(real && x >= 0));
            let values = [x, index, correction, x.abs(), outside];
            for (c, v) in columns.iter_mut().zip(values) {
                c.push(F::from_i64(v));
            }
            if real {
                output
                    .push(i32::try_from(x.max(0) + correction).map_err(|_| "activation overflow")?);
                magnitude.push(i128::from(x.abs()));
                excess.push(i128::from(outside));
            }
        }
        // Padded cells are outside the output domain, but their clamp equation still holds.
        // Their excess must be zero for the chosen table's domain around input zero.
        if lower > 0 || upper < 0 {
            return Err("activation domain must contain zero");
        }
        self.ranges.record(a, &[&magnitude, &excess])?;
        a.put(self.sign, Part::Bits(signs))?;
        self.lookup.record(a, &indices)?;
        Ok(ActivationTrace {
            output,
            columns,
            indices,
        })
    }

    pub(crate) fn reverse<'s, 'w, F: ProofField>(
        &'s self,
        q: Claim<F>,
        trace: impl Fn() -> (&'w ActivationTrace<F>, &'w Advice) + Sync,
        io: &mut Session<'_, '_, F>,

        lookups: &mut Batches<'s, 'w, F>,
    ) -> Result<Claim<F>> {
        let vars = self.shape.vars();
        io.pow(vars + 3)?;
        let mix = io.challenge(3)?;
        let random = io.challenge(vars)?;
        let lower = F::from_i32(self.table.lower);
        let last = F::from_u64((self.table.correction.len() - 1) as u64);
        let clamp = |v: &[F]| {
            dot(
                [
                    last * (v[0] - v[1] - lower) - v[5] * (v[1] + v[1] - last),
                    v[5] * v[1] * (v[1] - last),
                    v[0] - (v[4] + v[4] - F::one()) * v[3],
                ],
                mix.iter().copied(),
            )
        };
        let active = self.shape.padded(
            &vec![true; self.shape.rows * self.shape.width],
            F::zero(),
            F::from_bool,
        );
        let (point, v) = io.relation(
            instance(q.value, vars, 4),
            || {
                let [x, a, c, m, e] = &trace().0.columns;
                [
                    Column::borrowed(x),
                    Column::borrowed(a),
                    Column::borrowed(c),
                    Column::borrowed(m),
                    Column::Bits(trace().1.bits(self.sign), 1),
                    Column::borrowed(e),
                    Column::borrowed(&active),
                    Column::owned(weights(&q.point)),
                    Column::owned(weights(&random)),
                ]
            },
            |v| v[6] * v[7] * (v[4] * v[3] + v[2]) + v[8] * clamp(v),
            |p, v: &[F; 6]| {
                active.as_slice().evaluate(p) * eq(&q.point, p) * (v[4] * v[3] + v[2])
                    + eq(&random, p) * clamp(v)
            },
        )?;
        self.ranges.check(&[(&point, [v[3], v[5]])], io)?;
        io.openings.push(Opening::new(
            self.sign,
            Claim {
                point: point.clone(),
                value: v[4],
            },
        ));
        lookups.push(
            &[Table::Slice(self.table.correction)],
            &self.lookup,
            Read {
                cell: point.clone(),
                address: v[1],
                output: Box::new([v[2]]),
                witness: io.proving().then(|| {
                    let (trace, advice) = trace();
                    (trace.indices.as_slice(), advice)
                }),
            },
        );
        Ok(Claim { point, value: v[0] })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NormalizationKind {
    Layer,
    Rms,
}

impl NormalizationKind {
    fn stats(self) -> usize {
        if self == Self::Layer {
            9
        } else {
            6
        }
    }
}

#[derive(Clone, Copy)]
pub struct NormalizationParameters<'a> {
    pub kind: NormalizationKind,
    pub gain: &'a [i32],
    pub bias: &'a [i32],
    /// Epsilon quantized in the same units as the variance.
    pub epsilon_variance_units: i64,
}

pub(crate) struct Norm<'a> {
    shape: Shape,
    bits: usize,
    p: NormalizationParameters<'a>,
    rows: Ranges,
    cells: Ranges,
}

pub(crate) struct NormTrace<F> {
    pub(crate) output: Vec<i32>,
    rows: [Vec<F>; 9],
    cells: [Vec<F>; 5],
}

impl<'a> Norm<'a> {
    pub(crate) fn new(
        layout: &mut Layout,
        shape: Shape,
        bits: usize,
        p: NormalizationParameters<'a>,
    ) -> Result<Self> {
        if bits > 15
            || p.gain.len() != shape.width
            || p.bias.len() != shape.width
            || p.epsilon_variance_units <= 0
            || shape.width > 1 << 30
        {
            return Err("invalid normalization parameters");
        }
        let u = Range::unsigned;
        let rem = u(shape.width.next_power_of_two().ilog2().max(1) as usize);
        let row = Shape::new(shape.rows, 1)?;
        let ranges = [u(65), rem, rem, u(33), u(34), u(34), I32, rem, rem];
        let rows = Ranges::new(layout, row, &ranges[..p.kind.stats()])?;
        let cells = Ranges::new(
            layout,
            shape,
            [Range::signed(49), I32, u(33), u(33), u(bits)],
        )?;
        Ok(Self {
            shape,
            bits,
            p,
            rows,
            cells,
        })
    }

    pub(crate) fn bind(&self, t: &mut impl Transcript) {
        t.append(&(self.p.epsilon_variance_units as u64));
        t.append(&u64::from(self.p.kind == NormalizationKind::Rms));
        bind_i32(t, self.p.gain);
        bind_i32(t, self.p.bias);
    }

    pub(crate) fn execute<F: ProofField>(
        &self,
        input: &[i32],
        advice: &mut Advice,
    ) -> Result<NormTrace<F>> {
        let shape = self.shape;
        shape.check(input)?;
        let width = shape.width as i128;
        let scale = 1i128 << self.bits;
        let mut rows: [Vec<i128>; 9] = std::array::from_fn(|_| Vec::new());
        let mut cells: [Vec<i128>; 5] = std::array::from_fn(|_| Vec::new());
        let mut output = Vec::with_capacity(input.len());
        let layer = self.p.kind == NormalizationKind::Layer;
        for row in input.chunks_exact(shape.width) {
            let (mean, mrem) = if layer {
                let sum: i128 = row.iter().copied().map(i128::from).sum();
                (sum.div_euclid(width), sum.rem_euclid(width))
            } else {
                (0, 0)
            };
            let squares: i128 = row.iter().map(|&x| (i128::from(x) - mean).pow(2)).sum();
            let var = squares.div_euclid(width) + i128::from(self.p.epsilon_variance_units);
            let vrem = squares.rem_euclid(width);
            let dev = var.isqrt();
            let root = var - dev * dev;
            let stats = [
                var,
                vrem,
                width - 1 - vrem,
                dev,
                root,
                2 * dev - root,
                mean,
                mrem,
                width - 1 - mrem,
            ];
            for (column, &v) in rows.iter_mut().zip(&stats[..self.p.kind.stats()]) {
                column.push(v);
            }
            for (i, &x) in row.iter().enumerate() {
                let x = (i128::from(x) - mean) * scale;
                let z = x.div_euclid(dev);
                let rem = x.rem_euclid(dev);
                let affine = z * i128::from(self.p.gain[i]);
                let y = affine.div_euclid(scale) + i128::from(self.p.bias[i]);
                output.push(i32::try_from(y).map_err(|_| "normalization overflow")?);
                for (c, v) in
                    cells
                        .iter_mut()
                        .zip([z, y, rem, dev - 1 - rem, affine.rem_euclid(scale)])
                {
                    c.push(v);
                }
            }
        }
        self.rows.record(
            advice,
            &rows.each_ref().map(Vec::as_slice)[..self.p.kind.stats()],
        )?;
        self.cells
            .record(advice, &cells.each_ref().map(Vec::as_slice))?;
        Ok(NormTrace {
            output,
            rows: rows.map(|v| {
                if v.is_empty() {
                    Vec::new()
                } else {
                    Shape {
                        rows: shape.rows,
                        width: 1,
                    }
                    .padded(&v, F::zero(), F::from_i128)
                }
            }),
            cells: cells.map(|v| shape.padded(&v, F::zero(), F::from_i128)),
        })
    }

    pub(crate) fn reverse<'w, F: ProofField>(
        &self,
        claims: &[Claim<F>],
        trace: impl Fn() -> (&'w [i32], &'w NormTrace<F>),
        io: &mut Session<'_, '_, F>,
    ) -> Result<Claim<F>> {
        let shape = self.shape;
        let (row_vars, vars) = (shape.row_vars(), shape.vars());
        let (rows, width) = (1 << row_vars, shape.width.next_power_of_two());
        let scale = F::from_u64(1 << self.bits);
        let w = F::from_u64(shape.width as u64);
        let epsilon = F::from_i64(self.p.epsilon_variance_units);
        let stats = self.p.kind.stats();
        let checks = stats - 2;
        io.pow(row_vars + vars + checks + 3 + claims.len())?;
        let mix = io.challenge(checks + 3)?;
        let row_query = io.challenge(row_vars)?;
        let query = io.challenge(vars)?;
        let batch = io.challenge(claims.len())?;
        let active_col: Vec<_> = (0..width).map(|i| F::from_bool(i < shape.width)).collect();
        let first: Vec<_> = (0..width).map(|i| F::from_bool(i == 0)).collect();
        let row_mask: Vec<_> = (0..rows).map(|i| F::from_bool(i < shape.rows)).collect();
        let mask: Vec<_> = weights(&row_query)
            .into_iter()
            .zip(&row_mask)
            .map(|(q, &r)| q * r)
            .collect();
        let padded = |values| {
            Shape {
                rows: 1,
                width: shape.width,
            }
            .padded(values, F::zero(), F::from_i32)
        };
        let (gain, bias) = (padded(self.p.gain), padded(self.p.bias));
        // Batch the zero-sum moment constraints and affine relation with independent challenges.
        // Both reductions then use the same input and row-statistic evaluations.
        let expression = |values: &[F]| {
            let [x, a, first, m, gain, bias, active, output, eq, z, y, rem, gap, arem]: [F; 14] =
                values[..14].try_into().expect("normalization columns");
            let r = &values[14..];
            let [var, vr, vg, dev, rt, rg]: [F; 6] =
                r[..6].try_into().expect("normalization statistics");
            let (centered, mean_check) = if stats == 9 {
                let [mean, rem, gap]: [F; 3] = r[6..9].try_into().expect("mean statistics");
                (
                    x - mean,
                    mix[4] * (a * x - first * (w * mean + rem))
                        + mix[5] * first * (rem + gap - w + F::one()),
                )
            } else {
                (x, F::zero())
            };
            let moments = [
                a * centered * centered - first * (w * (var - epsilon) + vr),
                first * (vr + vg - w + F::one()),
                first * (var - dev * dev - rt),
                first * (dev + dev - rt - rg),
            ];
            let affine = [
                scale * centered - dev * z - rem,
                dev - F::one() - rem - gap,
                gain * z - scale * (y - bias) - arem,
            ];
            m * (mean_check + dot(moments, mix.iter().copied()))
                + active * (output * y + eq * dot(affine, mix[checks..].iter().copied()))
        };
        let sum = dot(claims.iter().map(|q| q.value), batch.iter().copied());
        let (end, [x, cs @ ..]) = io.fold(
            instance(sum, vars, 4),
            || {
                let (input, trace) = trace();
                let col = |v| Column::repeat(v, rows);
                let row = |v| Column::rows(v, width);
                let mut cols = vec![
                    Column::owned(shape.padded(input, F::zero(), F::from_i32)),
                    col(active_col.as_slice().into()),
                    col(first.as_slice().into()),
                    row(mask.as_slice().into()),
                    col(gain.as_slice().into()),
                    col(bias.as_slice().into()),
                    Column::owned(
                        row_mask
                            .iter()
                            .flat_map(|&r| active_col.iter().map(move |&c| r * c))
                            .collect(),
                    ),
                    Column::owned(query_column(claims, &batch)),
                    Column::owned(weights(&query)),
                ];
                cols.extend(trace.cells.iter().map(|c| Column::borrowed(c)));
                cols.extend(trace.rows[..stats].iter().map(|r| row(r.as_slice().into())));
                cols
            },
            [0, 9, 10, 11, 12, 13],
            expression,
        )?;
        let p = &end.challenges;
        let (r, c) = p.split_at(row_vars);
        let rs: [F; 9] = io.values_prefix(stats, || {
            std::array::from_fn(|i| {
                if i < stats {
                    trace().1.rows[i].as_slice().evaluate(r)
                } else {
                    F::zero()
                }
            })
        })?;
        io.check_value(&end, || {
            let a = active_col.as_slice().evaluate(c);
            let mut values = [
                x,
                a,
                first.as_slice().evaluate(c),
                mask.as_slice().evaluate(r),
                gain.as_slice().evaluate(c),
                bias.as_slice().evaluate(c),
                row_mask.as_slice().evaluate(r) * a,
                query_weight(claims, &batch, p),
                eq(&query, p),
            ]
            .to_vec();
            values.extend(cs);
            values.extend(&rs[..stats]);
            expression(&values)
        })?;
        self.rows.check(&[(r, &rs[..stats])], io)?;
        self.cells.check(&[(p, cs)], io)?;
        Ok(Claim {
            point: end.challenges,
            value: x,
        })
    }
}

#[cfg(all(test, feature = "akita"))]
mod tests {
    use super::*;

    #[test]
    fn normalization_uses_epsilon_in_variance_units() {
        let shape = Shape::new(1, 2).unwrap();
        for kind in [NormalizationKind::Layer, NormalizationKind::Rms] {
            let mut layout = Layout::default();
            let norm = Norm::new(
                &mut layout,
                shape,
                14,
                NormalizationParameters {
                    kind,
                    gain: &[1 << 14; 2],
                    bias: &[0; 2],
                    epsilon_variance_units: 268,
                },
            )
            .unwrap();
            let trace = norm
                .execute::<jolt_akita::AkitaField>(&[1, -1], &mut layout.witness())
                .unwrap();
            // The integer deviation is floor(sqrt(1 + 268)) = 16.
            assert_eq!(trace.output, [1024, -1024]);
        }
    }
}
