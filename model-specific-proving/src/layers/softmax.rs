use jolt_transcript::Transcript;
use rayon::prelude::*;
use std::sync::Arc;

use crate::protocol::{
    lookup::{Batches, Lookup, Read, Table},
    pcs::{bind_i32, Advice, Layout},
    range::{Range, Ranges},
    session::{instance, Claim, Column, ProofField, Session},
    tensor::{dot, eq, weights, Shape},
    Result,
};

#[derive(Clone, Copy)]
pub struct SoftmaxTables<'a> {
    pub probabilities: &'a [i32],
    pub sums: &'a [i32],
    pub limit: i32,
}

#[derive(Clone, Copy)]
pub enum SoftmaxMask {
    Full,
    Causal,
    Sliding(usize),
}

pub(crate) struct Softmax<'a> {
    shape: Shape,
    row_start: usize,
    mask: SoftmaxMask,
    tables: SoftmaxTables<'a>,
    previous: Table<'a>,
    rows: Ranges,
    excess: Ranges,
    lookup: Lookup,
}

pub(crate) struct SoftmaxTrace<F> {
    pub(crate) output: Vec<i32>,
    shifts: Vec<i64>,
    rows: [Vec<F>; 3],
    indices: Arc<Vec<u32>>,
}

pub(crate) struct SoftmaxRead<F> {
    cell: Vec<F>,
    address: F,
    values: [F; 3],
    row_ranges: [F; 3],
    excess: F,
}

impl<'a> Softmax<'a> {
    pub(crate) fn new(
        layout: &mut Layout,
        shape: Shape,
        mask: SoftmaxMask,
        tables: SoftmaxTables<'a>,
    ) -> Result<Self> {
        if !matches!(mask, SoftmaxMask::Full) && shape.rows != shape.width {
            return Err("invalid Softmax parameters");
        }
        Self::with_row_start(layout, shape, mask, tables, 0)
    }

    #[cfg(feature = "akita")]
    pub(crate) fn block(&self, layout: &mut Layout, start: usize, rows: usize) -> Result<Self> {
        Self::with_row_start(
            layout,
            Shape::new(rows, self.shape.width)?,
            self.mask,
            self.tables,
            start,
        )
    }

    fn with_row_start(
        layout: &mut Layout,
        shape: Shape,
        mask: SoftmaxMask,
        tables: SoftmaxTables<'a>,
        row_start: usize,
    ) -> Result<Self> {
        let len = tables.sums.len();
        if !(2..=1 << 31).contains(&len)
            || len != tables.probabilities.len()
            || !(1..i32::MAX).contains(&tables.limit)
            || tables.sums[len - 2] != 0
            || tables.probabilities[len - 1] != 0
            || shape.width > 1 << 30
            || tables
                .sums
                .iter()
                .chain(tables.probabilities)
                .any(|&x| x < 0)
            || tables.sums.windows(2).any(|w| w[0] < w[1])
            || matches!(mask, SoftmaxMask::Sliding(0))
            || (!matches!(mask, SoftmaxMask::Full) && row_start + shape.rows > shape.width)
        {
            return Err("invalid Softmax parameters");
        }
        let previous = Table::Prepend(tables.limit + 1, &tables.sums[..len - 1]);
        let active = match mask {
            SoftmaxMask::Sliding(n) => n.min(shape.width),
            _ => shape.width,
        };
        let bound =
            active as u64 * (tables.sums[0].max(tables.limit + 1) as u64) - tables.limit as u64 - 1;
        let bits = |bound: u64| Range::unsigned((64 - bound.leading_zeros()).max(1) as usize);
        let rows = Ranges::new(
            layout,
            Shape::new(shape.rows, 1)?,
            [Range::signed(34), bits(tables.limit as u64), bits(bound)],
        )?;
        let excess = Ranges::new(layout, shape, [Range::unsigned(34)])?;
        Ok(Self {
            shape,
            row_start,
            mask,
            tables,
            previous,
            rows,
            excess,
            lookup: Lookup::new(layout, shape.vars(), len)?,
        })
    }

    fn active(&self, row: usize) -> std::ops::Range<usize> {
        let row = self.row_start + row;
        match self.mask {
            SoftmaxMask::Full => 0..self.shape.width,
            SoftmaxMask::Causal => 0..row + 1,
            SoftmaxMask::Sliding(n) => (row + 1).saturating_sub(n)..row + 1,
        }
    }

    fn masks_at<F: ProofField>(&self, point: &[F]) -> (F, F) {
        let (row, col) = point.split_at(self.shape.row_vars());
        let columns = weights(col);
        let mut prefix = vec![F::zero()];
        let mut total = F::zero();
        for &weight in &columns {
            total += weight;
            prefix.push(total);
        }
        let mut mask = F::zero();
        let mut first = F::zero();
        for (row, weight) in weights(row).into_iter().take(self.shape.rows).enumerate() {
            let active = self.active(row);
            mask += weight * (prefix[active.end] - prefix[active.start]);
            first += weight * columns[active.start];
        }
        (mask, first)
    }

    pub(crate) fn bind(&self, t: &mut impl Transcript) {
        let mask = match self.mask {
            SoftmaxMask::Full => 0,
            SoftmaxMask::Causal => 1,
            SoftmaxMask::Sliding(n) => n as u64 + 1,
        };
        t.append(&mask);
        t.append(&(self.tables.limit as u64));
        bind_i32(t, self.tables.sums);
        bind_i32(t, self.tables.probabilities);
    }

    pub(crate) fn execute<F: ProofField>(
        &self,
        x: &[i32],
        advice: &mut Advice,
    ) -> Result<SoftmaxTrace<F>> {
        self.shape.check(x)?;
        let last = (self.tables.sums.len() - 1) as i64;
        let limit = i64::from(self.tables.limit);
        let mut output = Vec::with_capacity(x.len());
        let mut raw_rows: [Vec<i64>; 3] = std::array::from_fn(|_| Vec::new());
        let stride = self.shape.width.next_power_of_two();
        let mut indices = vec![0; 1 << self.shape.vars()];
        let mut excess = Vec::with_capacity(x.len());
        let sums = Table::Slice(self.tables.sums);
        for (r, row) in x.chunks_exact(self.shape.width).enumerate() {
            let active = self.active(r);
            let sum = |shift: i64, table: Table<'_>| {
                row[active.clone()]
                    .iter()
                    .map(|&x| i64::from(table.at((shift - i64::from(x)).min(last) as usize)))
                    .sum::<i64>()
            };
            let mut lo = i64::from(*row[active.clone()].iter().max().expect("valid row"));
            let mut hi = lo + last;
            while lo < hi {
                let mid = (lo + hi).div_euclid(2);
                if sum(mid, sums) <= limit {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            let gap = limit - sum(lo, sums);
            let prev = sum(lo, self.previous) - limit - 1;
            for (c, v) in raw_rows.iter_mut().zip([lo, gap, prev]) {
                c.push(v);
            }
            for (i, &score) in row.iter().enumerate() {
                let real = active.contains(&i);
                let difference = if real { lo - i64::from(score) } else { 0 };
                let index = difference.min(last) as usize;
                let prob = self.tables.probabilities[index];
                output.push(if real { prob } else { 0 });
                indices[r * stride + i] = index as u32;
                excess.push(difference - index as i64);
            }
        }
        let indices = Arc::new(indices);
        self.lookup.record(advice, Arc::clone(&indices))?;
        self.rows
            .record(advice, &raw_rows.each_ref().map(Vec::as_slice))?;
        self.excess.record(advice, &[&excess])?;
        let row_shape = Shape {
            rows: self.shape.rows,
            width: 1,
        };
        let shifts = raw_rows[0].clone();
        let rows = raw_rows.map(|c| row_shape.padded(&c, F::zero(), F::from_i64));
        Ok(SoftmaxTrace {
            output,
            shifts,
            rows,
            indices,
        })
    }

    pub(crate) fn reverse<'w, F: ProofField>(
        &self,
        claims: &[Claim<F>],
        trace: impl Fn(usize) -> (&'w SoftmaxTrace<F>, &'w [i32]) + Sync,
        io: &mut Session<'_, '_, F>,
    ) -> Result<Vec<(Claim<F>, SoftmaxRead<F>)>> {
        let vars = self.shape.vars();
        let rows = self.shape.row_vars();
        let stride = self.shape.width.next_power_of_two();
        io.pow(vars + rows + 3)?;
        let mix = io.challenge(3)?;
        let clamp = io.challenge(vars)?;
        let total = io.challenge(rows)?;
        let limit = F::from_i32(self.tables.limit);
        let last = F::from_u64((self.tables.sums.len() - 1) as u64);
        let expression = |v: &[F]| {
            let condition = v[6] - v[0] - v[1] - v[5] + mix[0] * v[5] * (v[1] - last);
            let sum = v[9] * (mix[1] * v[3] + mix[2] * v[4])
                + v[10] * (mix[1] * (v[7] - limit) - mix[2] * (v[8] + limit + F::one()));
            v[11] * v[9] * v[2] + v[12] * v[9] * condition + v[13] * sum
        };
        let batch = io.point(claims.len())?;
        let sum = dot(claims.iter().map(|q| q.value), batch.iter().copied());
        let columns = |i: usize| {
            let (t, scores) = trace(i);
            let score = move |row, col| {
                if row < self.shape.rows && col < self.shape.width {
                    i64::from(scores[row * self.shape.width + col])
                } else {
                    0
                }
            };
            let size = 1 << vars;
            std::array::from_fn::<_, 14, _>(|j| match j {
                0 => Column::source(size, move |i| F::from_i64(score(i / stride, i % stride))),
                1 => Column::source(size, move |i| F::from_u64(t.indices[i] as u64)),
                2 => Column::source(size, move |i| {
                    F::from_i32(self.tables.probabilities[t.indices[i] as usize])
                }),
                3 => Column::source(size, move |i| {
                    F::from_i32(self.tables.sums[t.indices[i] as usize])
                }),
                4 => Column::source(size, move |i| {
                    F::from_i32(self.previous.at(t.indices[i] as usize))
                }),
                5 => Column::source(size, move |i| {
                    let (row, col) = (i / stride, i % stride);
                    if row < self.shape.rows && self.active(row).contains(&col) {
                        F::from_i64(t.shifts[row] - score(row, col) - t.indices[i] as i64)
                    } else {
                        F::zero()
                    }
                }),
                6..=8 => Column::rows(t.rows[j - 6].as_slice().into(), stride),
                9 => Column::source(size, |i| {
                    let row = i / stride;
                    F::from_bool(row < self.shape.rows && self.active(row).contains(&(i % stride)))
                }),
                10 => Column::source(size, |i| {
                    let row = i / stride;
                    F::from_bool(row < self.shape.rows && self.active(row).start == i % stride)
                }),
                11 => Column::equality(&claims[i].point),
                12 => Column::equality(&clamp),
                _ => Column::rows(weights(&total).into(), stride),
            })
        };
        let (end, values) = io.fold_weighted(
            instance(sum, vars, 4),
            &batch,
            || (0..claims.len()).into_par_iter().map(columns).collect(),
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            9,
            expression,
        )?;
        let point = &end.challenges;
        io.check_value(&end, || {
            let mut x = [F::zero(); 14];
            let (mask, first) = self.masks_at(point);
            x[9..].copy_from_slice(&[
                mask,
                first,
                F::zero(),
                eq(&clamp, point),
                eq(&total, &point[..rows]),
            ]);
            claims
                .iter()
                .zip(&values)
                .zip(batch)
                .map(|((q, v), w)| {
                    x[..9].copy_from_slice(v);
                    x[11] = eq(&q.point, point);
                    w * expression(&x)
                })
                .sum()
        })?;
        Ok(values
            .into_iter()
            .map(|v| {
                (
                    Claim {
                        point: point.clone(),
                        value: v[0],
                    },
                    SoftmaxRead {
                        cell: point.clone(),
                        address: v[1],
                        values: [v[2], v[3], v[4]],
                        row_ranges: [v[6], v[7], v[8]],
                        excess: v[5],
                    },
                )
            })
            .collect())
    }

    pub(crate) fn defer<'s, 'w, F: ProofField>(
        reads: Vec<(&'s Self, SoftmaxRead<F>)>,
        trace: impl Fn(usize) -> &'w SoftmaxTrace<F>,
        io: &mut Session<'_, '_, F>,

        lookups: &mut Batches<'s, 'w, F>,
    ) -> Result<()> {
        for (i, (op, read)) in reads.into_iter().enumerate() {
            op.rows
                .check(&[(&read.cell[..op.shape.row_vars()], read.row_ranges)], io)?;
            op.excess.check(&[(&read.cell, [read.excess])], io)?;
            lookups.push(
                &[
                    Table::Slice(op.tables.probabilities),
                    Table::Slice(op.tables.sums),
                    op.previous,
                ],
                &op.lookup,
                Read {
                    cell: read.cell,
                    address: read.address,
                    output: Box::new(read.values),
                    witness: io.proving().then(|| trace(i).indices.as_slice()),
                },
            );
        }
        Ok(())
    }
}
