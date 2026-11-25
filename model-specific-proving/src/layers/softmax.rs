use jolt_transcript::Transcript;
use rayon::prelude::*;
use std::{borrow::Cow, sync::Arc};

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
    excess: Option<Ranges>,
    tiles: Vec<Tile>,
    levels: Vec<usize>,
}

const TILE: usize = 64;

struct Tile {
    row: usize,
    col: usize,
    shape: Shape,
    lookup: Lookup,
}

impl Tile {
    fn new(
        layout: &mut Layout,
        row: usize,
        col: usize,
        shape: Shape,
        table: usize,
    ) -> Result<Self> {
        Ok(Self {
            row,
            col,
            shape,
            lookup: Lookup::new(layout, shape.vars(), table)?,
        })
    }

    fn read<F: ProofField>(&self, row_point: &[F], col_point: &[F]) -> (F, Vec<F>) {
        let split = |point: &[F], start: usize, vars: usize| {
            let fixed = point.len() - vars;
            point[..fixed]
                .iter()
                .enumerate()
                .fold(F::one(), |weight, (i, &r)| {
                    weight
                        * if start >> (point.len() - i - 1) & 1 == 0 {
                            F::one() - r
                        } else {
                            r
                        }
                })
        };
        let rows = self.shape.row_vars();
        let cols = self.shape.vars() - rows;
        let scale = split(row_point, self.row, rows) * split(col_point, self.col, cols);
        (
            scale,
            [
                &row_point[row_point.len() - rows..],
                &col_point[col_point.len() - cols..],
            ]
            .concat(),
        )
    }
}

pub(crate) struct SoftmaxTrace<F> {
    shifts: Vec<i64>,
    rows: [Vec<F>; 3],
    indices: Vec<Arc<Vec<u32>>>,
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
        score_bound: Option<u128>,
    ) -> Result<Self> {
        if !matches!(mask, SoftmaxMask::Full) && shape.rows != shape.width {
            return Err("invalid Softmax parameters");
        }
        Self::with_row_start(layout, shape, mask, tables, 0, score_bound)
    }

    #[cfg(feature = "akita")]
    pub(crate) fn block(&self, layout: &mut Layout, start: usize, rows: usize) -> Result<Self> {
        Self::with_row_start(
            layout,
            Shape::new(rows, self.shape.width)?,
            self.mask,
            self.tables,
            start,
            None,
        )
    }

    fn with_row_start(
        layout: &mut Layout,
        shape: Shape,
        mask: SoftmaxMask,
        tables: SoftmaxTables<'a>,
        row_start: usize,
        score_bound: Option<u128>,
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
        // At row_max + step, every entry is at least step below the shift,
        // and even the longest row has sum at most the public limit.
        let step = tables
            .sums
            .partition_point(|&v| (v as u64) * active as u64 > tables.limit as u64);
        let no_clip = score_bound.is_some_and(|bound| {
            bound <= i32::MAX as u128 && 2 * bound + (step as u128) < len as u128
        });
        let excess = (!no_clip)
            .then(|| Ranges::new(layout, shape, [Range::unsigned(34)]))
            .transpose()?;
        let mut tiles = Vec::new();
        let mut levels = Vec::new();
        if matches!(mask, SoftmaxMask::Causal)
            && row_start == 0
            && shape.rows == shape.width
            && shape.width > TILE
        {
            for start in (0..shape.rows).step_by(TILE) {
                tiles.push(Tile::new(
                    layout,
                    start,
                    start,
                    Shape::new(TILE, TILE)?,
                    len,
                )?);
            }
            let mut side = TILE;
            while side < shape.width {
                levels.push(tiles.len());
                for row in (side..shape.rows).step_by(2 * side) {
                    tiles.push(Tile::new(
                        layout,
                        row,
                        row - side,
                        Shape::new(side, side)?,
                        len,
                    )?);
                }
                side *= 2;
            }
        } else {
            tiles.push(Tile::new(layout, 0, 0, shape, len)?);
        }
        Ok(Self {
            shape,
            row_start,
            mask,
            tables,
            previous,
            rows,
            excess,
            tiles,
            levels,
        })
    }

    pub(super) fn has_excess(&self) -> bool {
        self.excess.is_some()
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

    fn location(&self, row: usize, col: usize) -> Option<(usize, usize)> {
        if row >= self.shape.rows || col >= self.shape.width {
            return None;
        }
        if self.levels.is_empty() {
            return Some((0, row * self.shape.width.next_power_of_two() + col));
        }
        let blocks = (row ^ col) / TILE;
        let tile = if blocks == 0 {
            row / TILE
        } else if row > col {
            let high = (row ^ col).ilog2() as usize;
            self.levels[blocks.ilog2() as usize] + (row >> (high + 1))
        } else {
            return None;
        };
        let tile_plan = &self.tiles[tile];
        Some((
            tile,
            (row - tile_plan.row) * tile_plan.shape.width + col - tile_plan.col,
        ))
    }

    fn index<F>(&self, trace: &SoftmaxTrace<F>, row: usize, col: usize) -> usize {
        self.location(row, col)
            .map_or(0, |(tile, offset)| trace.indices[tile][offset] as usize)
    }

    pub(super) fn probabilities<'s, F>(
        &'s self,
        trace: &'s SoftmaxTrace<F>,
    ) -> impl Iterator<Item = Cow<'s, [i32]>> + 's
    where
        F: 's,
    {
        (0..self.shape.rows).map(move |row| {
            let active = self.active(row);
            let mut output = vec![0; self.shape.width];
            for col in active {
                output[col] = self.tables.probabilities[self.index(trace, row, col)];
            }
            Cow::Owned(output)
        })
    }

    pub(super) fn bind_probabilities<F: ProofField>(
        &self,
        trace: &SoftmaxTrace<F>,
        point: &[F],
    ) -> Vec<F> {
        let mut output = vec![F::zero(); self.shape.width.next_power_of_two()];
        for (row, weight) in weights(point).into_iter().take(self.shape.rows).enumerate() {
            for col in self.active(row) {
                output[col] +=
                    weight * F::from_i32(self.tables.probabilities[self.index(trace, row, col)]);
            }
        }
        output
    }

    pub(crate) fn bind(&self, t: &mut impl Transcript) {
        let mask = match self.mask {
            SoftmaxMask::Full => 0,
            SoftmaxMask::Causal => 1,
            SoftmaxMask::Sliding(n) => n as u64 + 1,
        };
        t.append(&mask);
        t.append(&u64::from(self.has_excess()));
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
        let mut raw_rows: [Vec<i64>; 3] = std::array::from_fn(|_| Vec::new());
        let mut indices = self
            .tiles
            .iter()
            .map(|tile| vec![0; 1 << tile.shape.vars()])
            .collect::<Vec<_>>();
        let mut excess = self.excess.as_ref().map(|_| Vec::with_capacity(x.len()));
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
                if real {
                    let (tile, offset) = self.location(r, i).expect("active Softmax cell");
                    indices[tile][offset] = index as u32;
                }
                if let Some(excess) = &mut excess {
                    excess.push(difference - index as i64);
                } else if difference != index as i64 {
                    return Err("Softmax exceeds the public score bound");
                }
            }
        }
        let indices = indices.into_iter().map(Arc::new).collect::<Vec<_>>();
        for (tile, values) in self.tiles.iter().zip(&indices) {
            tile.lookup.record(advice, Arc::clone(values))?;
        }
        self.rows
            .record(advice, &raw_rows.each_ref().map(Vec::as_slice))?;
        if let (Some(ranges), Some(excess)) = (&self.excess, excess) {
            ranges.record(advice, &[&excess])?;
        }
        let row_shape = Shape {
            rows: self.shape.rows,
            width: 1,
        };
        let shifts = raw_rows[0].clone();
        let rows = raw_rows.map(|c| row_shape.padded(&c, F::zero(), F::from_i64));
        Ok(SoftmaxTrace {
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
            let condition = v[6] - v[0] - v[1]
                + if self.has_excess() {
                    v[5] * (mix[0] * (v[1] - last) - F::one())
                } else {
                    F::zero()
                };
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
            let address = move |i: usize| self.index(t, i / stride, i % stride);
            std::array::from_fn::<_, 14, _>(|j| match j {
                0 => Column::source(size, move |i| F::from_i64(score(i / stride, i % stride))),
                1 => Column::source(size, move |i| F::from_u64(address(i) as u64)),
                2 => Column::source(size, move |i| {
                    F::from_i32(self.tables.probabilities[address(i)])
                }),
                3 => Column::source(size, move |i| F::from_i32(self.tables.sums[address(i)])),
                4 => Column::source(size, move |i| F::from_i32(self.previous.at(address(i)))),
                5 if !self.has_excess() => Column::repeat(vec![F::zero()].into(), size),
                5 => Column::source(size, move |i| {
                    let (row, col) = (i / stride, i % stride);
                    if row < self.shape.rows && self.active(row).contains(&col) {
                        F::from_i64(t.shifts[row] - score(row, col) - address(i) as i64)
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
            instance(sum, vars, if self.has_excess() { 4 } else { 3 }),
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
            if let Some(excess) = &op.excess {
                excess.check(&[(&read.cell, [read.excess])], io)?;
            }
            let tables = [
                Table::Slice(op.tables.probabilities),
                Table::Slice(op.tables.sums),
                op.previous,
            ];
            if let [tile] = op.tiles.as_slice() {
                lookups.push(
                    &tables,
                    &tile.lookup,
                    Read {
                        cell: read.cell,
                        address: read.address,
                        output: Box::new(read.values),
                        witness: io.proving().then(|| trace(i).indices[0].as_slice()),
                    },
                );
                continue;
            }

            let (row, col) = read.cell.split_at(op.shape.row_vars());
            let defaults = [
                F::zero(),
                F::from_i32(tables[0].at(0)),
                F::from_i32(tables[1].at(0)),
                F::from_i32(tables[2].at(0)),
            ];
            let mut sum = defaults;
            for (j, tile) in op.tiles.iter().enumerate() {
                let (weight, cell) = tile.read(row, col);
                let witness = io.proving().then(|| trace(i).indices[j].as_slice());
                let values = io.values(|| {
                    let (r, c) = cell.split_at(tile.shape.row_vars());
                    let columns = weights(c);
                    let mut values = [F::zero(); 4];
                    for (indices, row_weight) in witness
                        .expect("Softmax tile witness")
                        .chunks_exact(columns.len())
                        .zip(weights(r))
                    {
                        let mut partial = [F::zero(); 4];
                        for (&address, &column_weight) in indices.iter().zip(&columns) {
                            partial[0] += column_weight * F::from_u64(u64::from(address));
                            for (value, table) in partial[1..].iter_mut().zip(tables) {
                                *value += column_weight * F::from_i32(table.at(address as usize));
                            }
                        }
                        for (v, part) in values.iter_mut().zip(partial) {
                            *v += row_weight * part;
                        }
                    }
                    values
                })?;
                for ((total, value), default) in sum.iter_mut().zip(values).zip(defaults) {
                    *total += weight * (value - default);
                }
                lookups.push(
                    &tables,
                    &tile.lookup,
                    Read {
                        cell,
                        address: values[0],
                        output: Box::new([values[1], values[2], values[3]]),
                        witness,
                    },
                );
            }
            if sum != [read.address, read.values[0], read.values[1], read.values[2]] {
                return Err("invalid tiled Softmax lookup evaluation");
            }
        }
        Ok(())
    }
}
