use jolt_transcript::Transcript;
use rayon::prelude::*;
use std::{borrow::Cow, sync::Arc};

use crate::protocol::{
    lookup::{Addresses, Batches, Lookup, Read, Table},
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
}

pub(crate) struct Softmax<'a> {
    shape: Shape,
    mask: SoftmaxMask,
    tables: SoftmaxTables<'a>,
    previous: Table<'a>,
    combined: Table<'a>,
    radix: u128,
    rows: Ranges,
    excess: Option<Ranges>,
    compact: bool,
    may_clip: bool,
    tiles: Vec<Tile>,
    levels: Vec<usize>,
}

const TILE: usize = 64;

#[derive(Clone, Default)]
pub(crate) struct ClipSpec {
    /// None encodes every causal block, Some encodes a sorted list of occupied blocks.
    blocks: Option<Vec<usize>>,
    bits: usize,
    width: usize,
    pub(super) narrow_score: bool,
}

impl ClipSpec {
    pub(crate) fn bind(&self, t: &mut impl Transcript) {
        t.append(&(self.bits as u64));
        t.append(&(self.width as u64));
        t.append(&u64::from(self.narrow_score));
        t.append(&u64::from(self.blocks.is_some()));
        if let Some(blocks) = &self.blocks {
            t.append(&(blocks.len() as u64));
            for &block in blocks {
                t.append(&(block as u64));
            }
        }
    }
}

pub(crate) enum ClipPlan {
    Overflow(OverflowPlan),
    Narrow(Ranges),
}

impl ClipPlan {
    pub(super) fn narrows_score(&self) -> bool {
        matches!(self, Self::Narrow(_))
    }
}

pub(crate) struct OverflowPlan {
    ranges: Ranges,
    blocks: Vec<usize>,
    shape: Shape,
    per_row: usize,
}

impl OverflowPlan {
    fn values(
        &self,
        overflow: &std::collections::BTreeMap<(usize, usize), Box<[u32; TILE]>>,
    ) -> Vec<u32> {
        let mut values = Vec::with_capacity(self.blocks.len() * self.shape.width);
        for &block in &self.blocks {
            let row = block / self.per_row;
            let start = block % self.per_row * self.shape.width;
            for c in (0..self.shape.width).step_by(TILE) {
                let source = overflow.get(&(row, (start + c) / TILE));
                if let Some(source) = source {
                    values.extend_from_slice(&source[..self.shape.width.min(TILE)]);
                } else {
                    values.resize(values.len() + self.shape.width.min(TILE), 0);
                }
            }
        }
        values
    }

    fn check<F: ProofField>(
        &self,
        q: &[F],
        value: F,
        witness: impl FnOnce() -> Vec<u32>,
        io: &mut Session<'_, '_, F>,
    ) -> Result<()> {
        let (row, col) = q.split_at(q.len() / 2);
        let row = weights::<F>(row);
        let col = weights::<F>(col);
        let mut coefficients = Vec::with_capacity(1 << self.shape.vars());
        for &block in &self.blocks {
            let weight = row[block / self.per_row];
            let start = (block % self.per_row) * self.shape.width;
            coefficients.extend(col[start..][..self.shape.width].iter().map(|&c| weight * c));
        }
        coefficients.resize(1 << self.shape.vars(), F::zero());
        let evaluation = io.public_dot(value, &coefficients, || {
            self.shape
                .padded(&witness(), F::zero(), |x| F::from_u64(u64::from(x)))
        })?;
        self.ranges
            .check(&[(&evaluation.point, [evaluation.value])], io)
    }
}

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
    indices: Vec<Arc<Addresses>>,
    overflow: std::collections::BTreeMap<(usize, usize), Box<[u32; 64]>>,
}

pub(crate) struct SoftmaxRead<F> {
    cell: Vec<F>,
    address: F,
    values: [F; 2],
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
        compact: bool,
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
            || (matches!(mask, SoftmaxMask::Causal) && shape.rows != shape.width)
        {
            return Err("invalid Softmax parameters");
        }
        let previous = Table::Prepend(tables.limit + 1, &tables.sums[..len - 1]);
        let active = shape.width;
        let bound =
            active as u64 * (tables.sums[0].max(tables.limit + 1) as u64) - tables.limit as u64 - 1;
        let bits = |bound: u64| Range::unsigned((64 - bound.leading_zeros()).max(1) as usize);
        let rows = Ranges::new(
            layout,
            Shape::new(shape.rows, 1)?,
            [Range::signed(34), bits(tables.limit as u64), bits(bound)],
        )?;
        // Bounds on the two integer row residuals, including the full ranges of the slacks.
        let limit = tables.limit as u128;
        let max_sum = shape.width as u128 * tables.sums[0] as u128;
        let max_gap = limit.next_power_of_two() * 2 - 1;
        let max_prev_sum = shape.width as u128 * tables.sums[0].max(tables.limit + 1) as u128;
        let max_prev_gap = u128::from(bound).next_power_of_two() * 2 - 1;
        let radix = max_sum + limit + max_gap + 1;
        let packed_bound = (radix - 1) + radix * (max_prev_sum + limit + 1 + max_prev_gap);
        layout.integer_bits = layout
            .integer_bits
            .max(u128::BITS - packed_bound.leading_zeros() + 1);
        let combined = Table::SoftmaxSums {
            sums: tables.sums,
            first: tables.limit + 1,
            radix,
        };
        // At row_max + step, every entry is at least step below the shift,
        // and even the longest row has sum at most the public limit.
        let step = tables
            .sums
            .partition_point(|&v| (v as u64) * active as u64 > tables.limit as u64);
        let no_clip = score_bound.is_some_and(|bound| {
            bound <= i32::MAX as u128 && 2 * bound + (step as u128) < len as u128
        });
        let excess = (!no_clip && !compact)
            .then(|| Ranges::new(layout, shape, [Range::unsigned(34)]))
            .transpose()?;
        let mut tiles = Vec::new();
        let mut levels = Vec::new();
        if matches!(mask, SoftmaxMask::Causal) && shape.rows == shape.width && shape.width > TILE {
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
            mask,
            tables,
            previous,
            combined,
            radix,
            rows,
            excess,
            compact,
            may_clip: !no_clip,
            tiles,
            levels,
        })
    }

    pub(super) fn has_excess(&self) -> bool {
        self.may_clip
    }

    pub(super) fn clip_spec<F>(&self, trace: &SoftmaxTrace<F>) -> ClipSpec {
        let max = trace
            .overflow
            .values()
            .flat_map(|v| v.iter())
            .copied()
            .max()
            .unwrap_or(0);
        if max == 0 {
            return ClipSpec {
                narrow_score: self.narrow_score(trace),
                ..Default::default()
            };
        }
        let (width, blocks) = [TILE, TILE * 2, TILE * 4]
            .map(|width| width.min(self.shape.width))
            .into_iter()
            .map(|width| {
                let per_row = self.shape.width / width;
                let mut blocks: Vec<_> = trace
                    .overflow
                    .keys()
                    .map(|&(row, col)| row * per_row + col * TILE / width)
                    .collect();
                blocks.dedup();
                (width, blocks)
            })
            .min_by_key(|(width, blocks)| (blocks.len().next_power_of_two() * width, blocks.len()))
            .expect("nonempty clip layout candidates");
        let all: usize = (0..self.shape.rows).map(|row| row / width + 1).sum();
        ClipSpec {
            bits: ((u32::BITS - max.leading_zeros()) as usize).next_multiple_of(4),
            width,
            blocks: (blocks.len().next_power_of_two() < all.next_power_of_two()).then_some(blocks),
            narrow_score: false,
        }
    }

    pub(super) fn clip_plan(
        &self,
        layout: &mut Layout,
        spec: &ClipSpec,
    ) -> Result<Option<ClipPlan>> {
        if spec.bits == 0 {
            if spec.blocks.is_some() || spec.width != 0 {
                return Err("invalid empty Softmax clip plan");
            }
            if !spec.narrow_score {
                return Ok(None);
            }
            if !self.can_narrow_score() {
                return Err("invalid Softmax score range plan");
            }
            return Ok(Some(ClipPlan::Narrow(Ranges::new(
                layout,
                Shape::new(self.shape.rows, 1)?,
                [Range::signed(31)],
            )?)));
        }
        if spec.narrow_score {
            return Err("invalid clipped Softmax score range");
        }
        if !self.compact || !self.may_clip || spec.bits > 32 || !spec.bits.is_multiple_of(4) {
            return Err("invalid Softmax clip width");
        }
        let width = spec.width;
        if ![TILE, TILE * 2, TILE * 4]
            .iter()
            .any(|&w| w.min(self.shape.width) == width)
        {
            return Err("invalid Softmax clip block width");
        }
        let per_row = self.shape.width / width;
        let blocks = if let Some(blocks) = &spec.blocks {
            if blocks.is_empty()
                || blocks.windows(2).any(|pair| pair[0] >= pair[1])
                || blocks.iter().any(|&id| {
                    id / per_row >= self.shape.rows || (id % per_row) * width > id / per_row
                })
            {
                return Err("invalid Softmax clip blocks");
            }
            blocks.clone()
        } else {
            (0..self.shape.rows)
                .flat_map(|row| (0..=row / width).map(move |col| row * per_row + col))
                .collect()
        };
        let shape = Shape::new(blocks.len(), width)?;
        Ok(Some(ClipPlan::Overflow(OverflowPlan {
            ranges: Ranges::new(layout, shape, [Range::unsigned(spec.bits)])?,
            blocks,
            shape,
            per_row,
        })))
    }

    pub(super) fn record_clip<F>(
        &self,
        plan: &ClipPlan,
        trace: &SoftmaxTrace<F>,
        advice: &mut Advice,
    ) -> Result<()> {
        match plan {
            ClipPlan::Overflow(plan) => {
                plan.ranges.record(advice, &[&plan.values(&trace.overflow)])
            }
            ClipPlan::Narrow(range) => range.record(advice, &[&trace.shifts]),
        }
    }

    fn can_narrow_score(&self) -> bool {
        self.compact && self.may_clip && self.tables.sums.len() - 1 <= 1 << 30
    }

    pub(super) fn narrow_score<F>(&self, trace: &SoftmaxTrace<F>) -> bool {
        self.can_narrow_score()
            && trace.overflow.is_empty()
            && trace
                .shifts
                .iter()
                .all(|shift| (-(1i64 << 30)..(1i64 << 30)).contains(shift))
    }

    fn active(&self, row: usize) -> std::ops::Range<usize> {
        match self.mask {
            SoftmaxMask::Full => 0..self.shape.width,
            SoftmaxMask::Causal => 0..row + 1,
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
            .map_or(0, |(tile, offset)| trace.indices[tile].get(offset) as usize)
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
        t.append(&u64::from(matches!(self.mask, SoftmaxMask::Causal)));
        t.append(&u64::from(self.has_excess()));
        t.append(&(self.tables.limit as u64));
        bind_i32(t, self.tables.sums);
        bind_i32(t, self.tables.probabilities);
    }

    pub(super) fn score_tiles(&self) -> impl Iterator<Item = (usize, usize, Shape)> + '_ {
        self.tiles.iter().map(|t| (t.row, t.col, t.shape))
    }

    pub(super) fn score<F>(&self, trace: &SoftmaxTrace<F>, row: usize, col: usize) -> i32 {
        if row >= self.shape.rows || col > row {
            return 0;
        }
        let excess = trace
            .overflow
            .get(&(row, col / 64))
            .map_or(0, |values| values[col % 64]);
        (trace.shifts[row] - self.index(trace, row, col) as i64 - i64::from(excess)) as i32
    }

    pub(crate) fn execute<F: ProofField>(
        &self,
        x: &[i32],
        advice: &mut Advice,
    ) -> Result<SoftmaxTrace<F>> {
        self.shape.check(x)?;
        self.execute_rows(
            |r| &x[r * self.shape.width..][..self.shape.width],
            false,
            advice,
        )
    }

    pub(super) fn execute_causal<'w, F: ProofField>(
        &self,
        row: impl Fn(usize) -> &'w [i32],
        advice: &mut Advice,
    ) -> Result<SoftmaxTrace<F>> {
        self.execute_rows(row, true, advice)
    }

    fn execute_rows<'w, F: ProofField>(
        &self,
        score_row: impl Fn(usize) -> &'w [i32],
        causal: bool,
        advice: &mut Advice,
    ) -> Result<SoftmaxTrace<F>> {
        let last = (self.tables.sums.len() - 1) as i64;
        let limit = i64::from(self.tables.limit);
        let mut raw_rows: [Vec<i64>; 3] = std::array::from_fn(|_| Vec::new());
        let mut indices = self
            .tiles
            .iter()
            .map(|tile| {
                Addresses::zeros(
                    1 << tile.shape.vars(),
                    self.tables.sums.len().ilog2() as usize,
                )
            })
            .collect::<Vec<_>>();
        let mut excess = self
            .excess
            .as_ref()
            .map(|_| Vec::with_capacity(self.shape.rows * self.shape.width));
        let mut overflow = std::collections::BTreeMap::new();
        let sums = Table::Slice(self.tables.sums);
        for r in 0..self.shape.rows {
            let row = score_row(r);
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
                    indices[tile].set(offset, index as u32);
                }
                if let Some(excess) = &mut excess {
                    let amount = difference - index as i64;
                    excess.push(amount);
                    if causal && amount != 0 {
                        overflow
                            .entry((r, i / 64))
                            .or_insert_with(|| Box::new([0; 64]))[i % 64] =
                            amount.try_into().map_err(|_| "Softmax excess overflow")?;
                    }
                } else if difference != index as i64 {
                    if !causal || !self.may_clip {
                        return Err("Softmax exceeds the public score bound");
                    }
                    overflow
                        .entry((r, i / TILE))
                        .or_insert_with(|| Box::new([0; TILE]))[i % TILE] = (difference
                        - index as i64)
                        .try_into()
                        .map_err(|_| "Softmax excess overflow")?;
                }
            }
            if let Some(excess) = &mut excess {
                excess.resize((r + 1) * self.shape.width, 0);
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
            overflow,
        })
    }

    pub(crate) fn reverse<'w, F: ProofField>(
        &self,
        claims: &[Claim<F>],
        clips: &[Option<ClipPlan>],
        trace: impl Fn(usize) -> (&'w SoftmaxTrace<F>, Option<&'w [i32]>) + Sync,
        io: &mut Session<'_, '_, F>,
    ) -> Result<Vec<(Claim<F>, SoftmaxRead<F>)>> {
        let vars = self.shape.vars();
        let rows = self.shape.row_vars();
        let stride = self.shape.width.next_power_of_two();
        io.pow(vars + rows + 1)?;
        let mix = io.challenge(1)?[0];
        let clamp = io.challenge(vars)?;
        let total = io.challenge(rows)?;
        let limit = F::from_i32(self.tables.limit);
        let last = F::from_u64((self.tables.sums.len() - 1) as u64);
        let radix = F::from_u128(self.radix);
        let has_excess = self.excess.is_some()
            || clips
                .iter()
                .any(|clip| matches!(clip, Some(ClipPlan::Overflow(_))));
        let expression = |v: &[F]| {
            let condition = v[5] - v[0] - v[1]
                + if has_excess {
                    v[4] * (mix * (v[1] - last) - F::one())
                } else {
                    F::zero()
                };
            let sum = v[8] * v[3] + v[9] * (v[6] - limit - radix * (v[7] + limit + F::one()));
            v[10] * v[8] * v[2] + v[11] * v[8] * condition + v[12] * sum
        };
        let batch = io.point(claims.len())?;
        let sum = dot(claims.iter().map(|q| q.value), batch.iter().copied());
        let columns = |i: usize| {
            let (t, scores) = trace(i);
            let score = move |row, col| {
                if row < self.shape.rows && col < self.shape.width {
                    i64::from(scores.map_or_else(
                        || self.score(t, row, col),
                        |dense| dense[row * self.shape.width + col],
                    ))
                } else {
                    0
                }
            };
            let size = 1 << vars;
            let address = move |i: usize| self.index(t, i / stride, i % stride);
            std::array::from_fn::<_, 13, _>(|j| match j {
                0 => Column::source(size, move |i| F::from_i64(score(i / stride, i % stride))),
                1 => Column::source(size, move |i| F::from_u64(address(i) as u64)),
                2 => Column::source(size, move |i| {
                    F::from_i32(self.tables.probabilities[address(i)])
                }),
                3 => Column::source(size, move |i| self.combined.field_at(address(i))),
                4 if self.excess.is_none()
                    && !matches!(clips.get(i), Some(Some(ClipPlan::Overflow(_)))) =>
                {
                    Column::repeat(vec![F::zero()].into(), size)
                }
                4 => Column::source(size, move |i| {
                    let (row, col) = (i / stride, i % stride);
                    if row < self.shape.rows && self.active(row).contains(&col) {
                        F::from_i64(t.shifts[row] - score(row, col) - address(i) as i64)
                    } else {
                        F::zero()
                    }
                }),
                5..=7 => Column::rows(t.rows[j - 5].as_slice().into(), stride),
                8 => Column::source(size, |i| {
                    let row = i / stride;
                    F::from_bool(row < self.shape.rows && self.active(row).contains(&(i % stride)))
                }),
                9 => Column::source(size, |i| {
                    let row = i / stride;
                    F::from_bool(row < self.shape.rows && self.active(row).start == i % stride)
                }),
                10 => Column::equality(&claims[i].point),
                11 => Column::equality(&clamp),
                _ => Column::rows(weights(&total).into(), stride),
            })
        };
        let (end, values) = io.fold_weighted_masked(
            instance(sum, vars, if has_excess { 4 } else { 3 }),
            &batch,
            Some([8, 9]),
            || (0..claims.len()).into_par_iter().map(columns).collect(),
            [0, 1, 2, 3, 4, 5, 6, 7],
            8,
            expression,
        )?;
        let point = &end.challenges;
        io.check_value(&end, || {
            let mut x = [F::zero(); 13];
            let (mask, first) = self.masks_at(point);
            x[8..].copy_from_slice(&[
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
                    x[..8].copy_from_slice(v);
                    x[10] = eq(&q.point, point);
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
                        values: [v[2], v[3]],
                        row_ranges: [v[5], v[6], v[7]],
                        excess: v[4],
                    },
                )
            })
            .collect())
    }

    pub(crate) fn defer<'s, 'w, F: ProofField>(
        reads: Vec<(&'s Self, SoftmaxRead<F>)>,
        clips: &[Option<ClipPlan>],
        trace: impl Fn(usize) -> &'w SoftmaxTrace<F>,
        io: &mut Session<'_, '_, F>,

        lookups: &mut Batches<'s, 'w, F>,
    ) -> Result<()> {
        for (i, (op, read)) in reads.into_iter().enumerate() {
            op.rows
                .check(&[(&read.cell[..op.shape.row_vars()], read.row_ranges)], io)?;
            if let Some(excess) = &op.excess {
                excess.check(&[(&read.cell, [read.excess])], io)?;
            } else {
                match clips.get(i).and_then(Option::as_ref) {
                    Some(ClipPlan::Overflow(clip)) => clip.check(
                        &read.cell,
                        read.excess,
                        || clip.values(&trace(i).overflow),
                        io,
                    )?,
                    clip => {
                        if !read.excess.is_zero() {
                            return Err("unexpected Softmax excess");
                        }
                        if let Some(ClipPlan::Narrow(range)) = clip {
                            range.check(
                                &[(&read.cell[..op.shape.row_vars()], [read.row_ranges[0]])],
                                io,
                            )?;
                        }
                    }
                }
            }
            let tables = [Table::Slice(op.tables.probabilities), op.combined];
            if let [tile] = op.tiles.as_slice() {
                lookups.push(
                    &tables,
                    &tile.lookup,
                    Read {
                        cell: read.cell,
                        address: read.address,
                        output: Box::new(read.values),
                        witness: io.proving().then(|| trace(i).indices[0].as_ref()),
                    },
                );
                continue;
            }

            let (row, col) = read.cell.split_at(op.shape.row_vars());
            let defaults = [F::zero(), tables[0].field_at(0), tables[1].field_at(0)];
            let mut sum = defaults;
            for (j, tile) in op.tiles.iter().enumerate() {
                let (weight, cell) = tile.read(row, col);
                let witness = io.proving().then(|| trace(i).indices[j].as_ref());
                let values = io.values(|| {
                    let (r, c) = cell.split_at(tile.shape.row_vars());
                    let columns = weights(c);
                    let mut values = [F::zero(); 3];
                    let indices = witness.expect("Softmax tile witness");
                    for (row, row_weight) in weights(r).into_iter().enumerate() {
                        let mut partial = [F::zero(); 3];
                        for (col, &column_weight) in columns.iter().enumerate() {
                            let address = indices.get(row * columns.len() + col);
                            partial[0] += column_weight * F::from_u64(u64::from(address));
                            for (value, table) in partial[1..].iter_mut().zip(tables) {
                                *value += column_weight * table.field_at::<F>(address as usize);
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
                        output: Box::new([values[1], values[2]]),
                        witness,
                    },
                );
            }
            if sum != [read.address, read.values[0], read.values[1]] {
                return Err("invalid tiled Softmax lookup evaluation");
            }
        }
        Ok(())
    }
}
