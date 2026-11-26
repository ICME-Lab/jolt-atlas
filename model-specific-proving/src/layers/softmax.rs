use std::sync::Arc;

use jolt_field::{One, Ring, Zero};

use super::{SoftmaxMask, SoftmaxTables};
use crate::protocol::{
    field::F,
    lookup::{Addresses, Lookup, Lookups, Table},
    opening::{Layout, Packed, Request},
    range::{Ranges, Width},
    sumcheck::{Column, Session},
    tensor::{equality, inner, weights, Claim, Shape},
    Result,
};

#[derive(Clone, Copy)]
pub(crate) struct Tile {
    pub row: usize,
    pub col: usize,
    pub shape: Shape,
}

impl Tile {
    pub(crate) fn restrict(self, row: &[F], col: &[F]) -> (F, Vec<F>) {
        fn axis(point: &[F], start: usize, width: usize) -> (F, &[F]) {
            let (fixed, free) = point.split_at(point.len() - width);
            let weight = fixed
                .iter()
                .enumerate()
                .map(|(i, &q)| {
                    if start >> (point.len() - i - 1) & 1 != 0 {
                        q
                    } else {
                        F::one() - q
                    }
                })
                .product();
            (weight, free)
        }
        let (wr, row) = axis(row, self.row, self.shape.row_bits());
        let (wc, col) = axis(col, self.col, self.shape.bits() - self.shape.row_bits());
        (wr * wc, [row, col].concat())
    }
}

pub(crate) fn tiles(n: usize, mask: SoftmaxMask) -> Result<Vec<Tile>> {
    let make = |row, col, r, c| {
        Ok(Tile {
            row,
            col,
            shape: Shape::new(r, c)?,
        })
    };
    if mask == SoftmaxMask::Full || n <= 64 || !n.is_power_of_two() {
        return Ok(vec![make(0, 0, n, n)?]);
    }
    let mut tiles = Vec::new();
    for row in (0..n).step_by(64) {
        tiles.push(make(row, row, 64, 64)?);
    }
    let mut width = 64;
    while width < n {
        for row in (width..n).step_by(2 * width) {
            tiles.push(make(row, row - width, width, width)?);
        }
        width *= 2;
    }
    Ok(tiles)
}

struct Block {
    tile: Tile,
    read: Lookup,
    overflow: Ranges<1>,
}

pub(crate) struct Soft<'a> {
    shape: Shape,
    blocks: Vec<Block>,
    mask: SoftmaxMask,
    tables: SoftmaxTables<'a>,
    rows: Ranges<3>,
    radix: u128,
}

pub(crate) struct SoftTrace {
    addresses: Vec<Arc<Addresses>>,
    overflows: Vec<Option<Vec<i64>>>,
    stats: Vec<[i64; 3]>,
}

impl<'a> Soft<'a> {
    pub(crate) fn new(
        layout: &mut Layout,
        n: usize,
        mask: SoftmaxMask,
        tables: SoftmaxTables<'a>,
        tiles: &[Tile],
    ) -> Result<Self> {
        if tables.limit == i32::MAX
            || tables.sums.len() != tables.probabilities.len()
            || tables.sums.len() < 2
            || !tables.sums.len().is_power_of_two()
            || tables.limit <= 0
            || tables.sums.iter().any(|&x| x < 0)
            || tables.sums.windows(2).any(|w| w[0] < w[1])
            || i128::from(*tables.sums.last().unwrap()) * n as i128 > i128::from(tables.limit)
        {
            return Err("invalid Softmax table");
        }
        let maximum = n as u128 * (tables.sums[0].max(tables.limit.saturating_add(1)) as u128);
        let slack = maximum.next_power_of_two().ilog2() as usize + 1;
        if 2 * slack + 3 >= 121 {
            return Err("Softmax range exceeds the proof field");
        }
        let radix = (maximum + (1u128 << slack) + tables.limit as u128).next_power_of_two();
        let rows = Ranges::new(
            layout,
            Shape::new(n, 1)?,
            [
                Width::signed(34),
                Width::unsigned(slack),
                Width::unsigned(slack),
            ],
        );
        let mut blocks = Vec::new();
        for &tile in tiles {
            let read = Lookup::new(layout, tile.shape.bits(), tables.sums.len())?;
            let overflow = Ranges::adaptive(layout, tile.shape, 34);
            blocks.push(Block {
                tile,
                read,
                overflow,
            });
        }
        Ok(Self {
            shape: Shape::new(n, n)?,
            blocks,
            mask,
            tables,
            rows,
            radix,
        })
    }

    fn active(&self, row: usize) -> usize {
        if self.mask == SoftmaxMask::Causal {
            row + 1
        } else {
            self.shape.width
        }
    }

    fn tables(&self) -> [Table<'a>; 2] {
        [
            Table::Direct(self.tables.probabilities),
            Table::Softmax {
                sums: self.tables.sums,
                limit: self.tables.limit,
                radix: self.radix,
            },
        ]
    }

    pub(crate) fn find(&self, row: usize, col: usize) -> (usize, usize) {
        if self.blocks.len() == 1 {
            return (0, row * self.shape.width.next_power_of_two() + col);
        }
        let diff = (row ^ col) >> 6;
        let id = if diff == 0 {
            row >> 6
        } else {
            let level = diff.ilog2() as usize;
            let diagonal = self.shape.rows >> 6;
            2 * diagonal - (diagonal >> level) + (row >> (level + 7))
        };
        let tile = self.blocks[id].tile;
        (id, (row - tile.row) * tile.shape.width + col - tile.col)
    }

    pub(crate) fn execute(
        &self,
        score: impl Fn(usize, usize) -> i32,
        packed: &mut Packed<'_>,
    ) -> Result<SoftTrace> {
        let mut addresses: Vec<_> = self
            .blocks
            .iter()
            .map(|block| Addresses::new(1 << block.read.cells, block.read.bits))
            .collect();
        let mut overflows: Vec<_> = self.blocks.iter().map(|_| None::<Vec<i64>>).collect();
        let mut stats = Vec::with_capacity(self.shape.rows);
        let max_address = (self.tables.sums.len() - 1) as i64;
        let limit = i64::from(self.tables.limit);
        let mut scores = Vec::with_capacity(self.shape.width);
        for row in 0..self.shape.rows {
            scores.clear();
            scores.extend((0..self.active(row)).map(|col| i64::from(score(row, col))));
            let exceeds_limit = |shift: i64| {
                let mut sum = 0;
                for &score in &scores {
                    let address = (shift - score).min(max_address) as usize;
                    sum += i64::from(self.tables.sums[address]);
                    if sum > limit {
                        return true;
                    }
                }
                false
            };
            let mut lo = *scores.iter().max().expect("nonempty row");
            let mut hi = lo + max_address;
            while lo < hi {
                let mid = (lo + hi).div_euclid(2);
                if exceeds_limit(mid) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            let mut sum = 0;
            let mut previous = 0;
            for (col, &value) in scores.iter().enumerate() {
                let (block, index) = self.find(row, col);
                let difference = lo - value;
                let address = difference.min(max_address);
                addresses[block].set(index, address as usize);
                sum += i64::from(self.tables.sums[address as usize]);
                previous += i64::from(if address == 0 {
                    self.tables.limit + 1
                } else {
                    self.tables.sums[address as usize - 1]
                });
                let overflow = difference - address;
                if overflow != 0 {
                    let shape = self.blocks[block].tile.shape;
                    let stride = shape.width.next_power_of_two();
                    let values = overflows[block].get_or_insert_with(|| vec![0; shape.len()]);
                    values[index / stride * shape.width + index % stride] = overflow;
                }
            }
            stats.push([lo, limit - sum, previous - limit - 1]);
        }
        let column: [Vec<i64>; 3] =
            std::array::from_fn(|i| stats.iter().map(|row| row[i]).collect());
        self.rows
            .record(packed, column.each_ref().map(Vec::as_slice))?;
        for ((block, addresses), overflow) in self.blocks.iter().zip(&addresses).zip(&overflows) {
            block.read.record(packed, addresses)?;
            block
                .overflow
                .record_adaptive(packed, overflow.as_deref().unwrap_or_default())?;
        }
        Ok(SoftTrace {
            addresses: addresses.into_iter().map(Arc::new).collect(),
            overflows,
            stats,
        })
    }

    pub(crate) fn probability(&self, trace: &SoftTrace, row: usize, col: usize) -> i32 {
        if row >= self.shape.rows || col >= self.active(row) {
            return 0;
        }
        let (block, index) = self.find(row, col);
        self.tables.probabilities[trace.addresses[block].get(index)]
    }

    pub(crate) fn reverse<'w>(
        &self,
        io: &mut Session<'_, '_>,
        requests: &mut Vec<Request>,
        lookups: &mut Lookups<'a>,
        claim: Claim,
        witness: impl Fn() -> &'w SoftTrace,
        score: impl Fn(usize, usize) -> i32 + Sync,
    ) -> Result<Claim> {
        let (vars, rb) = (self.shape.bits(), self.shape.row_bits());
        let stride = self.shape.width.next_power_of_two();
        let len = 1 << vars;
        let rand = io.random(vars + rb + 1, vars + rb + 1)?;
        let (p, r) = rand[..vars + rb].split_at(vars);
        let mix = rand[vars + rb];
        let tables = self.tables();
        let last = F::from_u64((self.tables.sums.len() - 1) as u64);
        let radix = F::from_u128(self.radix);
        let limit = F::from_i32(self.tables.limit);
        let formula = |v: &[F]| {
            let [score, index, prob, table, extra, shift, less, more, active, first, query, check, row] =
                v.try_into().expect("Softmax columns");
            let clamp = shift - score - index - extra + mix * extra * (index - last);
            let totals =
                active * table - first * (limit - less - radix * (limit + F::one() + more));
            active * (query * prob + check * clamp) + row * totals
        };
        let rw = weights(r);
        let (end, v) = io.fold(
            claim.value,
            vars,
            4,
            || {
                let trace = witness();
                let score = &score;
                let coordinates = |i| (i / stride, i % stride);
                let cell = move |i| {
                    let (row, col) = coordinates(i);
                    (row < self.shape.rows && col < self.active(row)).then(|| self.find(row, col))
                };
                let stats = |column: usize| {
                    let mut values: Vec<_> = trace
                        .stats
                        .iter()
                        .map(|row| F::from_i64(row[column]))
                        .collect();
                    values.resize(1 << rb, F::zero());
                    Column::axis(len, vars - rb, values)
                };
                vec![
                    Column::source(len, move |i| {
                        let (row, col) = coordinates(i);
                        if cell(i).is_some() {
                            F::from_i32(score(row, col))
                        } else {
                            F::zero()
                        }
                    }),
                    Column::source(len, move |i| {
                        F::from_u64(cell(i).map_or(0, |(b, j)| trace.addresses[b].get(j)) as u64)
                    }),
                    Column::source(len, move |i| {
                        tables[0].value(cell(i).map_or(0, |(b, j)| trace.addresses[b].get(j)))
                    }),
                    Column::source(len, move |i| {
                        tables[1].value(cell(i).map_or(0, |(b, j)| trace.addresses[b].get(j)))
                    }),
                    Column::source(len, move |i| {
                        F::from_i64(cell(i).map_or(0, |(b, j)| {
                            let shape = self.blocks[b].tile.shape;
                            let stride = shape.width.next_power_of_two();
                            trace.overflows[b]
                                .as_ref()
                                .map_or(0, |v| v[j / stride * shape.width + j % stride])
                        }))
                    }),
                    stats(0),
                    stats(1),
                    stats(2),
                    Column::row_mask(
                        self.shape.rows,
                        self.shape.width,
                        self.mask == SoftmaxMask::Causal,
                    ),
                    Column::row_mask(self.shape.rows, 1, false),
                    Column::split_equality(&claim.point),
                    Column::split_equality(p),
                    Column::axis(len, vars - rb, rw.clone()),
                ]
            },
            [0, 1, 2, 3, 4, 5, 6, 7],
            formula,
        )?;
        let (rpoint, cpoint) = end.challenges.split_at(rb);
        let mut prefix = vec![F::zero()];
        for w in weights(cpoint) {
            prefix.push(prefix[prefix.len() - 1] + w);
        }
        let rows = weights(rpoint);
        let active = inner(
            rows.iter().copied(),
            (0..self.shape.rows).map(|i| prefix[self.active(i)]),
        );
        let row_active: F = rows.iter().take(self.shape.rows).copied().sum();
        io.expect(&end, || {
            let mut values = v.to_vec();
            values.extend([
                active,
                row_active * prefix[1],
                equality(&claim.point, &end.challenges),
                equality(p, &end.challenges),
                inner(rows.iter().copied(), rw.iter().copied()),
            ]);
            formula(&values)
        })?;
        self.rows
            .open(io, requests, rpoint, v[5..].try_into().expect("row stats"))?;

        let defaults = [F::zero(), tables[0].value(0), tables[1].value(0), F::zero()];
        let mut full = defaults;
        for (i, block) in self.blocks.iter().enumerate() {
            let (factor, point) = block.tile.restrict(rpoint, cpoint);
            let [a, b, c, d] = io.values(|| {
                let shape = block.tile.shape;
                let trace = witness();
                let mut result = [F::zero(); 4];
                for (j, w) in weights(&point).into_iter().enumerate() {
                    let address = trace.addresses[i].get(j);
                    for (dst, val) in result.iter_mut().zip([
                        F::from_u64(address as u64),
                        tables[0].value(address),
                        tables[1].value(address),
                    ]) {
                        *dst += w * val;
                    }
                }
                if let Some(overflow) = &trace.overflows[i] {
                    result[3] = shape.evaluate(overflow, &point);
                }
                result
            })?;
            for ((sum, value), base) in full.iter_mut().zip([a, b, c, d]).zip(defaults) {
                *sum += factor * (value - base);
            }
            block.overflow.open(io, requests, &point, [d])?;
            lookups.read(
                &block.read,
                &tables,
                point,
                a,
                vec![b, c],
                io.proving().then(|| witness().addresses[i].clone()),
            );
        }
        if !io.proving() && full != v[1..5] {
            return Err("inconsistent Softmax tiles");
        }
        Ok(Claim::new(end.challenges, v[0]))
    }
}
