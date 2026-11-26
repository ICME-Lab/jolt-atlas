use crate::{
    advice::{Advice, Requests, Slot},
    channel::{Channel, Result},
    field::F,
    fold::{agreement, basis, Column},
    layers::{SoftmaxMask, SoftmaxTables},
    lookup::{Lookup, Lookups, Table},
    norm::{Claim, Width},
    tensor::Shape,
    tiles::Tile,
};
use jolt_field::{One, Ring, Zero};
use rayon::prelude::*;

pub(crate) struct Soft<'a> {
    pub shape: Shape,
    pub mask: SoftmaxMask,
    tables: SoftmaxTables<'a>,
    stride: u128,
    slack: u8,
    tiles: Vec<Tile>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct SoftTrace {
    pub probabilities: Option<Vec<i32>>,
    lookups: Vec<TileTrace>,
    stats: Option<Vec<[i64; 3]>>,
    overflow: Option<Vec<i64>>,
    stats_slots: [Slot; 3],
}

#[derive(serde::Serialize, serde::Deserialize)]
struct TileTrace {
    lookup: Lookup,
    overflow: Slot,
    remainder: Slot,
    score: Option<Slot>,
}

impl<'a> Soft<'a> {
    pub fn new(rows: usize, mask: SoftmaxMask, tables: SoftmaxTables<'a>) -> Result<Self> {
        let size = tables.sums.len();
        if size != tables.probabilities.len()
            || size < 2
            || !size.is_power_of_two()
            || tables.limit <= 0
            || tables.limit == i32::MAX
            || tables.sums.iter().any(|&x| x < 0)
            || tables.sums.windows(2).any(|x| x[1] > x[0])
            || i128::from(tables.sums[size - 1]) * rows as i128 > i128::from(tables.limit)
        {
            return Err("invalid Softmax parameters");
        }
        let max = rows as u128 * ((tables.limit + 1).max(tables.sums[0]) as u128);
        let slack = (max.ilog2() + 2) as u8;
        let stride = (max + (1 << slack) + tables.limit as u128).next_power_of_two();
        if slack > 55 {
            return Err("Softmax range exceeds field");
        }
        Ok(Self {
            shape: Shape::new(rows, rows)?,
            mask,
            tables,
            stride,
            slack,
            tiles: Tile::for_attention(rows, mask == SoftmaxMask::Causal),
        })
    }

    pub fn count(&self, row: usize) -> usize {
        if row >= self.shape.rows {
            0
        } else if self.mask == SoftmaxMask::Causal {
            row + 1
        } else {
            self.shape.cols
        }
    }

    fn table(&self) -> Table<'a> {
        Table::Pair(
            self.tables.probabilities,
            self.tables.sums,
            self.tables.limit,
            self.stride,
        )
    }

    pub fn prepare(
        &self,
        advice: &mut Advice,
        bits: u32,
        commit_scores: bool,
        compute: Option<impl Fn(usize, usize) -> i128 + Sync>,
    ) -> Result<SoftTrace> {
        let (scores, remainders): (Option<Vec<i32>>, Option<Vec<u32>>) =
            if let Some(compute) = compute {
                let width = self.shape.cols;
                let mask = (1i128 << bits) - 1;
                let data: Vec<_> = (0..self.shape.rows)
                    .into_par_iter()
                    .map(|r| {
                        let active = self.count(r);
                        let mut row = vec![(0i32, 0u32); width];
                        for (c, value) in row[..active].iter_mut().enumerate() {
                            let raw = compute(r, c);
                            *value = (
                                i32::try_from(raw >> bits).map_err(|_| "QK score exceeds i32")?,
                                (raw & mask) as u32,
                            );
                        }
                        Ok(row)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let (a, b) = data.into_iter().flatten().unzip();
                (Some(a), Some(b))
            } else {
                (None, None)
            };
        let last = (self.tables.sums.len() - 1) as i64;
        let n = self.shape.rows;
        let w = self.shape.stride();
        let limit = i64::from(self.tables.limit);
        let mut stats = None;
        let mut overflow = None;
        let mut probabilities = None;
        let mut addresses = None;
        if let Some(scores) = &scores {
            let rows: Vec<_> = scores
                .par_chunks_exact(n)
                .enumerate()
                .map(|(r, row)| {
                    let row = &row[..self.count(r)];
                    let max = i64::from(*row.iter().max().unwrap());
                    let sum = |shift: i64, prev: bool| {
                        row.iter()
                            .map(|&v| {
                                let a = (shift - i64::from(v)).min(last) - i64::from(prev);
                                if a < 0 {
                                    limit + 1
                                } else {
                                    i64::from(self.tables.sums[a.min(last) as usize])
                                }
                            })
                            .sum::<i64>()
                    };
                    let (mut a, mut b) = (max, max + last);
                    while a < b {
                        let mid = a + (b - a) / 2;
                        let above = row.iter().try_fold(0i64, |total, &v| {
                            let address = (mid - i64::from(v)).min(last) as usize;
                            let total = total + i64::from(self.tables.sums[address]);
                            (total <= limit).then_some(total)
                        });
                        if above.is_some() {
                            b = mid;
                        } else {
                            a = mid + 1;
                        }
                    }
                    [a, limit - sum(a, false), sum(a, true) - limit - 1]
                })
                .collect();
            let mut extra = vec![0; n * n];
            let mut prob = vec![0; n * n];
            let mut codes = vec![0; self.shape.padded()];
            extra
                .par_chunks_mut(n)
                .zip(prob.par_chunks_mut(n))
                .zip(codes.par_chunks_mut(w))
                .zip(scores.par_chunks_exact(n))
                .zip(rows.par_iter())
                .enumerate()
                .for_each(|(r, ((((extra, prob), codes), scores), stat))| {
                    for c in 0..self.count(r) {
                        let d = stat[0] - i64::from(scores[c]);
                        let v = d.min(last);
                        codes[c] = v as u32;
                        prob[c] = self.tables.probabilities[v as usize];
                        extra[c] = d - v;
                    }
                });
            stats = Some(rows);
            overflow = extra.iter().any(|&v| v != 0).then_some(extra);
            probabilities = Some(prob);
            addresses = Some(codes);
        }
        let lookups = self
            .tiles
            .iter()
            .map(|&tile| {
                let shape = tile.shape();
                let addresses = addresses.as_ref().map(|a| {
                    (0..shape.padded())
                        .map(|i| {
                            let (r, c) = tile.at(i);
                            a[r * w + c]
                        })
                        .collect()
                });
                let lookup = Lookup::record(advice, shape, self.tables.sums.len(), addresses)?;
                let at = |i| {
                    let (r, c) = tile.at(i);
                    (r < n && c < n).then_some(r * n + c)
                };
                let extra = advice.column(shape, Width::Unsigned(34), |i| {
                    i128::from(
                        at(i)
                            .and_then(|j| overflow.as_ref().map(|v| v[j]))
                            .unwrap_or(0),
                    )
                })?;
                let rem = advice.column(shape, Width::Unsigned(bits as u8), |i| {
                    i128::from(at(i).map(|j| remainders.as_ref().unwrap()[j]).unwrap_or(0))
                })?;
                let score = commit_scores
                    .then(|| {
                        advice.column(shape, Width::Signed(32), |i| {
                            i128::from(at(i).map(|j| scores.as_ref().unwrap()[j]).unwrap_or(0))
                        })
                    })
                    .transpose()?;
                Ok(TileTrace {
                    lookup,
                    overflow: extra,
                    remainder: rem,
                    score,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let rows = Shape::new(n, 1)?;
        let mut record =
            |i, width| advice.column(rows, width, |j| i128::from(stats.as_ref().unwrap()[j][i]));
        let stats_slots = [
            record(0, Width::Signed(34))?,
            record(1, Width::Unsigned(self.slack))?,
            record(2, Width::Unsigned(self.slack))?,
        ];
        Ok(SoftTrace {
            probabilities,
            lookups,
            stats,
            overflow,
            stats_slots,
        })
    }

    pub fn addresses(&self, trace: &SoftTrace, advice: Option<&Advice>) -> Option<Vec<u32>> {
        let advice = advice?;
        let mut addresses = vec![0; self.shape.padded()];
        let stride = self.shape.stride();
        for (&tile, block) in self.tiles.iter().zip(&trace.lookups) {
            for i in 0..tile.size * tile.size {
                let (r, c) = tile.at(i);
                addresses[r * stride + c] = block.lookup.code(advice, i);
            }
        }
        Some(addresses)
    }

    #[inline]
    fn at(&self, trace: &SoftTrace, address: u32, row: usize, col: usize) -> [F; 2] {
        if col >= self.count(row) {
            return [F::zero(); 2];
        }
        let i = row * self.shape.cols + col;
        let overflow = trace.overflow.as_ref().map_or(0, |v| v[i]);
        let shift = trace.stats.as_ref().unwrap()[row][0];
        [
            F::from_i64(shift - i64::from(address) - overflow),
            F::from_i64(overflow),
        ]
    }

    pub fn reverse(
        &self,
        io: &mut Channel<'_, '_>,
        requests: &mut Requests,
        lookups: &mut Lookups<'a>,
        trace: &SoftTrace,
        product: F,
        output_row: &[F],
        values: Option<&[F]>,
        codes: Option<&[u32]>,
        bits: u32,
    ) -> Result<(Claim<F>, Vec<F>, F)> {
        let vars = self.shape.bits();
        let rb = self.shape.row_bits();
        let n = self.shape.padded();
        let stride = self.shape.stride();
        let draw = io.sample(vars + rb + 3, vars + rb + 3)?;
        let (spot, tail) = draw.split_at(vars);
        let (rows, a) = tail.split_at(rb);
        let last = F::from_u64((self.tables.sums.len() - 1) as u64);
        let radix = F::from_u128(self.stride);
        let limit = F::from_i32(self.tables.limit);
        let table = self.table();
        let zero_prob = F::from_i32(self.tables.probabilities[0]);
        let zero_sum = F::from_i32(self.tables.sums[0]) + radix * (limit + F::one());
        let relation = |v: &[F; 11]| {
            let [address, p, table, overflow, under, over, mask, first, query, check, row] = *v;
            let inactive = F::one() - mask;
            let clamp =
                a[0] * overflow * (last - address) + inactive * (a[1] * address + a[2] * overflow);
            let row_sum = first * (under - limit - radix * (over + limit + F::one()))
                + table
                + mask * zero_sum;
            query * (p + mask * zero_prob) + check * clamp + row * row_sum
        };
        let mut weights = basis(rows);
        weights[self.shape.rows..].fill(F::zero());
        let (end, v) = io.fold(
            product,
            vars,
            3,
            || {
                let addresses = codes.unwrap();
                let numbers = |i| {
                    let mut v: Vec<_> = trace
                        .stats
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|s| F::from_i64(s[i]))
                        .collect();
                    v.resize(stride, F::zero());
                    Column::table(v, n, stride)
                };
                [
                    Column::source(n, |i| F::from_u32(addresses[i])),
                    Column::source(n, |i| table.values(addresses[i] as usize).0),
                    Column::source(n, |i| table.values(addresses[i] as usize).1),
                    Column::source(n, |i| {
                        self.at(trace, addresses[i], i / stride, i % stride)[1]
                    }),
                    numbers(1),
                    numbers(2),
                    Column::source(n, |i| F::from_bool(i % stride < self.count(i / stride))),
                    Column::table((0..stride).map(|i| F::from_bool(i == 0)).collect(), n, 1),
                    Column::Product(basis(output_row), values.unwrap().to_vec()),
                    Column::table(basis(spot), n, 1),
                    Column::table(weights.clone(), n, stride),
                ]
            },
            [0, 1, 2, 3, 4, 5],
            relation,
        )?;
        let point = end.challenges;
        let (r, c) = point.split_at(rb);
        let cw = basis(c);
        let [value] = io.values(|| [cw.iter().zip(values.unwrap()).map(|(&w, &v)| w * v).sum()])?;
        let mut prefix = vec![F::zero()];
        for w in cw {
            prefix.push(*prefix.last().unwrap() + w);
        }
        let rw = basis(r);
        let masked_rows: Vec<_> = rw
            .iter()
            .enumerate()
            .map(|(i, &w)| w * prefix[self.count(i)])
            .collect();
        let mask = masked_rows.iter().copied().sum();
        let row = rw.into_iter().zip(weights).map(|(a, b)| a * b).sum();
        let [i, p, t, diff, under, over] = v;
        if relation(&[
            i,
            p,
            t,
            diff,
            under,
            over,
            mask,
            prefix[1],
            agreement(output_row, r) * value,
            agreement(spot, &point),
            row,
        ]) != end.output_claim
        {
            return Err("invalid Softmax terminal evaluation");
        }
        let [shift_slot, under_slot, over_slot] = trace.stats_slots;
        requests.push(under_slot, r, under);
        requests.push(over_slot, r, over);
        let advice = io.advice();
        let mut whole = [F::zero(); 6];
        for (&tile, trace_tile) in self.tiles.iter().zip(&trace.lookups) {
            let (factor, local) = tile.restrict(r, c);
            let values = io.values(|| {
                let codes = codes.unwrap();
                let advice = advice.unwrap();
                let mut acc = [F::zero(); 6];
                let (rp, cp) = local.split_at(tile.size.ilog2() as usize);
                let cp = basis(cp);
                for (i, w) in basis(rp).into_iter().enumerate() {
                    let mut totals = [F::zero(); 6];
                    for (j, &weight) in cp.iter().enumerate() {
                        let address = codes[(tile.row + i) * stride + tile.col + j];
                        let rem = F::from_u32(
                            advice.value(trace_tile.remainder, i * tile.size + j) as u32,
                        );
                        let [score, overflow] = self.at(trace, address, tile.row + i, tile.col + j);
                        let (p, t) = table.values(address as usize);
                        for (v, b) in totals.iter_mut().zip([
                            score,
                            F::from_u32(address),
                            p,
                            t,
                            overflow,
                            rem,
                        ]) {
                            *v += weight * b;
                        }
                    }
                    for (a, t) in acc.iter_mut().zip(totals) {
                        *a += w * t;
                    }
                }
                acc
            })?;
            for (a, b) in whole.iter_mut().zip(values) {
                *a += factor * b;
            }
            let [score, address, p, t, extra, rem] = values;
            if let Some(slot) = trace_tile.score {
                requests.push(slot, &local, score);
            }
            requests.push(trace_tile.overflow, &local, extra);
            requests.push(trace_tile.remainder, &local, rem);
            lookups.read(&trace_tile.lookup, table, local, address, (p, t));
        }
        if whole[1..5] != [i, p, t, diff] {
            return Err("invalid Softmax tile evaluations");
        }
        // The score is the masked row shift minus the address and clamp excess.
        // After fixing the column point, only the row variables are left.
        let shift = io.dot(whole[0] + i + diff, masked_rows, || {
            let mut values: Vec<_> = trace
                .stats
                .as_ref()
                .unwrap()
                .iter()
                .map(|row| F::from_i64(row[0]))
                .collect();
            values.resize(stride, F::zero());
            values
        })?;
        requests.push(shift_slot, &shift.point, shift.value);
        let v_point = c.to_vec();
        Ok((
            Claim {
                point,
                value: whole[0] * F::from_u64(1 << bits) + whole[5],
            },
            v_point,
            value,
        ))
    }
}
