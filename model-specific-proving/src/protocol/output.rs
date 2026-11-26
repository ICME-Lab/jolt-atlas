use jolt_field::{One, Ring, Zero};
use rayon::prelude::*;

use super::{
    field::F,
    opening::{Layout, Packed, Request},
    range::{Ranges, Width},
    sumcheck::Session,
    tensor::{bind_rows, inner, weights, Claim, Shape},
    Candidate, Result, TopKConfig,
};
use crate::layers::linear::wide_dot;

struct Block {
    row: usize,
    col: usize,
    bounds: Ranges<2>,
}

pub(crate) struct Ranking<'a> {
    input: Shape,
    weights: &'a [i32],
    configuration: &'a TopKConfig,
    blocks: Vec<Block>,
    scale: i128,
    one_sided: bool,
}

fn order(c: &Candidate) -> (i64, usize) {
    (-i64::from(c.logit), c.id)
}

impl<'a> Ranking<'a> {
    pub(crate) fn new(
        layout: &mut Layout,
        input: Shape,
        weights: &'a [i32],
        bits: usize,
        configuration: &'a TopKConfig,
        bound: &[u128],
    ) -> Result<Self> {
        let cfg = configuration;
        let vocab = weights.len() / input.width;
        if cfg.positions.is_empty()
            || cfg.positions.end > input.rows
            || cfg.k == 0
            || cfg.k > vocab
            || !cfg.rows_per_block.is_power_of_two()
            || !cfg.vocab_per_block.is_power_of_two()
            || weights.len() != vocab * input.width
            || bound.len() != input.width
        {
            return Err("invalid Top-K configuration");
        }
        let scale = 1i128 << bits;
        let maximum = weights
            .chunks_exact(input.width)
            .map(|w| {
                w.iter()
                    .zip(bound)
                    .map(|(&x, &m)| i128::from(x.unsigned_abs()) * m as i128)
                    .sum::<i128>()
            })
            .max()
            .ok_or("empty vocabulary")?;
        let one_sided = maximum <= scale * i128::from(i32::MAX);
        let span = maximum
            + if one_sided {
                maximum + scale
            } else {
                maximum.max(scale << 32)
            };
        let bit_width = span.ilog2() as usize + 1;
        let limits = [
            Width::unsigned(if one_sided { 0 } else { bit_width }),
            Width::unsigned(bit_width),
        ];
        let mut blocks = Vec::new();
        let row_limit = cfg.rows_per_block.min((512 * 1024 / vocab).max(1)).ilog2();
        let row_limit = 1 << row_limit;
        let chunk = |start: usize, end: usize, limit: usize| {
            let n = (1 << (end - start).ilog2()).min(limit);
            if start == 0 {
                n
            } else {
                n.min(1 << start.trailing_zeros())
            }
        };
        let mut row = cfg.positions.start;
        while row < cfg.positions.end {
            let height = chunk(row, cfg.positions.end, row_limit);
            let mut col = 0;
            while col < vocab {
                let count = chunk(col, vocab, cfg.vocab_per_block);
                let bounds = Ranges::new(layout, Shape::new(height, count)?, limits);
                blocks.push(Block { row, col, bounds });
                col += count;
            }
            row += height;
        }
        Ok(Self {
            input,
            scale,
            weights,
            configuration,
            blocks,
            one_sided,
        })
    }

    pub(crate) fn validate(&self, candidates: &[Vec<Candidate>]) -> Result<()> {
        let config = self.configuration;
        if candidates.len() != config.positions.len() {
            return Err("wrong number of Top-K rows");
        }
        let vocab = self.weights.len() / self.input.width;
        for row in candidates {
            if row.len() != config.k
                || row.iter().any(|c| c.id >= vocab)
                || row.windows(2).any(|p| order(&p[0]) >= order(&p[1]))
            {
                return Err("invalid Top-K candidates");
            }
            let mut ids: Vec<_> = row.iter().map(|c| c.id).collect();
            ids.sort_unstable();
            if ids.windows(2).any(|w| w[0] == w[1]) {
                return Err("duplicate Top-K candidate");
            }
        }
        Ok(())
    }

    fn limits(&self, row: &[Candidate], id: usize) -> (i128, i128, bool) {
        if let Some(candidate) = row.iter().find(|c| c.id == id) {
            let base = self.scale * i128::from(candidate.logit);
            (base, base + self.scale - 1, true)
        } else {
            let kth = row.last().expect("validated Top-K");
            let logit = i128::from(kth.logit) + i128::from(id > kth.id);
            (
                self.scale * i128::from(i32::MIN),
                self.scale * logit - 1,
                false,
            )
        }
    }

    pub(crate) fn select(
        &self,
        hidden: &[i32],
        packed: &mut Packed<'_>,
    ) -> Result<Vec<Vec<Candidate>>> {
        let cfg = self.configuration;
        let width = self.input.width;
        let mut result = Vec::with_capacity(cfg.positions.len());
        for blocks in self.blocks.chunk_by(|a, b| a.row == b.row) {
            let first = &blocks[0];
            let mut values = Vec::new();
            let start = result.len();
            for position in first.row..first.row + first.bounds.shape.rows {
                let h = &hidden[position * width..][..width];
                let raw: Vec<_> = self
                    .weights
                    .par_chunks_exact(width)
                    .with_min_len(128)
                    .map(|w| wide_dot(h, w))
                    .collect();
                let mut row = raw
                    .iter()
                    .enumerate()
                    .map(|(id, &value)| {
                        Ok(Candidate {
                            id,
                            logit: i32::try_from(value.div_euclid(self.scale))
                                .map_err(|_| "Top-K overflow")?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                if cfg.k < row.len() {
                    row.select_nth_unstable_by_key(cfg.k, order);
                    row.truncate(cfg.k);
                }
                row.sort_unstable_by_key(order);
                values.push(raw);
                result.push(row);
            }
            for block in blocks {
                let mut lower = Vec::new();
                let mut upper = Vec::with_capacity(block.bounds.shape.len());
                for (raw, row) in values.iter().zip(&result[start..]) {
                    for index in block.col..block.col + block.bounds.shape.width {
                        let (min, max, _) = self.limits(row, index);
                        if !self.one_sided {
                            lower.push(raw[index] - min);
                        }
                        upper.push(max - raw[index]);
                    }
                }
                block.bounds.record(packed, [&lower, &upper])?;
            }
        }
        Ok(result)
    }

    pub(crate) fn reverse<'w>(
        &self,
        io: &mut Session<'_, '_>,
        requests: &mut Vec<Request>,
        candidates: &[Vec<Candidate>],
        hidden: impl Fn() -> &'w [i32],
    ) -> Result<Vec<Claim>> {
        let width = self.input.width;
        let mut claims = Vec::new();
        for block in &self.blocks {
            let shape = block.bounds.shape;
            let point = io.random(shape.bits(), shape.bits())?;
            let (row, col) = point.split_at(shape.row_bits());
            let public = bind_rows(
                width,
                &self.weights[block.col * width..][..shape.width * width],
                col,
            );
            let private = || {
                bind_rows(
                    width,
                    &hidden()[block.row * width..][..shape.rows * width],
                    row,
                )
            };
            let [value] = io.values(|| [inner(private(), public.iter().copied())])?;
            let claim = io.dot(value, &public, private)?;
            let low = self.scale * i128::from(i32::MIN);
            let lo = F::from_i128(low);
            let mut lower = F::zero();
            let mut hi = F::zero();
            let cw = weights(col);
            let mut prefix = Vec::with_capacity(cw.len() + 1);
            prefix.push(F::zero());
            for &weight in &cw {
                prefix.push(prefix.last().copied().unwrap() + weight);
            }
            for (i, wr) in weights(row).into_iter().enumerate() {
                let candidates = &candidates[block.row + i - self.configuration.positions.start];
                let kth = candidates.last().expect("validated Top-K");
                let split = (kth.id + 1).saturating_sub(block.col).min(shape.width);
                let upper = self.scale * i128::from(kth.logit) - 1;
                let mut row_lo = lo;
                let mut row_hi =
                    F::from_i128(upper) + (F::one() - prefix[split]) * F::from_i128(self.scale);
                for candidate in candidates {
                    let j = candidate.id.wrapping_sub(block.col);
                    if j >= shape.width {
                        continue;
                    }
                    let base = self.scale * i128::from(candidate.logit);
                    let default = upper + i128::from(candidate.id > kth.id) * self.scale;
                    row_hi += cw[j] * F::from_i128(base + self.scale - 1 - default);
                    if self.one_sided {
                        let exact: Vec<_> = (0..shape.bits())
                            .rev()
                            .map(|b| F::from_bool((i * shape.width + j) >> b & 1 == 1))
                            .collect();
                        block
                            .bounds
                            .high_bits_zero(requests, exact, self.scale.ilog2() as usize);
                    } else {
                        row_lo += cw[j] * F::from_i128(base - low);
                    }
                }
                hi += wr * row_hi;
                lower += wr * row_lo;
            }
            block.bounds.open(
                io,
                requests,
                &point,
                [
                    if self.one_sided {
                        F::zero()
                    } else {
                        value - lower
                    },
                    hi - value,
                ],
            )?;
            let prefix = self.input.row_bits() - row.len();
            let mut point: Vec<_> = (0..prefix)
                .rev()
                .map(|b| F::from_bool((block.row / shape.rows) >> b & 1 == 1))
                .collect();
            point.extend(row);
            point.extend(claim.point);
            claims.push(Claim::new(point, claim.value));
        }
        Ok(claims)
    }
}
