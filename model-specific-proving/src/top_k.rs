use crate::{
    advice::{Advice, Requests, Slot},
    channel::{Channel, Result},
    field::F,
    fold::basis,
    norm::{Claim, Width},
    protocol::{Candidate, TopKConfig},
    tensor::Shape,
};
use jolt_field::{Ring, Zero};
use rayon::prelude::*;

pub(crate) struct Top<'a> {
    config: &'a TopKConfig,
    tokens: &'a [i32],
    input: Shape,
    scale: i128,
}

pub(crate) struct Check {
    row: usize,
    col: usize,
    shape: Shape,
    low: Slot,
    high: Slot,
}

fn order(value: &Candidate) -> (i64, usize) {
    (-i64::from(value.logit), value.id)
}

impl<'a> Top<'a> {
    pub fn new(config: &'a TopKConfig, tokens: &'a [i32], input: Shape, bits: u32) -> Result<Self> {
        let count = tokens.len() / input.cols;
        if config.positions.is_empty()
            || config.positions.end > input.rows
            || config.k == 0
            || config.k > count
            || !config.rows_per_block.is_power_of_two()
            || !config.vocab_per_block.is_power_of_two()
        {
            return Err("invalid Top-K configuration");
        }
        Ok(Self {
            config,
            tokens,
            input,
            scale: 1 << bits,
        })
    }

    pub fn validate(&self, public: &[Vec<Candidate>]) -> Result<()> {
        let vocab = self.tokens.len() / self.input.cols;
        if public.len() != self.config.positions.len()
            || public.iter().any(|r| {
                r.len() != self.config.k
                    || r.iter().any(|c| c.id >= vocab)
                    || r.windows(2).any(|w| order(&w[0]) >= order(&w[1]))
                    || r.iter()
                        .enumerate()
                        .any(|(i, c)| r[..i].iter().any(|a| a.id == c.id))
            })
        {
            return Err("invalid Top-K results");
        }
        Ok(())
    }

    fn bounds(&self, row: &[Candidate], candidate: usize) -> (i128, i128) {
        if let Some(found) = row.iter().find(|c| c.id == candidate) {
            let lower = self.scale * i128::from(found.logit);
            return (lower, lower + self.scale - 1);
        }
        let last = row.last().unwrap();
        let upper = self.scale * (i128::from(last.logit) + i128::from(candidate > last.id)) - 1;
        (self.scale * i128::from(i32::MIN), upper)
    }

    fn eval_bounds(&self, candidates: &[Candidate], start: usize, weights: &[F]) -> (F, F) {
        let last = candidates.last().unwrap();
        let min = self.scale * i128::from(i32::MIN);
        let max = self.scale * i128::from(last.logit) - 1;
        let above = (last.id + 1).saturating_sub(start);
        let suffix: F = weights.iter().skip(above).copied().sum();
        let mut lo = F::from_i128(min);
        let mut hi = F::from_i128(max) + suffix * F::from_i128(self.scale);
        for candidate in candidates {
            if let Some(&w) = candidate.id.checked_sub(start).and_then(|i| weights.get(i)) {
                let base = max + self.scale * i128::from(candidate.id > last.id);
                let value = self.scale * i128::from(candidate.logit);
                lo += w * F::from_i128(value - min);
                hi += w * F::from_i128(value + self.scale - 1 - base);
            }
        }
        (lo, hi)
    }

    pub fn select(&self, values: &[i32]) -> Result<(Vec<Vec<Candidate>>, Vec<i128>)> {
        let width = self.input.cols;
        let vocab = self.tokens.len() / width;
        let mut sums = Vec::new();
        let mut results = Vec::new();
        for block in self.config.positions.clone() {
            let x = &values[block * width..][..width];
            let raw: Vec<i128> = self
                .tokens
                .par_chunks_exact(width)
                .with_min_len(128)
                .map(|weight| {
                    weight
                        .iter()
                        .zip(x)
                        .map(|(&a, &b)| i128::from(a) * i128::from(b))
                        .sum()
                })
                .collect();
            let mut candidates = raw
                .iter()
                .enumerate()
                .map(|(id, &y)| {
                    Ok(Candidate {
                        id,
                        logit: i32::try_from(y.div_euclid(self.scale))
                            .map_err(|_| "Top-K logit overflow")?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            if self.config.k < vocab {
                candidates.select_nth_unstable_by_key(self.config.k, order);
                candidates.truncate(self.config.k);
            }
            candidates.sort_unstable_by_key(order);
            results.push(candidates);
            sums.extend(raw);
        }
        Ok((results, sums))
    }

    pub fn record(
        &self,
        advice: &mut Advice,
        public: &[Vec<Candidate>],
        raw: Option<&[i128]>,
    ) -> Result<Vec<Check>> {
        self.validate(public)?;
        let vocab = self.tokens.len() / self.input.cols;
        let cap = self.config.vocab_per_block.min(1 << 17);
        let ends = |from: usize, to: usize, max: usize| {
            let sz = (1 << (to - from).ilog2()).min(max);
            if from == 0 {
                sz
            } else {
                sz.min(1 << from.trailing_zeros())
            }
        };
        let mut checks = Vec::new();
        let mut r = self.config.positions.start;
        let bits = self.scale.ilog2() as u8 + 32;
        while r < self.config.positions.end {
            let rows = ends(r, self.config.positions.end, self.config.rows_per_block);
            let mut c = 0;
            while c < vocab {
                let cols = ends(c, vocab, cap);
                let shape = Shape::new(rows, cols)?;
                let range = |i: usize| {
                    let row = r + i / cols - self.config.positions.start;
                    let col = c + i % cols;
                    let (lo, hi) = self.bounds(&public[row], col);
                    (
                        raw.unwrap()[row * vocab + col] - lo,
                        hi - raw.unwrap()[row * vocab + col],
                    )
                };
                let lo = advice.column(shape, Width::Unsigned(bits), |i| range(i).0)?;
                let hi = advice.column(shape, Width::Unsigned(bits), |i| range(i).1)?;
                checks.push(Check {
                    row: r,
                    col: c,
                    shape,
                    low: lo,
                    high: hi,
                });
                c += cols;
            }
            r += rows;
        }
        Ok(checks)
    }

    pub fn reverse(
        &self,
        io: &mut Channel<'_, '_>,
        requests: &mut Requests,
        checks: &[Check],
        public: &[Vec<Candidate>],
        values: Option<&[i32]>,
    ) -> Result<Vec<Claim<F>>> {
        let width = self.input.cols;
        let stride = self.input.stride();
        let mut claims = Vec::new();
        for blocks in checks.chunk_by(|a, b| a.row == b.row) {
            let first = &blocks[0];
            let rows = first.shape.rows;
            let row_bits = first.shape.row_bits();
            let row = if row_bits == 0 {
                Vec::new()
            } else {
                io.sample(row_bits, row_bits)?
            };
            let rw = basis(&row);
            let hidden = values.map(|input| {
                let mut x = vec![F::zero(); stride];
                for (r, &a) in rw.iter().enumerate() {
                    for (out, &b) in x.iter_mut().zip(&input[(first.row + r) * width..][..width]) {
                        *out += a * F::from_i32(b);
                    }
                }
                x
            });
            let mut terms = Vec::with_capacity(blocks.len());
            for check in blocks {
                let cols = check.shape.bits() - row_bits;
                let col = if cols == 0 {
                    Vec::new()
                } else {
                    io.sample(cols, cols)?
                };
                let cw = basis(&col);
                let point: Vec<_> = row.iter().chain(&col).copied().collect();
                let mut weight = vec![F::zero(); stride];
                let mut lower = F::zero();
                let mut upper = F::zero();
                for (&a, matrix) in cw
                    .iter()
                    .zip(self.tokens[check.col * width..].chunks_exact(width))
                {
                    for (out, &b) in weight.iter_mut().zip(matrix) {
                        *out += a * F::from_i32(b);
                    }
                }
                for (r, &a) in rw.iter().enumerate() {
                    let (min, max) = self.eval_bounds(
                        &public[check.row + r - self.config.positions.start],
                        check.col,
                        &cw,
                    );
                    lower += a * min;
                    upper += a * max;
                }
                let [raw] = io.values(|| {
                    [hidden
                        .as_ref()
                        .unwrap()
                        .iter()
                        .zip(&weight)
                        .map(|(&a, &b)| a * b)
                        .sum()]
                })?;
                requests.push(check.low, &point, raw - lower);
                requests.push(check.high, &point, upper - raw);
                terms.push((raw, weight));
            }
            let mix = io.sample(terms.len(), terms.len())?;
            let mut sum = F::zero();
            let mut weight = vec![F::zero(); stride];
            for ((raw, projected), a) in terms.into_iter().zip(mix) {
                sum += a * raw;
                for (out, w) in weight.iter_mut().zip(projected) {
                    *out += a * w;
                }
            }
            let end = io.dot(sum, weight, || hidden.unwrap())?;
            let prefix = self.input.row_bits() - row_bits;
            let index = first.row / rows;
            let point = (0..prefix)
                .rev()
                .map(|i| F::from_bool(index & (1 << i) != 0))
                .chain(row)
                .chain(end.point)
                .collect();
            claims.push(Claim {
                point,
                value: end.value,
            });
        }
        Ok(claims)
    }
}
