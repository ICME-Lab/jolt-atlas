use crate::{
    advice::{Advice, Requests, Slot},
    channel::{Channel, Result},
    field::F,
    fold::basis,
    norm::{Claim, Width},
    protocol::{Candidate, TopKConfig},
    tensor::Shape,
};
use jolt_field::Ring;
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
        for check in checks {
            let d = check.shape;
            let point = io.sample(d.bits(), d.bits())?;
            let (row, col) = point.split_at(d.row_bits());
            let cw = basis(col);
            let rw = basis(row);
            let mut weight = vec![F::from_u64(0); stride];
            let mut lower = F::from_u64(0);
            let mut upper = F::from_u64(0);
            for (&a, matrix) in cw
                .iter()
                .zip(self.tokens[check.col * width..].chunks_exact(width))
            {
                for (out, &b) in weight.iter_mut().zip(matrix) {
                    *out += a * F::from_i32(b);
                }
            }
            for (r, &a) in rw.iter().enumerate() {
                for (c, &b) in cw.iter().enumerate() {
                    let (min, max) = self.bounds(
                        &public[check.row + r - self.config.positions.start],
                        check.col + c,
                    );
                    lower += a * b * F::from_i128(min);
                    upper += a * b * F::from_i128(max);
                }
            }
            let bound = || {
                let input = values.unwrap();
                let mut x = vec![F::from_u64(0); stride];
                for (r, &a) in rw.iter().enumerate() {
                    for (out, &b) in x.iter_mut().zip(&input[(check.row + r) * width..][..width]) {
                        *out += a * F::from_i32(b);
                    }
                }
                x
            };
            let [raw] =
                io.values(|| [bound().into_iter().zip(&weight).map(|(a, &b)| a * b).sum()])?;
            requests.push(check.low, &point, raw - lower);
            requests.push(check.high, &point, upper - raw);
            let end = io.dot(raw, weight, bound)?;
            let prefix = self.input.row_bits() - row.len();
            let index = check.row / d.rows;
            let point: Vec<_> = (0..prefix)
                .rev()
                .map(|i| F::from_bool(index & (1 << i) != 0))
                .chain(row.iter().copied())
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
