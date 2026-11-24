//! Prove the selected logits of a decoder without committing the vocabulary tensor.
use super::{
    pcs::{Advice, Layout},
    range::{Range, Ranges},
    session::{Claim, ProofField, Session},
    tensor::{bind_rows, dot, integer_dot, weights, Shape},
    Result,
};
use jolt_transcript::Transcript;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub id: usize,
    pub logit: i32,
}

fn order(c: Candidate) -> (i64, usize) {
    (-i64::from(c.logit), c.id)
}

pub struct TopKConfig {
    pub positions: std::ops::Range<usize>,
    pub k: usize,
    pub rows_per_block: usize,
    pub vocab_per_block: usize,
}

struct Block {
    row: usize,
    col: usize,
    ranges: Ranges,
}

pub(crate) struct TopKPlan {
    pub(crate) layout: Layout,
    blocks: Vec<Block>,
    rows: usize,
    vocab: usize,
    shape: Shape,
    real_rows: usize,
    max_raw: i128,
    scale: i128,
}

impl TopKPlan {
    pub(crate) fn new(
        mut layout: Layout,
        input: Shape,
        fractional_bits: u8,
        weights: &[i32],
        cfg: &TopKConfig,
    ) -> Result<Self> {
        let real_rows = input.rows;
        let vocab = weights.len() / input.width;
        if cfg.positions.is_empty()
            || cfg.positions.end > real_rows
            || cfg.k == 0
            || cfg.k > vocab
            || !cfg.rows_per_block.is_power_of_two()
            || !cfg.vocab_per_block.is_power_of_two()
        {
            return Err("invalid Top-K configuration");
        }
        let shape = Shape::new(cfg.rows_per_block, cfg.vocab_per_block)?;
        let rows = real_rows.next_power_of_two();
        if shape.rows > rows || shape.width > vocab.next_power_of_two() {
            return Err("invalid Top-K block size");
        }
        let scale = 1i128 << fractional_bits;
        let max_raw = weights
            .chunks_exact(input.width)
            .map(|w| w.iter().map(|&x| i128::from(x).abs()).sum::<i128>() * (1 << 31))
            .max()
            .ok_or("empty vocabulary")?;
        let max_difference = max_raw + max_raw.max(scale * ((1i128 << 31) + 1));
        let bits = (128 - max_difference.leading_zeros()) as usize;
        if bits >= 120 {
            return Err("unsupported Top-K range");
        }
        let mut blocks = Vec::new();
        for row in (0..rows).step_by(shape.rows) {
            for col in (0..vocab).step_by(shape.width) {
                blocks.push(Block {
                    row,
                    col,
                    ranges: Ranges::new(&mut layout, shape, [Range::unsigned(bits); 2])?,
                });
            }
        }
        Ok(Self {
            layout,
            blocks,
            rows,
            vocab,
            shape,
            real_rows,
            max_raw,
            scale,
        })
    }

    pub(crate) fn candidates(&self, cfg: &TopKConfig, candidates: &[Vec<Candidate>]) -> Result<()> {
        if candidates.len() != cfg.positions.len() {
            return Err("invalid Top-K row count");
        }
        for row in candidates {
            if row.len() != cfg.k
                || row.iter().any(|c| c.id >= self.vocab)
                || row.windows(2).any(|p| order(p[0]) >= order(p[1]))
            {
                return Err("invalid Top-K candidates");
            }
            let mut ids: Vec<_> = row.iter().map(|c| c.id).collect();
            ids.sort_unstable();
            if ids.windows(2).any(|p| p[0] == p[1]) {
                return Err("duplicate Top-K token");
            }
        }
        Ok(())
    }

    fn selected_bounds(&self, logit: i32) -> (i128, i128) {
        let lower = self.scale * i128::from(logit);
        (lower, lower + self.scale - 1)
    }

    pub(crate) fn select(
        &self,
        cfg: &TopKConfig,
        hidden: &[i32],
        weights: &[i32],
        width: usize,
    ) -> Result<Vec<Vec<Candidate>>> {
        cfg.positions
            .clone()
            .map(|row| {
                let mut values = weights
                    .chunks_exact(width)
                    .enumerate()
                    .map(|(id, w)| {
                        let raw = integer_dot(&hidden[row * width..][..width], w);
                        let logit = i32::try_from(raw.div_euclid(self.scale))
                            .map_err(|_| "Top-K logit overflow")?;
                        Ok(Candidate { id, logit })
                    })
                    .collect::<Result<Vec<_>>>()?;
                if values.len() > cfg.k {
                    values.select_nth_unstable_by_key(cfg.k, |c| order(*c));
                }
                values.truncate(cfg.k);
                values.sort_unstable_by_key(|c| order(*c));
                Ok(values)
            })
            .collect()
    }

    pub(crate) fn record(
        &self,
        cfg: &TopKConfig,
        candidates: &[Vec<Candidate>],
        hidden: &[i32],
        public: &[i32],
        width: usize,
        advice: &mut Advice,
    ) -> Result<()> {
        let by_id = candidates_by_id(candidates);
        for block in &self.blocks {
            let mut less = Vec::new();
            let mut greater = Vec::new();
            for row in block.row..block.row + self.shape.rows {
                let selected_row = cfg
                    .positions
                    .contains(&row)
                    .then(|| row - cfg.positions.start);
                let kth = selected_row.map(|i| candidates[i].last().expect("nonempty candidates"));
                let choices = selected_row.map_or(&[][..], |i| by_id[i].as_slice());
                let start = choices.partition_point(|c| c.id < block.col);
                let mut selected = choices[start..].iter().peekable();
                for col in block.col..block.col + self.shape.width {
                    let (lower, upper) = if row >= self.real_rows || col >= self.vocab {
                        (0, 0)
                    } else if let Some(kth) = kth {
                        if selected.peek().is_some_and(|c| c.id == col) {
                            self.selected_bounds(selected.next().expect("peeked candidate").logit)
                        } else {
                            let maximum = i128::from(kth.logit) + i128::from(col > kth.id);
                            (self.scale * i128::from(i32::MIN), self.scale * maximum - 1)
                        }
                    } else {
                        (-self.max_raw, self.max_raw)
                    };
                    let raw = if row < self.real_rows && col < self.vocab {
                        integer_dot(
                            &hidden[row * width..][..width],
                            &public[col * width..][..width],
                        )
                    } else {
                        0
                    };
                    less.push(raw - lower);
                    greater.push(upper - raw);
                }
            }
            block.ranges.record(advice, &[&less, &greater])?;
        }
        Ok(())
    }

    pub(crate) fn reduce<'w, F: ProofField>(
        &self,
        cfg: &TopKConfig,
        candidates: &[Vec<Candidate>],
        public: &[i32],
        width: usize,
        trace: impl Fn() -> &'w [i32],
        io: &mut Session<'_, '_, F>,
    ) -> Result<Vec<Claim<F>>> {
        let row_vars = self.shape.row_vars();
        let mut inputs = Vec::new();
        let by_id = candidates_by_id(candidates);
        for block in &self.blocks {
            let query = io.point(self.shape.vars())?;
            let (row, col) = query.split_at(row_vars);
            let real = self.vocab.saturating_sub(block.col).min(self.shape.width);
            let w = bind_rows(width, &public[block.col * width..][..real * width], col);
            let hidden = || {
                let end = (block.row + self.shape.rows).min(self.real_rows);
                bind_rows(
                    width,
                    &trace()[block.row.min(end) * width..end * width],
                    row,
                )
            };
            let [raw] = io.values(|| [dot(hidden(), w.iter().copied())])?;
            let Claim { point, value } = io.public_dot(raw, &w, hidden)?;
            let rweights = weights(row);
            let cweights = weights(col);
            let mut lower = F::zero();
            let mut upper = F::zero();
            for (i, &r) in rweights.iter().enumerate() {
                let at = block.row + i;
                if at >= self.real_rows {
                    continue;
                }
                let mass: F = cweights.iter().take(real).copied().sum();
                if !cfg.positions.contains(&at) {
                    lower -= r * mass * F::from_i128(self.max_raw);
                    upper += r * mass * F::from_i128(self.max_raw);
                    continue;
                }
                let row_index = at - cfg.positions.start;
                let kth = candidates[row_index].last().expect("validated candidates");
                let below = (kth.id + 1).saturating_sub(block.col).min(real);
                let below: F = cweights.iter().take(below).copied().sum();
                let min = self.scale * i128::from(i32::MIN);
                let max = self.scale * i128::from(kth.logit) - 1;
                lower += r * mass * F::from_i128(min);
                upper += r * (mass * F::from_i128(max) + (mass - below) * F::from_i128(self.scale));
                let choices = &by_id[row_index];
                let start = choices.partition_point(|c| c.id < block.col);
                for c in choices[start..]
                    .iter()
                    .take_while(|c| c.id < block.col + real)
                {
                    let j = c.id - block.col;
                    let (lo, hi) = self.selected_bounds(c.logit);
                    let default = max + self.scale * i128::from(c.id > kth.id);
                    lower += r * cweights[j] * F::from_i128(lo - min);
                    upper += r * cweights[j] * F::from_i128(hi - default);
                }
            }
            block
                .ranges
                .check(&[(&query, [raw - lower, upper - raw])], io)?;
            let extra = self.rows.ilog2() as usize - row_vars;
            let mut full: Vec<_> = (0..extra)
                .rev()
                .map(|i| F::from_bool((block.row >> row_vars) & (1 << i) != 0))
                .collect();
            full.extend_from_slice(row);
            full.extend(point);
            inputs.push(Claim { point: full, value });
        }
        Ok(inputs)
    }
}

fn candidates_by_id(candidates: &[Vec<Candidate>]) -> Vec<Vec<Candidate>> {
    candidates
        .iter()
        .map(|row| {
            let mut sorted = row.clone();
            sorted.sort_unstable_by_key(|candidate| candidate.id);
            sorted
        })
        .collect()
}

pub(crate) fn bind(t: &mut impl Transcript, cfg: &TopKConfig, candidates: &[Vec<Candidate>]) {
    for value in [
        cfg.positions.start,
        cfg.positions.end,
        cfg.k,
        cfg.rows_per_block,
        cfg.vocab_per_block,
    ] {
        t.append(&(value as u64));
    }
    for row in candidates {
        for c in row {
            t.append(&(c.id as u64));
            t.append(&(i64::from(c.logit) as u64));
        }
    }
}
