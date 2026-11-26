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
    one_sided: bool,
    max_raw: i128,
    scale: i128,
}

impl TopKPlan {
    pub(crate) fn new(
        mut layout: Layout,
        input: Shape,
        fractional_bits: u8,
        weights: &[i32],
        hidden_bounds: &[u128],
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
        let rows = real_rows.next_power_of_two();
        if cfg.rows_per_block > rows || cfg.vocab_per_block > vocab.next_power_of_two() {
            return Err("invalid Top-K block size");
        }
        let scale = 1i128 << fractional_bits;
        let max_raw = weights
            .chunks_exact(input.width)
            .map(|w| {
                w.iter()
                    .zip(hidden_bounds)
                    .map(|(&x, &bound)| i128::from(x).abs() * bound as i128)
                    .sum::<i128>()
            })
            .max()
            .ok_or("empty vocabulary")?;
        let one_sided = max_raw <= scale * i128::from(i32::MAX);
        let max_difference = max_raw
            + if one_sided {
                max_raw + scale - 1
            } else {
                max_raw.max(scale * ((1i128 << 31) + 1))
            };
        let bits = (128 - max_difference.leading_zeros()) as usize;
        if bits >= 120 {
            return Err("unsupported Top-K range");
        }
        let ranges = [
            Range::unsigned(if one_sided { 0 } else { bits }),
            Range::unsigned(bits),
        ];

        // Limit the temporary raw scores to a few rows of a wide vocabulary.
        let scratch_rows = ((8 << 20) / (vocab * size_of::<i128>())).max(1);
        let row_limit = cfg.rows_per_block.min(1 << scratch_rows.ilog2());
        let mut blocks = Vec::new();
        let mut row = cfg.positions.start;
        while row < cfg.positions.end {
            let available = cfg.positions.end - row;
            let alignment = if row == 0 {
                row_limit
            } else {
                1 << row.trailing_zeros()
            };
            let block_rows = (1 << available.ilog2()).min(row_limit).min(alignment);
            let mut col = 0;
            while col < vocab {
                let block_width = (1 << (vocab - col).ilog2()).min(cfg.vocab_per_block);
                blocks.push(Block {
                    row,
                    col,
                    ranges: Ranges::new(&mut layout, Shape::new(block_rows, block_width)?, ranges)?,
                });
                col += block_width;
            }
            row += block_rows;
        }
        Ok(Self {
            layout,
            blocks,
            rows,
            vocab,
            one_sided,
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
                || row.iter().any(|c| {
                    let (lo, hi) = self.selected_bounds(c.logit);
                    c.id >= self.vocab || lo > self.max_raw || hi < -self.max_raw
                })
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

    pub(crate) fn select_and_record(
        &self,
        cfg: &TopKConfig,
        hidden: &[i32],
        weights: &[i32],
        width: usize,
        advice: &mut Advice,
    ) -> Result<Vec<Vec<Candidate>>> {
        let mut candidates = Vec::with_capacity(cfg.positions.len());
        for blocks in self.blocks.chunk_by(|a, b| a.row == b.row) {
            let first = &blocks[0];
            let mut scores = Vec::with_capacity(first.ranges.shape.rows);
            for row in first.row..first.row + first.ranges.shape.rows {
                let hidden = &hidden[row * width..][..width];
                let raw: Vec<_> = weights
                    .chunks_exact(width)
                    .map(|w| integer_dot(hidden, w))
                    .collect();
                let mut row_candidates = raw
                    .iter()
                    .enumerate()
                    .map(|(id, &raw)| {
                        let logit = i32::try_from(raw.div_euclid(self.scale))
                            .map_err(|_| "Top-K logit overflow")?;
                        Ok(Candidate { id, logit })
                    })
                    .collect::<Result<Vec<_>>>()?;
                if row_candidates.len() > cfg.k {
                    row_candidates.select_nth_unstable_by_key(cfg.k, |c| order(*c));
                    row_candidates.truncate(cfg.k);
                }
                row_candidates.sort_unstable_by_key(|c| order(*c));
                scores.push(raw);
                candidates.push(row_candidates);
            }
            let selected = &candidates[candidates.len() - scores.len()..];
            let by_id = candidates_by_id(selected);
            for block in blocks {
                let shape = block.ranges.shape;
                let mut lower_slack = Vec::new();
                let mut upper_slack = Vec::with_capacity(shape.rows * shape.width);
                for (i, (raw, ids)) in scores.iter().zip(&by_id).enumerate() {
                    let kth = selected[i][cfg.k - 1];
                    let start = ids.partition_point(|c| c.id < block.col);
                    let mut choices = ids[start..].iter().peekable();
                    for col in block.col..block.col + shape.width {
                        let (lo, hi) = if choices.peek().is_some_and(|c| c.id == col) {
                            self.selected_bounds(choices.next().expect("selected candidate").logit)
                        } else {
                            let maximum = i128::from(kth.logit) + i128::from(col > kth.id);
                            (self.scale * i128::from(i32::MIN), self.scale * maximum - 1)
                        };
                        if !self.one_sided {
                            lower_slack.push(raw[col] - lo);
                        }
                        upper_slack.push(hi - raw[col]);
                    }
                }
                block
                    .ranges
                    .record_packed(advice, &[&lower_slack, &upper_slack])?;
            }
        }
        Ok(candidates)
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
        let mut inputs = Vec::new();
        let by_id = candidates_by_id(candidates);
        for block in &self.blocks {
            let shape = block.ranges.shape;
            let row_vars = shape.row_vars();
            let query = io.point(shape.vars())?;
            let (row, col) = query.split_at(row_vars);
            let w = bind_rows(
                width,
                &public[block.col * width..][..shape.width * width],
                col,
            );
            let hidden = || {
                bind_rows(
                    width,
                    &trace()[block.row * width..][..shape.rows * width],
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
                let row_index = at - cfg.positions.start;
                let kth = candidates[row_index].last().expect("validated candidates");
                let below = (kth.id + 1).saturating_sub(block.col).min(shape.width);
                let below: F = cweights.iter().take(below).copied().sum();
                let min = self.scale * i128::from(i32::MIN);
                let max = self.scale * i128::from(kth.logit) - 1;
                if !self.one_sided {
                    lower += r * F::from_i128(min);
                }
                upper += r * (F::from_i128(max) + (F::one() - below) * F::from_i128(self.scale));
                let choices = &by_id[row_index];
                let start = choices.partition_point(|c| c.id < block.col);
                for c in choices[start..]
                    .iter()
                    .take_while(|c| c.id < block.col + shape.width)
                {
                    let j = c.id - block.col;
                    let (lo, hi) = self.selected_bounds(c.logit);
                    let default = max + self.scale * i128::from(c.id > kth.id);
                    if self.one_sided {
                        let point = (0..shape.vars())
                            .rev()
                            .map(|bit| F::from_bool((i * shape.width + j) >> bit & 1 != 0))
                            .collect();
                        block
                            .ranges
                            .check_last_bits(point, self.scale.ilog2() as usize, io);
                    } else {
                        lower += r * cweights[j] * F::from_i128(lo - min);
                    }
                    upper += r * cweights[j] * F::from_i128(hi - default);
                }
            }
            block.ranges.check(
                &[(
                    &query,
                    [
                        if self.one_sided {
                            F::zero()
                        } else {
                            raw - lower
                        },
                        upper - raw,
                    ],
                )],
                io,
            )?;
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
