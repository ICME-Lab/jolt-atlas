use std::sync::Arc;

use super::super::linear::{narrow_dot, quotient_remainder};
use super::Softmax;
use crate::protocol::{
    pcs::{Advice, Layout},
    range::{Range, Ranges, I32},
    session::{instance, Claim, Column, ProofField, Session},
    tensor::{dot, eq, integer_dot, weights, Shape},
    Result,
};

pub(super) struct Causal {
    key: Shape,
    tiles: Vec<Tile>,
    bits: usize,
    factor: i32,
    check_score: bool,
}

struct Tile {
    row: usize,
    col: usize,
    shape: Shape,
    range: Ranges,
}

pub(super) type Remainders = Vec<Arc<Vec<u8>>>;

impl Causal {
    pub(super) fn new(
        layout: &mut Layout,
        key: Shape,
        bits: usize,
        factor: i32,
        softmax: &Softmax<'_>,
    ) -> Result<Self> {
        let shift = (factor.trailing_zeros() as usize).min(bits);
        let factor = factor >> shift;
        let bits = bits - shift;
        let check_score = softmax.has_excess();
        let ranges = [
            if check_score { I32 } else { Range::unsigned(0) },
            Range::unsigned(bits),
        ];
        layout.integer_bits = layout.integer_bits.max(
            key.width.next_power_of_two().ilog2()
                + (u32::BITS - factor.unsigned_abs().leading_zeros())
                + 65,
        );
        let tiles = softmax
            .score_tiles()
            .map(|(row, col, shape)| {
                Ok(Tile {
                    row,
                    col,
                    shape,
                    range: Ranges::new(layout, shape, ranges)?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            key,
            tiles,
            bits,
            factor,
            check_score,
        })
    }

    pub(super) fn execute(
        &self,
        query: &[i32],
        key: &[i32],
        advice: &mut Advice,
    ) -> Result<(Vec<i32>, Remainders)> {
        self.key.check(query)?;
        self.key.check(key)?;
        let size = self.key.rows;
        let max_key = u128::from(key.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0));
        let mut scores = vec![0; size * (size + 1) / 2];
        let mut digits = Vec::with_capacity(self.tiles.len());
        for tile in &self.tiles {
            let mut remainders = vec![0; tile.shape.rows * tile.shape.width];
            let mut checked = if self.check_score {
                vec![0; remainders.len()]
            } else {
                Vec::new()
            };
            for r in 0..tile.shape.rows {
                let row = tile.row + r;
                let q = &query[row * self.key.width..][..self.key.width];
                let norm: u128 = q.iter().map(|v| u128::from(v.unsigned_abs())).sum();
                let narrow = norm * max_key <= i64::MAX as u128;
                for c in 0..tile.shape.width.min(row + 1 - tile.col) {
                    let col = tile.col + c;
                    let k = &key[col * self.key.width..][..self.key.width];
                    let dot = if narrow {
                        i128::from(narrow_dot(q, k))
                    } else {
                        integer_dot(q, k)
                    };
                    let (score, remainder) =
                        quotient_remainder(dot * i128::from(self.factor), self.bits)?;
                    let index = r * tile.shape.width + c;
                    remainders[index] = remainder;
                    if self.check_score {
                        checked[index] = score;
                    }
                    scores[row * (row + 1) / 2 + col] = score;
                }
            }
            tile.range.record(advice, &[&checked, &remainders])?;
            digits.push(advice.range_digits(tile.range.slot));
        }
        Ok((scores, digits))
    }

    pub(super) fn narrow(&self, advice: &mut Advice, digits: &mut Remainders) {
        for (tile, digits) in self.tiles.iter().zip(digits) {
            let cells = 1 << tile.shape.vars();
            let suffix = if self.bits == 0 {
                vec![0; cells]
            } else {
                digits[8 * cells..].to_vec()
            };
            *digits = Arc::new(suffix);
            advice.replace_digits(tile.range.slot, Arc::clone(digits));
        }
    }

    pub(super) fn narrow_layout(&self, layout: &mut Layout) {
        for tile in &self.tiles {
            layout.replace_bits(tile.range.suffix(1).slot);
        }
    }

    pub(super) fn reverse<'w, F: ProofField>(
        &self,
        q: Claim<F>,
        narrow: bool,
        digits: impl Fn() -> &'w Remainders + Sync,
        io: &mut Session<'_, '_, F>,
    ) -> Result<Claim<F>> {
        let (row, col) = q.point.split_at(self.key.row_vars());
        let check_score = self.check_score && !narrow;
        let in_range = |i: usize, j: usize| {
            let tile = &self.tiles[i];
            let bit = if check_score { 32 } else { 0 };
            let cells = 1 << tile.shape.vars();
            let lanes = if narrow {
                1 << self.bits.next_power_of_two().ilog2().min(2)
            } else {
                1 << tile.range.slot.digit_vars()
            };
            let value = |offset: usize, bits: usize| {
                (0..bits).step_by(lanes).fold(0u32, |v, k| {
                    let digit = digits()[i][((offset + k) / lanes) * cells + j];
                    v | (u32::from(digit) << k)
                }) & ((1u32 << (bits.min(31))) - 1 | (u32::MAX * u32::from(bits == 32)))
            };
            let rem = value(bit, self.bits);
            let score = if check_score {
                i64::from(value(0, 32)) - (1i64 << 31)
            } else {
                0
            };
            [F::from_u64(u64::from(rem)), F::from_i64(score)]
        };

        let mut rem = F::zero();
        let mut score = F::zero();
        for (i, tile) in self.tiles.iter().enumerate() {
            let cols = tile.shape.width;
            let (scale, point) = tile.point(row, col);
            let [r, s] = io.values_prefix(1 + usize::from(check_score), || {
                let (r, c) = point.split_at(tile.shape.row_vars());
                let cols_w = weights(c);
                let mut result = [F::zero(); 2];
                for (j, wr) in weights(r).into_iter().enumerate() {
                    for (k, &wc) in cols_w.iter().enumerate() {
                        let w = wr * wc;
                        let v = in_range(i, j * cols + k);
                        result[0] += w * v[0];
                        result[1] += w * v[1];
                    }
                }
                result
            })?;
            tile.check(narrow, &point, r, s, io)?;
            rem += scale * r;
            score += scale * s;
        }
        if check_score && score != q.value {
            return Err("invalid causal score evaluation");
        }

        let diagonals: Vec<_> = self
            .tiles
            .iter()
            .enumerate()
            .filter(|(_, t)| t.row == t.col)
            .collect();
        let side = diagonals[0].1.shape.width;
        if side > 1 {
            let vars = diagonals[0].1.shape.vars();
            // Draw the mix and evaluation point after the same PoW.
            io.pow(vars + diagonals.len() + usize::from(check_score))?;
            let mix = io.challenge(diagonals.len())?;
            let blend = if check_score {
                io.challenge(1)?[0]
            } else {
                F::zero()
            };
            let p = io.challenge(vars)?;
            let mask = |i: usize| F::from_bool(i % side > i / side);
            let (end, values) = io.fold_weighted(
                instance(F::zero(), vars, 3),
                &mix,
                || {
                    diagonals
                        .iter()
                        .map(|(i, _)| {
                            let i = *i;
                            [
                                Column::source(side * side, move |j| in_range(i, j)[0]),
                                Column::source(side * side, move |j| in_range(i, j)[1]),
                                Column::source(side * side, mask),
                                Column::equality(&p),
                            ]
                        })
                        .collect()
                },
                [0, 1],
                1 + usize::from(check_score),
                |v| (v[0] + blend * v[1]) * v[2] * v[3],
            )?;
            let point = &end.challenges;
            let (r, c) = point.split_at(point.len() / 2);
            io.check_value(&end, || {
                let mut prefix = F::zero();
                let selected = (0..side)
                    .rev()
                    .zip(weights::<F>(r).into_iter().rev())
                    .zip(weights::<F>(c).into_iter().rev())
                    .map(|((_, wr), wc)| {
                        let term = wr * prefix;
                        prefix += wc;
                        term
                    })
                    .sum::<F>();
                let value = dot(
                    values.iter().map(|v| v[0] + blend * v[1]),
                    mix.iter().copied(),
                );
                value * selected * eq(&p, &point)
            })?;
            for ((_, tile), [r, s]) in diagonals.into_iter().zip(values) {
                tile.check(narrow, &point, r, s, io)?;
            }
        }
        Ok(Claim {
            point: q.point,
            value: F::from_u64(1 << self.bits) * q.value + rem,
        })
    }

    pub(super) fn product<'w, F: ProofField>(
        &self,
        claims: &[Claim<F>],
        query: impl Fn(usize) -> &'w [i32] + Sync,
        key: impl Fn() -> &'w [i32] + Sync,
        io: &mut Session<'_, '_, F>,
    ) -> Result<(Vec<Claim<F>>, Claim<F>)> {
        let n = self.key.rows;
        let width = self.key.width.next_power_of_two();
        let rows = self.key.row_vars();
        let (row, col) = claims[0].point.split_at(rows);
        let col_weights = weights::<F>(col);
        let mix = io.point(claims.len())?;
        let scale = F::from_i32(self.factor);
        let sum = dot(claims.iter().map(|c| c.value), mix.iter().copied());
        let row_weights = weights(row);
        let (point, [mixed_query, prefix]) = io.relation(
            instance(sum, self.key.vars(), 3),
            || {
                let mut prefix = Vec::with_capacity(n * width);
                let mut running = vec![F::zero(); width];
                for (k, w) in key().chunks_exact(self.key.width).zip(&col_weights) {
                    for (dst, &src) in running.iter_mut().zip(k) {
                        *dst += *w * F::from_i32(src);
                    }
                    prefix.extend_from_slice(&running);
                }
                let mut mixed = vec![F::zero(); n * width];
                for (i, &weight) in mix.iter().enumerate() {
                    for (dst, src) in mixed
                        .chunks_exact_mut(width)
                        .zip(query(i).chunks_exact(self.key.width))
                    {
                        for (dst, &q) in dst.iter_mut().zip(src) {
                            *dst += weight * F::from_i32(q);
                        }
                    }
                }
                [
                    Column::owned(mixed),
                    Column::owned(prefix),
                    Column::rows(row_weights.as_slice().into(), width),
                ]
            },
            |v| scale * v[0] * v[1] * v[2],
            |p, &[q, k]| scale * q * k * eq(row, &p[..rows]),
        )?;
        let (r, h) = point.split_at(rows);
        let queries = mix
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let [value] = io.values(|| [self.key.eval(query(i), &point)])?;
                Ok(Claim {
                    point: point.clone(),
                    value,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if dot(queries.iter().map(|q| q.value), mix) != mixed_query {
            return Err("invalid causal query batch evaluation");
        }
        let mut suffix = weights::<F>(r);
        for j in (0..n - 1).rev() {
            let next = suffix[j + 1];
            suffix[j] += next;
        }
        for (x, y) in suffix.iter_mut().zip(col_weights) {
            *x *= y;
        }
        let key_claim = io.public_dot(prefix, &suffix, || {
            let w = weights::<F>(h);
            key()
                .chunks_exact(self.key.width)
                .map(|k| dot(k.iter().copied().map(F::from_i32), w.iter().copied()))
                .collect()
        })?;
        Ok((
            queries,
            Claim {
                point: [key_claim.point.as_slice(), h].concat(),
                value: key_claim.value,
            },
        ))
    }
}

impl Tile {
    fn check<F: ProofField>(
        &self,
        narrow: bool,
        point: &[F],
        rem: F,
        score: F,
        io: &mut Session<'_, '_, F>,
    ) -> Result<()> {
        if narrow {
            self.range.suffix(1).check(&[(point, [rem])], io)
        } else {
            self.range.check(&[(point, [score, rem])], io)
        }
    }

    fn point<F: ProofField>(&self, row: &[F], col: &[F]) -> (F, Vec<F>) {
        let r = row.len() - self.shape.row_vars();
        let c = col.len() - (self.shape.vars() - self.shape.row_vars());
        let fixed = |p: &[F], offset: usize, bits: usize| {
            eq(
                &p[..bits],
                &(0..bits)
                    .map(|i| F::from_bool(offset >> (p.len() - i - 1) & 1 == 1))
                    .collect::<Vec<_>>(),
            )
        };
        (
            fixed(row, self.row, r) * fixed(col, self.col, c),
            [&row[r..], &col[c..]].concat(),
        )
    }
}
