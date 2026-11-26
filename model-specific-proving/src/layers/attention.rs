use std::{cell::OnceCell, sync::Arc};

use jolt_field::{Ring, Zero};

use super::{
    linear::{narrow_dot, wide_dot, Linear, Rotation, Rounded, Rounding},
    norm::{Norm, NormTrace},
    softmax::{tiles, Soft, SoftTrace, Tile},
    Attention, NormalizationParameters, Projection, Rotary, SoftmaxMask,
};
use crate::protocol::{
    field::F,
    lookup::{Addresses, Lookups},
    opening::{Layout, Packed, Request},
    sumcheck::{Column, Session},
    tensor::{bind_rows, equality, inner, weights, Claim, Shape},
    Result,
};

struct Project<'a> {
    linear: Linear<'a>,
    norm: Option<Norm<'a>>,
    rotation: Option<Rotation<'a>>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ProjectTrace {
    linear: Rounded,
    norm: Option<NormTrace>,
    rotation: Option<Rounded>,
}

impl ProjectTrace {
    fn normalized(&self) -> &[i32] {
        self.norm
            .as_ref()
            .map_or(self.linear.values.as_slice(), |norm| &norm.values)
    }

    fn values(&self) -> &[i32] {
        self.rotation
            .as_ref()
            .map_or_else(|| self.normalized(), |rot| &rot.values)
    }
}

impl<'a> Project<'a> {
    fn new(
        layout: &mut Layout,
        input: Shape,
        bits: usize,
        weights: Projection<'a>,
        norm: Option<NormalizationParameters<'a>>,
        rope: Option<Rotary<'a>>,
    ) -> Result<Self> {
        let linear = Linear::new(layout, input, weights, bits)?;
        let shape = linear.shape();
        let norm = norm
            .map(|params| Norm::new(layout, shape, bits, params))
            .transpose()?;
        let rotation = rope
            .map(|table| Rotation::new(layout, shape, bits, table))
            .transpose()?;
        Ok(Self {
            linear,
            norm,
            rotation,
        })
    }

    fn execute(&self, input: &[i32], packed: &mut Packed<'_>) -> Result<ProjectTrace> {
        let linear = self.linear.execute(input, packed)?;
        let norm = self
            .norm
            .as_ref()
            .map(|norm| norm.execute(&linear.values, packed))
            .transpose()?;
        let y = norm
            .as_ref()
            .map_or(linear.values.as_slice(), |norm| &norm.values);
        let rotation = self
            .rotation
            .as_ref()
            .map(|rot| rot.execute(y, packed))
            .transpose()?;
        Ok(ProjectTrace {
            linear,
            norm,
            rotation,
        })
    }

    fn reverse<'w>(
        &self,
        io: &mut Session<'_, '_>,
        requests: &mut Vec<Request>,
        mut claim: Claim,
        witness: impl Fn() -> (&'w ProjectTrace, &'w [i32]),
    ) -> Result<Claim> {
        if let Some(rot) = &self.rotation {
            claim = rot.reverse(io, requests, claim, || {
                let w = witness().0;
                (w.rotation.as_ref().expect("rotary trace"), w.normalized())
            })?;
        }
        if let Some(norm) = &self.norm {
            claim = norm.reverse(io, requests, &[claim], || {
                let w = witness().0;
                (
                    w.norm.as_ref().expect("Q/K normalization trace"),
                    &w.linear.values,
                )
            })?;
        }
        self.linear.reverse(io, requests, claim, || {
            let (w, x) = witness();
            (&w.linear, x)
        })
    }
}

struct Head<'a> {
    scores: Vec<Rounding>,
    soft: Soft<'a>,
    value: Rounding,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct HeadTrace {
    scores: Vec<Rounded>,
    soft: SoftTrace,
    value: Rounded,
}

pub(crate) struct AttentionPlan<'a> {
    input: Shape,
    query: Vec<Project<'a>>,
    groups: Vec<(Project<'a>, Linear<'a>)>,
    heads: Vec<Head<'a>>,
    tiles: Vec<Tile>,
    rotary_factor: i32,
    causal: bool,
    joined: Shape,
    output: Linear<'a>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct AttentionTrace {
    queries: Vec<ProjectTrace>,
    groups: Vec<(ProjectTrace, Rounded)>,
    heads: Vec<HeadTrace>,
    output: Rounded,
}

impl AttentionTrace {
    pub(crate) fn take_addresses(&mut self) -> Vec<Vec<Arc<Addresses>>> {
        for project in self
            .queries
            .iter_mut()
            .chain(self.groups.iter_mut().map(|(key, _)| key))
        {
            if let Some(norm) = &mut project.norm {
                norm.values = Vec::new();
            }
        }
        for head in &mut self.heads {
            for scores in &mut head.scores {
                scores.values = Vec::new();
            }
        }
        self.heads
            .iter_mut()
            .map(|head| std::mem::take(&mut head.soft.addresses))
            .collect()
    }

    pub(crate) fn restore_addresses(&mut self, addresses: Vec<Vec<Arc<Addresses>>>) {
        for (head, addresses) in self.heads.iter_mut().zip(addresses) {
            head.soft.addresses = addresses;
        }
    }

    pub(crate) fn values(&self) -> &[i32] {
        &self.output.values
    }
}

impl<'a> AttentionPlan<'a> {
    pub(crate) fn new(
        layout: &mut Layout,
        input: Shape,
        bits: usize,
        p: Attention<'a>,
        head_norm: Option<[NormalizationParameters<'a>; 2]>,
    ) -> Result<Self> {
        if p.groups.is_empty()
            || p.queries.is_empty()
            || !p.queries.len().is_multiple_of(p.groups.len())
            || p.factor <= 0
        {
            return Err("invalid attention parameters");
        }
        let query = p
            .queries
            .into_iter()
            .map(|w| Project::new(layout, input, bits, w, head_norm.map(|n| n[0]), p.rotary))
            .collect::<Result<Vec<_>>>()?;
        let groups = p
            .groups
            .into_iter()
            .map(|[k, v]| {
                Ok((
                    Project::new(layout, input, bits, k, head_norm.map(|n| n[1]), p.rotary)?,
                    Linear::new(layout, input, v, bits)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let value = groups[0].1.shape();
        let key = groups[0].0.linear.shape();
        if query.iter().any(|q| q.linear.shape() != key)
            || groups
                .iter()
                .any(|(k, v)| k.linear.shape() != key || v.shape() != value)
            || (query.len() > 1 && !value.width.is_power_of_two())
        {
            return Err("inconsistent attention head dimensions");
        }
        let cancel = (p.factor.trailing_zeros() as usize).min(2 * bits);
        let rotary_factor = p.factor >> cancel;
        let tiles = tiles(input.rows, p.mask)?;
        let causal = p.mask == SoftmaxMask::Causal;
        let mut heads = Vec::new();
        for _ in &query {
            let scores = tiles
                .iter()
                .map(|tile| Rounding::new(layout, tile.shape, 2 * bits - cancel))
                .collect();
            let soft = Soft::new(layout, input.rows, p.mask, p.tables, &tiles)?;
            heads.push(Head {
                scores,
                soft,
                value: Rounding::new(layout, value, bits),
            });
        }
        let joined = Shape::new(input.rows, heads.len() * value.width)?;
        let output = Linear::new(layout, joined, p.output, bits)?;
        if output.shape() != input {
            return Err("wrong attention output shape");
        }
        Ok(Self {
            input,
            query,
            groups,
            heads,
            tiles,
            rotary_factor,
            causal,
            joined,
            output,
        })
    }

    pub(crate) fn execute(&self, input: &[i32], packed: &mut Packed<'_>) -> Result<AttentionTrace> {
        let queries = self
            .query
            .iter()
            .map(|q| q.execute(input, packed))
            .collect::<Result<Vec<_>>>()?;
        let groups = self
            .groups
            .iter()
            .map(|(k, v)| Ok((k.execute(input, packed)?, v.execute(input, packed)?)))
            .collect::<Result<Vec<_>>>()?;
        let per_group = queries.len() / groups.len();
        let key_width = self.query[0].linear.shape().width;
        let value_width = self.groups[0].1.shape().width;
        let n = self.input.rows;
        let mut heads = Vec::with_capacity(queries.len());
        for (group, (key, value)) in groups.iter().enumerate() {
            let values = &value.values;
            let columns: Vec<Vec<i32>> = (0..value_width)
                .map(|c| {
                    values
                        .iter()
                        .skip(c)
                        .step_by(value_width)
                        .copied()
                        .collect()
                })
                .collect();
            let max_v = values
                .iter()
                .map(|x| u128::from(x.unsigned_abs()))
                .max()
                .unwrap_or(0);
            let key = key.values();
            let max_k = key
                .iter()
                .map(|x| u128::from(x.unsigned_abs()))
                .max()
                .unwrap_or(0);
            for i in group * per_group..(group + 1) * per_group {
                let head = &self.heads[i];
                let query = queries[i].values();
                let narrow: Vec<_> = query
                    .chunks_exact(key_width)
                    .map(|q| {
                        q.iter().map(|x| u128::from(x.unsigned_abs())).sum::<u128>() * max_k
                            <= i64::MAX as u128
                    })
                    .collect();
                let scores: Vec<_> = self
                    .tiles
                    .iter()
                    .zip(&head.scores)
                    .map(|(tile, proof)| {
                        proof.execute(packed, |r, c| {
                            let (row, col) = (tile.row + r, tile.col + c);
                            if self.causal && col > row {
                                return 0;
                            }
                            let q = &query[row * key_width..][..key_width];
                            let k = &key[col * key_width..][..key_width];
                            i128::from(self.rotary_factor)
                                * if narrow[row] {
                                    narrow_dot(q, k)
                                } else {
                                    wide_dot(q, k)
                                }
                        })
                    })
                    .collect::<Result<_>>()?;
                let score = |r, c| self.score(i, &scores, r, c);
                let soft = head.soft.execute(score, packed)?;
                let probabilities: Vec<Vec<i32>> = (0..n)
                    .map(|r| {
                        (0..if self.causal { r + 1 } else { n })
                            .map(|c| head.soft.probability(&soft, r, c))
                            .collect()
                    })
                    .collect();
                let narrow: Vec<_> = probabilities
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|x| u128::from(x.unsigned_abs()))
                            .sum::<u128>()
                            * max_v
                            <= i64::MAX as u128
                    })
                    .collect();
                let value = head.value.execute(packed, |r, c| {
                    let row = &probabilities[r];
                    let v = &columns[c][..row.len()];
                    if narrow[r] {
                        narrow_dot(row, v)
                    } else {
                        wide_dot(row, v)
                    }
                })?;
                heads.push(HeadTrace {
                    scores,
                    soft,
                    value,
                });
            }
        }
        let concatenated = self.concatenate(&heads, value_width);
        let output = self.output.execute(&concatenated, packed)?;
        Ok(AttentionTrace {
            queries,
            groups,
            heads,
            output,
        })
    }

    fn concatenate(&self, heads: &[HeadTrace], width: usize) -> Vec<i32> {
        let mut values = Vec::with_capacity(self.joined.len());
        for row in 0..self.joined.rows {
            for head in heads {
                values.extend_from_slice(&head.value.values[row * width..][..width]);
            }
        }
        values
    }

    pub(crate) fn restore(&self, trace: &mut AttentionTrace) -> Result<()> {
        for (plan, project) in self.query.iter().zip(&mut trace.queries).chain(
            self.groups
                .iter()
                .map(|(key, _)| key)
                .zip(trace.groups.iter_mut().map(|(key, _)| key)),
        ) {
            if let (Some(norm), Some(trace)) = (&plan.norm, &mut project.norm) {
                norm.restore(trace)?;
            }
        }
        for (head, witness) in self.heads.iter().zip(&mut trace.heads) {
            for (block, rounded) in witness.scores.iter_mut().enumerate() {
                rounded.values = head.soft.restore_scores(&witness.soft, block)?;
            }
        }
        Ok(())
    }

    fn score(&self, head: usize, scores: &[Rounded], row: usize, col: usize) -> i32 {
        let (tile, index) = self.heads[head].soft.find(row, col);
        let width = self.tiles[tile].shape.width;
        let stride = width.next_power_of_two();
        scores[tile].values[index / stride * width + index % stride]
    }

    fn qk<'w>(
        &self,
        io: &mut Session<'_, '_>,
        claim: Claim,
        witness: impl Fn() -> (&'w [i32], &'w [i32]),
    ) -> Result<[Claim; 2]> {
        let shape = self.query[0].linear.shape();
        let width = shape.width.next_power_of_two();
        let (qr, kr) = claim.point.split_at(shape.row_bits());
        let n = shape.rows;
        let row_weights = weights(qr);
        let key_weights = weights(kr);
        let factor = F::from_i32(self.rotary_factor);
        let (end, [q, k]) = io.fold(
            claim.value,
            shape.bits(),
            3,
            || {
                let (query, key) = witness();
                let mut prefix = Vec::with_capacity((1 << shape.row_bits()) * width);
                if self.causal {
                    let mut current = vec![F::zero(); width];
                    for (key, &w) in key.chunks_exact(shape.width).zip(&key_weights) {
                        for (sum, &v) in current.iter_mut().zip(key) {
                            *sum += F::from_i32(v) * w;
                        }
                        prefix.extend_from_slice(&current);
                    }
                } else {
                    let bound = bind_rows(shape.width, key, kr);
                    for _ in 0..n {
                        prefix.extend_from_slice(&bound);
                    }
                }
                prefix.resize(1 << shape.bits(), F::zero());
                vec![
                    Column::source(1 << shape.bits(), move |i| shape.at(query, i)),
                    Column::Dense(prefix),
                    Column::axis(1 << shape.bits(), width.ilog2() as usize, row_weights),
                ]
            },
            [0, 1],
            |c| factor * c[0] * c[1] * c[2],
        )?;
        let (rp, cols) = end.challenges.split_at(shape.row_bits());
        io.expect(&end, || factor * q * k * equality(qr, rp))?;
        let mut suffix = weights(rp);
        suffix[n..].fill(F::zero());
        if self.causal {
            for i in (0..n - 1).rev() {
                let next = suffix[i + 1];
                suffix[i] += next;
            }
        } else {
            let sum = suffix.iter().copied().sum();
            suffix[..n].fill(sum);
        }
        for (w, coefficient) in suffix.iter_mut().zip(key_weights) {
            *w *= coefficient;
        }
        let k = io.dot(k, &suffix, || {
            let cols = weights(cols);
            let mut values: Vec<_> = witness()
                .1
                .chunks_exact(shape.width)
                .map(|row| inner(cols.iter().copied(), row.iter().copied().map(F::from_i32)))
                .collect();
            values.resize(suffix.len(), F::zero());
            values
        })?;
        let key = Claim::new([k.point.as_slice(), cols].concat(), k.value);
        Ok([Claim::new(end.challenges, q), key])
    }

    pub(crate) fn reverse<'w>(
        &self,
        io: &mut Session<'_, '_>,
        requests: &mut Vec<Request>,
        reads: &mut Lookups<'a>,
        claim: Claim,
        witness: impl Fn() -> (&'w AttentionTrace, &'w [i32]) + Sync,
    ) -> Result<Claim> {
        let shape = self.groups[0].1.shape();
        let bound_heads = OnceCell::new();
        let joined = self.output.reverse_bound(
            io,
            requests,
            claim,
            || &witness().0.output.remainder,
            |rows| {
                let heads = bound_heads.get_or_init(|| {
                    witness()
                        .0
                        .heads
                        .iter()
                        .map(|head| bind_rows(shape.width, &head.value.values, rows))
                        .collect::<Vec<_>>()
                });
                let mut joined = Vec::with_capacity(self.joined.width.next_power_of_two());
                for head in heads {
                    joined.extend_from_slice(&head[..shape.width]);
                }
                joined.resize(self.joined.width.next_power_of_two(), F::zero());
                joined
            },
        )?;
        let (rows, c) = joined.point.split_at(shape.row_bits());
        let (head_bits, channels) = c.split_at(c.len() - (shape.bits() - shape.row_bits()));
        let point = [rows, channels].concat();
        let per_group = self.heads.len() / self.groups.len();
        let mut input = Vec::new();
        let mut values = Vec::new();
        for (g, (key, value)) in self.groups.iter().enumerate() {
            let mut key_claims = Vec::new();
            let mut value_claims = Vec::new();
            let bound_v = OnceCell::new();
            let bind_v = || {
                bound_v
                    .get_or_init(|| {
                        let cols = weights(channels);
                        let mut projected: Vec<_> = witness().0.groups[g]
                            .1
                            .values
                            .chunks_exact(shape.width)
                            .map(|row| {
                                inner(cols.iter().copied(), row.iter().copied().map(F::from_i32))
                            })
                            .collect();
                        projected.resize(shape.rows.next_power_of_two(), F::zero());
                        projected
                    })
                    .clone()
            };
            for i in g * per_group..(g + 1) * per_group {
                let head = &self.heads[i];
                let trace = || &witness().0.heads[i];
                let [v] = io.values(|| {
                    [inner(
                        weights(channels),
                        bound_heads.get().expect("bound head outputs")[i]
                            .iter()
                            .copied(),
                    )]
                })?;
                values.push(v);
                let value_output = Claim::new(point.clone(), v);
                let raw = head
                    .value
                    .recover(io, requests, &value_output, || &trace().value.remainder)?;
                let size = shape.rows.next_power_of_two();
                let (end, [probability, value_input]) = io.fold(
                    raw,
                    shape.row_bits(),
                    2,
                    || {
                        let weights_r = weights(rows);
                        let mut a = vec![F::zero(); size];
                        for (r, &w) in weights_r.iter().take(shape.rows).enumerate() {
                            for (c, p) in a
                                .iter_mut()
                                .take(if self.causal { r + 1 } else { shape.rows })
                                .enumerate()
                            {
                                *p += w.mul_signed(i128::from(head.soft.probability(
                                    &trace().soft,
                                    r,
                                    c,
                                )));
                            }
                        }
                        vec![Column::Dense(a), Column::Dense(bind_v())]
                    },
                    [0, 1],
                    |v| v[0] * v[1],
                )?;
                io.expect(&end, || probability * value_input)?;
                let (score, tiles) = head.soft.reverse(
                    io,
                    requests,
                    reads,
                    Claim::new([rows, &end.challenges].concat(), probability),
                    || &trace().soft,
                    |j| &trace().scores[j],
                )?;
                value_claims.push(Claim::new(end.challenges, value_input));
                let mut raw = F::zero();
                let mut whole_score = F::zero();
                for (rounding, tile) in head.scores.iter().zip(tiles) {
                    let [value] = io.values(|| [tile.qk[0]])?;
                    whole_score += tile.factor * value;
                    let claim = Claim::new(tile.point, value);
                    raw +=
                        tile.factor * rounding.recover_at(io, requests, &claim, || tile.qk[1])?;
                }
                if !io.proving() && whole_score != score.value {
                    return Err("inconsistent QK tiles");
                }
                let [q, k] = self.qk(io, Claim::new(score.point, raw), || {
                    let w = witness().0;
                    (w.queries[i].values(), w.groups[g].0.values())
                })?;
                input.push(self.query[i].reverse(io, requests, q, || {
                    let (w, x) = witness();
                    (&w.queries[i], x)
                })?);
                key_claims.push(k);
            }
            let combine = |io: &mut Session<'_, '_>,
                           shape: Shape,
                           claims: &[Claim],
                           x: &dyn Fn() -> &'w [i32]| {
                io.combine(claims, || {
                    let x = x();
                    (0..1 << shape.bits()).map(|i| shape.at(x, i)).collect()
                })
            };
            let k = combine(io, key.linear.shape(), &key_claims, &|| {
                witness().0.groups[g].0.values()
            })?;
            let v = io.combine(&value_claims, bind_v)?;
            let v = Claim::new([v.point.as_slice(), channels].concat(), v.value);
            input.push(key.reverse(io, requests, k, || {
                let (w, x) = witness();
                (&w.groups[g].0, x)
            })?);
            input.push(value.reverse(io, requests, v, || {
                let (w, x) = witness();
                (&w.groups[g].1, x)
            })?);
        }
        if !io.proving() && joined.value != inner(weights(head_bits), values) {
            return Err("invalid head concatenation");
        }
        io.combine(&input, || {
            let x = witness().1;
            (0..1 << self.input.bits())
                .map(|i| self.input.at(x, i))
                .collect()
        })
    }
}
