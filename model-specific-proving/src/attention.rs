use crate::{
    advice::{Advice, Requests, Slot},
    channel::{Channel, Result},
    field::F,
    fold::{agreement, basis, Column},
    layers::{Attention, NormalizationParameters, Projection, Rotary},
    linear,
    lookup::Lookups,
    norm::{Claim, Norm, Witness},
    rounding::Rounded,
    softmax::{Soft, SoftTrace},
    tensor::Shape,
};
use jolt_field::{Ring, Zero};

pub(crate) struct Context<'a> {
    pub settings: &'a Attention<'a>,
    pub shape: Shape,
    pub bits: u32,
    pub head_norms: Option<[NormalizationParameters<'a>; 2]>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Projected {
    transformed: Rounded,
    norm: Option<(Option<Witness>, Vec<Slot>)>,
    rotary: Option<Rounded>,
}

impl Projected {
    fn execute(
        advice: &mut Advice,
        shape: Shape,
        bits: u32,
        input: Option<&[i32]>,
        projection: Projection<'_>,
        norm: Option<NormalizationParameters<'_>>,
        rotation: Option<Rotary<'_>>,
    ) -> Result<Self> {
        let transformed = linear::project(advice, shape, bits, projection, input)?;
        let p = transformed.values.as_deref();
        let norm = norm
            .map(|params| {
                let plan = params.at(shape.rows, bits)?;
                let output = p.map(|v| plan.witness(v)).transpose()?;
                let slots = plan.record(advice, p, output.as_ref())?;
                Ok((output, slots))
            })
            .transpose()?;
        let values = norm
            .as_ref()
            .map_or(p, |(x, _)| x.as_ref().map(|x| x.result.as_slice()));
        let rotary = rotation
            .map(|r| r.apply(advice, transformed.shape, bits, values))
            .transpose()?;
        Ok(Self {
            transformed,
            norm,
            rotary,
        })
    }

    fn values(&self) -> Option<&[i32]> {
        if let Some(rope) = &self.rotary {
            rope.values.as_deref()
        } else if let Some((x, _)) = &self.norm {
            x.as_ref().map(|w| w.result.as_slice())
        } else {
            self.transformed.values.as_deref()
        }
    }

    fn reverse(
        &self,
        io: &mut Channel<'_, '_>,
        requests: &mut Requests,
        ctx: &Context<'_>,
        norm: Option<NormalizationParameters<'_>>,
        mut claims: Vec<Claim<F>>,
    ) -> Result<Vec<Claim<F>>> {
        if let Some((r, rounded)) = ctx.settings.rotary.zip(self.rotary.as_ref()) {
            let p = self.transformed.values.as_deref();
            let v = self
                .norm
                .as_ref()
                .map_or(p, |(x, _)| x.as_ref().map(|x| x.result.as_slice()));
            claims = vec![r.reverse(io, requests, rounded, &claims, v)?];
        }
        if let Some((norm, (trace, slots))) = norm.zip(self.norm.as_ref()) {
            let n = norm.at(ctx.shape.rows, ctx.bits)?;
            claims = vec![n.reduce_model(
                io,
                self.transformed.values.as_deref(),
                trace.as_ref(),
                &claims,
                &[],
                slots,
                requests,
            )?];
        }
        Ok(claims)
    }
}

impl<'a> NormalizationParameters<'a> {
    pub(crate) fn at(self, rows: usize, bits: u32) -> Result<Norm<'a>> {
        let epsilon = u64::try_from(self.epsilon_variance_units)
            .map_err(|_| "invalid normalization epsilon")?;
        Ok(Norm {
            rows,
            bits,
            epsilon,
            center: self.kind == crate::layers::NormalizationKind::Layer,
            gain: self.gain,
            bias: self.bias,
        })
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Head {
    soft: SoftTrace,
    value: Rounded,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct Trace {
    queries: Vec<Projected>,
    keys: Vec<Projected>,
    values: Vec<Rounded>,
    heads: Vec<Head>,
    joined: Option<Vec<i32>>,
    pub output: Rounded,
}

fn narrow(a: &[i32], b: &[i32], terms: usize) -> bool {
    let maximum = |v: &[i32]| {
        v.iter()
            .map(|x| u128::from(x.unsigned_abs()))
            .max()
            .unwrap_or(0)
    };
    maximum(a) * maximum(b) * terms as u128 <= i64::MAX as u128
}

fn score_fits(
    params: [NormalizationParameters<'_>; 2],
    rope: Option<Rotary<'_>>,
    bits: u32,
    factor: i32,
) -> bool {
    let scale = 1u128 << bits;
    let sqrt = |n: u128| {
        let x = n.isqrt();
        x + u128::from(x * x < n)
    };
    let rotation = rope.map(|r| {
        r.cos
            .iter()
            .zip(r.sin)
            .map(|(&c, &s)| {
                sqrt(u128::from(c.unsigned_abs()).pow(2) + u128::from(s.unsigned_abs()).pow(2))
            })
            .max()
            .unwrap_or(0)
    });
    let bound = |p: NormalizationParameters<'_>| {
        let root = u128::try_from(p.epsilon_variance_units).ok()?.isqrt();
        if root == 0 {
            return None;
        }
        let width = sqrt(p.gain.len() as u128);
        let gain = p.gain.iter().map(|x| u128::from(x.unsigned_abs())).max()?;
        let bias = p.bias.iter().map(|x| u128::from(x.unsigned_abs())).max()?;
        let quotient = width * ((scale * (root + 1)).div_ceil(root) + 1);
        let norm = (gain * quotient).div_ceil(scale) + width * (bias + 1);
        Some(rotation.map_or(norm, |r| (r * norm).div_ceil(scale) + width))
    };
    bound(params[0])
        .zip(bound(params[1]))
        .and_then(|(q, k)| q.checked_mul(k))
        .and_then(|p| p.checked_mul(factor as u128))
        .is_some_and(|raw| raw.div_ceil(scale * scale) <= i32::MAX as u128)
}

impl<'a> Context<'a> {
    pub fn execute(&self, advice: &mut Advice, input: Option<&[i32]>) -> Result<Trace> {
        let p = self.settings;
        if p.groups.is_empty() || p.queries.is_empty() || p.queries.len() % p.groups.len() != 0 {
            return Err("invalid grouped Attention dimensions");
        }
        let head = p.queries[0].bias.len();
        let channels = p.groups[0][1].bias.len();
        if !head.is_power_of_two()
            || !channels.is_power_of_two()
            || p.queries.iter().any(|q| q.bias.len() != head)
            || p.groups
                .iter()
                .any(|[k, v]| k.bias.len() != head || v.bias.len() != channels)
        {
            return Err("invalid Attention head width");
        }
        let factor = p.factor;
        if factor <= 0 {
            return Err("invalid score scale");
        }
        let commit_scores = !self
            .head_norms
            .is_some_and(|p| score_fits(p, self.settings.rotary, self.bits, factor));
        let shift = factor.trailing_zeros().min(self.bits * 2);
        let factor = factor >> shift;
        let n = self.shape.rows;
        let soft = Soft::new(n, p.mask, p.tables)?;
        let build = |advice: &mut Advice, p0, i| {
            Projected::execute(
                advice,
                self.shape,
                self.bits,
                input,
                p0,
                self.head_norms.map(|x| x[i]),
                p.rotary,
            )
        };
        let queries = p
            .queries
            .iter()
            .map(|&q| build(advice, q, 0))
            .collect::<Result<Vec<_>>>()?;
        let keys = p
            .groups
            .iter()
            .map(|[k, _]| build(advice, *k, 1))
            .collect::<Result<Vec<_>>>()?;
        let values = p
            .groups
            .iter()
            .map(|[_, v]| linear::project(advice, self.shape, self.bits, *v, input))
            .collect::<Result<Vec<_>>>()?;
        let value_columns: Vec<_> = values
            .iter()
            .map(|v| {
                v.values.as_ref().map(|x| {
                    (0..channels)
                        .flat_map(|c| x[c..].iter().step_by(channels).copied())
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let mut heads = Vec::new();
        let group = p.queries.len() / p.groups.len();
        for (i, q) in queries.iter().enumerate() {
            let k = &keys[i / group];
            let v = &values[i / group];
            let mut probability = soft.prepare(
                advice,
                2 * self.bits - shift,
                commit_scores,
                q.values().zip(k.values()).map(|(q, k)| {
                    let use_narrow = narrow(q, k, head);
                    move |r, c| {
                        i128::from(factor)
                            * crate::tensor::dot(
                                &q[r * head..][..head],
                                &k[c * head..][..head],
                                use_narrow,
                            )
                    }
                }),
            )?;
            let value = Rounded::evaluate(
                advice,
                v.shape,
                self.bits,
                true,
                probability
                    .probabilities
                    .as_deref()
                    .zip(value_columns[i / group].as_deref())
                    .map(|(prob, vals)| {
                        let soft = &soft;
                        let compact = narrow(prob, vals, n);
                        move |r, c| {
                            let count = soft.count(r);
                            crate::tensor::dot(
                                &prob[r * n..][..count],
                                &vals[c * n..][..count],
                                compact,
                            )
                        }
                    }),
            )?;
            probability.probabilities = None;
            heads.push(Head {
                soft: probability,
                value,
            });
        }
        let all = Shape::new(n, channels * heads.len())?;
        let joined = input.map(|_| {
            let mut joined = Vec::with_capacity(n * all.cols);
            for row in 0..n {
                for h in &heads {
                    joined.extend_from_slice(
                        &h.value.values.as_ref().unwrap()[row * channels..][..channels],
                    );
                }
            }
            joined
        });
        let output = linear::project(advice, all, self.bits, p.output, joined.as_deref())?;
        Ok(Trace {
            queries,
            keys,
            values,
            heads,
            joined,
            output,
        })
    }

    pub fn reverse(
        &self,
        io: &mut Channel<'_, '_>,
        requests: &mut Requests,
        lookups: &mut Lookups<'a>,
        trace: &Trace,
        output: Claim<F>,
        input: Option<&[i32]>,
    ) -> Result<Vec<Claim<F>>> {
        let p = self.settings;
        let channels = trace.values[0].shape.cols;
        let head_bits = channels.ilog2() as usize;
        let group = trace.heads.len() / trace.keys.len();
        let all = Shape::new(self.shape.rows, channels * trace.heads.len())?;
        let claim = linear::reverse(
            io,
            requests,
            all,
            self.bits,
            &[(&trace.output, p.output, output)],
            trace.joined.as_deref(),
        )?
        .remove(0);
        let (r, c) = claim.point.split_at(all.row_bits());
        let (heads, col) = c.split_at(c.len() - head_bits);
        let weights = basis(heads);
        let point = [r, col].concat();
        let mut value = F::zero();
        let soft = Soft::new(self.shape.rows, p.mask, p.tables)?;
        let mut projections = Vec::new();
        let shift = p.factor.trailing_zeros().min(2 * self.bits);
        for (g, key) in trace.keys.iter().enumerate() {
            let v = &trace.values[g];
            let bound_v = v.values.as_ref().map(|v| {
                let weight = basis(col);
                let mut rows: Vec<F> = v
                    .chunks_exact(channels)
                    .map(|row| {
                        row.iter()
                            .zip(&weight)
                            .map(|(&x, &w)| w.mul_integer(i128::from(x)))
                            .sum()
                    })
                    .collect();
                rows.resize(self.shape.rows.next_power_of_two(), F::zero());
                rows
            });
            let mut k_claims = Vec::new();
            for i in g * group..(g + 1) * group {
                let head = &trace.heads[i];
                let [y] = io.values(|| {
                    [head
                        .value
                        .shape
                        .eval(head.value.values.as_ref().unwrap(), &point)]
                })?;
                value += weights[i] * y;
                let y = Claim {
                    point: point.clone(),
                    value: y,
                };
                let pv = head.value.raw(io, requests, &y)?;
                let codes = soft.addresses(&head.soft);
                let [prob, vc] =
                    values_product(io, &soft, pv, r, col, codes.as_deref(), bound_v.as_deref())?;
                let score = soft.reverse(
                    io,
                    requests,
                    lookups,
                    &head.soft,
                    prob,
                    codes.as_deref(),
                    2 * self.bits - shift,
                )?;
                let q = &trace.queries[i];
                let [q_claim, k_claim] = score_product(
                    io,
                    &soft,
                    q.transformed.shape,
                    &score.point,
                    score.value,
                    p.factor >> shift,
                    q.values().zip(key.values()),
                )?;
                let query_claims = q.reverse(
                    io,
                    requests,
                    self,
                    self.head_norms.map(|n| n[0]),
                    vec![q_claim],
                )?;
                projections.extend(
                    query_claims
                        .into_iter()
                        .map(|claim| (&q.transformed, p.queries[i], claim)),
                );
                k_claims.push(k_claim);
                projections.push((v, p.groups[g][1], vc));
            }
            let key_claims =
                key.reverse(io, requests, self, self.head_norms.map(|n| n[1]), k_claims)?;
            projections.extend(
                key_claims
                    .into_iter()
                    .map(|claim| (&key.transformed, p.groups[g][0], claim)),
            );
        }
        if value != claim.value {
            return Err("invalid Attention head concatenation");
        }
        linear::reverse(io, requests, self.shape, self.bits, &projections, input)
    }
}

fn values_product(
    io: &mut Channel<'_, '_>,
    soft: &Soft<'_>,
    raw: F,
    row: &[F],
    col: &[F],
    addresses: Option<&[u32]>,
    values: Option<&[F]>,
) -> Result<[Claim<F>; 2]> {
    let n = soft.shape.rows;
    let size = n.next_power_of_two();
    let (end, [p, v]) = io.fold(
        raw,
        soft.shape.row_bits(),
        2,
        || {
            let mut ps = vec![F::zero(); size];
            for (r, w) in basis(row).into_iter().take(n).enumerate() {
                for j in 0..soft.count(r) {
                    ps[j] += w.mul_integer(i128::from(soft.probability(addresses.unwrap(), r, j)));
                }
            }
            [
                Column::table(ps, size, 1),
                Column::table(values.unwrap().to_vec(), size, 1),
            ]
        },
        [0, 1],
        |v| v[0] * v[1],
    )?;
    if end.output_claim != p * v {
        return Err("invalid PV terminal evaluation");
    }
    Ok([
        Claim {
            point: [row, &end.challenges].concat(),
            value: p,
        },
        Claim {
            point: [&end.challenges, col].concat(),
            value: v,
        },
    ])
}

fn score_product(
    io: &mut Channel<'_, '_>,
    soft: &Soft<'_>,
    shape: Shape,
    point: &[F],
    value: F,
    factor: i32,
    witness: Option<(&[i32], &[i32])>,
) -> Result<[Claim<F>; 2]> {
    let (qr, kr) = point.split_at(shape.row_bits());
    let q_weights = basis(qr);
    let k_weights = basis(kr);
    let d = shape.cols;
    let size = shape.padded();
    let f = F::from_i32(factor);
    let (end, [q, k]) = io.fold(
        value,
        shape.bits(),
        3,
        || {
            let (q, k) = witness.unwrap();
            let mut prefix = vec![F::zero(); size];
            let mut acc = vec![F::zero(); d];
            let mut last = 0;
            for r in 0..shape.rows {
                let count = soft.count(r);
                for (j, &w) in k_weights.iter().enumerate().take(count).skip(last) {
                    for (out, &v) in acc.iter_mut().zip(&k[j * d..][..d]) {
                        *out += w * F::from_i32(v);
                    }
                }
                last = count;
                prefix[r * d..][..d].copy_from_slice(&acc);
            }
            [
                Column::source(size, |i| shape.at(q, i)),
                Column::table(prefix, size, 1),
                Column::table(q_weights.clone(), size, d),
            ]
        },
        [0, 1],
        |v| f * v[0] * v[1] * v[2],
    )?;
    let (r, c) = end.challenges.split_at(shape.row_bits());
    if end.output_claim != f * q * k * agreement(qr, r) {
        return Err("invalid QK terminal evaluation");
    }

    let mut col_weights = basis(r);
    col_weights[shape.rows..].fill(F::zero());
    if soft.mask == crate::layers::SoftmaxMask::Causal {
        for j in (0..shape.rows - 1).rev() {
            let next = col_weights[j + 1];
            col_weights[j] += next;
        }
    } else {
        let sum = col_weights.iter().copied().sum();
        col_weights[..shape.rows].fill(sum);
    }
    for (a, b) in col_weights.iter_mut().zip(k_weights) {
        *a *= b;
    }
    let key = io.dot(k, col_weights, || {
        let weights = basis(c);
        let mut columns: Vec<_> = witness
            .unwrap()
            .1
            .chunks_exact(d)
            .map(|r| {
                r.iter()
                    .zip(&weights)
                    .map(|(&a, &b)| F::from_i32(a) * b)
                    .sum()
            })
            .collect();
        columns.resize(shape.rows.next_power_of_two(), F::zero());
        columns
    })?;
    Ok([
        Claim {
            point: end.challenges.clone(),
            value: q,
        },
        Claim {
            point: [key.point.as_slice(), c].concat(),
            value: key.value,
        },
    ])
}
