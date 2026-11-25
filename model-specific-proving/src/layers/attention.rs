//! Witnesses and reverse proofs for attention layers.

use jolt_transcript::Transcript;

use crate::protocol::{
    lookup::Batches,
    pcs::{Advice, Layout},
    session::{Claim, ProofField, Session},
    tensor::{dot, weights, Shape},
    Result,
};

use super::{
    linear::{Linear, Product, ProductKind, Projection, Rotary, Rotate, Rounded},
    nonlinear::{Norm, NormTrace, NormalizationParameters},
    softmax::{ClipPlan, ClipSpec, Softmax, SoftmaxMask, SoftmaxTables, SoftmaxTrace},
};

#[cfg(feature = "akita")]
mod blocked;
mod causal;
#[cfg(feature = "akita")]
pub use blocked::{BlockedAttention, BlockedAttentionProof};
use causal::{Causal, Remainders};

pub struct Attention<'a> {
    pub queries: Vec<Projection<'a>>,
    pub groups: Vec<[Projection<'a>; 2]>,
    pub output: Projection<'a>,
    pub rotary: Option<Rotary<'a>>,
    pub factor: i32,
    pub mask: SoftmaxMask,
    pub tables: SoftmaxTables<'a>,
}

pub(crate) struct AttentionPlan<'a> {
    queries: Vec<Project<'a>>,
    groups: Vec<(Project<'a>, Project<'a>)>,
    heads: Vec<Head<'a>>,
    output: Linear<'a>,
    joined: Shape,
    key: Shape,
    value: Shape,
    group_size: usize,
    factor: i32,
}

struct Project<'a> {
    linear: Linear<'a>,
    norm: Option<Norm<'a>>,
    rotary: Option<Rotate<'a>>,
}
struct ProjectTrace<F> {
    linear: Rounded,
    norm: Option<NormTrace<F>>,
    rotary: Option<Rounded>,
}

impl<F> ProjectTrace<F> {
    fn output(&self) -> &[i32] {
        self.rotary
            .as_ref()
            .map_or_else(|| self.before_rope(), |t| &t.output)
    }

    fn before_rope(&self) -> &[i32] {
        self.norm
            .as_ref()
            .map_or(&self.linear.output, |t| &t.output)
    }
}

impl<'a> Project<'a> {
    fn new(
        layout: &mut Layout,
        input: Shape,
        bits: usize,
        w: Projection<'a>,
        norm: Option<NormalizationParameters<'a>>,
        rotary: Option<Rotary<'a>>,
    ) -> Result<Self> {
        let linear = Linear::new(layout, input, w, bits, false)?;
        let norm = norm
            .map(|p| Norm::new(layout, linear.shape(), bits, p))
            .transpose()?;
        let rotary = rotary
            .map(|r| Rotate::new(layout, linear.shape(), bits, r))
            .transpose()?;
        Ok(Self {
            linear,
            norm,
            rotary,
        })
    }

    fn bind(&self, t: &mut impl Transcript) {
        self.linear.bind(t);
        t.append(&u64::from(self.norm.is_some()));
        if let Some(norm) = &self.norm {
            norm.bind(t);
        }
        t.append(&u64::from(self.rotary.is_some()));
        if let Some(r) = &self.rotary {
            r.bind(t);
        }
    }

    fn execute<F: ProofField>(&self, x: &[i32], a: &mut Advice) -> Result<ProjectTrace<F>> {
        let linear = self.linear.execute(x, a)?;
        let norm = self
            .norm
            .as_ref()
            .map(|n| n.execute(&linear.output, a))
            .transpose()?;
        let rotary = self
            .rotary
            .as_ref()
            .map(|r| r.execute(norm.as_ref().map_or(&linear.output, |n| &n.output), a))
            .transpose()?;
        Ok(ProjectTrace {
            linear,
            norm,
            rotary,
        })
    }

    fn reverse<'w, F: ProofField>(
        &self,
        mut q: Claim<F>,
        trace: impl Fn() -> (&'w [i32], &'w ProjectTrace<F>),
        io: &mut Session<'_, '_, F>,
    ) -> Result<Claim<F>> {
        if let Some(r) = &self.rotary {
            q = r.reverse(
                q,
                || {
                    let (_, w) = trace();
                    (w.before_rope(), w.rotary.as_ref().expect("RoPE trace"))
                },
                io,
            )?;
        }
        if let Some(norm) = &self.norm {
            q = norm.reverse(
                &[q],
                || {
                    let (_, w) = trace();
                    (
                        w.linear.output.as_slice(),
                        w.norm.as_ref().expect("QK norm trace"),
                    )
                },
                io,
            )?;
        }
        self.linear
            .reverse(q, || (trace().0, &trace().1.linear), io)
    }
}

struct Head<'a> {
    score: Option<Product>,
    causal: Option<Causal>,
    softmax: Softmax<'a>,
    value: Product,
}
struct HeadTrace<F> {
    score: Option<Rounded>,
    remainders: Remainders,
    softmax: SoftmaxTrace<F>,
    value: Rounded,
}
pub(crate) struct AttentionTrace<F> {
    queries: Vec<ProjectTrace<F>>,
    groups: Vec<(ProjectTrace<F>, ProjectTrace<F>)>,
    heads: Vec<HeadTrace<F>>,
    joined: Vec<i32>,
    output: Rounded,
}

impl<F> AttentionTrace<F> {
    pub(crate) fn output(&self) -> &[i32] {
        &self.output.output
    }
}

impl<'a> AttentionPlan<'a> {
    pub(crate) fn new(
        layout: &mut Layout,
        input: Shape,
        bits: usize,
        p: Attention<'a>,
        input_bound: Option<u128>,
        qk_norm: Option<[NormalizationParameters<'a>; 2]>,
    ) -> Result<Self> {
        Self::build(layout, input, bits, p, false, input_bound, qk_norm)
    }

    fn build(
        layout: &mut Layout,
        input: Shape,
        bits: usize,
        p: Attention<'a>,
        blocked: bool,
        input_bound: Option<u128>,
        qk_norm: Option<[NormalizationParameters<'a>; 2]>,
    ) -> Result<Self> {
        if p.groups.is_empty()
            || p.queries.is_empty()
            || !p.queries.len().is_multiple_of(p.groups.len())
            || p.factor <= 0
            || bits > 15
        {
            return Err("invalid attention configuration");
        }
        let score_bound = input_bound
            .filter(|_| qk_norm.is_none() && matches!(p.mask, SoftmaxMask::Causal))
            .and_then(|bound| {
                let scale = 1u128 << bits;
                let rotary = p.rotary.map_or(scale, |r| {
                    r.cos
                        .iter()
                        .zip(r.sin)
                        .map(|(&c, &s)| u128::from(c.unsigned_abs()) + u128::from(s.unsigned_abs()))
                        .max()
                        .unwrap_or(scale)
                });
                let project_bound = |proj: Projection<'_>| {
                    (proj.output_bound(input.width, bound, scale) * rotary)
                        .div_ceil(scale)
                        .min(1 << 31)
                };
                let q = p
                    .queries
                    .iter()
                    .copied()
                    .map(project_bound)
                    .max()
                    .expect("queries");
                let k = p
                    .groups
                    .iter()
                    .map(|pair| project_bound(pair[0]))
                    .max()
                    .expect("keys");
                (q * k * p.queries[0].bias.len() as u128)
                    .checked_mul(p.factor as u128)
                    .map(|product| product.div_ceil(scale * scale))
            });
        let group_size = p.queries.len() / p.groups.len();
        let queries = p
            .queries
            .into_iter()
            .map(|q| Project::new(layout, input, bits, q, qk_norm.map(|n| n[0]), p.rotary))
            .collect::<Result<Vec<_>>>()?;
        let groups = p
            .groups
            .into_iter()
            .map(|[k, v]| {
                Ok((
                    Project::new(layout, input, bits, k, qk_norm.map(|n| n[1]), p.rotary)?,
                    Project::new(layout, input, bits, v, None, None)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let key = queries[0].linear.shape();
        let value = groups[0].1.linear.shape();
        if key.width > 1 << 30
            || queries.iter().any(|q| q.linear.shape() != key)
            || groups
                .iter()
                .any(|(k, v)| k.linear.shape() != key || v.linear.shape() != value)
            || (queries.len() > 1 && !value.width.is_power_of_two())
        {
            return Err("inconsistent head dimensions");
        }
        let square = Shape::new(input.rows, input.rows)?;
        let mut block_template = Layout::default();
        let head_layout = if blocked {
            &mut block_template
        } else {
            &mut *layout
        };
        let heads = (0..queries.len())
            .map(|_| {
                let compact = !blocked
                    && matches!(p.mask, SoftmaxMask::Causal)
                    && input.rows.is_power_of_two();
                let softmax =
                    Softmax::new(head_layout, square, p.mask, p.tables, score_bound, compact)?;
                let causal = compact
                    .then(|| Causal::new(head_layout, key, bits * 2, p.factor, &softmax))
                    .transpose()?;
                Ok(Head {
                    // Without clipping, Softmax binds the integer score to its
                    // bounded row shift and table index at every used position.
                    score: causal
                        .is_none()
                        .then(|| {
                            Product::with_output_range(
                                head_layout,
                                square,
                                bits * 2,
                                ProductKind::Matrix {
                                    inner: key.width,
                                    transpose: true,
                                    factor: p.factor,
                                },
                                softmax.has_excess(),
                            )
                        })
                        .transpose()?,
                    causal,
                    softmax,
                    value: Product::new(
                        head_layout,
                        value,
                        bits,
                        ProductKind::Matrix {
                            inner: input.rows,
                            transpose: false,
                            factor: 1,
                        },
                    )?,
                })
            })
            .collect::<Result<_>>()?;
        let width = queries
            .len()
            .checked_mul(value.width)
            .ok_or("attention width overflow")?;
        let joined = Shape::new(input.rows, width)?;
        let output = Linear::new(layout, joined, p.output, bits, false)?;
        if output.shape() != input {
            return Err("invalid attention output shape");
        }
        Ok(Self {
            queries,
            groups,
            heads,
            output,
            joined,
            key,
            value,
            group_size,
            factor: p.factor,
        })
    }

    pub(crate) fn bind(&self, t: &mut impl Transcript) {
        t.append(&[
            self.queries.len() as u64,
            self.groups.len() as u64,
            self.factor as u64,
        ]);
        for q in &self.queries {
            q.bind(t);
        }
        for (k, v) in &self.groups {
            k.bind(t);
            v.bind(t);
        }
        self.heads[0].softmax.bind(t);
        self.output.bind(t);
    }

    pub(crate) fn execute<F: ProofField>(
        &self,
        x: &[i32],
        a: &mut Advice,
    ) -> Result<AttentionTrace<F>> {
        let queries = self
            .queries
            .iter()
            .map(|p| p.execute(x, a))
            .collect::<Result<Vec<_>>>()?;
        let groups = self
            .groups
            .iter()
            .map(|(k, v)| Ok((k.execute(x, a)?, v.execute(x, a)?)))
            .collect::<Result<Vec<_>>>()?;
        let heads = self
            .heads
            .iter()
            .enumerate()
            .map(|(i, h)| {
                let (k, v) = &groups[i / self.group_size];
                let (score, remainders, softmax) = if let Some(causal) = &h.causal {
                    let (scores, mut remainders) =
                        causal.execute(queries[i].output(), k.output(), a)?;
                    let softmax = h
                        .softmax
                        .execute_causal(|r| &scores[r * (r + 1) / 2..][..r + 1], a)?;
                    if h.softmax.narrow_score(&softmax) {
                        causal.narrow(a, &mut remainders);
                    }
                    (None, remainders, softmax)
                } else {
                    let score = h.score.as_ref().expect("full QK").execute(
                        queries[i].output(),
                        k.output(),
                        a,
                    )?;
                    let softmax = h.softmax.execute(&score.output, a)?;
                    (Some(score), Vec::new(), softmax)
                };
                let value =
                    h.value
                        .execute_matrix(h.softmax.probabilities(&softmax), v.output(), a)?;
                Ok(HeadTrace {
                    score,
                    remainders,
                    softmax,
                    value,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut joined = Vec::with_capacity(self.joined.rows * self.joined.width);
        for row in 0..self.joined.rows {
            for h in &heads {
                joined.extend_from_slice(
                    &h.value.output[row * self.value.width..][..self.value.width],
                );
            }
        }
        let output = self.output.execute(&joined, a)?;
        Ok(AttentionTrace {
            queries,
            groups,
            heads,
            joined,
            output,
        })
    }

    pub(crate) fn clip_specs<F>(&self, trace: &AttentionTrace<F>) -> Vec<ClipSpec> {
        self.heads
            .iter()
            .zip(&trace.heads)
            .map(|(h, t)| h.softmax.clip_spec(&t.softmax))
            .collect()
    }

    pub(crate) fn clip_plan(
        &self,
        layout: &mut Layout,
        specs: &[ClipSpec],
    ) -> Result<Vec<Option<ClipPlan>>> {
        if self.heads.len() != specs.len() {
            return Err("invalid Softmax clip head count");
        }
        self.heads
            .iter()
            .zip(specs)
            .map(|(h, s)| {
                let plan = h.softmax.clip_plan(layout, s)?;
                if s.narrow_score {
                    h.causal
                        .as_ref()
                        .ok_or("invalid causal score plan")?
                        .narrow_layout(layout);
                }
                Ok(plan)
            })
            .collect()
    }

    pub(crate) fn record_clip<F>(
        &self,
        plans: &[Option<ClipPlan>],
        trace: &AttentionTrace<F>,
        advice: &mut Advice,
    ) -> Result<()> {
        for ((head, plan), w) in self.heads.iter().zip(plans).zip(&trace.heads) {
            if let Some(plan) = plan {
                head.softmax.record_clip(plan, &w.softmax, advice)?;
            }
        }
        Ok(())
    }

    pub(crate) fn reverse<'s, 'w, F: ProofField>(
        &'s self,
        q: Claim<F>,
        clips: &[Option<ClipPlan>],
        trace: impl Fn() -> (&'w [i32], &'w AttentionTrace<F>) + Sync,
        io: &mut Session<'_, '_, F>,

        lookups: &mut Batches<'s, 'w, F>,
    ) -> Result<Vec<Claim<F>>> {
        let t = || trace().1;
        let claim = self
            .output
            .reverse(q, || (&t().joined[..], &t().output), io)?;
        let (row, col) = claim.point.split_at(self.joined.row_vars());
        let (head, channel) =
            col.split_at(col.len() - self.value.width.next_power_of_two().ilog2() as usize);
        let point = [row, channel].concat();
        let mut values = Vec::with_capacity(self.heads.len());
        for i in 0..self.heads.len() {
            let [v] = io.values(|| [self.value.eval(&t().heads[i].value.output, &point)])?;
            values.push(v);
        }
        if !io.proving() && claim.value != dot(weights(head), values.iter().copied()) {
            return Err("invalid head concatenation");
        }
        let mut keys: Vec<Vec<_>> = self.groups.iter().map(|_| Vec::new()).collect();
        let mut vals: Vec<Vec<_>> = self.groups.iter().map(|_| Vec::new()).collect();
        let mut inputs = Vec::new();
        let mut probabilities = Vec::with_capacity(self.heads.len());
        for (i, (h, value)) in self.heads.iter().zip(values).enumerate() {
            let group = i / self.group_size;
            let hw = || &t().heads[i];
            let [p, v] = h.value.reverse_matrix(
                Claim {
                    point: point.clone(),
                    value,
                },
                || (t().groups[group].1.output(), &hw().value),
                |point| h.softmax.bind_probabilities(&hw().softmax, point),
                io,
            )?;
            probabilities.push(p);
            vals[group].push(v);
        }
        let softmax = self.heads[0].softmax.reverse(
            &probabilities,
            clips,
            |i| {
                let head = &t().heads[i];
                (
                    &head.softmax,
                    head.score.as_ref().map(|score| score.output.as_slice()),
                )
            },
            io,
        )?;
        let mut reads = Vec::with_capacity(self.heads.len());
        let mut scores = softmax.into_iter();
        for group in 0..self.groups.len() {
            let start = group * self.group_size;
            let heads = &self.heads[start..][..self.group_size];
            if let Some(causal) = &heads[0].causal {
                let mut raw = Vec::with_capacity(heads.len());
                for (index, h) in heads.iter().enumerate() {
                    let i = start + index;
                    let (score, read) = scores.next().expect("head score claim");
                    raw.push(
                        h.causal.as_ref().expect("causal head").reverse(
                            score,
                            clips
                                .get(i)
                                .and_then(Option::as_ref)
                                .is_some_and(ClipPlan::narrows_score),
                            || &t().heads[i].remainders,
                            io,
                        )?,
                    );
                    reads.push((&h.softmax, read));
                }
                let (queries, key) = causal.product(
                    &raw,
                    |i| t().queries[start + i].output(),
                    || t().groups[group].0.output(),
                    io,
                )?;
                keys[group].push(key);
                for (index, q) in queries.into_iter().enumerate() {
                    let i = start + index;
                    inputs.push(self.queries[i].reverse(q, || (trace().0, &t().queries[i]), io)?);
                }
            } else {
                for (index, h) in heads.iter().enumerate() {
                    let i = start + index;
                    let (score, read) = scores.next().expect("head score claim");
                    let [q, k] = h.score.as_ref().expect("full QK").reverse(
                        score,
                        || {
                            (
                                t().queries[i].output(),
                                t().groups[group].0.output(),
                                t().heads[i].score.as_ref().expect("full QK trace"),
                            )
                        },
                        io,
                    )?;
                    inputs.push(self.queries[i].reverse(q, || (trace().0, &t().queries[i]), io)?);
                    keys[group].push(k);
                    reads.push((&h.softmax, read));
                }
            }
        }
        for (i, (k, v)) in self.groups.iter().enumerate() {
            let g = || &t().groups[i];
            let kc = io.merge(&keys[i], || {
                self.key.padded(g().0.output(), F::zero(), F::from_i32)
            })?;
            let vc = io.merge(&vals[i], || {
                self.value.padded(g().1.output(), F::zero(), F::from_i32)
            })?;
            inputs.push(k.reverse(kc, || (trace().0, &g().0), io)?);
            inputs.push(v.reverse(vc, || (trace().0, &g().1), io)?);
        }
        Softmax::defer(reads, clips, |i| &t().heads[i].softmax, io, lookups)?;
        Ok(inputs)
    }
}
