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
    softmax::{Softmax, SoftmaxMask, SoftmaxTables, SoftmaxTrace},
};

#[cfg(feature = "akita")]
mod blocked;
#[cfg(feature = "akita")]
pub use blocked::{BlockedAttention, BlockedAttentionProof};

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
    rotary: Option<Rotate<'a>>,
}
struct ProjectTrace {
    linear: Rounded,
    rotary: Option<Rounded>,
}

impl ProjectTrace {
    fn output(&self) -> &[i32] {
        self.rotary
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
        rotary: Option<Rotary<'a>>,
    ) -> Result<Self> {
        let linear = Linear::new(layout, input, w, bits, false)?;
        let rotary = rotary
            .map(|r| Rotate::new(layout, linear.shape(), bits, r))
            .transpose()?;
        Ok(Self { linear, rotary })
    }

    fn bind(&self, t: &mut impl Transcript) {
        self.linear.bind(t);
        t.append(&u64::from(self.rotary.is_some()));
        if let Some(r) = &self.rotary {
            r.bind(t);
        }
    }

    fn execute(&self, x: &[i32], a: &mut Advice) -> Result<ProjectTrace> {
        let linear = self.linear.execute(x, a)?;
        let rotary = self
            .rotary
            .as_ref()
            .map(|r| r.execute(&linear.output, a))
            .transpose()?;
        Ok(ProjectTrace { linear, rotary })
    }

    fn reverse<'w, F: ProofField>(
        &self,
        mut q: Claim<F>,
        trace: impl Fn() -> (&'w [i32], &'w ProjectTrace),
        io: &mut Session<'_, '_, F>,
    ) -> Result<Claim<F>> {
        if let Some(r) = &self.rotary {
            q = r.reverse(
                q,
                || {
                    let (_, w) = trace();
                    (
                        w.linear.output.as_slice(),
                        w.rotary.as_ref().expect("RoPE trace"),
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
    score: Product,
    softmax: Softmax<'a>,
    value: Product,
}
struct HeadTrace<F> {
    score: Rounded,
    softmax: SoftmaxTrace<F>,
    value: Rounded,
}
pub(crate) struct AttentionTrace<F> {
    queries: Vec<ProjectTrace>,
    groups: Vec<(ProjectTrace, ProjectTrace)>,
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
    ) -> Result<Self> {
        Self::build(layout, input, bits, p, false, input_bound)
    }

    fn build(
        layout: &mut Layout,
        input: Shape,
        bits: usize,
        p: Attention<'a>,
        blocked: bool,
        input_bound: Option<u128>,
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
            .filter(|_| matches!(p.mask, SoftmaxMask::Causal))
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
            .map(|q| Project::new(layout, input, bits, q, p.rotary))
            .collect::<Result<Vec<_>>>()?;
        let groups = p
            .groups
            .into_iter()
            .map(|[k, v]| {
                Ok((
                    Project::new(layout, input, bits, k, p.rotary)?,
                    Project::new(layout, input, bits, v, None)?,
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
                let softmax = Softmax::new(head_layout, square, p.mask, p.tables, score_bound)?;
                Ok(Head {
                    // Without clipping, Softmax binds the integer score to its
                    // bounded row shift and table index at every used position.
                    score: Product::with_output_range(
                        head_layout,
                        square,
                        bits * 2,
                        ProductKind::Matrix {
                            inner: key.width,
                            transpose: true,
                            factor: p.factor,
                        },
                        softmax.has_excess(),
                    )?,
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
                let score = h.score.execute(queries[i].output(), k.output(), a)?;
                let softmax = h.softmax.execute(&score.output, a)?;
                let value = h.value.execute(&softmax.output, v.output(), a)?;
                Ok(HeadTrace {
                    score,
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

    pub(crate) fn reverse<'s, 'w, F: ProofField>(
        &'s self,
        q: Claim<F>,
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
            let [p, v] = h.value.reverse(
                Claim {
                    point: point.clone(),
                    value,
                },
                || {
                    (
                        &hw().softmax.output,
                        t().groups[group].1.output(),
                        &hw().value,
                    )
                },
                io,
            )?;
            probabilities.push(p);
            vals[group].push(v);
        }
        let softmax = self.heads[0].softmax.reverse(
            &probabilities,
            |i| {
                let head = &t().heads[i];
                (&head.softmax, head.score.output.as_slice())
            },
            io,
        )?;
        let mut reads = Vec::with_capacity(self.heads.len());
        for (i, (h, (score, read))) in self.heads.iter().zip(softmax).enumerate() {
            let group = i / self.group_size;
            let [q, k] = h.score.reverse(
                score,
                || {
                    (
                        t().queries[i].output(),
                        t().groups[group].0.output(),
                        &t().heads[i].score,
                    )
                },
                io,
            )?;
            inputs.push(self.queries[i].reverse(q, || (trace().0, &t().queries[i]), io)?);
            keys[group].push(k);
            reads.push((&h.softmax, read));
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
        Softmax::defer(reads, |i| &t().heads[i].softmax, io, lookups)?;
        Ok(inputs)
    }
}
