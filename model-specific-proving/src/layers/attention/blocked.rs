//! Akita attention proofs with a single live block of quadratic witness data.

use jolt_akita::{AkitaField as F, AkitaScheme};
use jolt_field::{Ring, Zero};
use jolt_transcript::{protocol_tag, Transcript};

use crate::protocol::{
    pcs::{bind_i32, plan, AkitaAdviceSetup, Proof},
    range::{Ranges, I32},
};

use super::*;

type AkitaProof = Proof<AkitaScheme>;

/// Proves an attention projection for a public input and output without retaining
/// all query rows' private advice. Projections and head outputs are committed once.
pub struct BlockedAttention<'a> {
    attention: AttentionPlan<'a>,
    input: Shape,
    bits: u8,
    layout: Layout,
    blocks: Vec<Block<'a>>,
    queries: Vec<Ranges>,
    keys: Vec<Ranges>,
    values: Vec<Ranges>,
    joined: Ranges,
}

pub struct BlockedAttentionProof {
    projections: AkitaProof,
    blocks: Vec<AkitaProof>,
}

struct Block<'a> {
    start: usize,
    rows: usize,
    layout: Layout,
    heads: Vec<Head<'a>>,
}

struct Projections {
    queries: Vec<ProjectTrace>,
    groups: Vec<(ProjectTrace, ProjectTrace)>,
}

struct Trace {
    advice: Advice,
    projected: Projections,
    joined: Vec<i32>,
    output: Rounded,
}

impl<'a> BlockedAttention<'a> {
    pub fn new(
        rows: usize,
        width: usize,
        bits: u8,
        block_rows: usize,
        attention: Attention<'a>,
    ) -> Result<Self> {
        if !block_rows.is_power_of_two() {
            return Err("attention block size must be a power of two");
        }
        let input = Shape::new(rows, width)?;
        let mut layout = Layout::default();
        let attention =
            AttentionPlan::build(&mut layout, input, bits.into(), attention, true, None)?;
        let mut ranges = |shape, count| {
            (0..count)
                .map(|_| Ranges::new(&mut layout, shape, [I32]))
                .collect::<Result<Vec<_>>>()
        };
        let queries = ranges(attention.key, attention.queries.len())?;
        let keys = ranges(attention.key, attention.groups.len())?;
        let values = ranges(attention.value, attention.groups.len())?;
        let joined = Ranges::new(&mut layout, attention.joined, [I32])?;
        let mut blocks = Vec::new();
        let mut start = 0;
        while start < rows {
            let remaining = (rows - start).min(block_rows);
            let size = 1 << remaining.ilog2();
            blocks.push(Block::new(&attention, bits.into(), start, size)?);
            start += size;
        }
        Ok(Self {
            attention,
            input,
            bits,
            layout,
            blocks,
            queries,
            keys,
            values,
            joined,
        })
    }

    /// Computes the public attention output and proves it. Each block is opened
    /// before the next block's advice is constructed.
    pub fn prove(
        &self,
        input: &[i32],
        setup: &AkitaAdviceSetup,
    ) -> Result<(Vec<i32>, BlockedAttentionProof)> {
        self.input.check(input)?;
        let mut advice = self.layout.witness();
        let projected = Projections {
            queries: self
                .attention
                .queries
                .iter()
                .map(|p| p.execute(input, &mut advice))
                .collect::<Result<_>>()?,
            groups: self
                .attention
                .groups
                .iter()
                .map(|(k, v)| {
                    Ok((
                        k.execute(input, &mut advice)?,
                        v.execute(input, &mut advice)?,
                    ))
                })
                .collect::<Result<_>>()?,
        };
        let mut joined =
            Vec::with_capacity(self.attention.joined.rows * self.attention.joined.width);
        for block in &self.blocks {
            let (_, heads) = block.execute(&self.attention, &projected)?;
            for row in 0..block.rows {
                for head in &heads {
                    let width = self.attention.value.width;
                    joined.extend_from_slice(&head.value.output[row * width..][..width]);
                }
            }
        }
        for (range, trace) in self.queries.iter().zip(&projected.queries) {
            range.record(&mut advice, &[trace.output()])?;
        }
        for ((k, v), (kt, vt)) in self.keys.iter().zip(&self.values).zip(&projected.groups) {
            k.record(&mut advice, &[kt.output()])?;
            v.record(&mut advice, &[vt.output()])?;
        }
        self.joined.record(&mut advice, &[&joined])?;
        let output = self.attention.output.execute(&joined, &mut advice)?;
        let public = output.output.clone();
        let trace = Trace {
            advice,
            projected,
            joined,
            output,
        };
        let mut blocks = Vec::with_capacity(self.blocks.len());
        let projections = Proof::prove(
            &protocol_tag(b"atlas/blocked-attention/v1"),
            &self.layout,
            trace,
            |w| &mut w.advice,
            &plan::<AkitaScheme>(&self.layout)?,
            setup,
            |t| self.bind(t, input, &public),
            |w, io| {
                self.reduce(
                    input,
                    &public,
                    || &w,
                    io,
                    |block, seed| {
                        let projected = &w.projected;
                        let witness = block.execute(&self.attention, projected)?;
                        let mut claims = Vec::new();
                        let proof = Proof::prove(
                            &protocol_tag(b"atlas/attention-block/v1"),
                            &block.layout,
                            witness,
                            |w| &mut w.0,
                            &plan::<AkitaScheme>(&block.layout)?,
                            setup,
                            |t| block.bind(t, seed),
                            |w, io| {
                                claims = block.reduce(&self.attention, io, || {
                                    (&w.0, w.1.as_slice(), projected)
                                })?;
                                Ok(())
                            },
                        )?;
                        blocks.push(proof);
                        Ok(claims)
                    },
                )
            },
        )?;
        Ok((
            public,
            BlockedAttentionProof {
                projections,
                blocks,
            },
        ))
    }

    pub fn verify(
        &self,
        input: &[i32],
        output: &[i32],
        proof: &BlockedAttentionProof,
        setup: &AkitaAdviceSetup,
    ) -> Result<()> {
        self.input.check(input)?;
        self.input.check(output)?;
        if proof.blocks.len() != self.blocks.len() {
            return Err("invalid attention block count");
        }
        let mut blocks = proof.blocks.iter();
        proof.projections.verify(
            &protocol_tag(b"atlas/blocked-attention/v1"),
            &self.layout,
            setup,
            |t| self.bind(t, input, output),
            |io| {
                self.reduce(
                    input,
                    output,
                    || unreachable!(),
                    io,
                    |block, seed| {
                        let mut claims = Vec::new();
                        blocks.next().ok_or("missing attention block")?.verify(
                            &protocol_tag(b"atlas/attention-block/v1"),
                            &block.layout,
                            setup,
                            |t| block.bind(t, seed),
                            |io| {
                                claims = block.reduce(&self.attention, io, || unreachable!())?;
                                Ok(())
                            },
                        )?;
                        Ok(claims)
                    },
                )
            },
        )
    }

    fn bind(&self, t: &mut impl Transcript, input: &[i32], output: &[i32]) {
        for value in [self.input.rows, self.input.width, usize::from(self.bits)] {
            t.append(&(value as u64));
        }
        self.attention.bind(t);
        for block in &self.blocks {
            t.append(&(block.rows as u64));
        }
        bind_i32(t, input);
        bind_i32(t, output);
    }

    fn reduce<'w>(
        &self,
        input: &[i32],
        output: &[i32],
        trace: impl Fn() -> &'w Trace,
        io: &mut Session<'_, '_, F>,
        mut run: impl FnMut(&Block<'_>, F) -> Result<Vec<[Claim<F>; 4]>>,
    ) -> Result<()> {
        let seed = io.challenge(1)?[0];
        let point = io.point(self.input.vars())?;
        let output_claim = Claim {
            value: self.input.eval(output, &point),
            point,
        };
        let result = self.attention.output.reverse(
            output_claim,
            || (&trace().joined, &trace().output),
            io,
        )?;
        let mut joined = vec![result];
        let mut queries = vec![Vec::new(); self.queries.len()];
        let mut keys = vec![Vec::new(); self.keys.len()];
        let mut values = vec![Vec::new(); self.values.len()];
        for block in &self.blocks {
            let claims = run(block, seed)?;
            for claim in claims.iter().flatten() {
                io.bind_values(&claim.point);
                io.bind_values(&[claim.value]);
            }
            for (i, [mut q, k, v, mut y]) in claims.into_iter().enumerate() {
                let cols = self.attention.joined.vars() - self.attention.value.vars();
                let local_rows = block.rows.ilog2() as usize;
                y.point.splice(local_rows..local_rows, coordinates(i, cols));
                let high = self.input.row_vars() - local_rows;
                for claim in [&mut q, &mut y] {
                    claim
                        .point
                        .splice(..0, coordinates(block.start / block.rows, high));
                }
                let group = i / self.attention.group_size;
                queries[i].push(q);
                keys[group].push(k);
                values[group].push(v);
                joined.push(y);
            }
        }
        let result = io.merge(&joined, || {
            self.attention
                .joined
                .padded(&trace().joined, F::zero(), F::from_i32)
        })?;
        self.joined.check(&[(&result.point, [result.value])], io)?;
        let mut project = |plan: &Project<'_>,
                           range: &Ranges,
                           claims: &[Claim<F>],
                           witness: &dyn Fn() -> &'w ProjectTrace| {
            let shape = plan.linear.shape();
            let q = io.merge(claims, || {
                shape.padded(witness().output(), F::zero(), F::from_i32)
            })?;
            range.check(&[(&q.point, [q.value])], io)?;
            let q = plan.reverse(q, || (input, witness()), io)?;
            if !io.proving() && q.value != self.input.eval(input, &q.point) {
                return Err("invalid attention projection input");
            }
            Ok(())
        };
        for (i, p) in self.attention.queries.iter().enumerate() {
            project(p, &self.queries[i], &queries[i], &|| {
                &trace().projected.queries[i]
            })?;
        }
        for (i, (k, v)) in self.attention.groups.iter().enumerate() {
            project(k, &self.keys[i], &keys[i], &|| {
                &trace().projected.groups[i].0
            })?;
            project(v, &self.values[i], &values[i], &|| {
                &trace().projected.groups[i].1
            })?;
        }
        Ok(())
    }
}

impl<'a> Block<'a> {
    fn new(attention: &AttentionPlan<'a>, bits: usize, start: usize, rows: usize) -> Result<Self> {
        let mut layout = Layout::default();
        let heads = attention
            .heads
            .iter()
            .map(|head| {
                Ok(Head {
                    score: Product::new(
                        &mut layout,
                        Shape::new(rows, attention.key.rows)?,
                        bits * 2,
                        ProductKind::Matrix {
                            inner: attention.key.width,
                            transpose: true,
                            factor: attention.factor,
                        },
                    )?,
                    softmax: head.softmax.block(&mut layout, start, rows)?,
                    value: Product::new(
                        &mut layout,
                        Shape::new(rows, attention.value.width)?,
                        bits,
                        ProductKind::Matrix {
                            inner: attention.key.rows,
                            transpose: false,
                            factor: 1,
                        },
                    )?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            start,
            rows,
            layout,
            heads,
        })
    }

    fn bind(&self, t: &mut impl Transcript, seed: F) {
        t.append(&seed);
        t.append(&(self.start as u64));
        t.append(&(self.rows as u64));
    }

    fn query<'w>(&self, p: &AttentionPlan<'_>, witness: &'w Projections, head: usize) -> &'w [i32] {
        &witness.queries[head].output()[self.start * p.key.width..][..self.rows * p.key.width]
    }

    fn execute(
        &self,
        p: &AttentionPlan<'_>,
        projected: &Projections,
    ) -> Result<(Advice, Vec<HeadTrace<F>>)> {
        let mut advice = self.layout.witness();
        let mut trace = Vec::with_capacity(self.heads.len());
        for (i, head) in self.heads.iter().enumerate() {
            let (k, v) = &projected.groups[i / p.group_size];
            let score = head
                .score
                .execute(self.query(p, projected, i), k.output(), &mut advice)?;
            let softmax = head.softmax.execute(&score.output, &mut advice)?;
            let value = head.value.execute_matrix(
                head.softmax.probabilities(&softmax),
                v.output(),
                &mut advice,
            )?;
            trace.push(HeadTrace {
                score,
                softmax,
                value,
            });
        }
        Ok((advice, trace))
    }

    fn reduce<'w>(
        &self,
        p: &AttentionPlan<'_>,
        io: &mut Session<'_, '_, F>,
        witness: impl Fn() -> (&'w Advice, &'w [HeadTrace<F>], &'w Projections) + Sync,
    ) -> Result<Vec<[Claim<F>; 4]>> {
        let shape = Shape::new(self.rows, p.value.width)?;
        let mut lookups = Batches::default();
        let mut claims = Vec::with_capacity(self.heads.len());
        for (i, head) in self.heads.iter().enumerate() {
            let wt = || &witness().1[i];
            let projected = || witness().2;
            let group = i / p.group_size;
            let y = fresh(shape, || &wt().value.output, io)?;
            let [prob, v] = head.value.reverse_matrix(
                y.clone(),
                || (projected().groups[group].1.output(), &wt().value),
                |point| head.softmax.bind_probabilities(&wt().softmax, point),
                io,
            )?;
            let mut scores =
                head.softmax
                    .reverse(&[prob], |_| (&wt().softmax, &wt().score.output), io)?;
            let (score, read) = scores.pop().ok_or("missing attention score")?;
            let [q, k] = head.score.reverse(
                score,
                || {
                    (
                        self.query(p, projected(), i),
                        projected().groups[group].0.output(),
                        &wt().score,
                    )
                },
                io,
            )?;
            Softmax::defer(
                vec![(&head.softmax, read)],
                |_| &wt().softmax,
                io,
                &mut lookups,
            )?;
            claims.push([q, k, v, y]);
        }
        lookups.finish(io)?;
        Ok(claims)
    }
}

fn fresh<'a>(
    shape: Shape,
    values: impl FnOnce() -> &'a [i32],
    io: &mut Session<'_, '_, F>,
) -> Result<Claim<F>> {
    let point = io.point(shape.vars())?;
    let [value] = io.values(|| [shape.eval(values(), &point)])?;
    Ok(Claim { point, value })
}

fn coordinates(value: usize, bits: usize) -> impl Iterator<Item = F> {
    (0..bits)
        .rev()
        .map(move |bit| F::from_bool(value & (1 << bit) != 0))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use jolt_akita::{AkitaChunkProfile, AkitaScheduleArtifacts};

    use super::*;
    use crate::layers::{SoftmaxMask, SoftmaxTables};

    #[test]
    fn blocked_attention_matches_unpartitioned_attention() {
        let config = || {
            let column = Projection {
                weights: &[4, -2],
                bias: &[0],
            };
            Attention {
                queries: vec![column; 2],
                groups: vec![[column; 2]],
                output: Projection {
                    weights: &[4, 0, 0, 4],
                    bias: &[0; 2],
                },
                rotary: None,
                factor: 4,
                mask: SoftmaxMask::Causal,
                tables: SoftmaxTables {
                    probabilities: &[4, 2, 1, 0],
                    sums: &[4, 2, 0, 0],
                    limit: 4,
                },
            }
        };
        let input = [4, 2, 4, -2, -2, 3, 2, 0, 2, 1];
        let mut layout = Layout::default();
        let reference =
            AttentionPlan::new(&mut layout, Shape::new(5, 2).unwrap(), 2, config(), None).unwrap();
        let expected = reference
            .execute::<F>(&input, &mut layout.witness())
            .unwrap();
        let artifacts: Arc<_> =
            AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2));
        let setup = AkitaAdviceSetup::new(artifacts)
            .unwrap()
            .with_group_row_limit(512)
            .unwrap();
        let plan = BlockedAttention::new(5, 2, 2, 2, config()).unwrap();
        let (output, proof) = plan.prove(&input, &setup).unwrap();
        assert_eq!(output, expected.output());
        plan.verify(&input, &output, &proof, &setup).unwrap();
    }
}
