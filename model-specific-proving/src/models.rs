use jolt_transcript::{protocol_tag, prover_transcript, Keccak, Transcript};

use crate::{
    layers::{
        attention::{AttentionPlan, AttentionTrace},
        linear::{Linear, Rounded},
        mlp::{FeedForward, FeedTrace},
        norm::{Norm, NormTrace},
        Attention, Mlp, NormalizationParameters, Projection,
    },
    protocol::{
        lookup::Lookups,
        opening::{Layout, Packed, Request},
        output::Ranking,
        range::{Ranges, I32},
        sumcheck::Session,
        tensor::{Claim, Shape},
        Candidate, Proof, Result, TopKConfig,
    },
};

#[derive(Clone, Copy)]
pub struct PublicEmbeddings<'a> {
    pub width: usize,
    pub tokens: &'a [i32],
    pub positions: Option<&'a [i32]>,
    pub segments: Option<&'a [i32]>,
}

pub struct Layer<'a> {
    pub attention: Attention<'a>,
    pub mlp: Mlp<'a>,
    pub attention_norm: NormalizationParameters<'a>,
    pub mlp_norm: NormalizationParameters<'a>,
}

pub enum Architecture<'a> {
    Decoder {
        final_norm: NormalizationParameters<'a>,
        logits_bias: &'a [i32],
    },
    Encoder {
        embedding_norm: NormalizationParameters<'a>,
    },
}

struct Block<'a> {
    attention: AttentionPlan<'a>,
    mlp: FeedForward<'a>,
    norms: [Norm<'a>; 2],
    residuals: [Ranges<1>; 2],
}

struct Trace {
    attention: AttentionTrace,
    mlp: FeedTrace,
    norms: [NormTrace; 2],
    residuals: [Vec<i32>; 2],
}

impl Trace {
    fn output(&self, pre: bool) -> &[i32] {
        if pre {
            &self.residuals[1]
        } else {
            &self.norms[1].values
        }
    }
}

impl<'a> Block<'a> {
    fn execute(&self, input: &[i32], pre: bool, packed: &mut Packed<'_>) -> Result<Trace> {
        let (attention, first, norm1) = residual(
            pre,
            input,
            &self.norms[0],
            &self.residuals[0],
            packed,
            |x, packed| self.attention.execute(x, packed),
            AttentionTrace::values,
        )?;
        let hidden = if pre { &first } else { &norm1.values };
        let (mlp, output, norm2) = residual(
            pre,
            hidden,
            &self.norms[1],
            &self.residuals[1],
            packed,
            |x, packed| self.mlp.execute(x, packed),
            FeedTrace::values,
        )?;
        Ok(Trace {
            attention,
            mlp,
            norms: [norm1, norm2],
            residuals: [first, output],
        })
    }

    fn reverse<'w>(
        &self,
        io: &mut Session<'_, '_>,
        requests: &mut Vec<Request>,
        lookups: &mut Lookups<'a>,
        claims: &[Claim],
        pre: bool,
        witness: impl Fn() -> (&'w Trace, &'w [i32]) + Sync,
    ) -> Result<Vec<Claim>> {
        let point = if pre {
            join(io, requests, &self.residuals[1], claims, || {
                &witness().0.residuals[1]
            })?
        } else {
            let c = self.norms[1].reverse(io, requests, claims, || {
                let w = witness().0;
                (&w.norms[1], &w.residuals[1])
            })?;
            self.residuals[1].open(io, requests, &c.point, [c.value])?;
            c
        };
        let shape = self.residuals[0].shape;
        let [hidden, mlp] = split(io, shape, point, || {
            if pre {
                &witness().0.residuals[0]
            } else {
                &witness().0.norms[0].values
            }
        })?;
        let mlp = self.mlp.reverse(io, requests, lookups, mlp, || {
            let w = witness().0;
            (
                &w.mlp,
                if pre {
                    &w.norms[1].values
                } else {
                    &w.norms[0].values
                },
            )
        })?;
        let sum = if pre {
            let q = self.norms[1].reverse(io, requests, &[mlp], || {
                let w = witness().0;
                (&w.norms[1], &w.residuals[0])
            })?;
            join(io, requests, &self.residuals[0], &[hidden, q], || {
                &witness().0.residuals[0]
            })?
        } else {
            let q = self.norms[0].reverse(io, requests, &[hidden, mlp], || {
                let w = witness().0;
                (&w.norms[0], &w.residuals[0])
            })?;
            self.residuals[0].open(io, requests, &q.point, [q.value])?;
            q
        };
        let [skip, attention] = split(io, shape, sum, || witness().1)?;
        let attention = self
            .attention
            .reverse(io, requests, lookups, attention, || {
                let (w, x) = witness();
                (&w.attention, if pre { &w.norms[0].values } else { x })
            })?;
        let attention = if pre {
            self.norms[0].reverse(io, requests, &[attention], || {
                (&witness().0.norms[0], witness().1)
            })?
        } else {
            attention
        };
        Ok(vec![attention, skip])
    }
}

fn residual<T>(
    pre: bool,
    input: &[i32],
    norm: &Norm<'_>,
    range: &Ranges<1>,
    packed: &mut Packed<'_>,
    execute: impl FnOnce(&[i32], &mut Packed<'_>) -> Result<T>,
    output: impl FnOnce(&T) -> &[i32],
) -> Result<(T, Vec<i32>, NormTrace)> {
    let prepared = pre.then(|| norm.execute(input, packed)).transpose()?;
    let op = execute(prepared.as_ref().map_or(input, |t| &t.values), packed)?;
    let sum = input
        .iter()
        .zip(output(&op))
        .map(|(&x, &y)| x.checked_add(y).ok_or("residual overflow"))
        .collect::<Result<Vec<_>>>()?;
    range.record(packed, [&sum])?;
    let norm = match prepared {
        Some(n) => n,
        None => norm.execute(&sum, packed)?,
    };
    Ok((op, sum, norm))
}

fn split<'w>(
    io: &mut Session<'_, '_>,
    shape: Shape,
    claim: Claim,
    input: impl FnOnce() -> &'w [i32],
) -> Result<[Claim; 2]> {
    let [skip] = io.values(|| [shape.evaluate(input(), &claim.point)])?;
    Ok([
        Claim::new(claim.point.clone(), skip),
        Claim::new(claim.point, claim.value - skip),
    ])
}

fn join<'w>(
    io: &mut Session<'_, '_>,
    requests: &mut Vec<Request>,
    range: &Ranges<1>,
    claims: &[Claim],
    input: impl FnOnce() -> &'w [i32],
) -> Result<Claim> {
    let point = io.random(range.shape.bits(), range.shape.bits())?;
    let [value] = io.values(|| [range.shape.evaluate(input(), &point)])?;
    let combined = Claim::new(point, value);
    for claim in claims.iter().chain([&combined]) {
        range.open(io, requests, &claim.point, [claim.value])?;
    }
    Ok(combined)
}

#[derive(Clone, Copy)]
enum Output<'a> {
    Logits(&'a [i32]),
    Ranked(&'a TopKConfig, &'a [Vec<Candidate>]),
}

impl Output<'_> {
    fn domain(&self) -> [u8; 32] {
        protocol_tag(match self {
            Self::Logits(_) => b"atlas/transformer/logits/v1",
            Self::Ranked(..) => b"atlas/transformer/top-k/v1",
        })
    }

    fn bind(&self, t: &mut impl Transcript) {
        match self {
            Self::Logits(y) => bind_i32(t, y),
            Self::Ranked(cfg, candidates) => {
                t.append(&[
                    cfg.positions.start as u64,
                    cfg.positions.end as u64,
                    cfg.k as u64,
                    cfg.rows_per_block as u64,
                    cfg.vocab_per_block as u64,
                ]);
                for candidate in candidates.iter().flatten() {
                    t.append(&(candidate.id as u64));
                    t.append(&(i64::from(candidate.logit) as u64));
                }
            }
        }
    }
}

pub struct Model<'a> {
    shape: Shape,
    bits: usize,
    embeddings: PublicEmbeddings<'a>,
    layers: Vec<Block<'a>>,
    norm: Norm<'a>,
    decoder_bias: Option<&'a [i32]>,
    layout: Layout,
    digest: [u8; 32],
}

struct Run {
    norms: NormTrace,
    layers: Vec<Trace>,
}

impl<'a> Model<'a> {
    pub fn new(
        rows: usize,
        bits: u8,
        embeddings: PublicEmbeddings<'a>,
        layers: Vec<Layer<'a>>,
        architecture: Architecture<'a>,
    ) -> Result<Self> {
        Self::with_query_norms(rows, bits, embeddings, layers, architecture, &[])
    }

    pub(crate) fn with_query_norms(
        rows: usize,
        bits: u8,
        embeddings: PublicEmbeddings<'a>,
        layers: Vec<Layer<'a>>,
        architecture: Architecture<'a>,
        qk: &[Option<[NormalizationParameters<'a>; 2]>],
    ) -> Result<Self> {
        let shape = Shape::new(rows, embeddings.width)?;
        if bits > 15 || layers.is_empty() || !embeddings.tokens.len().is_multiple_of(shape.width) {
            return Err("invalid model dimensions");
        }
        let bits = usize::from(bits);
        let mut layout = Layout::default();
        let (params, decoder_bias) = match architecture {
            Architecture::Decoder {
                final_norm,
                logits_bias,
            } => {
                if embeddings.segments.is_some()
                    || logits_bias.len() * shape.width != embeddings.tokens.len()
                    || logits_bias.iter().any(|&v| v != 0)
                {
                    return Err("invalid decoder logits");
                }
                (final_norm, Some(logits_bias))
            }
            Architecture::Encoder { embedding_norm } => (embedding_norm, None),
        };
        let digest = model_digest(
            shape,
            bits,
            &embeddings,
            &layers,
            params,
            decoder_bias.is_some(),
            qk,
        );
        let norm = Norm::new(&mut layout, shape, bits, params)?;
        let mut blocks = Vec::with_capacity(layers.len());
        for (index, p) in layers.into_iter().enumerate() {
            let norms = [
                Norm::new(&mut layout, shape, bits, p.attention_norm)?,
                Norm::new(&mut layout, shape, bits, p.mlp_norm)?,
            ];
            let attention = AttentionPlan::new(
                &mut layout,
                shape,
                bits,
                p.attention,
                qk.get(index).copied().flatten(),
            )?;
            let mlp = FeedForward::new(&mut layout, shape, bits, p.mlp)?;
            let residuals = std::array::from_fn(|_| Ranges::new(&mut layout, shape, [I32]));
            blocks.push(Block {
                attention,
                mlp,
                norms,
                residuals,
            });
        }
        Ok(Self {
            shape,
            bits,
            embeddings,
            layers: blocks,
            norm,
            decoder_bias,
            layout,
            digest,
        })
    }

    fn input(&self, tokens: &[usize], segments: Option<&[usize]>) -> Result<Vec<i32>> {
        let width = self.shape.width;
        if tokens.len() != self.shape.rows
            || segments
                .is_some_and(|s| s.len() != tokens.len() || self.embeddings.segments.is_none())
        {
            return Err("invalid sequence dimensions");
        }
        let mut input = Vec::with_capacity(self.shape.len());
        for (row, &token) in tokens.iter().enumerate() {
            let tables = [
                (Some(self.embeddings.tokens), token),
                (self.embeddings.positions, row),
                (self.embeddings.segments, segments.map_or(0, |s| s[row])),
            ];
            let mut values = vec![0i64; width];
            for (table, index) in tables {
                if let Some(table) = table {
                    let slice = table
                        .chunks_exact(width)
                        .nth(index)
                        .ok_or("invalid token or embedding index")?;
                    for (sum, &x) in values.iter_mut().zip(slice) {
                        *sum += i64::from(x);
                    }
                }
            }
            for value in values {
                input.push(i32::try_from(value).map_err(|_| "embedding overflow")?);
            }
        }
        Ok(input)
    }

    fn execute(&self, input: &[i32], packed: &mut Packed<'_>) -> Result<Run> {
        let pre = self.decoder_bias.is_some();
        let mut norms = if pre {
            None
        } else {
            Some(self.norm.execute(input, packed)?)
        };
        let mut hidden = norms.as_ref().map_or(input, |n| &n.values);
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            layers.push(layer.execute(hidden, pre, packed)?);
            hidden = layers.last().unwrap().output(pre);
        }
        if pre {
            norms = Some(self.norm.execute(hidden, packed)?);
        }
        Ok(Run {
            norms: norms.expect("model normalization"),
            layers,
        })
    }

    fn reverse<'w>(
        &self,
        io: &mut Session<'_, '_>,
        input: &[i32],
        mut claims: Vec<Claim>,
        trace: impl Fn() -> &'w Run + Sync,
        mut requests: Vec<Request>,
    ) -> Result<Vec<Request>> {
        let pre = self.decoder_bias.is_some();
        let mut lookups = Lookups::default();
        if pre {
            claims = vec![self.norm.reverse(io, &mut requests, &claims, || {
                let run = trace();
                (&run.norms, run.layers.last().unwrap().output(true))
            })?];
        }
        for (i, block) in self.layers.iter().enumerate().rev() {
            claims = block.reverse(io, &mut requests, &mut lookups, &claims, pre, || {
                let run = trace();
                let input = if i > 0 {
                    run.layers[i - 1].output(pre)
                } else if pre {
                    input
                } else {
                    &run.norms.values
                };
                (&run.layers[i], input)
            })?;
        }
        if !pre {
            claims = vec![self
                .norm
                .reverse(io, &mut requests, &claims, || (&trace().norms, input))?];
        }
        if claims
            .iter()
            .any(|claim| self.shape.evaluate(input, &claim.point) != claim.value)
        {
            return Err("invalid input embedding evaluation");
        }
        lookups.finish(io, &mut requests)?;
        Ok(requests)
    }

    fn bind(
        &self,
        t: &mut impl Transcript,
        tokens: &[usize],
        segments: Option<&[usize]>,
        output: Output<'_>,
    ) {
        t.append_bytes(&self.digest);
        for &x in tokens {
            t.append(&(x as u64));
        }
        t.append(&u64::from(segments.is_some()));
        if let Some(segments) = segments {
            for &x in segments {
                t.append(&(x as u64));
            }
        }
        output.bind(t);
    }

    fn logits(&self, layout: &mut Layout) -> Result<Option<Linear<'a>>> {
        self.decoder_bias
            .map(|bias| {
                Linear::new(
                    layout,
                    self.shape,
                    Projection {
                        weights: self.embeddings.tokens,
                        bias,
                    },
                    self.bits,
                )
            })
            .transpose()
    }

    fn reverse_logits<'w>(
        &self,
        io: &mut Session<'_, '_>,
        input: &[i32],
        y: &[i32],
        logits: Option<&Linear<'_>>,
        run: impl Fn() -> &'w Run + Sync,
        values: impl Fn() -> &'w Rounded,
    ) -> Result<Vec<Request>> {
        let shape = logits.map_or(self.shape, Linear::shape);
        let point = io.random(shape.bits(), shape.bits())?;
        let claim = Claim::new(point.clone(), shape.evaluate(y, &point));
        let mut requests = Vec::new();
        let claim = if let Some(logits) = logits {
            logits.reverse(io, &mut requests, claim, || (values(), &run().norms.values))?
        } else {
            claim
        };
        self.reverse(io, input, vec![claim], run, requests)
    }

    pub fn prove(&self, tokens: &[usize], segments: Option<&[usize]>) -> Result<(Vec<i32>, Proof)> {
        let input = self.input(tokens, segments)?;
        let mut layout = self.layout.clone();
        let logits = self.logits(&mut layout)?;
        let mut packed = layout.packed();
        let run = self.execute(&input, &mut packed)?;
        let score = logits
            .as_ref()
            .map(|l| l.execute(&run.norms.values, &mut packed))
            .transpose()?;
        let y = score
            .as_ref()
            .map_or_else(|| run.layers.last().unwrap().output(false), |v| &v.values)
            .to_vec();
        let values = &y;
        let output = Output::Logits(values);
        let proof = layout.finish(&packed.widths)?.prove(
            &output.domain(),
            packed,
            |t| self.bind(t, tokens, segments, output),
            move |io| {
                self.reverse_logits(
                    io,
                    &input,
                    values,
                    logits.as_ref(),
                    || &run,
                    || score.as_ref().unwrap(),
                )
            },
        )?;
        Ok((y, proof))
    }

    pub fn verify(
        &self,
        tokens: &[usize],
        segments: Option<&[usize]>,
        y: &[i32],
        proof: &Proof,
    ) -> Result<()> {
        let input = self.input(tokens, segments)?;
        let mut layout = self.layout.clone();
        let logits = self.logits(&mut layout)?;
        if y.len() != logits.as_ref().map_or(self.shape, Linear::shape).len() {
            return Err("invalid output dimensions");
        }
        let output = Output::Logits(y);
        layout.finish(&proof.widths)?.verify(
            &output.domain(),
            proof,
            |t| self.bind(t, tokens, segments, output),
            |io| {
                self.reverse_logits(
                    io,
                    &input,
                    y,
                    logits.as_ref(),
                    || unreachable!(),
                    || unreachable!(),
                )
            },
        )
    }

    pub fn cls<'b>(&self, output: &'b [i32]) -> Result<&'b [i32]> {
        if output.len() != self.shape.len() {
            return Err("invalid encoder output");
        }
        Ok(&output[..self.shape.width])
    }

    fn ranking<'cfg>(
        &'cfg self,
        layout: &mut Layout,
        config: &'cfg TopKConfig,
    ) -> Result<Ranking<'cfg>> {
        if self.decoder_bias.is_none() {
            return Err("Top-K requires a decoder");
        }
        Ranking::new(
            layout,
            self.shape,
            self.embeddings.tokens,
            self.bits,
            config,
            &self.norm.bounds(),
        )
    }

    pub fn prove_top_k(
        &self,
        tokens: &[usize],
        config: &TopKConfig,
    ) -> Result<(Vec<Vec<Candidate>>, Proof)> {
        let input = self.input(tokens, None)?;
        let mut layout = self.layout.clone();
        let ranking = self.ranking(&mut layout, config)?;
        let mut packed = layout.packed();
        let run = self.execute(&input, &mut packed)?;
        let candidates = ranking.select(&run.norms.values, &mut packed)?;
        let selected = &candidates;
        let output = Output::Ranked(config, selected);
        let proof = layout.finish(&packed.widths)?.prove(
            &output.domain(),
            packed,
            |t| self.bind(t, tokens, None, output),
            move |io| {
                let mut requests = Vec::new();
                let claims = ranking.reverse(io, &mut requests, selected, || &run.norms.values)?;
                self.reverse(io, &input, claims, || &run, requests)
            },
        )?;
        Ok((candidates, proof))
    }

    pub fn verify_top_k(
        &self,
        tokens: &[usize],
        config: &TopKConfig,
        candidates: &[Vec<Candidate>],
        proof: &Proof,
    ) -> Result<()> {
        let input = self.input(tokens, None)?;
        let mut layout = self.layout.clone();
        let ranking = self.ranking(&mut layout, config)?;
        ranking.validate(candidates)?;
        let output = Output::Ranked(config, candidates);
        layout.finish(&proof.widths)?.verify(
            &output.domain(),
            proof,
            |t| self.bind(t, tokens, None, output),
            |io| {
                let mut requests = Vec::new();
                let claims = ranking.reverse(io, &mut requests, candidates, || unreachable!())?;
                self.reverse(io, &input, claims, || unreachable!(), requests)
            },
        )
    }
}

fn bind_i32(t: &mut impl Transcript, values: &[i32]) {
    t.append(&(values.len() as u64));
    let mut buffer = [0; 16384];
    for block in values.chunks(buffer.len() / 4) {
        for (&value, bytes) in block.iter().zip(buffer.chunks_exact_mut(4)) {
            bytes.copy_from_slice(&value.to_le_bytes());
        }
        t.append_bytes(&buffer[..4 * block.len()]);
    }
}

fn model_digest(
    shape: Shape,
    bits: usize,
    embeddings: &PublicEmbeddings<'_>,
    layers: &[Layer<'_>],
    norm: NormalizationParameters<'_>,
    decoder: bool,
    qk: &[Option<[NormalizationParameters<'_>; 2]>],
) -> [u8; 32] {
    let mut t = prover_transcript::<Keccak>(&protocol_tag(b"atlas/transformer/parameters/v1"), b"");
    t.append(&[
        shape.rows as u64,
        shape.width as u64,
        bits as u64,
        layers.len() as u64,
        u64::from(decoder),
    ]);
    for table in [
        Some(embeddings.tokens),
        embeddings.positions,
        embeddings.segments,
    ] {
        t.append(&u64::from(table.is_some()));
        if let Some(values) = table {
            bind_i32(&mut t, values);
        }
    }
    let normalization = |t: &mut jolt_transcript::ProverState<Keccak>,
                         n: NormalizationParameters<'_>| {
        t.append(&[n.epsilon_variance_units as u64, n.kind as u64]);
        bind_i32(t, n.gain);
        bind_i32(t, n.bias);
    };
    for (i, layer) in layers.iter().enumerate() {
        let attention = &layer.attention;
        t.append(&[
            attention.queries.len() as u64,
            attention.groups.len() as u64,
            attention.factor as u64,
            attention.mask as u64,
        ]);
        let projections = attention
            .queries
            .iter()
            .chain(attention.groups.iter().flatten())
            .copied()
            .chain([attention.output, layer.mlp.up, layer.mlp.down])
            .chain(layer.mlp.gate);
        t.append(&u64::from(layer.mlp.gate.is_some()));
        for p in projections {
            bind_i32(&mut t, p.weights);
            bind_i32(&mut t, p.bias);
        }
        t.append(&u64::from(attention.rotary.is_some()));
        if let Some(rope) = attention.rotary {
            bind_i32(&mut t, rope.cos);
            bind_i32(&mut t, rope.sin);
        }
        t.append(&(attention.tables.limit as u64));
        bind_i32(&mut t, attention.tables.sums);
        bind_i32(&mut t, attention.tables.probabilities);
        t.append(&(layer.mlp.activation.lower as u64));
        bind_i32(&mut t, layer.mlp.activation.correction);
        normalization(&mut t, layer.attention_norm);
        normalization(&mut t, layer.mlp_norm);
        let heads = qk.get(i).copied().flatten();
        t.append(&u64::from(heads.is_some()));
        if let Some(heads) = heads {
            for n in heads {
                normalization(&mut t, n);
            }
        }
    }
    normalization(&mut t, norm);
    t.verifier_message_as(32, |bytes| bytes.try_into().expect("32 digest bytes"))
}
