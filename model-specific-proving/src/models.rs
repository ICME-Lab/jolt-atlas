use bincode::Options;
use std::{
    borrow::Cow,
    io::{Read, Seek, SeekFrom, Write},
};

use crate::{
    advice::{Advice, Requests, Slot},
    attention,
    channel::{Channel, Messages, Result},
    field::F,
    layers::{Attention, Mlp, NormalizationParameters},
    linear,
    lookup::Lookups,
    mlp::Feed,
    norm::{Claim, Width, Witness},
    projection::Projection,
    protocol::{Candidate, Header, Proof, TopKConfig},
    residual,
    rounding::Rounded,
    tensor::Shape,
    top_k::Top,
};
use jolt_transcript::{prover_transcript, verifier_transcript, Keccak, Transcript};

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

pub struct Model<'a> {
    shape: Shape,
    bits: u32,
    embeddings: PublicEmbeddings<'a>,
    layers: Vec<Layer<'a>>,
    qk_norms: Vec<Option<[NormalizationParameters<'a>; 2]>>,
    norm: NormalizationParameters<'a>,
    decoder: Option<&'a [i32]>,
    digest: [u8; 32],
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Normalized {
    trace: Option<Witness>,
    slots: Vec<Slot>,
}

impl Normalized {
    fn forward(
        advice: &mut Advice,
        shape: Shape,
        bits: u32,
        params: NormalizationParameters<'_>,
        input: Option<&[i32]>,
    ) -> Result<Self> {
        if params.gain.len() != shape.cols {
            return Err("invalid layer normalization width");
        }
        let plan = params.at(shape.rows, bits)?;
        let trace = input.map(|v| plan.witness(v)).transpose()?;
        let slots = plan.record(advice, input, trace.as_ref())?;
        Ok(Self { trace, slots })
    }
    fn values(&self) -> Option<&[i32]> {
        self.trace.as_ref().map(|w| w.result.as_slice())
    }
    fn reverse(
        &self,
        io: &mut Channel<'_, '_>,
        requests: &mut Requests,
        shape: Shape,
        bits: u32,
        params: NormalizationParameters<'_>,
        input: Option<&[i32]>,
        output: &[Claim<F>],
        carry: &[Claim<F>],
    ) -> Result<Claim<F>> {
        params.at(shape.rows, bits)?.reduce_model(
            io,
            input,
            self.trace.as_ref(),
            output,
            carry,
            &self.slots,
            requests,
        )
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Sum {
    values: Option<Vec<i32>>,
    range: Slot,
}

impl Sum {
    fn forward(advice: &mut Advice, shape: Shape, x: Option<&[i32]>, f: &Rounded) -> Result<Self> {
        if f.shape != shape {
            return Err("invalid residual dimensions");
        }
        let values = x
            .zip(f.values.as_deref())
            .map(|(x, f)| residual::add(x, f))
            .transpose()?;
        let range = advice.column(shape, Width::Signed(32), |i| {
            i128::from(values.as_ref().unwrap()[i])
        })?;
        Ok(Self { values, range })
    }

    fn split(
        &self,
        io: &mut Channel<'_, '_>,
        requests: &mut Requests,
        shape: Shape,
        claim: Claim<F>,
        x: Option<&[i32]>,
    ) -> Result<(Claim<F>, Claim<F>)> {
        requests.push(self.range, &claim.point, claim.value);
        let [first] = io.values(|| [shape.eval(x.unwrap(), &claim.point)])?;
        let a = Claim {
            point: claim.point.clone(),
            value: first,
        };
        let b = Claim {
            point: claim.point,
            value: claim.value - first,
        };
        Ok((a, b))
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Block {
    a: Normalized,
    attention: attention::Trace,
    after_attention: Sum,
    m: Normalized,
    mlp: Feed,
    after_mlp: Sum,
}

impl Block {
    fn output(&self, decoder: bool) -> Option<&[i32]> {
        if decoder {
            self.after_mlp.values.as_deref()
        } else {
            self.m.values()
        }
    }
}

struct Run {
    blocks: Vec<Saved>,
    trace: Option<std::cell::RefCell<std::fs::File>>,
    norm: Normalized,
}

enum Saved {
    InMemory(Box<Block>),
    OnDisk { offset: u64, output: Vec<i32> },
}

impl Saved {
    fn output(&self, decoder: bool) -> Option<&[i32]> {
        match self {
            Self::InMemory(block) => block.output(decoder),
            Self::OnDisk { output, .. } => Some(output),
        }
    }
}

impl Run {
    fn output(&self, decoder: bool) -> Option<&[i32]> {
        if decoder {
            self.norm.values()
        } else {
            self.blocks.last().unwrap().output(false)
        }
    }

    fn block(&self, index: usize) -> Result<BlockRef<'_>> {
        match &self.blocks[index] {
            Saved::InMemory(b) => Ok(BlockRef::Borrowed(b)),
            Saved::OnDisk { offset, .. } => {
                let mut file = self.trace.as_ref().unwrap().borrow_mut();
                file.seek(SeekFrom::Start(*offset))
                    .map_err(|_| "could not locate layer trace")?;
                codec()
                    .deserialize_from(std::io::BufReader::new(&mut *file))
                    .map(BlockRef::Owned)
                    .map_err(|_| "could not read layer trace")
            }
        }
    }
}
enum BlockRef<'a> {
    Borrowed(&'a Block),
    Owned(Block),
}

impl std::ops::Deref for BlockRef<'_> {
    type Target = Block;
    fn deref(&self) -> &Block {
        match self {
            Self::Borrowed(block) => block,
            Self::Owned(block) => block,
        }
    }
}

enum Output<'a> {
    All(&'a [i32]),
    Top(&'a TopKConfig, &'a [Vec<Candidate>]),
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
        qk_norms: &[Option<[NormalizationParameters<'a>; 2]>],
    ) -> Result<Self> {
        let shape = Shape::new(rows, embeddings.width)?;
        if bits > 15
            || layers.is_empty()
            || embeddings.tokens.is_empty()
            || embeddings.tokens.len() % shape.cols != 0
            || qk_norms.len() > layers.len()
        {
            return Err("invalid model dimensions");
        }
        let (norm, decoder) = match architecture {
            Architecture::Decoder {
                final_norm,
                logits_bias,
            } => {
                if embeddings.segments.is_some()
                    || logits_bias.len() != embeddings.tokens.len() / shape.cols
                    || logits_bias.iter().any(|&v| v != 0)
                {
                    return Err("invalid decoder logits");
                }
                (final_norm, Some(logits_bias))
            }
            Architecture::Encoder { embedding_norm } => (embedding_norm, None),
        };
        let mut model = Self {
            shape,
            bits: bits.into(),
            embeddings,
            layers,
            qk_norms: qk_norms.to_vec(),
            norm,
            decoder,
            digest: [0; 32],
        };
        let tag = jolt_transcript::protocol_tag(b"atlas/transformer/model/v2");
        let mut transcript = prover_transcript::<Keccak>(&tag, b"");
        model.bind_model(&mut transcript);
        model.digest = transcript.verifier_message_as(32, |bytes| bytes.try_into().unwrap());
        Ok(model)
    }

    fn context(&self, i: usize) -> attention::Context<'_> {
        attention::Context {
            settings: &self.layers[i].attention,
            shape: self.shape,
            bits: self.bits,
            head_norms: self.qk_norms.get(i).copied().flatten(),
        }
    }

    fn input(&self, tokens: &[usize], segments: Option<&[usize]>) -> Result<Vec<i32>> {
        let width = self.shape.cols;
        if tokens.len() != self.shape.rows
            || segments
                .is_some_and(|s| s.len() != tokens.len() || self.embeddings.segments.is_none())
        {
            return Err("invalid sequence");
        }
        let mut output = Vec::with_capacity(tokens.len() * width);
        for (r, &token) in tokens.iter().enumerate() {
            let lookup = |table: Option<&'a [i32]>, index| {
                table
                    .map(|v| {
                        v.chunks_exact(width)
                            .nth(index)
                            .ok_or("invalid embedding index")
                    })
                    .transpose()
            };
            let token = lookup(Some(self.embeddings.tokens), token)?.unwrap();
            let position = lookup(self.embeddings.positions, r)?;
            let segment = lookup(self.embeddings.segments, segments.map_or(0, |s| s[r]))?;
            for c in 0..width {
                let y = i64::from(token[c])
                    + position.map_or(0, |x| i64::from(x[c]))
                    + segment.map_or(0, |x| i64::from(x[c]));
                output.push(i32::try_from(y).map_err(|_| "embedding exceeds i32")?);
            }
        }
        Ok(output)
    }

    fn forward(&self, advice: &mut Advice, input: Option<&[i32]>) -> Result<Run> {
        let (shape, bits) = (self.shape, self.bits);
        let pre = self.decoder.is_some();
        let mut norm = if pre {
            None
        } else {
            Some(Normalized::forward(advice, shape, bits, self.norm, input)?)
        };
        let mut x = norm.as_ref().map_or(input, Normalized::values);
        let mut blocks: Vec<Saved> = Vec::with_capacity(self.layers.len());
        let mut trace = if input.is_some() && self.shape.rows >= 128 {
            Some(tempfile::tempfile().map_err(|_| "could not create trace file")?)
        } else {
            None
        };
        for (i, p) in self.layers.iter().enumerate() {
            let ctx = self.context(i);
            let mut a = if pre {
                Some(Normalized::forward(
                    advice,
                    shape,
                    bits,
                    p.attention_norm,
                    x,
                )?)
            } else {
                None
            };
            let attention = ctx.execute(advice, a.as_ref().map_or(x, Normalized::values))?;
            let after_attention = Sum::forward(advice, shape, x, &attention.output)?;
            if !pre {
                a = Some(Normalized::forward(
                    advice,
                    shape,
                    bits,
                    p.attention_norm,
                    after_attention.values.as_deref(),
                )?);
            }
            let a = a.unwrap();
            let h = if pre {
                after_attention.values.as_deref()
            } else {
                a.values()
            };
            let mut m = if pre {
                Some(Normalized::forward(advice, shape, bits, p.mlp_norm, h)?)
            } else {
                None
            };
            let mlp = Feed::prepare(
                advice,
                p.mlp,
                shape,
                bits,
                m.as_ref().map_or(h, Normalized::values),
            )?;
            let after_mlp = Sum::forward(advice, shape, h, &mlp.output)?;
            if !pre {
                m = Some(Normalized::forward(
                    advice,
                    shape,
                    bits,
                    p.mlp_norm,
                    after_mlp.values.as_deref(),
                )?);
            }
            let block = Block {
                a,
                attention,
                after_attention,
                m: m.unwrap(),
                mlp,
                after_mlp,
            };
            let saved = if let Some(file) = &mut trace {
                let offset = file
                    .stream_position()
                    .map_err(|_| "could not locate layer trace")?;
                let mut writer = std::io::BufWriter::new(file);
                codec()
                    .serialize_into(&mut writer, &block)
                    .map_err(|_| "could not store layer trace")?;
                writer.flush().map_err(|_| "could not store layer trace")?;
                Saved::OnDisk {
                    offset,
                    output: block.output(pre).unwrap().to_vec(),
                }
            } else {
                Saved::InMemory(Box::new(block))
            };
            blocks.push(saved);
            x = blocks.last().unwrap().output(pre);
        }
        if pre {
            norm = Some(Normalized::forward(advice, shape, bits, self.norm, x)?);
        }
        Ok(Run {
            blocks,
            trace: trace.map(std::cell::RefCell::new),
            norm: norm.unwrap(),
        })
    }

    fn reverse(
        &self,
        io: &mut Channel<'_, '_>,
        requests: &mut Requests,
        input: &[i32],
        run: &Run,
        mut claims: Vec<Claim<F>>,
    ) -> Result<()> {
        let pre = self.decoder.is_some();
        let (shape, bits) = (self.shape, self.bits);
        let mut lookups = Lookups::default();
        if pre {
            claims = vec![run.norm.reverse(
                io,
                requests,
                shape,
                bits,
                self.norm,
                run.blocks.last().unwrap().output(true),
                &claims,
                &[],
            )?];
        }
        for i in (0..run.blocks.len()).rev() {
            let b = run.block(i)?;
            let p = &self.layers[i];
            let x = if i > 0 {
                run.blocks[i - 1].output(pre)
            } else if pre {
                Some(input)
            } else {
                run.norm.values()
            };
            let h = if pre {
                b.after_attention.values.as_deref()
            } else {
                b.a.values()
            };
            let output = if pre {
                claims.pop().unwrap()
            } else {
                b.m.reverse(
                    io,
                    requests,
                    shape,
                    bits,
                    p.mlp_norm,
                    b.after_mlp.values.as_deref(),
                    &claims,
                    &[],
                )?
            };
            let (skip, mlp) = b.after_mlp.split(io, requests, shape, output, h)?;
            let mut mlp = b.mlp.reverse(
                io,
                requests,
                &mut lookups,
                p.mlp,
                shape,
                bits,
                mlp,
                if pre { b.m.values() } else { h },
            )?;
            let mid = if pre {
                b.m.reverse(io, requests, shape, bits, p.mlp_norm, h, &mlp, &[skip])?
            } else {
                mlp.push(skip);
                b.a.reverse(
                    io,
                    requests,
                    shape,
                    bits,
                    p.attention_norm,
                    b.after_attention.values.as_deref(),
                    &mlp,
                    &[],
                )?
            };
            let (skip, att) = b.after_attention.split(io, requests, shape, mid, x)?;
            let mut from_att = self.context(i).reverse(
                io,
                requests,
                &mut lookups,
                &b.attention,
                att,
                if pre { b.a.values() } else { x },
            )?;
            claims = if pre {
                vec![b.a.reverse(
                    io,
                    requests,
                    shape,
                    bits,
                    p.attention_norm,
                    x,
                    &from_att,
                    &[skip],
                )?]
            } else {
                from_att.push(skip);
                from_att
            };
        }
        if !pre {
            claims = vec![run.norm.reverse(
                io,
                requests,
                shape,
                bits,
                self.norm,
                Some(input),
                &claims,
                &[],
            )?];
        }
        if claims
            .iter()
            .any(|q| self.shape.eval(input, &q.point) != q.value)
        {
            return Err("invalid public embedding evaluation");
        }
        tracing::info_span!("atlas lookup").in_scope(|| lookups.finish(io, requests))
    }

    fn bind_model(&self, t: &mut impl Transcript) {
        fn norm(t: &mut impl Transcript, p: NormalizationParameters<'_>) {
            t.append(&[p.kind as u64, p.epsilon_variance_units as u64]);
            integers(t, p.gain);
            integers(t, p.bias);
        }
        fn affine(t: &mut impl Transcript, p: Projection<'_>) {
            integers(t, p.weights);
            integers(t, p.bias);
        }
        t.append(&[
            self.shape.rows as u64,
            self.shape.cols as u64,
            self.bits as u64,
            self.decoder.is_some() as u64,
            self.layers.len() as u64,
        ]);
        integers(t, self.embeddings.tokens);
        integers(t, self.embeddings.positions.unwrap_or_default());
        integers(t, self.embeddings.segments.unwrap_or_default());
        norm(t, self.norm);
        for (i, layer) in self.layers.iter().enumerate() {
            norm(t, layer.attention_norm);
            norm(t, layer.mlp_norm);
            let a = &layer.attention;
            t.append(&[
                a.mask as u64,
                a.factor as u64,
                a.tables.limit as u64,
                a.queries.len() as u64,
                a.groups.len() as u64,
            ]);
            for &q in &a.queries {
                affine(t, q);
            }
            for &[k, v] in &a.groups {
                affine(t, k);
                affine(t, v);
            }
            affine(t, a.output);
            integers(t, a.tables.probabilities);
            integers(t, a.tables.sums);
            if let Some(r) = a.rotary {
                t.append(&1u64);
                integers(t, r.cos);
                integers(t, r.sin);
            } else {
                t.append(&0u64);
            }
            if let Some(qk) = self.qk_norms.get(i).copied().flatten() {
                t.append(&1u64);
                norm(t, qk[0]);
                norm(t, qk[1]);
            } else {
                t.append(&0u64);
            }
            let p = layer.mlp;
            affine(t, p.up);
            affine(t, p.down);
            if let Some(gate) = p.gate {
                t.append(&1u64);
                affine(t, gate);
            } else {
                t.append(&0u64);
            }
            t.append(&(p.activation.lower as u64));
            integers(t, p.activation.correction);
        }
    }

    fn bind(
        &self,
        t: &mut impl Transcript,
        tokens: &[usize],
        segments: Option<&[usize]>,
        expected: Output<'_>,
    ) {
        t.append_bytes(&self.digest);
        for &token in tokens {
            t.append(&(token as u64));
        }
        t.append(&(segments.is_some() as u64));
        if let Some(segments) = segments {
            for &s in segments {
                t.append(&(s as u64));
            }
        }
        match expected {
            Output::All(y) => {
                t.append(&0u64);
                integers(t, y);
            }
            Output::Top(config, y) => {
                t.append(&[
                    1,
                    config.positions.start as u64,
                    config.positions.end as u64,
                    config.k as u64,
                    config.rows_per_block as u64,
                    config.vocab_per_block as u64,
                ]);
                for value in y.iter().flatten() {
                    t.append(&[value.id as u64, value.logit as u64]);
                }
            }
        }
    }

    fn logits(&self, advice: &mut Advice, value: Option<&[i32]>) -> Result<Option<Rounded>> {
        self.decoder
            .map(|bias| {
                linear::project(
                    advice,
                    self.shape,
                    self.bits,
                    Projection {
                        weights: self.embeddings.tokens,
                        bias,
                    },
                    value,
                )
            })
            .transpose()
    }

    fn output_claims(
        &self,
        io: &mut Channel<'_, '_>,
        requests: &mut Requests,
        logits: Option<&Rounded>,
        output: &[i32],
        hidden: Option<&[i32]>,
    ) -> Result<Vec<Claim<F>>> {
        let shape = logits.map_or(self.shape, |m| m.shape);
        if output.len() != shape.rows * shape.cols {
            return Err("invalid model output dimensions");
        }
        let point = io.sample(shape.bits(), shape.bits())?;
        let claim = Claim {
            value: shape.eval(output, &point),
            point,
        };
        if let Some((l, bias)) = logits.zip(self.decoder) {
            linear::reverse(
                io,
                requests,
                self.shape,
                self.bits,
                &[(
                    l,
                    Projection {
                        weights: self.embeddings.tokens,
                        bias,
                    },
                    claim,
                )],
                hidden,
            )
        } else {
            Ok(vec![claim])
        }
    }

    pub fn cls<'b>(&self, output: &'b [i32]) -> Result<&'b [i32]> {
        if self.decoder.is_some() || output.len() != self.shape.rows * self.shape.cols {
            return Err("invalid encoder output");
        }
        Ok(&output[..self.shape.cols])
    }

    pub fn prove(&self, tokens: &[usize], segments: Option<&[usize]>) -> Result<(Vec<i32>, Proof)> {
        let input = self.input(tokens, segments)?;
        let mut advice = Advice::new(true)?;
        let run = tracing::info_span!("atlas forward")
            .in_scope(|| self.forward(&mut advice, Some(&input)))?;
        let hidden = run.output(self.decoder.is_some());
        let logits = self.logits(&mut advice, hidden)?;
        let output = logits
            .as_ref()
            .and_then(|x| x.values.as_deref())
            .unwrap_or(hidden.unwrap())
            .to_vec();
        let mut evaluations = Vec::new();
        let header = self.prove_with(
            &mut advice,
            tokens,
            segments,
            Output::All(&output),
            |one| {
                evaluations.push(one);
                Ok(())
            },
            |io, requests| {
                let claims = self.output_claims(io, requests, logits.as_ref(), &output, hidden)?;
                self.reverse(io, requests, &input, &run, claims)
            },
        )?;
        Ok((
            output,
            Proof {
                header,
                evaluations,
            },
        ))
    }

    pub fn verify(
        &self,
        tokens: &[usize],
        segments: Option<&[usize]>,
        output: &[i32],
        proof: &Proof,
    ) -> Result<()> {
        if proof.evaluations.len() != proof.header.roots.len() {
            return Err("wrong number of BitZ proofs");
        }
        let input = self.input(tokens, segments)?;
        let mut advice = Advice::new(false)?;
        let run = self.forward(&mut advice, None)?;
        let logits = self.logits(&mut advice, None)?;
        self.verify_with(
            &mut advice,
            tokens,
            segments,
            Output::All(output),
            &proof.header,
            |i| {
                proof
                    .evaluations
                    .get(i)
                    .map(Cow::Borrowed)
                    .ok_or("missing BitZ evaluation")
            },
            |io, requests| {
                let claims = self.output_claims(io, requests, logits.as_ref(), output, None)?;
                self.reverse(io, requests, &input, &run, claims)
            },
        )
    }

    pub fn prove_top_k(
        &self,
        tokens: &[usize],
        cfg: &TopKConfig,
    ) -> Result<(Vec<Vec<Candidate>>, Proof)> {
        let mut evaluations = Vec::new();
        let (output, header) = self.prove_top_k_with(tokens, cfg, |one| {
            evaluations.push(one);
            Ok(())
        })?;
        Ok((
            output,
            Proof {
                header,
                evaluations,
            },
        ))
    }

    fn prove_top_k_with(
        &self,
        tokens: &[usize],
        cfg: &TopKConfig,
        emit: impl FnMut(crate::bitz::Proof) -> Result<()>,
    ) -> Result<(Vec<Vec<Candidate>>, Header)> {
        if self.decoder.is_none() {
            return Err("Top-K requires decoder");
        }
        let top = Top::new(cfg, self.embeddings.tokens, self.shape, self.bits)?;
        let input = self.input(tokens, None)?;
        let mut advice = Advice::new(true)?;
        let run = tracing::info_span!("atlas forward")
            .in_scope(|| self.forward(&mut advice, Some(&input)))?;
        let hidden = run.output(true);
        let (results, raw) = top.select(hidden.unwrap())?;
        let rank = top.record(&mut advice, &results, Some(&raw))?;
        drop(raw);
        let proof = self.prove_with(
            &mut advice,
            tokens,
            None,
            Output::Top(cfg, &results),
            emit,
            |io, requests| {
                let claims = top.reverse(io, requests, &rank, &results, hidden)?;
                self.reverse(io, requests, &input, &run, claims)
            },
        )?;
        Ok((results, proof))
    }

    pub fn verify_top_k(
        &self,
        tokens: &[usize],
        cfg: &TopKConfig,
        results: &[Vec<Candidate>],
        proof: &Proof,
    ) -> Result<()> {
        if proof.evaluations.len() != proof.header.roots.len() {
            return Err("wrong number of BitZ proofs");
        }
        self.verify_top_k_with(tokens, cfg, results, &proof.header, |i| {
            proof
                .evaluations
                .get(i)
                .map(Cow::Borrowed)
                .ok_or("missing BitZ evaluation")
        })
    }

    fn verify_top_k_with<'p>(
        &self,
        tokens: &[usize],
        cfg: &TopKConfig,
        results: &[Vec<Candidate>],
        proof: &'p Header,
        read: impl FnMut(usize) -> Result<Cow<'p, crate::bitz::Proof>>,
    ) -> Result<()> {
        if self.decoder.is_none() {
            return Err("Top-K requires decoder");
        }
        let top = Top::new(cfg, self.embeddings.tokens, self.shape, self.bits)?;
        let input = self.input(tokens, None)?;
        let mut advice = Advice::new(false)?;
        let run = self.forward(&mut advice, None)?;
        let rank = top.record(&mut advice, results, None)?;
        self.verify_with(
            &mut advice,
            tokens,
            None,
            Output::Top(cfg, results),
            proof,
            read,
            |io, requests| {
                let claims = top.reverse(io, requests, &rank, results, None)?;
                self.reverse(io, requests, &input, &run, claims)
            },
        )
    }

    fn prove_with(
        &self,
        advice: &mut Advice,
        tokens: &[usize],
        segments: Option<&[usize]>,
        expected: Output<'_>,
        emit: impl FnMut(crate::bitz::Proof) -> Result<()>,
        work: impl FnOnce(&mut Channel<'_, '_>, &mut Requests) -> Result<()>,
    ) -> Result<Header> {
        let tag = jolt_transcript::protocol_tag(b"atlas/fixed-point-transformer/v2");
        let mut transcript = prover_transcript::<Keccak>(&tag, b"");
        self.bind(&mut transcript, tokens, segments, expected);
        let widths = advice.widths();
        transcript.append_bytes(&widths);
        let plan = advice.buckets()?;
        let roots = tracing::info_span!("atlas commit")
            .in_scope(|| plan.commit(advice, &mut transcript))?;
        let mut state = Messages::default();
        let mut requests = Requests(Vec::new());
        tracing::info_span!("atlas reduce").in_scope(|| {
            work(
                &mut Channel::Send(&mut transcript, &mut state),
                &mut requests,
            )
        })?;
        tracing::info_span!("atlas open")
            .in_scope(|| plan.prove(advice, &roots, requests, &mut transcript, emit))?;
        Ok(Header {
            state,
            widths,
            transcript: transcript.narg_string().to_vec(),
            roots,
        })
    }

    fn verify_with<'p>(
        &self,
        advice: &mut Advice,
        tokens: &[usize],
        segments: Option<&[usize]>,
        expected: Output<'_>,
        proof: &'p Header,
        read: impl FnMut(usize) -> Result<Cow<'p, crate::bitz::Proof>>,
        work: impl FnOnce(&mut Channel<'_, '_>, &mut Requests) -> Result<()>,
    ) -> Result<()> {
        let tag = jolt_transcript::protocol_tag(b"atlas/fixed-point-transformer/v2");
        let mut transcript = verifier_transcript::<Keccak>(&tag, b"", &proof.transcript);
        self.bind(&mut transcript, tokens, segments, expected);
        advice.bind_widths(&proof.widths)?;
        transcript.append_bytes(&proof.widths);
        let plan = advice.buckets()?;
        plan.bind(&proof.roots, &mut transcript)?;
        let mut io = Channel::checking(&mut transcript, &proof.state);
        let mut requests = Requests(Vec::new());
        work(&mut io, &mut requests)?;
        io.finish()?;
        plan.verify(advice, requests, &proof.roots, &mut transcript, read)?;
        transcript.check_eof().map_err(|_| "extra transcript bytes")
    }

    pub fn prove_top_k_to(
        &self,
        tokens: &[usize],
        cfg: &TopKConfig,
        mut out: impl Write + Seek,
    ) -> Result<Vec<Vec<Candidate>>> {
        if out
            .seek(SeekFrom::End(0))
            .map_err(|_| "could not inspect proof file")?
            != 0
        {
            return Err("proof destination is not empty");
        }
        out.write_all(&[0; 16])
            .map_err(|_| "could not initialize proof file")?;
        let (result, header) = self.prove_top_k_with(tokens, cfg, |one| {
            codec()
                .serialize_into(&mut out, &one)
                .map_err(|_| "could not write BitZ proof")
        })?;
        let end = out
            .stream_position()
            .map_err(|_| "could not locate proof metadata")?;
        codec()
            .serialize_into(&mut out, &header)
            .map_err(|_| "could not write proof metadata")?;
        out.rewind()
            .and_then(|_| out.write_all(MAGIC))
            .and_then(|_| out.write_all(&end.to_le_bytes()))
            .and_then(|_| out.flush())
            .map_err(|_| "could not finalize proof")?;
        Ok(result)
    }

    pub fn verify_top_k_from(
        &self,
        tokens: &[usize],
        cfg: &TopKConfig,
        result: &[Vec<Candidate>],
        mut input: impl Read + Seek,
    ) -> Result<()> {
        let size = input
            .seek(SeekFrom::End(0))
            .map_err(|_| "could not inspect proof file")?;
        let mut prefix = [0; 16];
        input
            .rewind()
            .and_then(|_| input.read_exact(&mut prefix))
            .map_err(|_| "truncated proof")?;
        let end = u64::from_le_bytes(prefix[8..].try_into().unwrap());
        if &prefix[..8] != MAGIC || end < 16 || end > size {
            return Err("invalid proof header");
        }
        input
            .seek(SeekFrom::Start(end))
            .map_err(|_| "could not locate proof metadata")?;
        let mut metadata = (&mut input).take(size - end);
        let header = codec()
            .with_limit(size - end)
            .deserialize_from(&mut metadata)
            .map_err(|_| "invalid proof metadata")?;
        if metadata.limit() != 0 {
            return Err("extra proof metadata");
        }
        input
            .seek(SeekFrom::Start(16))
            .map_err(|_| "could not locate BitZ proofs")?;
        let mut groups = input.take(end - 16);
        self.verify_top_k_with(tokens, cfg, result, &header, |_| {
            codec()
                .with_limit(groups.limit())
                .deserialize_from(&mut groups)
                .map(Cow::Owned)
                .map_err(|_| "invalid BitZ proof")
        })?;
        if groups.limit() != 0 {
            return Err("extra BitZ proofs");
        }
        Ok(())
    }
}

fn integers(t: &mut impl Transcript, x: &[i32]) {
    t.append(&(x.len() as u64));
    for chunk in x.chunks(1024) {
        let bytes: Vec<_> = chunk.iter().flat_map(|x| x.to_le_bytes()).collect();
        t.append_bytes(&bytes);
    }
}

const MAGIC: &[u8; 8] = b"ATLASMP2";

fn codec() -> impl Options {
    bincode::DefaultOptions::new().with_fixint_encoding()
}
