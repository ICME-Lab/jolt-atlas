use bincode::Options;
use std::{
    borrow::Cow,
    io::{BufWriter, Read, Seek, SeekFrom, Write},
};

use crate::{
    advice::{Advice, Buckets, Requests, Slot},
    attention,
    channel::{Channel, Messages, Result},
    field::F,
    layers::{Attention, Mlp, NormalizationParameters},
    lookup::Lookups,
    mlp::{gate_fits, Feed},
    norm::{Claim, Normalized, Width},
    projection::Projection,
    protocol::{Candidate, Header, TopKConfig},
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
    bounded_gates: Vec<bool>,
    norm: NormalizationParameters<'a>,
    decoder: bool,
    digest: [u8; 32],
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

    fn take_output(&mut self, decoder: bool) -> Vec<i32> {
        if decoder {
            self.after_mlp.values.take().unwrap()
        } else {
            std::mem::take(&mut self.m.trace.as_mut().unwrap().result)
        }
    }
}

struct Run {
    blocks: Vec<Saved>,
    trace: Option<memmap2::Mmap>,
    norm: Normalized,
}

enum Saved {
    InMemory(Box<Block>),
    OnDisk { offset: u64, body: u64 },
}

impl Run {
    fn output(&self, decoder: bool) -> Option<&[i32]> {
        if decoder {
            self.norm.values()
        } else if let Saved::InMemory(block) = self.blocks.last().unwrap() {
            block.output(false)
        } else {
            unreachable!()
        }
    }

    fn layer_output(&self, index: usize, decoder: bool) -> Result<Option<Cow<'_, [i32]>>> {
        match &self.blocks[index] {
            Saved::InMemory(b) => Ok(b.output(decoder).map(Cow::Borrowed)),
            Saved::OnDisk { offset, .. } => self.read(*offset).map(|v| Some(Cow::Owned(v))),
        }
    }

    fn block(
        &self,
        index: usize,
        prefetched: Option<Block>,
        norm_output: Option<Cow<'_, [i32]>>,
    ) -> Result<BlockRef<'_>> {
        match &self.blocks[index] {
            Saved::InMemory(b) => Ok(BlockRef::Borrowed(b)),
            Saved::OnDisk { body, .. } => {
                let mut block = prefetched.map_or_else(|| self.read(*body), Ok)?;
                if let Some(output) = norm_output {
                    block.m.trace.as_mut().unwrap().result = output.into_owned();
                }
                Ok(BlockRef::Owned(block))
            }
        }
    }
    fn read<T: serde::de::DeserializeOwned>(&self, offset: u64) -> Result<T> {
        codec()
            .deserialize_from(&self.trace.as_ref().unwrap()[offset as usize..])
            .map_err(|_| "could not read layer record")
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

type WindowFlush<'a, 'm> =
    dyn FnMut(&mut Channel<'_, '_>, &mut Requests, usize, Lookups<'m>) -> Result<()> + 'a;

const WINDOW_LAYERS: usize = 4;

fn bitz_window_tag() -> [u8; 32] {
    jolt_transcript::protocol_tag(b"atlas/bitz-window/v1")
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
            Architecture::Decoder { final_norm } if embeddings.segments.is_none() => {
                (final_norm, true)
            }
            Architecture::Decoder { .. } => return Err("decoder does not use segments"),
            Architecture::Encoder { embedding_norm } => (embedding_norm, false),
        };
        let bounded_gates = layers
            .iter()
            .map(|p| {
                decoder
                    && p.mlp
                        .gate
                        .is_some_and(|gate| gate_fits(p.mlp_norm, gate, bits.into()))
            })
            .collect();
        let mut model = Self {
            shape,
            bits: bits.into(),
            embeddings,
            layers,
            qk_norms: qk_norms.to_vec(),
            bounded_gates,
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
        let pre = self.decoder;
        let mut norm = if pre {
            None
        } else {
            Some(Normalized::forward(advice, shape, bits, self.norm, input)?)
        };
        let mut x = norm.as_ref().map_or(input, Normalized::values);
        let mut blocks: Vec<Saved> = Vec::with_capacity(self.layers.len());
        let mut trace = (input.is_some() && self.shape.rows >= 128 && self.layers.len() > 1)
            .then(tempfile::tempfile)
            .transpose()
            .map_err(|_| "could not create trace file")?;
        advice.cut();
        let mut last = None;
        for (i, p) in self.layers.iter().enumerate() {
            let ctx = self.context(i);
            let norm = |advice: &mut Advice, params, input: Option<&[i32]>| {
                Normalized::forward(advice, shape, bits, params, input)
            };
            let a = pre.then(|| norm(advice, p.attention_norm, x)).transpose()?;
            let attention = ctx.execute(advice, a.as_ref().map_or(x, Normalized::values))?;
            let after_attention = Sum::forward(advice, shape, x, &attention.output)?;
            let a = match a {
                Some(a) => a,
                None => norm(advice, p.attention_norm, after_attention.values.as_deref())?,
            };
            let h = if pre {
                after_attention.values.as_deref()
            } else {
                a.values()
            };
            let m = pre.then(|| norm(advice, p.mlp_norm, h)).transpose()?;
            let mlp = Feed::prepare(
                advice,
                p.mlp,
                shape,
                bits,
                m.as_ref().map_or(h, Normalized::values),
                self.bounded_gates[i],
            )?;
            let after_mlp = Sum::forward(advice, shape, h, &mlp.output)?;
            let m = match m {
                Some(m) => m,
                None => norm(advice, p.mlp_norm, after_mlp.values.as_deref())?,
            };
            let mut block = Block {
                a,
                attention,
                after_attention,
                m,
                mlp,
                after_mlp,
            };
            let saved = if let Some(file) = trace.as_mut().filter(|_| i + 1 < self.layers.len()) {
                let _write = tracing::info_span!("atlas trace layer write").entered();
                let offset = file
                    .stream_position()
                    .map_err(|_| "could not locate layer trace")?;
                let output = block.take_output(pre);
                let mut writer = std::io::BufWriter::new(file);
                codec()
                    .serialize_into(&mut writer, &output)
                    .map_err(|_| "could not store layer output")?;
                let body = writer
                    .stream_position()
                    .map_err(|_| "could not locate layer trace")?;
                last = Some(output);
                codec()
                    .serialize_into(&mut writer, &block)
                    .map_err(|_| "could not store layer trace")?;
                writer.flush().map_err(|_| "could not store layer trace")?;
                Saved::OnDisk { offset, body }
            } else {
                Saved::InMemory(Box::new(block))
            };
            advice.cut();
            blocks.push(saved);
            x = match blocks.last().unwrap() {
                Saved::InMemory(block) => block.output(pre),
                Saved::OnDisk { .. } => last.as_deref(),
            };
        }
        if pre {
            norm = Some(Normalized::forward(advice, shape, bits, self.norm, x)?);
        }
        Ok(Run {
            blocks,
            trace: trace
                // The private trace file is complete and will only be read from here.
                .map(|file| unsafe { memmap2::Mmap::map(&file) })
                .transpose()
                .map_err(|_| "could not map layer trace")?,
            norm: norm.unwrap(),
        })
    }

    fn reverse<'m>(
        &'m self,
        io: &mut Channel<'_, '_>,
        requests: &mut Requests,
        input: &[i32],
        run: &Run,
        mut claims: Vec<Claim<F>>,
        flush: &mut WindowFlush<'_, 'm>,
    ) -> Result<()> {
        let pre = self.decoder;
        let (shape, bits) = (self.shape, self.bits);
        let mut lookups = Lookups::default();
        let reduce_end_norm = |io: &mut Channel<'_, '_>,
                               requests: &mut Requests,
                               input: Option<&[i32]>,
                               claims: &[Claim<F>]| {
            run.norm
                .reverse(io, requests, shape, bits, self.norm, input, claims, &[])
        };
        if pre {
            let last = tracing::info_span!("atlas last output")
                .in_scope(|| run.layer_output(run.blocks.len() - 1, true))?;
            claims = vec![tracing::info_span!("atlas final norm")
                .in_scope(|| reduce_end_norm(io, requests, last.as_deref(), &claims))?];
        }
        let mut norm_output = if pre {
            None
        } else {
            run.layer_output(run.blocks.len() - 1, false)?
        };
        let mut prefetched = None;
        for i in (0..run.blocks.len()).rev() {
            let b = tracing::info_span!("atlas load layer")
                .in_scope(|| run.block(i, prefetched.take(), norm_output.take()))?;
            prefetched = std::thread::scope(|scope| -> Result<Option<Block>> {
                let next = match i.checked_sub(1).and_then(|i| run.blocks.get(i)) {
                    Some(Saved::OnDisk { body, .. }) => Some(scope.spawn(|| run.read(*body))),
                    _ => None,
                };
                let p = &self.layers[i];
                let previous = i
                    .checked_sub(1)
                    .map(|j| run.layer_output(j, pre))
                    .transpose()?
                    .flatten();
                let x = if i > 0 {
                    previous.as_deref()
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
                let mut mlp = tracing::info_span!("atlas mlp reverse").in_scope(|| {
                    b.mlp.reverse(
                        io,
                        requests,
                        &mut lookups,
                        p.mlp,
                        shape,
                        bits,
                        mlp,
                        if pre { b.m.values() } else { h },
                    )
                })?;
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
                let mut from_att =
                    tracing::info_span!("atlas attention reverse").in_scope(|| {
                        self.context(i).reverse(
                            io,
                            requests,
                            &mut lookups,
                            &b.attention,
                            att,
                            if pre { b.a.values() } else { x },
                        )
                    })?;
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
                if !pre {
                    norm_output = previous;
                }
                next.map(|job| job.join().map_err(|_| "layer prefetch failed")?)
                    .transpose()
            })?;
            if i != 0 && i % WINDOW_LAYERS == 0 {
                flush(io, requests, i, std::mem::take(&mut lookups))?;
            }
        }
        if !pre {
            claims = vec![reduce_end_norm(io, requests, Some(input), &claims)?];
        }
        if claims
            .iter()
            .any(|q| self.shape.eval(input, &q.point) != q.value)
        {
            return Err("invalid public embedding evaluation");
        }
        tracing::info_span!("atlas chain end").in_scope(|| ());
        flush(io, requests, 0, lookups)
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
            self.decoder as u64,
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
            t.append(&(a.rotary.is_some() as u64));
            if let Some(r) = a.rotary {
                integers(t, r.cos);
                integers(t, r.sin);
            }
            let qk = self.qk_norms.get(i).copied().flatten();
            t.append(&(qk.is_some() as u64));
            if let Some(qk) = qk {
                norm(t, qk[0]);
                norm(t, qk[1]);
            }
            let p = layer.mlp;
            affine(t, p.up);
            affine(t, p.down);
            t.append(&(p.gate.is_some() as u64));
            if let Some(gate) = p.gate {
                affine(t, gate);
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

    pub fn is_decoder(&self) -> bool {
        self.decoder
    }

    fn embedding_claim(&self, io: &mut Channel<'_, '_>, output: &[i32]) -> Result<Vec<Claim<F>>> {
        let point = io.sample(self.shape.bits(), self.shape.bits())?;
        Ok(vec![Claim {
            value: self.shape.eval(output, &point),
            point,
        }])
    }

    pub fn cls<'b>(&self, output: &'b [i32]) -> Result<&'b [i32]> {
        if self.decoder || output.len() != self.shape.rows * self.shape.cols {
            return Err("invalid encoder output");
        }
        Ok(&output[..self.shape.cols])
    }

    fn setup(
        &self,
        tokens: &[usize],
        segments: Option<&[usize]>,
        proving: bool,
    ) -> Result<(Vec<i32>, Advice, Run)> {
        let input = self.input(tokens, segments)?;
        let mut advice = Advice::new(proving)?;
        let run = tracing::info_span!("atlas forward")
            .in_scope(|| self.forward(&mut advice, proving.then_some(input.as_slice())))?;
        Ok((input, advice, run))
    }

    pub fn prove_to(
        &self,
        tokens: &[usize],
        segments: Option<&[usize]>,
        out: impl Write + Seek + Send,
    ) -> Result<Vec<i32>> {
        if self.decoder {
            return Err("decoder requires Top-K");
        }
        let (input, mut advice, run) = self.setup(tokens, segments, true)?;
        let output = run.output(false).unwrap().to_vec();
        let mut file = ProofWriter::new(out)?;
        let header = self.prove_with(
            &mut advice,
            tokens,
            segments,
            Output::All(&output),
            |group| file.group(group),
            |io, requests, flush| {
                let claims = self.embedding_claim(io, &output)?;
                self.reverse(io, requests, &input, &run, claims, flush)
            },
        )?;
        file.finish(&header)?;
        Ok(output)
    }

    pub fn verify_from(
        &self,
        tokens: &[usize],
        segments: Option<&[usize]>,
        output: &[i32],
        input: impl Read + Seek,
    ) -> Result<()> {
        self.cls(output)?;
        let (header, mut groups) = read_proof(input)?;
        let (input, mut advice, run) = self.setup(tokens, segments, false)?;
        self.verify_with(
            &mut advice,
            tokens,
            segments,
            Output::All(output),
            &header,
            &mut groups,
            |io, requests, flush| {
                let claims = self.embedding_claim(io, output)?;
                self.reverse(io, requests, &input, &run, claims, flush)
            },
        )
    }

    fn prove_top_k_with(
        &self,
        tokens: &[usize],
        cfg: &TopKConfig,
        emit: impl FnMut(crate::bitz::Proof) -> Result<()> + Send,
    ) -> Result<(Vec<Vec<Candidate>>, Header)> {
        if !self.decoder {
            return Err("Top-K requires decoder");
        }
        let top = Top::new(cfg, self.embeddings.tokens, self.shape, self.bits)?;
        let (input, mut advice, run) = self.setup(tokens, None, true)?;
        let hidden = run.output(true);
        let (results, raw) = top.select(hidden.unwrap())?;
        let rank = top.record(&mut advice, &results, Some(&raw))?;
        drop(raw);
        let expected = &results;
        let proof = self.prove_with(
            &mut advice,
            tokens,
            None,
            Output::Top(cfg, &results),
            emit,
            move |io, requests, flush| {
                let claims = tracing::info_span!("atlas top reverse")
                    .in_scope(|| top.reverse(io, requests, &rank, expected, run.output(true)))?;
                self.reverse(io, requests, &input, &run, claims, flush)
            },
        )?;
        Ok((results, proof))
    }

    pub fn verify_top_k_from(
        &self,
        tokens: &[usize],
        cfg: &TopKConfig,
        results: &[Vec<Candidate>],
        file: impl Read + Seek,
    ) -> Result<()> {
        if !self.decoder {
            return Err("Top-K requires decoder");
        }
        let (header, mut groups) = read_proof(file)?;
        let top = Top::new(cfg, self.embeddings.tokens, self.shape, self.bits)?;
        let (input, mut advice, run) = self.setup(tokens, None, false)?;
        let rank = top.record(&mut advice, results, None)?;
        self.verify_with(
            &mut advice,
            tokens,
            None,
            Output::Top(cfg, results),
            &header,
            &mut groups,
            |io, requests, flush| {
                let claims = top.reverse(io, requests, &rank, results, None)?;
                self.reverse(io, requests, &input, &run, claims, flush)
            },
        )
    }

    fn prove_with<'m>(
        &'m self,
        advice: &mut Advice,
        tokens: &[usize],
        segments: Option<&[usize]>,
        expected: Output<'_>,
        emit: impl FnMut(crate::bitz::Proof) -> Result<()> + Send,
        work: impl FnOnce(&mut Channel<'_, '_>, &mut Requests, &mut WindowFlush<'_, 'm>) -> Result<()>,
    ) -> Result<Header> {
        let tag = jolt_transcript::protocol_tag(b"atlas/window-bitz/v1");
        let mut transcript = prover_transcript::<Keccak>(&tag, b"");
        self.bind(&mut transcript, tokens, segments, expected);
        let widths = advice.widths();
        transcript.append_bytes(&widths);
        let width = WINDOW_LAYERS;
        transcript.append(&(width as u64));
        let plan = advice.windows(width)?;
        let owners = Buckets::owners(&plan, widths.len());
        let roots = tracing::info_span!("atlas commit").in_scope(|| -> Result<Vec<_>> {
            advice.seal()?;
            let mut roots = Vec::new();
            for window in &plan {
                roots.extend(window.commit(advice, &mut transcript)?);
            }
            Ok(roots)
        })?;
        let mut state = Messages::default();
        let mut requests = Requests(Vec::new());
        let windows = plan.len();
        let advice = &*advice;
        type Job<'a> = (usize, [u8; 32], Lookups<'a>, Requests);
        type Ready = (usize, crate::channel::Window, [u8; 32], Requests);
        type Done = (crate::channel::Window, [u8; 32]);

        let prepare = |(w, seed, lookups, mut local): Job<'m>| -> Result<Ready> {
            let (proof, digest, lookup) = tracing::info_span!("atlas lookup")
                .in_scope(|| crate::channel::Window::prove(seed, lookups, advice))?;
            local.0.extend(lookup.0);
            Ok((w, proof, digest, local))
        };
        let mut emit = emit;
        let plan = &plan;
        let window_roots = &roots;
        let open = move |(w, mut proof, digest, local): Ready| -> Result<Done> {
            let mut bitz = prover_transcript::<Keccak>(&bitz_window_tag(), &digest);
            let start: usize = plan[..w].iter().map(Buckets::len).sum();
            tracing::info_span!("atlas open").in_scope(|| {
                plan[w].prove(
                    advice,
                    &window_roots[start..start + plan[w].len()],
                    local,
                    &mut bitz,
                    |proof| {
                        tracing::info_span!("atlas group complete").in_scope(|| ());
                        tracing::info_span!("atlas bitz write").in_scope(|| emit(proof))
                    },
                )
            })?;
            proof.bitz = bitz.narg_string().to_vec();
            Ok((
                proof,
                bitz.verifier_message_as(32, |v| v.try_into().unwrap()),
            ))
        };
        let mut digests = Vec::with_capacity(windows);
        let mut record = |io: &mut Channel<'_, '_>, (proof, digest): Done| {
            io.push_window(proof);
            digests.push(digest);
        };
        let mut index = 0;
        let mut io = Channel::Send(&mut transcript, &mut state, advice);
        let mut job = |io: &mut Channel<'_, '_>,
                       requests: &mut Requests,
                       layer,
                       lookups|
         -> Result<Job<'m>> {
            let part = plan.get(index).ok_or("extra Lookup window")?;
            let seed = io.window_seed(layer, (layer + width).min(self.layers.len()));
            let local = part.take(requests, &owners, index);
            let next = (index, seed, lookups, local);
            index += 1;
            Ok(next)
        };
        tracing::info_span!("atlas reduce").in_scope(|| -> Result<()> {
            // A caller running inside Rayon must be available for work scheduled in its pool.
            if windows == 1
                || rayon::current_thread_index().is_some()
                || rayon::current_num_threads() == 1
            {
                let mut open = open;
                return work(
                    &mut io,
                    &mut requests,
                    &mut |io, requests, layer, lookups| {
                        let ready = prepare(job(io, requests, layer, lookups)?)?;
                        record(io, open(ready)?);
                        Ok(())
                    },
                );
            }
            std::thread::scope(|scope| -> Result<()> {
                let (submit, jobs) = std::sync::mpsc::sync_channel::<Job<'m>>(1);
                // While BitZ consumes one window, Lookup can prepare one more.
                let (handoff, ready) = std::sync::mpsc::sync_channel::<Result<Ready>>(0);
                let (send, done) = std::sync::mpsc::channel::<Result<Done>>();
                scope.spawn(move || {
                    for job in jobs {
                        if handoff.send(prepare(job)).is_err() {
                            break;
                        }
                    }
                });
                scope.spawn(move || {
                    let mut open = open;
                    for prepared in ready {
                        let opened = prepared.and_then(&mut open);
                        let failed = opened.is_err();
                        if send.send(opened).is_err() || failed {
                            break;
                        }
                    }
                });
                let result = work(
                    &mut io,
                    &mut requests,
                    &mut |io, requests, layer, lookups| {
                        while let Ok(result) = done.try_recv() {
                            record(io, result?);
                        }
                        let next = job(io, requests, layer, lookups)?;
                        if tracing::info_span!("atlas pipeline backpressure")
                            .in_scope(|| submit.send(next))
                            .is_err()
                        {
                            return match done.recv() {
                                Ok(Err(error)) => Err(error),
                                _ => Err("Lookup pipeline stopped"),
                            };
                        }
                        Ok(())
                    },
                );
                drop(submit);
                result?;
                for result in done {
                    record(&mut io, result?);
                }
                Ok(())
            })
        })?;
        if index != windows || digests.len() != windows || !requests.0.is_empty() {
            return Err("unfinished window requests");
        }
        io.bind_windows(&digests);
        Ok(Header {
            state,
            widths,
            transcript: transcript.narg_string().to_vec(),
            roots,
        })
    }

    fn verify_with<'m>(
        &'m self,
        advice: &mut Advice,
        tokens: &[usize],
        segments: Option<&[usize]>,
        expected: Output<'_>,
        proof: &Header,
        groups: &mut std::io::Take<impl Read>,
        work: impl FnOnce(&mut Channel<'_, '_>, &mut Requests, &mut WindowFlush<'_, 'm>) -> Result<()>,
    ) -> Result<()> {
        let tag = jolt_transcript::protocol_tag(b"atlas/window-bitz/v1");
        let mut transcript = verifier_transcript::<Keccak>(&tag, b"", &proof.transcript);
        self.bind(&mut transcript, tokens, segments, expected);
        advice.bind_widths(&proof.widths)?;
        transcript.append_bytes(&proof.widths);
        let width = WINDOW_LAYERS;
        transcript.append(&(width as u64));
        let plan = advice.windows(width)?;
        if plan.iter().map(Buckets::len).sum::<usize>() != proof.roots.len() {
            return Err("wrong number of BitZ roots");
        }
        let mut root = 0;
        for part in &plan {
            part.bind(&proof.roots[root..root + part.len()], &mut transcript)?;
            root += part.len();
        }
        let owners = Buckets::owners(&plan, proof.widths.len());
        let mut io = Channel::checking(&mut transcript, &proof.state);
        let mut requests = Requests(Vec::new());
        let mut digests = Vec::new();
        let mut index = 0;
        let mut root = 0;
        let advice = &*advice;
        let mut flush =
            |io: &mut Channel<'_, '_>, req: &mut Requests, layer, lookups| -> Result<()> {
                let part = plan.get(index).ok_or("extra Lookup window")?;
                let seed = io.window_seed(layer, (layer + width).min(self.layers.len()));
                let mut local = part.take(req, &owners, index);
                let window = io.take_window()?;
                let (digest, lookup) = window.verify(seed, lookups)?;
                local.0.extend(lookup.0);
                let mut bitz =
                    verifier_transcript::<Keccak>(&bitz_window_tag(), &digest, &window.bitz);
                part.verify(
                    advice,
                    local,
                    &proof.roots[root..root + part.len()],
                    &mut bitz,
                    || decode(groups),
                )?;
                digests.push(bitz.verifier_message_as(32, |v| v.try_into().unwrap()));
                bitz.check_eof()
                    .map_err(|_| "extra BitZ window transcript bytes")?;
                root += part.len();
                index += 1;
                Ok(())
            };
        work(&mut io, &mut requests, &mut flush)?;
        if index != plan.len() || !requests.0.is_empty() {
            return Err("unfinished window requests");
        }
        io.bind_windows(&digests);
        io.finish()?;
        if groups.limit() != 0 {
            return Err("extra BitZ proofs");
        }
        transcript.check_eof().map_err(|_| "extra transcript bytes")
    }

    pub fn prove_top_k_to(
        &self,
        tokens: &[usize],
        cfg: &TopKConfig,
        out: impl Write + Seek + Send,
    ) -> Result<Vec<Vec<Candidate>>> {
        let mut file = ProofWriter::new(out)?;
        let (result, header) = self.prove_top_k_with(tokens, cfg, |group| file.group(group))?;
        file.finish(&header)?;
        Ok(result)
    }
}

fn integers(t: &mut impl Transcript, x: &[i32]) {
    t.append(&(x.len() as u64));
    for chunk in x.chunks(1024) {
        let bytes: Vec<_> = chunk.iter().flat_map(|x| x.to_le_bytes()).collect();
        t.append_bytes(&bytes);
    }
}

const MAGIC: &[u8; 8] = b"ATLASBW1";

struct ProofWriter<W: Write>(BufWriter<W>);

impl<W: Write + Seek> ProofWriter<W> {
    fn new(mut out: W) -> Result<Self> {
        if out
            .seek(SeekFrom::End(0))
            .map_err(|_| "could not inspect proof file")?
            != 0
        {
            return Err("proof destination is not empty");
        }
        let mut out = BufWriter::with_capacity(256 * 1024, out);
        out.write_all(&[0; 16])
            .map_err(|_| "could not initialize proof file")?;
        Ok(Self(out))
    }

    fn group(&mut self, proof: crate::bitz::Proof) -> Result<()> {
        codec()
            .serialize_into(&mut self.0, &proof)
            .map_err(|_| "could not write BitZ proof")
    }

    fn finish(mut self, header: &Header) -> Result<()> {
        let _span = tracing::info_span!("atlas proof metadata").entered();
        let end = self
            .0
            .stream_position()
            .map_err(|_| "could not locate proof metadata")?;
        codec()
            .serialize_into(&mut self.0, header)
            .map_err(|_| "could not write proof metadata")?;
        self.0
            .rewind()
            .and_then(|_| self.0.write_all(MAGIC))
            .and_then(|_| self.0.write_all(&end.to_le_bytes()))
            .and_then(|_| self.0.flush())
            .map_err(|_| "could not finalize proof")
    }
}

fn read_proof(mut input: impl Read + Seek) -> Result<(Header, std::io::Take<impl Read>)> {
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
    let header = decode(&mut metadata)?;
    if metadata.limit() != 0 {
        return Err("extra proof metadata");
    }
    input
        .seek(SeekFrom::Start(16))
        .map_err(|_| "could not locate BitZ proofs")?;
    Ok((header, input.take(end - 16)))
}

fn decode<T: serde::de::DeserializeOwned>(data: &mut std::io::Take<impl Read>) -> Result<T> {
    codec()
        .with_limit(data.limit())
        .deserialize_from(data)
        .map_err(|_| "invalid proof record")
}

fn codec() -> impl Options {
    bincode::DefaultOptions::new().with_fixint_encoding()
}
