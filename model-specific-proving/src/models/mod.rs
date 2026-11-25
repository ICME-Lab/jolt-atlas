//! The fixed decoder and encoder proof flows.

use jolt_transcript::{protocol_tag, Transcript};

use crate::{
    layers::{
        attention::{Attention, AttentionPlan, AttentionTrace},
        linear::{Linear, Projection, Rounded},
        mlp::{Mlp, MlpPlan, MlpTrace},
        nonlinear::{Norm, NormTrace, NormalizationParameters},
    },
    protocol::{
        lookup::Batches,
        pcs::{bind_i32, plan as advice_plan, Advice, Layout, ModelPcs, Proof},
        range::{Ranges, I32},
        session::{Claim, ProofField, Session},
        tensor::Shape,
        top_k::{self, TopKPlan},
        Result,
    },
};

#[cfg(feature = "safetensors-weights")]
pub(crate) mod bert;
#[cfg(feature = "safetensors-weights")]
pub(crate) mod gpt2;
#[cfg(feature = "safetensors-weights")]
pub(crate) mod qwen2;
#[cfg(feature = "safetensors-weights")]
use crate::{
    layers::{Activation, NormalizationKind, Rotary, SoftmaxMask, SoftmaxTables},
    weights::{Affine, Family},
};

pub use crate::protocol::{Candidate, TopKConfig};
#[cfg(feature = "safetensors-weights")]
pub use crate::weights::ModelWeights;

#[derive(Clone, Copy)]
pub struct PublicEmbeddings<'a> {
    pub width: usize,
    pub tokens: &'a [i32],
    pub positions: Option<&'a [i32]>,
    pub segments: Option<&'a [i32]>,
}

impl PublicEmbeddings<'_> {
    fn select(&self, tokens: &[usize], segments: Option<&[usize]>) -> Result<Vec<i32>> {
        if self.width == 0
            || segments.is_some_and(|s| self.segments.is_none() || s.len() != tokens.len())
        {
            return Err("invalid public embeddings");
        }
        let mut out = Vec::with_capacity(self.width * tokens.len());
        for (i, &token) in tokens.iter().enumerate() {
            let mut row = vec![0i64; self.width];
            for (table, index) in [
                (Some(self.tokens), token),
                (self.positions, i),
                (self.segments, segments.map_or(0, |ids| ids[i])),
            ] {
                if let Some(table) = table {
                    let src = table
                        .chunks_exact(self.width)
                        .nth(index)
                        .ok_or("invalid embedding index")?;
                    for (dst, &v) in row.iter_mut().zip(src) {
                        *dst += i64::from(v);
                    }
                }
            }
            for v in row {
                out.push(i32::try_from(v).map_err(|_| "embedding overflow")?);
            }
        }
        Ok(out)
    }

    fn bind(&self, t: &mut impl Transcript) {
        bind_i32(t, self.tokens);
        for table in [self.positions, self.segments] {
            t.append(&u64::from(table.is_some()));
            if let Some(table) = table {
                bind_i32(t, table);
            }
        }
    }
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
    bits: u8,
    embeddings: PublicEmbeddings<'a>,
    layers: Vec<LayerPlan<'a>>,
    head: Head<'a>,
    layout: Layout,
}

enum Head<'a> {
    Decoder { norm: Norm<'a>, logits: Linear<'a> },
    Encoder(Norm<'a>),
}

struct LayerPlan<'a> {
    attention: AttentionPlan<'a>,
    mlp: MlpPlan<'a>,
    attention_norm: Norm<'a>,
    mlp_norm: Norm<'a>,
    attention_sum: Ranges,
    output: Ranges,
}

struct LayerTrace<F> {
    attention: AttentionTrace<F>,
    mlp: MlpTrace<F>,
    attention_norm: NormTrace<F>,
    mlp_norm: NormTrace<F>,
    attention_sum: Vec<i32>,
    mlp_sum: Vec<i32>,
}

struct Witness<F> {
    advice: Advice,
    layers: Vec<LayerTrace<F>>,
    norm: NormTrace<F>,
    logits: Option<Rounded>,
}

impl<F> Witness<F> {
    fn output(&self) -> &[i32] {
        self.logits.as_ref().map_or_else(
            || &self.layers.last().expect("nonempty model").mlp_norm.output,
            |logits| &logits.output,
        )
    }
}

type ClipSpec = crate::layers::softmax::ClipSpec;
type ClipPlans = Vec<Vec<Option<crate::layers::softmax::ClipPlan>>>;

fn bind_clips(t: &mut impl Transcript, specs: &[Vec<ClipSpec>]) {
    for layer in specs {
        for head in layer {
            head.bind(t);
        }
    }
}

impl<'a> Model<'a> {
    pub fn new(
        rows: usize,
        bits: u8,
        embeddings: PublicEmbeddings<'a>,
        layers: Vec<Layer<'a>>,
        architecture: Architecture<'a>,
    ) -> Result<Self> {
        let shape = Shape::new(rows, embeddings.width)?;
        if layers.is_empty() || bits > 15 {
            return Err("invalid model dimensions");
        }
        let mut layout = Layout::default();
        let mut plans = Vec::with_capacity(layers.len());
        for p in layers {
            let input_bound = match &architecture {
                Architecture::Decoder { .. } => p.attention_norm.output_bound(shape.width, bits),
                Architecture::Encoder { .. } => None,
            };
            let attention =
                AttentionPlan::new(&mut layout, shape, bits.into(), p.attention, input_bound)?;
            let mlp_norm = match &architecture {
                Architecture::Decoder { .. } => p.mlp_norm,
                Architecture::Encoder { .. } => p.attention_norm,
            };
            let mlp = MlpPlan::new(
                &mut layout,
                shape,
                bits.into(),
                p.mlp,
                mlp_norm.output_bound(shape.width, bits),
            )?;
            let attention_norm = Norm::new(&mut layout, shape, bits.into(), p.attention_norm)?;
            let mlp_norm = Norm::new(&mut layout, shape, bits.into(), p.mlp_norm)?;
            let attention_sum = Ranges::new(&mut layout, shape, [I32])?;
            let output = Ranges::new(&mut layout, shape, [I32])?;
            plans.push(LayerPlan {
                attention,
                mlp,
                attention_norm,
                mlp_norm,
                attention_sum,
                output,
            });
        }
        let head = match architecture {
            Architecture::Decoder {
                final_norm,
                logits_bias,
            } => {
                if embeddings.segments.is_some() || logits_bias.iter().any(|&b| b != 0) {
                    return Err("unexpected decoder segments or logits bias");
                }
                let norm = Norm::new(&mut layout, shape, bits.into(), final_norm)?;
                let logits = Linear::new(
                    &mut layout,
                    shape,
                    Projection {
                        weights: embeddings.tokens,
                        bias: logits_bias,
                    },
                    bits.into(),
                    true,
                )?;
                Head::Decoder { norm, logits }
            }
            Architecture::Encoder { embedding_norm } => {
                Head::Encoder(Norm::new(&mut layout, shape, bits.into(), embedding_norm)?)
            }
        };
        Ok(Self {
            shape,
            bits,
            embeddings,
            layers: plans,
            head,
            layout,
        })
    }

    pub fn setup_num_vars(&self) -> usize {
        self.layout.num_vars().expect("valid model layout")
    }

    pub fn cls<'b>(&self, output: &'b [i32]) -> Result<&'b [i32]> {
        self.shape.check(output)?;
        Ok(&output[..self.shape.width])
    }

    fn domain(&self) -> [u8; 32] {
        match self.head {
            Head::Decoder { .. } => protocol_tag(b"atlas/model-specific/decoder/v2"),
            Head::Encoder(_) => protocol_tag(b"atlas/model-specific/encoder/v2"),
        }
    }

    fn bind(
        &self,
        t: &mut impl Transcript,
        tokens: &[usize],
        segments: Option<&[usize]>,
        output: &[i32],
    ) {
        for value in [
            self.shape.rows,
            self.shape.width,
            self.bits.into(),
            self.layers.len(),
        ] {
            t.append(&(value as u64));
        }
        self.embeddings.bind(t);
        for &token in tokens {
            t.append(&(token as u64));
        }
        t.append(&u64::from(segments.is_some()));
        if let Some(segments) = segments {
            for &s in segments {
                t.append(&(s as u64));
            }
        }
        for layer in &self.layers {
            layer.attention.bind(t);
            layer.attention_norm.bind(t);
            layer.mlp.bind(t);
            layer.mlp_norm.bind(t);
        }
        match &self.head {
            Head::Decoder { norm, .. } => norm.bind(t),
            Head::Encoder(norm) => norm.bind(t),
        }
        bind_i32(t, output);
    }

    fn forward<F: ProofField>(
        &self,
        input: &[i32],
        layout: &Layout,
        full_logits: bool,
    ) -> Result<Witness<F>> {
        let mut advice = layout.witness();
        let mut layers: Vec<LayerTrace<F>> = Vec::with_capacity(self.layers.len());
        let embedding = if let Head::Encoder(norm) = &self.head {
            Some(norm.execute(input, &mut advice)?)
        } else {
            None
        };
        for p in &self.layers {
            let x = layers.last().map_or_else(
                || embedding.as_ref().map_or(input, |e| &e.output),
                |w| {
                    if embedding.is_some() {
                        &w.mlp_norm.output
                    } else {
                        &w.mlp_sum
                    }
                },
            );
            let pre = embedding.is_none();
            let (attention, attention_sum, attention_norm) = sublayer(
                pre,
                x,
                (&p.attention_norm, &p.attention_sum),
                &mut advice,
                |source, advice| p.attention.execute(source, advice),
                AttentionTrace::output,
            )?;
            let skip = if pre {
                &attention_sum
            } else {
                &attention_norm.output
            };
            let (mlp, mlp_sum, mlp_norm) = sublayer(
                pre,
                skip,
                (&p.mlp_norm, &p.output),
                &mut advice,
                |source, advice| p.mlp.execute(source, advice),
                MlpTrace::output,
            )?;
            let layer = LayerTrace {
                attention,
                mlp,
                attention_norm,
                mlp_norm,
                attention_sum,
                mlp_sum,
            };
            layers.push(layer);
        }
        let (norm, logits) = match &self.head {
            Head::Decoder { norm, logits } => {
                let norm = norm.execute(&layers.last().unwrap().mlp_sum, &mut advice)?;
                let logits = full_logits
                    .then(|| logits.execute(&norm.output, &mut advice))
                    .transpose()?;
                (norm, logits)
            }
            Head::Encoder(_) => (embedding.expect("encoder embedding"), None),
        };
        Ok(Witness {
            advice,
            layers,
            norm,
            logits,
        })
    }

    fn reverse<'w, F: ProofField>(
        &self,
        input: &[i32],
        output: &[i32],
        clipping: &ClipPlans,
        witness: impl Fn() -> &'w Witness<F> + Sync,
        io: &mut Session<'_, '_, F>,
    ) -> Result<()> {
        let claim = match &self.head {
            Head::Decoder { logits, .. } => logits.reverse(
                point(logits.shape(), output, io)?,
                || {
                    let w = witness();
                    (&w.norm.output, w.logits.as_ref().expect("logits trace"))
                },
                io,
            )?,
            Head::Encoder(_) => point(self.shape, output, io)?,
        };
        self.reverse_layers(input, vec![claim], clipping, witness, io)
    }

    fn reverse_layers<'w, F: ProofField>(
        &self,
        input: &[i32],
        mut claims: Vec<Claim<F>>,
        clipping: &ClipPlans,
        witness: impl Fn() -> &'w Witness<F> + Sync,
        io: &mut Session<'_, '_, F>,
    ) -> Result<()> {
        let shape = self.shape;
        let pre = matches!(self.head, Head::Decoder { .. });
        let norm_trace = || &witness().norm;
        if let Head::Decoder { norm, .. } = &self.head {
            let last = || &witness().layers.last().expect("model layers").mlp_sum[..];
            claims = vec![norm.reverse(&claims, || (last(), norm_trace()), io)?];
        }
        let mut lookups = Batches::default();
        for (i, p) in self.layers.iter().enumerate().rev() {
            let w = || &witness().layers[i];
            let x = || match (i, pre) {
                (0, true) => input,
                (0, false) => &norm_trace().output,
                (_, true) => &witness().layers[i - 1].mlp_sum,
                (_, false) => &witness().layers[i - 1].mlp_norm.output,
            };
            let (skip, branch) = if pre {
                let q = boundary(shape, &p.output, &claims, || &w().mlp_sum, io)?;
                split(shape, None, q, || &w().attention_sum, io)?
            } else {
                let q = p
                    .mlp_norm
                    .reverse(&claims, || (&w().mlp_sum, &w().mlp_norm), io)?;
                split(shape, Some(&p.output), q, || &w().attention_norm.output, io)?
            };
            let source = || {
                if pre {
                    &w().mlp_norm.output[..]
                } else {
                    &w().attention_norm.output
                }
            };
            let branch = p
                .mlp
                .reverse(branch, source, || &w().mlp, io, &mut lookups)?;
            let (skip, branch) = if pre {
                let q =
                    p.mlp_norm
                        .reverse(&[branch], || (&w().attention_sum, &w().mlp_norm), io)?;
                let h = fresh(shape, || &w().attention_sum, io)?;
                let range = [&skip, &q, &h].map(|q| (q.point.as_slice(), [q.value]));
                p.attention_sum.check(&range, io)?;
                split(shape, None, h, x, io)?
            } else {
                let q = p.attention_norm.reverse(
                    &[skip, branch],
                    || (&w().attention_sum, &w().attention_norm),
                    io,
                )?;
                split(shape, Some(&p.attention_sum), q, x, io)?
            };
            let source = || {
                if pre {
                    &w().attention_norm.output[..]
                } else {
                    x()
                }
            };
            claims = p.attention.reverse(
                branch,
                &clipping[i],
                || (source(), &w().attention),
                io,
                &mut lookups,
            )?;
            if pre {
                claims =
                    vec![p
                        .attention_norm
                        .reverse(&claims, || (x(), &w().attention_norm), io)?];
            }
            claims.push(skip);
        }
        if let Head::Encoder(norm) = &self.head {
            claims = vec![norm.reverse(&claims, || (input, norm_trace()), io)?];
        }
        if !io.proving()
            && claims
                .iter()
                .any(|q| q.value != shape.eval(input, &q.point))
        {
            return Err("invalid public embedding evaluation");
        }
        lookups.finish(io)
    }

    fn input(&self, tokens: &[usize], segments: Option<&[usize]>) -> Result<Vec<i32>> {
        if tokens.len() != self.shape.rows {
            return Err("invalid sequence length");
        }
        self.embeddings.select(tokens, segments)
    }

    fn clip_plan(&self, layout: &mut Layout, specs: &[Vec<ClipSpec>]) -> Result<ClipPlans> {
        if specs.len() != self.layers.len() {
            return Err("invalid Softmax clip layer count");
        }
        self.layers
            .iter()
            .zip(specs)
            .map(|(layer, s)| layer.attention.clip_plan(layout, s))
            .collect()
    }

    fn prepare_clip<F>(
        &self,
        layout: &mut Layout,
        w: &mut Witness<F>,
    ) -> Result<(Vec<Vec<ClipSpec>>, ClipPlans)> {
        let specs: Vec<_> = self
            .layers
            .iter()
            .zip(&w.layers)
            .map(|(p, w)| p.attention.clip_specs(&w.attention))
            .collect();
        let plans = self.clip_plan(layout, &specs)?;
        w.advice.extend(layout);
        for ((p, layer), plans) in self.layers.iter().zip(&w.layers).zip(&plans) {
            p.attention
                .record_clip(plans, &layer.attention, &mut w.advice)?;
        }
        Ok((specs, plans))
    }

    pub fn prove<C: ModelPcs>(
        &self,
        tokens: &[usize],
        segments: Option<&[usize]>,
        setup: &C::AdviceProverSetup,
    ) -> Result<(Vec<i32>, Proof<C>)> {
        let input = self.input(tokens, segments)?;
        let mut w = self.forward::<C::Field>(&input, &self.layout, true)?;
        let mut layout = self.layout.clone();
        let (specs, clipping) = self.prepare_clip(&mut layout, &mut w)?;
        let prepared = advice_plan::<C>(&layout)?;
        let output = w.output().to_vec();
        let mut proof = Proof::prove(
            &self.domain(),
            &layout,
            w,
            |w| &mut w.advice,
            &prepared,
            setup,
            |t| {
                self.bind(t, tokens, segments, &output);
                bind_clips(t, &specs);
            },
            |w, io| self.reverse(&input, &output, &clipping, || &w, io),
        )?;
        proof.clipping = specs;
        Ok((output, proof))
    }

    pub fn verify<C: ModelPcs>(
        &self,
        tokens: &[usize],
        segments: Option<&[usize]>,
        output: &[i32],
        proof: &Proof<C>,
        setup: &C::AdviceVerifierSetup,
    ) -> Result<()> {
        let input = self.input(tokens, segments)?;
        match &self.head {
            Head::Decoder { logits, .. } => logits.shape().check(output)?,
            Head::Encoder(_) => self.shape.check(output)?,
        }
        let mut layout = self.layout.clone();
        let clipping = self.clip_plan(&mut layout, &proof.clipping)?;
        proof.verify(
            &self.domain(),
            &layout,
            setup,
            |t| {
                self.bind(t, tokens, segments, output);
                bind_clips(t, &proof.clipping);
            },
            |io| self.reverse(&input, output, &clipping, || unreachable!(), io),
        )
    }
}

impl Model<'_> {
    fn reverse_top_k<'w, F: ProofField>(
        &self,
        input: &[i32],
        plan: &TopKPlan,
        cfg: &TopKConfig,
        candidates: &[Vec<Candidate>],
        clipping: &ClipPlans,
        witness: impl Fn() -> &'w Witness<F> + Sync,
        io: &mut Session<'_, '_, F>,
    ) -> Result<()> {
        if !matches!(self.head, Head::Decoder { .. }) {
            return Err("not a decoder");
        }
        let claims = plan.reduce(
            cfg,
            candidates,
            self.embeddings.tokens,
            self.shape.width,
            || &witness().norm.output,
            io,
        )?;
        self.reverse_layers(input, claims, clipping, witness, io)
    }

    fn top_k_plan(&self, cfg: &TopKConfig) -> Result<TopKPlan> {
        if !matches!(&self.head, Head::Decoder { .. }) {
            return Err("invalid Top-K configuration");
        }
        let mut layout = self.layout.clone();
        layout.remove_last()?;
        TopKPlan::new(layout, self.shape, self.bits, self.embeddings.tokens, cfg)
    }

    fn bind_top_k(
        &self,
        t: &mut impl Transcript,
        tokens: &[usize],
        cfg: &TopKConfig,
        candidates: &[Vec<Candidate>],
    ) {
        self.bind(t, tokens, None, &[]);
        top_k::bind(t, cfg, candidates);
    }

    pub fn top_k_setup_num_vars(&self, cfg: &TopKConfig) -> Result<usize> {
        self.top_k_plan(cfg)?.layout.num_vars()
    }

    pub fn prove_top_k<C: ModelPcs>(
        &self,
        tokens: &[usize],
        cfg: &TopKConfig,
        setup: &C::AdviceProverSetup,
    ) -> Result<(Vec<Vec<Candidate>>, Proof<C>)> {
        let input = self.input(tokens, None)?;
        let plan = self.top_k_plan(cfg)?;
        let mut witness = self.forward::<C::Field>(&input, &plan.layout, false)?;
        let norm = &witness.norm;
        let public = self.embeddings.tokens;
        let width = self.shape.width;
        let candidates = plan.select(cfg, &norm.output, public, width)?;
        plan.record(
            cfg,
            &candidates,
            &norm.output,
            public,
            width,
            &mut witness.advice,
        )?;
        let mut layout = plan.layout.clone();
        let (specs, clipping) = self.prepare_clip(&mut layout, &mut witness)?;
        let prepared = advice_plan::<C>(&layout)?;
        let mut proof = Proof::prove(
            &protocol_tag(b"atlas/model-specific/top-k/v2"),
            &layout,
            witness,
            |w| &mut w.advice,
            &prepared,
            setup,
            |t| {
                self.bind_top_k(t, tokens, cfg, &candidates);
                bind_clips(t, &specs);
            },
            |w, io| self.reverse_top_k(&input, &plan, cfg, &candidates, &clipping, || &w, io),
        )?;
        proof.clipping = specs;
        Ok((candidates, proof))
    }

    pub fn verify_top_k<C: ModelPcs>(
        &self,
        tokens: &[usize],
        cfg: &TopKConfig,
        candidates: &[Vec<Candidate>],
        proof: &Proof<C>,
        setup: &C::AdviceVerifierSetup,
    ) -> Result<()> {
        let input = self.input(tokens, None)?;
        let plan = self.top_k_plan(cfg)?;
        plan.candidates(cfg, candidates)?;
        let mut layout = plan.layout.clone();
        let clipping = self.clip_plan(&mut layout, &proof.clipping)?;
        proof.verify(
            &protocol_tag(b"atlas/model-specific/top-k/v2"),
            &layout,
            setup,
            |t| {
                self.bind_top_k(t, tokens, cfg, candidates);
                bind_clips(t, &proof.clipping);
            },
            |io| {
                self.reverse_top_k(
                    &input,
                    &plan,
                    cfg,
                    candidates,
                    &clipping,
                    || unreachable!(),
                    io,
                )
            },
        )
    }
}

#[cfg(feature = "safetensors-weights")]
impl ModelWeights {
    pub fn model(&self, rows: usize) -> Result<Model<'_>> {
        if rows == 0 || rows > self.max_positions {
            return Err("invalid model sequence length");
        }
        match self.family {
            Family::Gpt2 => gpt2::model(self, rows),
            Family::Qwen2 => qwen2::model(self, rows),
            Family::Bert => bert::model(self, rows),
        }
    }

    pub(super) fn decoder(&self, kind: NormalizationKind) -> Architecture<'_> {
        Architecture::Decoder {
            final_norm: self.normalization(&self.parameters.norm, kind),
            logits_bias: &self.logits_bias,
        }
    }

    fn normalization<'a>(
        &self,
        affine: &'a Affine,
        kind: NormalizationKind,
    ) -> NormalizationParameters<'a> {
        NormalizationParameters {
            kind,
            gain: &affine.weights,
            bias: &affine.bias,
            epsilon_variance_units: self.epsilon_variance_units,
        }
    }

    pub(super) fn assemble<'a>(
        &'a self,
        rows: usize,
        kind: NormalizationKind,
        rotary: Option<Rotary<'a>>,
        architecture: Architecture<'a>,
    ) -> Result<Model<'a>> {
        let mask = match architecture {
            Architecture::Decoder { .. } => SoftmaxMask::Causal,
            Architecture::Encoder { .. } => SoftmaxMask::Full,
        };
        let width = self.width;
        let head = self.head_width;
        let tables = SoftmaxTables {
            probabilities: &self.softmax.out,
            sums: &self.softmax.sum,
            limit: self.softmax.sum_scale as i32,
        };
        let layers = self
            .parameters
            .layers
            .iter()
            .map(|layer| Layer {
                attention: Attention {
                    queries: layer.query.heads(width, head).collect(),
                    groups: layer
                        .key
                        .heads(width, head)
                        .zip(layer.value.heads(width, head))
                        .map(|(k, v)| [k, v])
                        .collect(),
                    output: layer.out.at(),
                    rotary,
                    mask,
                    tables,
                    factor: self.factor,
                },
                mlp: Mlp {
                    up: layer.up.at(),
                    down: layer.down.at(),
                    gate: layer.gate.as_ref().map(Affine::at),
                    activation: Activation {
                        lower: self.parameters.activation.0,
                        correction: &self.parameters.activation.1,
                    },
                },
                attention_norm: self.normalization(&layer.norm1, kind),
                mlp_norm: self.normalization(&layer.norm2, kind),
            })
            .collect();
        let embeddings = PublicEmbeddings {
            width,
            tokens: &self.parameters.tokens,
            positions: self.parameters.positions.as_deref(),
            segments: self.parameters.segments.as_deref(),
        };
        Model::new(rows, self.bits, embeddings, layers, architecture)
    }
}

#[cfg(feature = "safetensors-weights")]
impl Affine {
    fn at(&self) -> Projection<'_> {
        Projection {
            weights: &self.weights,
            bias: &self.bias,
        }
    }

    fn heads(&self, input: usize, head: usize) -> impl Iterator<Item = Projection<'_>> {
        self.weights
            .chunks_exact(input * head)
            .zip(self.bias.chunks_exact(head))
            .map(|(weights, bias)| Projection { weights, bias })
    }
}

fn sublayer<F: ProofField, T>(
    pre: bool,
    input: &[i32],
    (norm, range): (&Norm<'_>, &Ranges),
    advice: &mut Advice,
    execute: impl FnOnce(&[i32], &mut Advice) -> Result<T>,
    output: impl FnOnce(&T) -> &[i32],
) -> Result<(T, Vec<i32>, NormTrace<F>)> {
    let normalized = pre.then(|| norm.execute(input, advice)).transpose()?;
    let trace = execute(normalized.as_ref().map_or(input, |n| &n.output), advice)?;
    let sum = input
        .iter()
        .zip(output(&trace))
        .map(|(&a, &b)| a.checked_add(b).ok_or("residual overflow"))
        .collect::<Result<Vec<_>>>()?;
    range.record(advice, &[&sum])?;
    let normalized = match normalized {
        Some(n) => n,
        None => norm.execute(&sum, advice)?,
    };
    Ok((trace, sum, normalized))
}

fn point<F: ProofField>(shape: Shape, x: &[i32], io: &mut Session<'_, '_, F>) -> Result<Claim<F>> {
    let point = io.point(shape.vars())?;
    Ok(Claim {
        value: shape.eval(x, &point),
        point,
    })
}
fn fresh<'a, F: ProofField>(
    shape: Shape,
    x: impl FnOnce() -> &'a [i32],
    io: &mut Session<'_, '_, F>,
) -> Result<Claim<F>> {
    let point = io.point(shape.vars())?;
    let [value] = io.values(|| [shape.eval(x(), &point)])?;
    Ok(Claim { point, value })
}
fn boundary<'a, F: ProofField>(
    shape: Shape,
    range: &Ranges,
    claims: &[Claim<F>],
    trace: impl FnOnce() -> &'a [i32],
    io: &mut Session<'_, '_, F>,
) -> Result<Claim<F>> {
    let next = fresh(shape, trace, io)?;
    let checks: Vec<_> = claims
        .iter()
        .chain([&next])
        .map(|q| (q.point.as_slice(), [q.value]))
        .collect();
    range.check(&checks, io)?;
    Ok(next)
}
fn split<'a, F: ProofField>(
    shape: Shape,
    range: Option<&Ranges>,
    q: Claim<F>,
    left: impl FnOnce() -> &'a [i32],
    io: &mut Session<'_, '_, F>,
) -> Result<(Claim<F>, Claim<F>)> {
    if let Some(range) = range {
        range.check(&[(&q.point, [q.value])], io)?;
    }
    let [value] = io.values(|| [shape.eval(left(), &q.point)])?;
    Ok((
        Claim {
            point: q.point.clone(),
            value,
        },
        Claim {
            point: q.point,
            value: q.value - value,
        },
    ))
}
