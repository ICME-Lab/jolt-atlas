use std::{error::Error, fs, path::Path};

use atlas_trace::{
    semantics::{
        softmax,
        tables::{table, IndexRule, TableKind},
    },
    Scale, SemVersion,
};
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;

use crate::{
    layers::*,
    models::{Architecture, Layer, Model, PublicEmbeddings},
    protocol::Result,
};

type Load<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Family {
    Gpt,
    Qwen2,
    Qwen3,
    Bert,
}

struct Affine {
    w: Vec<i32>,
    b: Vec<i32>,
}

impl Affine {
    fn as_projection(&self) -> Projection<'_> {
        Projection {
            weights: &self.w,
            bias: &self.b,
        }
    }

    fn heads(&self, channels: usize, dim: usize) -> impl Iterator<Item = Projection<'_>> {
        self.w
            .chunks_exact(channels * dim)
            .zip(self.b.chunks_exact(dim))
            .map(|(weights, bias)| Projection { weights, bias })
    }
}

struct Block {
    query: Affine,
    key: Affine,
    value: Affine,
    output: Affine,
    up: Affine,
    gate: Option<Affine>,
    down: Affine,
    norm: [Affine; 2],
    qk: Option<[Affine; 2]>,
}

pub struct ModelWeights {
    family: Family,
    width: usize,
    head: usize,
    limit: usize,
    bits: u8,
    epsilon: i64,
    embeddings: Vec<i32>,
    positions: Option<Vec<i32>>,
    segments: Option<Vec<i32>>,
    blocks: Vec<Block>,
    norm: Affine,
    bias: Vec<i32>,
    activation: (i32, Vec<i32>),
    softmax: softmax::SoftmaxTables,
    theta: Option<f64>,
    rotary: Vec<std::sync::OnceLock<(Vec<i32>, Vec<i32>)>>,
}

struct Reader<'a> {
    data: SafeTensors<'a>,
    factor: f64,
}

impl Reader<'_> {
    fn read(&self, key: &str, shape: &[usize]) -> Load<Vec<i32>> {
        let tensor = self.data.tensor(key)?;
        if shape != tensor.shape() {
            return Err(format!("wrong tensor shape: {key}").into());
        }
        let (n, convert): (_, fn(&[u8]) -> f64) = match tensor.dtype() {
            Dtype::F16 => (2, |x| {
                half::f16::from_bits(u16::from_le_bytes(x.try_into().unwrap())).to_f64()
            }),
            Dtype::BF16 => (2, |x| {
                half::bf16::from_bits(u16::from_le_bytes(x.try_into().unwrap())).to_f64()
            }),
            Dtype::F32 => (4, |x| f64::from(f32::from_le_bytes(x.try_into().unwrap()))),
            Dtype::F64 => (8, |x| f64::from_le_bytes(x.try_into().unwrap())),
            _ => return Err(format!("unsupported weight encoding: {key}").into()),
        };
        tensor
            .data()
            .chunks_exact(n)
            .map(|v| {
                let y = (convert(v) * self.factor).round();
                if !y.is_finite() || y < i32::MIN as f64 || y > i32::MAX as f64 {
                    return Err(format!("weight exceeds i32: {key}").into());
                }
                Ok(y as i32)
            })
            .collect()
    }

    fn affine(
        &self,
        key: &str,
        from: usize,
        to: usize,
        bias: bool,
        transpose: bool,
    ) -> Load<Affine> {
        let name = format!("{key}.weight");
        let w = if transpose {
            let original = self.read(&name, &[from, to])?;
            (0..to)
                .flat_map(|c| original[c..].iter().step_by(to).copied())
                .collect()
        } else {
            self.read(&name, &[to, from])?
        };
        Ok(Affine {
            w,
            b: self.bias(key, to, bias)?,
        })
    }

    fn bias(&self, key: &str, cols: usize, present: bool) -> Load<Vec<i32>> {
        if present {
            self.read(&format!("{key}.bias"), &[cols])
        } else {
            Ok(vec![0; cols])
        }
    }

    fn norm(&self, key: &str, cols: usize, bias: bool) -> Load<Affine> {
        Ok(Affine {
            w: self.read(&format!("{key}.weight"), &[cols])?,
            b: self.bias(key, cols, bias)?,
        })
    }
}

impl ModelWeights {
    pub fn from_safetensors(directory: &Path, bits: u8) -> Load<Self> {
        if bits > 15 {
            return Err("unsupported fraction width".into());
        }
        let cfg: Value = serde_json::from_slice(&fs::read(directory.join("config.json"))?)?;
        let family = match cfg["model_type"].as_str() {
            Some("gpt2") => Family::Gpt,
            Some("qwen2") => Family::Qwen2,
            Some("qwen3") => Family::Qwen3,
            Some("bert") => Family::Bert,
            _ => return Err("unsupported model type".into()),
        };
        let gpt = family == Family::Gpt;
        let bert = family == Family::Bert;
        let q3 = family == Family::Qwen3;
        let qwen = !gpt && !bert;
        let count = |name: &str| -> Load<usize> {
            let n = cfg[name]
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .unwrap_or(0);
            if n == 0 || n > 1 << 20 {
                return Err(format!("invalid model dimension: {name}").into());
            }
            Ok(n)
        };
        let width = count(if gpt { "n_embd" } else { "hidden_size" })?;
        let heads = count(if gpt { "n_head" } else { "num_attention_heads" })?;
        let blocks = count(if gpt { "n_layer" } else { "num_hidden_layers" })?;
        let limit = count(if gpt {
            "n_positions"
        } else {
            "max_position_embeddings"
        })?;
        let vocab = count("vocab_size")?;
        let head = if qwen {
            count("head_dim").unwrap_or(width / heads)
        } else {
            width / heads
        };
        let kv = if qwen {
            count("num_key_value_heads").unwrap_or(heads)
        } else {
            heads
        };
        let inner = count(if gpt { "n_inner" } else { "intermediate_size" }).unwrap_or(width * 4);
        if !head.is_power_of_two()
            || heads % kv != 0
            || (!qwen && heads * head != width)
            || (qwen
                && (!cfg["rope_scaling"].is_null()
                    || cfg["use_sliding_window"].as_bool() == Some(true)))
            || (gpt
                && (cfg["scale_attn_by_inverse_layer_idx"].as_bool() == Some(true)
                    || cfg["scale_attn_weights"].as_bool() == Some(false)))
            || (bert
                && cfg["position_embedding_type"]
                    .as_str()
                    .is_some_and(|s| s != "absolute"))
        {
            return Err("unsupported transformer configuration".into());
        }
        let kind = if gpt {
            "gelu_new"
        } else if bert {
            "gelu"
        } else {
            "silu"
        };
        if cfg[if gpt {
            "activation_function"
        } else {
            "hidden_act"
        }]
        .as_str()
        .is_some_and(|x| x != kind)
        {
            return Err("unsupported activation".into());
        }
        let raw_epsilon = cfg[if gpt {
            "layer_norm_epsilon"
        } else if bert {
            "layer_norm_eps"
        } else {
            "rms_norm_eps"
        }]
        .as_f64()
        .ok_or("missing normalization epsilon")?;
        let scale = Scale::new(bits)?;
        let factor = f64::from(scale.denom_i32());
        let epsilon = (raw_epsilon * factor * factor).round().max(1.0);
        if !raw_epsilon.is_finite() || raw_epsilon < 0.0 || epsilon >= i64::MAX as f64 {
            return Err("invalid normalization epsilon".into());
        }
        let raw = fs::read(directory.join("model.safetensors"))?;
        let reader = Reader {
            data: SafeTensors::deserialize(&raw)?,
            factor,
        };
        let has = |name| reader.data.tensor(name).is_ok();
        let prefix = if gpt && has("transformer.wte.weight") {
            "transformer."
        } else if bert && has("bert.embeddings.word_embeddings.weight") {
            "bert."
        } else {
            ""
        };
        let word = if qwen {
            "model.embed_tokens.weight"
        } else if gpt {
            "wte.weight"
        } else {
            "embeddings.word_embeddings.weight"
        };
        let embeddings = reader.read(&format!("{prefix}{word}"), &[vocab, width])?;
        if !bert
            && has("lm_head.weight")
            && reader.read("lm_head.weight", &[vocab, width])? != embeddings
        {
            return Err("decoder requires tied embeddings".into());
        }
        let positions = if qwen {
            None
        } else {
            let field = if gpt {
                "wpe.weight"
            } else {
                "embeddings.position_embeddings.weight"
            };
            Some(reader.read(&format!("{prefix}{field}"), &[limit, width])?)
        };
        let segments = if bert {
            Some(reader.read(
                &format!("{prefix}embeddings.token_type_embeddings.weight"),
                &[count("type_vocab_size")?, width],
            )?)
        } else {
            None
        };
        let root = if qwen {
            "model.norm"
        } else if gpt {
            "ln_f"
        } else {
            "embeddings.LayerNorm"
        };
        let norm = reader.norm(&format!("{prefix}{root}"), width, !qwen)?;
        let mut layers = Vec::with_capacity(blocks);
        for i in 0..blocks {
            let stem = if qwen {
                format!("model.layers.{i}")
            } else if gpt {
                format!("{prefix}h.{i}")
            } else {
                format!("{prefix}encoder.layer.{i}")
            };
            let linear = |path, size, out, bias| {
                reader.affine(&format!("{stem}.{path}"), size, out, bias, gpt)
            };
            let normalization = |path, cols| reader.norm(&format!("{stem}.{path}"), cols, !qwen);
            let attention_bias = !q3 || cfg["attention_bias"].as_bool() == Some(true);
            let [query, key, value] = if gpt {
                let all = linear("attn.c_attn", width, 3 * width, true)?;
                std::array::from_fn(|j| Affine {
                    w: all.w[j * width * width..(j + 1) * width * width].to_vec(),
                    b: all.b[j * width..(j + 1) * width].to_vec(),
                })
            } else if bert {
                [
                    "attention.self.query",
                    "attention.self.key",
                    "attention.self.value",
                ]
                .map(|path| linear(path, width, width, true))
                .into_iter()
                .collect::<Load<Vec<_>>>()?
                .try_into()
                .ok()
                .unwrap()
            } else {
                [
                    linear("self_attn.q_proj", width, heads * head, attention_bias)?,
                    linear("self_attn.k_proj", width, kv * head, attention_bias)?,
                    linear("self_attn.v_proj", width, kv * head, attention_bias)?,
                ]
            };
            let (o, u, d, a, m) = match family {
                Family::Gpt => ("attn.c_proj", "mlp.c_fc", "mlp.c_proj", "ln_1", "ln_2"),
                Family::Bert => (
                    "attention.output.dense",
                    "intermediate.dense",
                    "output.dense",
                    "attention.output.LayerNorm",
                    "output.LayerNorm",
                ),
                _ => (
                    "self_attn.o_proj",
                    "mlp.up_proj",
                    "mlp.down_proj",
                    "input_layernorm",
                    "post_attention_layernorm",
                ),
            };
            layers.push(Block {
                query,
                key,
                value,
                output: linear(o, heads * head, width, !qwen || q3 && attention_bias)?,
                up: linear(u, width, inner, !qwen)?,
                gate: qwen
                    .then(|| linear("mlp.gate_proj", width, inner, false))
                    .transpose()?,
                down: linear(d, inner, width, !qwen)?,
                norm: [normalization(a, width)?, normalization(m, width)?],
                qk: if q3 {
                    Some([
                        normalization("self_attn.q_norm", head)?,
                        normalization("self_attn.k_norm", head)?,
                    ])
                } else {
                    None
                },
            });
        }
        let activation = if gpt {
            let bound = 8 * scale.denom_i32();
            let a = libm::sqrt(2.0 / std::f64::consts::PI);
            let entries = (0..2 * bound)
                .map(|i| {
                    let magnitude = f64::from((i - bound).abs());
                    let x = magnitude / factor;
                    let tail = 0.5 * magnitude * (1.0 - libm::tanh(a * (x + 0.044715 * x.powi(3))));
                    libm::floor(0.5 - tail) as i32
                })
                .collect();
            (-bound, entries)
        } else {
            let kind = if qwen {
                TableKind::Silu
            } else {
                TableKind::GeluErf
            };
            let t = table(kind, scale, SemVersion::V3);
            let IndexRule::SymClamp { bound_log } = t.index else {
                return Err("unsupported activation table".into());
            };
            (-(1 << bound_log), t.entries)
        };
        let theta = if qwen {
            Some(
                cfg["rope_theta"]
                    .as_f64()
                    .filter(|x| x.is_finite() && *x > 0.0)
                    .ok_or("invalid RoPE base")?,
            )
        } else {
            None
        };
        Ok(Self {
            family,
            width,
            head,
            bits,
            limit,
            epsilon: epsilon as i64,
            embeddings,
            positions,
            segments,
            blocks: layers,
            norm,
            bias: vec![0; vocab],
            activation,
            softmax: softmax::softmax_tables(scale, SemVersion::V3),
            theta,
            rotary: (0..=limit.next_power_of_two().ilog2())
                .map(|_| std::sync::OnceLock::new())
                .collect(),
        })
    }

    pub fn model<'a>(&'a self, rows: usize) -> Result<Model<'a>> {
        if rows == 0 || rows > self.limit {
            return Err("invalid sequence length");
        }
        let qwen = matches!(self.family, Family::Qwen2 | Family::Qwen3);
        let encoder = self.family == Family::Bert;
        let normalize = |p: &'a Affine| self.norm_params(p, qwen);
        let rotary = self.theta.map(|theta| {
            let size = rows.next_power_of_two();
            let slot = size.ilog2() as usize;
            let (cos, sin) = self.rotary[slot..]
                .iter()
                .find_map(std::sync::OnceLock::get)
                .unwrap_or_else(|| {
                    self.rotary[slot].get_or_init(|| {
                        let half = self.head / 2;
                        let factor = (1u64 << self.bits) as f64;
                        let frequencies: Vec<_> = (0..half)
                            .map(|i| libm::pow(theta, -((i * 2) as f64) / self.head as f64))
                            .collect();
                        (0..size.min(self.limit))
                            .flat_map(|r| {
                                frequencies.iter().map(move |&f| {
                                    let (sin, cos) = libm::sincos(r as f64 * f);
                                    ((cos * factor).round() as i32, (sin * factor).round() as i32)
                                })
                            })
                            .unzip()
                    })
                });
            let b = rows * self.head / 2;
            Rotary {
                cos: &cos[..b],
                sin: &sin[..b],
            }
        });
        let layers = self
            .blocks
            .iter()
            .map(|b| Layer {
                attention: Attention {
                    queries: b.query.heads(self.width, self.head).collect(),
                    groups: b
                        .key
                        .heads(self.width, self.head)
                        .zip(b.value.heads(self.width, self.head))
                        .map(|(a, b)| [a, b])
                        .collect(),
                    output: b.output.as_projection(),
                    rotary,
                    mask: if encoder {
                        SoftmaxMask::Full
                    } else {
                        SoftmaxMask::Causal
                    },
                    tables: SoftmaxTables {
                        probabilities: &self.softmax.out,
                        sums: &self.softmax.sum,
                        limit: self.softmax.sum_scale as i32,
                    },
                    factor: ((1u64 << self.bits) as f64 / libm::sqrt(self.head as f64)).round()
                        as i32,
                },
                mlp: Mlp {
                    gate: b.gate.as_ref().map(Affine::as_projection),
                    up: b.up.as_projection(),
                    down: b.down.as_projection(),
                    activation: Activation {
                        lower: self.activation.0,
                        correction: &self.activation.1,
                    },
                },
                attention_norm: normalize(&b.norm[0]),
                mlp_norm: normalize(&b.norm[1]),
            })
            .collect();
        let qk: Vec<_> = self
            .blocks
            .iter()
            .map(|b| b.qk.as_ref().map(|p| [normalize(&p[0]), normalize(&p[1])]))
            .collect();
        let norm = normalize(&self.norm);
        let architecture = if encoder {
            Architecture::Encoder {
                embedding_norm: norm,
            }
        } else {
            Architecture::Decoder {
                final_norm: norm,
                logits_bias: &self.bias,
            }
        };
        Model::with_query_norms(
            rows,
            self.bits,
            PublicEmbeddings {
                width: self.width,
                tokens: &self.embeddings,
                positions: self.positions.as_deref(),
                segments: self.segments.as_deref(),
            },
            layers,
            architecture,
            &qk,
        )
    }

    fn norm_params<'a>(&self, p: &'a Affine, qwen: bool) -> NormalizationParameters<'a> {
        NormalizationParameters {
            kind: if qwen {
                NormalizationKind::Rms
            } else {
                NormalizationKind::Layer
            },
            gain: &p.w,
            bias: &p.b,
            epsilon_variance_units: self.epsilon,
        }
    }
}
