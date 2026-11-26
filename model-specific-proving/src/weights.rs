use std::{error::Error, fs, path::Path};

use atlas_trace::{
    semantics::{
        softmax::{softmax_tables, SoftmaxTables as Tables},
        tables::{table, IndexRule, TableKind},
    },
    Scale, SemVersion,
};
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;

use crate::{
    layers::{
        Activation, Attention, Mlp, NormalizationKind, NormalizationParameters, Projection, Rotary,
        SoftmaxMask, SoftmaxTables,
    },
    models::{Architecture, Layer, Model, PublicEmbeddings},
    protocol::Result,
};

type Load<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Family {
    Gpt2,
    Qwen2,
    Qwen3,
    Bert,
}

struct Affine {
    weights: Vec<i32>,
    bias: Vec<i32>,
}

impl Affine {
    fn projection(&self) -> Projection<'_> {
        Projection {
            weights: &self.weights,
            bias: &self.bias,
        }
    }

    fn heads(&self, input: usize, width: usize) -> impl Iterator<Item = Projection<'_>> {
        self.weights
            .chunks_exact(input * width)
            .zip(self.bias.chunks_exact(width))
            .map(|(weights, bias)| Projection { weights, bias })
    }
}

struct Block {
    qkv: [Affine; 3],
    out: Affine,
    up: Affine,
    down: Affine,
    gate: Option<Affine>,
    norm: [Affine; 2],
    head_norm: Option<[Affine; 2]>,
}

pub struct ModelWeights {
    family: Family,
    width: usize,
    head: usize,
    bits: u8,
    limit: usize,
    epsilon: i64,
    tokens: Vec<i32>,
    positions: Option<Vec<i32>>,
    segments: Option<Vec<i32>>,
    layers: Vec<Block>,
    norm: Affine,
    bias: Vec<i32>,
    table: (i32, Vec<i32>),
    softmax: Tables,
    cos: Vec<i32>,
    sin: Vec<i32>,
}

struct Reader<'a> {
    data: SafeTensors<'a>,
    scale: f64,
}

impl Reader<'_> {
    fn tensor(&self, name: &str, shape: &[usize]) -> Load<Vec<i32>> {
        let tensor = self.data.tensor(name)?;
        if tensor.shape() != shape {
            return Err(format!("incorrect shape for {name}").into());
        }
        let (width, decode): (usize, fn(&[u8]) -> f64) = match tensor.dtype() {
            Dtype::F32 => (4, |x| f32::from_le_bytes(x.try_into().unwrap()).into()),
            Dtype::F64 => (8, |x| f64::from_le_bytes(x.try_into().unwrap())),
            Dtype::BF16 => (2, |x| {
                half::bf16::from_bits(u16::from_le_bytes(x.try_into().unwrap())).to_f64()
            }),
            Dtype::F16 => (2, |x| {
                half::f16::from_bits(u16::from_le_bytes(x.try_into().unwrap())).to_f64()
            }),
            _ => return Err(format!("unsupported tensor type for {name}").into()),
        };
        tensor
            .data()
            .chunks_exact(width)
            .map(|raw| {
                let value = (decode(raw) * self.scale).round();
                if !value.is_finite() || value < f64::from(i32::MIN) || value > f64::from(i32::MAX)
                {
                    return Err(format!("weight outside fixed-point range in {name}").into());
                }
                Ok(value as i32)
            })
            .collect()
    }

    fn affine(
        &self,
        name: &str,
        input: usize,
        output: usize,
        bias: bool,
        transpose: bool,
    ) -> Load<Affine> {
        let key = format!("{name}.weight");
        let weights = if transpose {
            let raw = self.tensor(&key, &[input, output])?;
            (0..output)
                .flat_map(|j| raw.iter().skip(j).step_by(output).copied())
                .collect()
        } else {
            self.tensor(&key, &[output, input])?
        };
        let bias = if bias {
            self.tensor(&format!("{name}.bias"), &[output])?
        } else {
            vec![0; output]
        };
        Ok(Affine { weights, bias })
    }

    fn norm(&self, name: &str, width: usize, bias: bool) -> Load<Affine> {
        let weights = self.tensor(&format!("{name}.weight"), &[width])?;
        let bias = if bias {
            self.tensor(&format!("{name}.bias"), &[width])?
        } else {
            vec![0; width]
        };
        Ok(Affine { weights, bias })
    }
}

fn number(config: &Value, key: &str) -> Load<usize> {
    config[key]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .ok_or_else(|| format!("missing or invalid {key}").into())
}

impl ModelWeights {
    pub fn from_safetensors(directory: &Path, bits: u8) -> Load<Self> {
        if bits > 15 {
            return Err("unsupported fractional precision".into());
        }
        let config: Value = serde_json::from_slice(&fs::read(directory.join("config.json"))?)?;
        let family = match config["model_type"].as_str() {
            Some("gpt2") => Family::Gpt2,
            Some("qwen2") => Family::Qwen2,
            Some("qwen3") => Family::Qwen3,
            Some("bert") => Family::Bert,
            _ => return Err("unsupported model family".into()),
        };
        let gpt = family == Family::Gpt2;
        let bert = family == Family::Bert;
        let qwen = !gpt && !bert;
        let qwen3 = family == Family::Qwen3;
        let width = number(&config, if gpt { "n_embd" } else { "hidden_size" })?;
        let heads = number(&config, if gpt { "n_head" } else { "num_attention_heads" })?;
        let count = number(&config, if gpt { "n_layer" } else { "num_hidden_layers" })?;
        let limit = number(
            &config,
            if gpt {
                "n_positions"
            } else {
                "max_position_embeddings"
            },
        )?;
        let vocab = number(&config, "vocab_size")?;
        let groups = if qwen {
            number(&config, "num_key_value_heads").unwrap_or(heads)
        } else {
            heads
        };
        let head = number(&config, "head_dim").unwrap_or(width / heads);
        let hidden =
            number(&config, if gpt { "n_inner" } else { "intermediate_size" }).unwrap_or(4 * width);
        if head == 0
            || !heads.is_multiple_of(groups)
            || (!qwen && head * heads != width)
            || (qwen
                && (head % 2 != 0
                    || config["tie_word_embeddings"].as_bool() != Some(true)
                    || !config["rope_scaling"].is_null()
                    || config["use_sliding_window"].as_bool() == Some(true)
                    || config["layer_types"].as_array().is_some_and(|types| {
                        types.len() != count
                            || types
                                .iter()
                                .any(|kind| kind.as_str() != Some("full_attention"))
                    })))
            || (gpt
                && (config["tie_word_embeddings"].as_bool() == Some(false)
                    || config["scale_attn_weights"].as_bool() == Some(false)
                    || config["scale_attn_by_inverse_layer_idx"].as_bool() == Some(true)
                    || config["reorder_and_upcast_attn"].as_bool() == Some(true)))
        {
            return Err("unsupported model configuration".into());
        }
        let expected = if qwen {
            "silu"
        } else if bert {
            "gelu"
        } else {
            "gelu_new"
        };
        let activation = config[if gpt {
            "activation_function"
        } else {
            "hidden_act"
        }]
        .as_str();
        if activation.is_some_and(|kind| kind != expected) {
            return Err("unsupported activation".into());
        }
        let epsilon_key = if gpt {
            "layer_norm_epsilon"
        } else if bert {
            "layer_norm_eps"
        } else {
            "rms_norm_eps"
        };
        let scale = Scale::new(bits)?;
        let denom = f64::from(scale.denom_i32());
        let epsilon = (config[epsilon_key]
            .as_f64()
            .ok_or("missing normalization epsilon")?
            * denom
            * denom)
            .round()
            .max(1.0);
        if !epsilon.is_finite() || epsilon >= i64::MAX as f64 {
            return Err("invalid normalization epsilon".into());
        }
        let bytes = fs::read(directory.join("model.safetensors"))?;
        let reader = Reader {
            data: SafeTensors::deserialize(&bytes)?,
            scale: denom,
        };
        let has = |key: &str| reader.data.tensor(key).is_ok();
        let bert_prefix = if has("bert.embeddings.word_embeddings.weight") {
            "bert."
        } else {
            ""
        };
        let gpt_prefix = if has("transformer.wte.weight") {
            "transformer."
        } else {
            ""
        };
        let embeddings = if qwen {
            "model.embed_tokens".to_owned()
        } else if gpt {
            format!("{gpt_prefix}wte")
        } else {
            format!("{bert_prefix}embeddings.word_embeddings")
        };
        let tokens = reader.tensor(&format!("{embeddings}.weight"), &[vocab, width])?;
        if !bert
            && has("lm_head.weight")
            && reader.tensor("lm_head.weight", &[vocab, width])? != tokens
        {
            return Err("untied logits are unsupported".into());
        }
        let positions = if qwen {
            None
        } else {
            let name = if gpt {
                format!("{gpt_prefix}wpe.weight")
            } else {
                format!("{bert_prefix}embeddings.position_embeddings.weight")
            };
            Some(reader.tensor(&name, &[limit, width])?)
        };
        let segments = if bert {
            Some(reader.tensor(
                &format!("{bert_prefix}embeddings.token_type_embeddings.weight"),
                &[number(&config, "type_vocab_size")?, width],
            )?)
        } else {
            None
        };
        let name = if qwen {
            "model.norm".to_owned()
        } else if gpt {
            format!("{gpt_prefix}ln_f")
        } else {
            format!("{bert_prefix}embeddings.LayerNorm")
        };
        let norm = reader.norm(&name, width, !qwen)?;

        let mut layers = Vec::new();
        for level in 0..count {
            let stem = if qwen {
                format!("model.layers.{level}")
            } else if gpt {
                format!("{gpt_prefix}h.{level}")
            } else {
                format!("{bert_prefix}encoder.layer.{level}")
            };
            let projection = |name, input, output, bias| {
                reader.affine(&format!("{stem}.{name}"), input, output, bias, gpt)
            };
            let normal = |name, width, bias| reader.norm(&format!("{stem}.{name}"), width, bias);
            let qkv = if gpt {
                let all = projection("attn.c_attn", width, 3 * width, true)?;
                std::array::from_fn(|i| Affine {
                    weights: all.weights[i * width * width..][..width * width].to_vec(),
                    bias: all.bias[i * width..][..width].to_vec(),
                })
            } else if qwen {
                let bias = !qwen3 || config["attention_bias"].as_bool() == Some(true);
                [
                    projection("self_attn.q_proj", width, heads * head, bias)?,
                    projection("self_attn.k_proj", width, groups * head, bias)?,
                    projection("self_attn.v_proj", width, groups * head, bias)?,
                ]
            } else {
                [
                    projection("attention.self.query", width, width, true)?,
                    projection("attention.self.key", width, width, true)?,
                    projection("attention.self.value", width, width, true)?,
                ]
            };
            let (out, up, down, norms) = match family {
                Family::Gpt2 => ("attn.c_proj", "mlp.c_fc", "mlp.c_proj", ["ln_1", "ln_2"]),
                Family::Bert => (
                    "attention.output.dense",
                    "intermediate.dense",
                    "output.dense",
                    ["attention.output.LayerNorm", "output.LayerNorm"],
                ),
                _ => (
                    "self_attn.o_proj",
                    "mlp.up_proj",
                    "mlp.down_proj",
                    ["input_layernorm", "post_attention_layernorm"],
                ),
            };
            let out = projection(
                out,
                heads * head,
                width,
                !qwen || qwen3 && config["attention_bias"].as_bool() == Some(true),
            )?;
            layers.push(Block {
                qkv,
                out,
                up: projection(up, width, hidden, !qwen)?,
                down: projection(down, hidden, width, !qwen)?,
                gate: qwen
                    .then(|| projection("mlp.gate_proj", width, hidden, false))
                    .transpose()?,
                norm: [
                    normal(norms[0], width, !qwen)?,
                    normal(norms[1], width, !qwen)?,
                ],
                head_norm: qwen3
                    .then(|| {
                        Ok::<_, Box<dyn Error>>([
                            normal("self_attn.q_norm", head, false)?,
                            normal("self_attn.k_norm", head, false)?,
                        ])
                    })
                    .transpose()?,
            });
        }
        let table = if gpt {
            let bound = 8 * scale.denom_i32();
            let factor = libm::sqrt(2.0 / std::f64::consts::PI);
            let entries = (-bound..bound)
                .map(|x| {
                    let n = f64::from(x.abs());
                    let z = n / denom;
                    libm::floor(
                        0.5 - n * (1.0 - libm::tanh(factor * (z + 0.044715 * z * z * z))) / 2.0,
                    ) as i32
                })
                .collect();
            (-bound, entries)
        } else {
            let t = table(
                if qwen {
                    TableKind::Silu
                } else {
                    TableKind::GeluErf
                },
                scale,
                SemVersion::V3,
            );
            let IndexRule::SymClamp { bound_log } = t.index else {
                return Err("invalid activation table".into());
            };
            (-(1i32 << bound_log), t.entries)
        };
        let (cos, sin) = if qwen {
            let theta = config["rope_theta"]
                .as_f64()
                .filter(|x| x.is_finite() && *x > 0.0)
                .ok_or("invalid RoPE theta")?;
            let frequency: Vec<_> = (0..head / 2)
                .map(|i| libm::pow(theta, -((2 * i) as f64) / head as f64))
                .collect();
            (0..limit)
                .flat_map(|i| frequency.iter().map(move |&w| w * i as f64))
                .map(|angle| {
                    let (sin, cos) = libm::sincos(angle);
                    ((cos * denom).round() as i32, (sin * denom).round() as i32)
                })
                .unzip()
        } else {
            (Vec::new(), Vec::new())
        };
        Ok(Self {
            family,
            width,
            head,
            bits,
            limit,
            epsilon: epsilon as i64,
            tokens,
            positions,
            segments,
            norm,
            layers,
            bias: vec![0; vocab],
            table,
            softmax: softmax_tables(scale, SemVersion::V3),
            cos,
            sin,
        })
    }

    pub fn model(&self, rows: usize) -> Result<Model<'_>> {
        if rows == 0 || rows > self.limit {
            return Err("invalid sequence length");
        }
        let bert = self.family == Family::Bert;
        let qwen = matches!(self.family, Family::Qwen2 | Family::Qwen3);
        fn norm(p: &Affine, qwen: bool, epsilon: i64) -> NormalizationParameters<'_> {
            NormalizationParameters {
                kind: if qwen {
                    NormalizationKind::Rms
                } else {
                    NormalizationKind::Layer
                },
                gain: &p.weights,
                bias: &p.bias,
                epsilon_variance_units: epsilon,
            }
        }
        let architecture = if bert {
            Architecture::Encoder {
                embedding_norm: norm(&self.norm, qwen, self.epsilon),
            }
        } else {
            Architecture::Decoder {
                final_norm: norm(&self.norm, qwen, self.epsilon),
                logits_bias: &self.bias,
            }
        };
        let rope = if qwen {
            let len = rows * self.head / 2;
            Some(Rotary {
                cos: &self.cos[..len],
                sin: &self.sin[..len],
            })
        } else {
            None
        };
        let mask = if bert {
            SoftmaxMask::Full
        } else {
            SoftmaxMask::Causal
        };
        let factor = (f64::from(1 << self.bits) / libm::sqrt(self.head as f64)).round() as i32;
        let tables = SoftmaxTables {
            probabilities: &self.softmax.out,
            sums: &self.softmax.sum,
            limit: self.softmax.sum_scale as i32,
        };
        let layers = self
            .layers
            .iter()
            .map(|layer| {
                let [q, k, v] = &layer.qkv;
                Layer {
                    attention: Attention {
                        queries: q.heads(self.width, self.head).collect(),
                        groups: k
                            .heads(self.width, self.head)
                            .zip(v.heads(self.width, self.head))
                            .map(|(k, v)| [k, v])
                            .collect(),
                        output: layer.out.projection(),
                        rotary: rope,
                        mask,
                        tables,
                        factor,
                    },
                    mlp: Mlp {
                        up: layer.up.projection(),
                        gate: layer.gate.as_ref().map(Affine::projection),
                        down: layer.down.projection(),
                        activation: Activation {
                            lower: self.table.0,
                            correction: &self.table.1,
                        },
                    },
                    attention_norm: norm(&layer.norm[0], qwen, self.epsilon),
                    mlp_norm: norm(&layer.norm[1], qwen, self.epsilon),
                }
            })
            .collect();
        let qk: Vec<_> = self
            .layers
            .iter()
            .map(|layer| {
                layer.head_norm.as_ref().map(|heads| {
                    [
                        norm(&heads[0], true, self.epsilon),
                        norm(&heads[1], true, self.epsilon),
                    ]
                })
            })
            .collect();
        Model::with_query_norms(
            rows,
            self.bits,
            PublicEmbeddings {
                width: self.width,
                tokens: &self.tokens,
                positions: self.positions.as_deref(),
                segments: self.segments.as_deref(),
            },
            layers,
            architecture,
            &qk,
        )
    }
}
