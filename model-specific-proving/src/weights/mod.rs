//! Owned fixed-point weights and parameters for a loaded model.

use std::{error::Error, fs, path::Path};

use atlas_trace::{
    semantics::{
        softmax::{softmax_tables, SoftmaxTables},
        tables::{table, IndexRule, TableKind},
    },
    Scale, SemVersion,
};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;

use crate::models::{bert, gpt2, qwen};

pub(crate) type LoadResult<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Family {
    Gpt2,
    Qwen2,
    Qwen3,
    Bert,
}

pub(crate) struct Affine {
    pub(crate) weights: Vec<i32>,
    pub(crate) bias: Vec<i32>,
}
pub(crate) struct LayerWeights {
    pub(crate) query: Affine,
    pub(crate) key: Affine,
    pub(crate) value: Affine,
    pub(crate) out: Affine,
    pub(crate) up: Affine,
    pub(crate) gate: Option<Affine>,
    pub(crate) down: Affine,
    pub(crate) norm1: Affine,
    pub(crate) norm2: Affine,
    pub(crate) qk_norm: Option<[Affine; 2]>,
}

pub(crate) struct Parameters {
    pub(crate) tokens: Vec<i32>,
    pub(crate) positions: Option<Vec<i32>>,
    pub(crate) segments: Option<Vec<i32>>,
    pub(crate) layers: Vec<LayerWeights>,
    pub(crate) norm: Affine,
    pub(crate) activation: (i32, Vec<i32>),
    pub(crate) rope: (Vec<i32>, Vec<i32>),
}

pub struct ModelWeights {
    pub(crate) family: Family,
    pub(crate) width: usize,
    pub(crate) head_width: usize,
    pub(crate) max_positions: usize,
    pub(crate) bits: u8,
    pub(crate) parameters: Parameters,
    pub(crate) logits_bias: Vec<i32>,
    pub(crate) softmax: SoftmaxTables,
    pub(crate) epsilon_variance_units: i64,
    pub(crate) factor: i32,
}

pub(crate) struct Tensors<'a> {
    data: SafeTensors<'a>,
    pub(crate) scale: Scale,
}

impl Tensors<'_> {
    pub(crate) fn has(&self, name: &str) -> bool {
        self.data.tensor(name).is_ok()
    }

    pub(crate) fn read(&self, name: &str, shape: &[usize]) -> LoadResult<Vec<i32>> {
        let tensor = self.data.tensor(name)?;
        if tensor.shape() != shape {
            return Err(format!("wrong shape for tensor {name}").into());
        }
        let scale = f64::from(self.scale.denom_i32());
        let quantize = |x: f32| {
            let fixed = (f64::from(x) * scale).round();
            if (f64::from(i32::MIN)..=f64::from(i32::MAX)).contains(&fixed) {
                Ok(fixed as i32)
            } else {
                Err(format!("invalid weight value in {name}").into())
            }
        };
        let bytes = tensor.data();
        match tensor.dtype() {
            Dtype::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| quantize(f32::from_le_bytes(b)))
                .collect(),
            Dtype::F16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&b| quantize(half::f16::from_le_bytes(b).to_f32()))
                .collect(),
            Dtype::BF16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&b| quantize(half::bf16::from_le_bytes(b).to_f32()))
                .collect(),
            _ => Err(format!("unsupported dtype for tensor {name}").into()),
        }
    }

    pub(crate) fn affine(
        &self,
        name: &str,
        shape: &[usize],
        output: usize,
        bias: bool,
    ) -> LoadResult<Affine> {
        Ok(Affine {
            weights: self.read(&format!("{name}.weight"), shape)?,
            bias: if bias {
                self.read(&format!("{name}.bias"), &[output])?
            } else {
                vec![0; output]
            },
        })
    }

    pub(crate) fn norm(&self, name: &str, width: usize, bias: bool) -> LoadResult<Affine> {
        self.affine(name, &[width], width, bias)
    }

    pub(crate) fn embeddings(&self, name: &str, shape: &[usize]) -> LoadResult<Vec<i32>> {
        let tokens = self.read(name, shape)?;
        if self.has("lm_head.weight") && self.read("lm_head.weight", shape)? != tokens {
            return Err("untied logits weights".into());
        }
        Ok(tokens)
    }
}

#[derive(Deserialize)]
pub(crate) struct Config {
    pub(crate) model_type: String,
    #[serde(alias = "n_embd")]
    pub(crate) hidden_size: usize,
    #[serde(alias = "n_head")]
    pub(crate) num_attention_heads: usize,
    #[serde(alias = "n_layer")]
    pub(crate) num_hidden_layers: usize,
    #[serde(alias = "n_positions")]
    pub(crate) max_position_embeddings: usize,
    #[serde(default, alias = "n_inner")]
    pub(crate) intermediate_size: Option<usize>,
    #[serde(alias = "activation_function")]
    pub(crate) hidden_act: String,
    pub(crate) vocab_size: usize,
    #[serde(default)]
    pub(crate) type_vocab_size: usize,
    #[serde(default)]
    pub(crate) num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub(crate) head_dim: Option<usize>,
    #[serde(default)]
    pub(crate) attention_bias: bool,
    #[serde(default)]
    pub(crate) layer_types: Option<Vec<String>>,
    #[serde(
        default,
        alias = "layer_norm_epsilon",
        alias = "layer_norm_eps",
        alias = "rms_norm_eps"
    )]
    pub(crate) epsilon: f64,
    #[serde(default)]
    pub(crate) rope_theta: Option<f64>,
    #[serde(default)]
    pub(crate) rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    pub(crate) use_sliding_window: bool,
    #[serde(default)]
    pub(crate) tie_word_embeddings: Option<bool>,
    #[serde(default = "default_scale_attn_weights")]
    pub(crate) scale_attn_weights: bool,
    #[serde(default)]
    pub(crate) scale_attn_by_inverse_layer_idx: bool,
    #[serde(default)]
    pub(crate) reorder_and_upcast_attn: bool,
    #[serde(default)]
    pub(crate) position_embedding_type: Option<String>,
    #[serde(default)]
    pub(crate) is_decoder: bool,
}

impl Config {
    pub(crate) fn head_width(&self) -> usize {
        if self.model_type == "qwen3" {
            if let Some(width) = self.head_dim {
                return width;
            }
        }
        self.hidden_size / self.num_attention_heads
    }
}

fn default_scale_attn_weights() -> bool {
    true
}

impl ModelWeights {
    pub fn from_safetensors(directory: &Path, bits: u8) -> LoadResult<Self> {
        let cfg: Config = serde_json::from_slice(&fs::read(directory.join("config.json"))?)?;
        type Load = fn(&Config, &Tensors<'_>, usize) -> LoadResult<Parameters>;
        let (family, act, load): (_, _, Load) = match cfg.model_type.as_str() {
            "gpt2" => (Family::Gpt2, gpt2::ACTIVATION, gpt2::load),
            "qwen2" => (Family::Qwen2, qwen::ACTIVATION, qwen::load),
            "qwen3" => (Family::Qwen3, qwen::ACTIVATION, qwen::load),
            "bert" => (Family::Bert, bert::ACTIVATION, bert::load),
            _ => return Err("unsupported model family".into()),
        };
        if bits > 15
            || cfg.hidden_act != act
            || cfg.hidden_size == 0
            || cfg.num_attention_heads == 0
            || ((family != Family::Qwen3 || cfg.head_dim.is_none())
                && !cfg.hidden_size.is_multiple_of(cfg.num_attention_heads))
            || cfg.head_width() == 0
            || cfg.num_hidden_layers == 0
            || !cfg.epsilon.is_finite()
            || cfg.epsilon <= 0.0
        {
            return Err("unsupported model configuration".into());
        }
        let width = cfg.hidden_size;
        let head = cfg.head_width();
        let hidden = cfg
            .intermediate_size
            .unwrap_or(width.checked_mul(4).ok_or("MLP width overflow")?);
        let scale = Scale::new(bits)?;
        let bytes = fs::read(directory.join("model.safetensors"))?;
        let tensors = Tensors {
            data: SafeTensors::deserialize(&bytes)?,
            scale,
        };
        let parameters = load(&cfg, &tensors, hidden)?;
        let denom = f64::from(scale.denom_i32());
        let epsilon_variance_units = (cfg.epsilon * (denom * denom)).round().max(1.0);
        if epsilon_variance_units >= i64::MAX as f64 {
            return Err("normalization epsilon overflow".into());
        }
        let factor = (denom / libm::sqrt(head as f64)).round() as i32;
        let logits_bias = vec![0; cfg.vocab_size];
        let softmax = softmax_tables(scale, SemVersion::V3);
        Ok(Self {
            family,
            width,
            head_width: head,
            max_positions: cfg.max_position_embeddings,
            bits,
            parameters,
            logits_bias,
            softmax,
            epsilon_variance_units: epsilon_variance_units as i64,
            factor,
        })
    }
}

pub(crate) fn transpose(values: &[i32], rows: usize, cols: usize) -> Vec<i32> {
    (0..cols)
        .flat_map(|j| (0..rows).map(move |i| values[i * cols + j]))
        .collect()
}

pub(crate) fn activation_table(kind: TableKind, scale: Scale) -> LoadResult<(i32, Vec<i32>)> {
    let spec = table(kind, scale, SemVersion::V3);
    let IndexRule::SymClamp { bound_log } = spec.index else {
        return Err("expected a clamped table".into());
    };
    Ok((-(1 << bound_log), spec.entries))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantizes_qwen2_epsilon_in_variance_units() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../models/qwen2-tiny");
        let weights = ModelWeights::from_safetensors(&root, 14).unwrap();
        assert_eq!(weights.epsilon_variance_units, 268);
    }

    #[test]
    fn gpt2_defaults_to_scaled_attention() {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("../models/gpt2-tiny");
        let mut json: serde_json::Value =
            serde_json::from_slice(&fs::read(directory.join("config.json")).unwrap()).unwrap();
        json.as_object_mut().unwrap().remove("scale_attn_weights");
        let mut cfg: Config = serde_json::from_value(json).unwrap();
        let bytes = fs::read(directory.join("model.safetensors")).unwrap();
        let tensors = Tensors {
            data: SafeTensors::deserialize(&bytes).unwrap(),
            scale: Scale::new(14).unwrap(),
        };
        let hidden = cfg.intermediate_size.unwrap_or(4 * cfg.hidden_size);
        assert!(gpt2::load(&cfg, &tensors, hidden).is_ok());
        cfg.scale_attn_weights = false;
        assert_eq!(
            gpt2::load(&cfg, &tensors, hidden)
                .err()
                .unwrap()
                .to_string(),
            "unsupported GPT-2 score scaling"
        );
    }
}
