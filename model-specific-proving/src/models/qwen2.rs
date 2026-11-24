//! Qwen2 uses a pre-RMSNorm decoder with causal grouped-query attention and RoPE.
//! Its MLP multiplies the SiLU-activated gate projection by the up projection.

use atlas_trace::{semantics::tables::TableKind, Scale};

use super::Model;
use crate::layers::{NormalizationKind, Rotary};
use crate::protocol::Result;
use crate::weights::{
    activation_table, Config, LayerWeights, LoadResult, ModelWeights, Parameters, Tensors,
};

pub(crate) const ACTIVATION: &str = "silu";

pub(super) fn model(weights: &ModelWeights, rows: usize) -> Result<Model<'_>> {
    let len = rows * (weights.head_width / 2);
    let rotary = Rotary {
        cos: &weights.parameters.rope.0[..len],
        sin: &weights.parameters.rope.1[..len],
    };
    let kind = NormalizationKind::Rms;
    weights.assemble(rows, kind, Some(rotary), weights.decoder(kind))
}

pub(crate) fn load(cfg: &Config, tensors: &Tensors<'_>, hidden: usize) -> LoadResult<Parameters> {
    let head = cfg.hidden_size / cfg.num_attention_heads;
    let groups = cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads);
    let theta = cfg.rope_theta.unwrap_or(0.0);
    if groups == 0
        || !cfg.num_attention_heads.is_multiple_of(groups)
        || !head.is_multiple_of(2)
        || cfg.tie_word_embeddings != Some(true)
        || cfg.use_sliding_window
        || cfg.rope_scaling.is_some()
        || !theta.is_finite()
        || theta <= 0.0
    {
        return Err("unsupported Qwen2 attention".into());
    }
    let width = cfg.hidden_size;
    let kv = groups * head;
    let tokens = tensors.embeddings("model.embed_tokens.weight", &[cfg.vocab_size, width])?;
    let norm = tensors.norm("model.norm", width, false)?;
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let name = format!("model.layers.{i}");
        let linear = |p, input, output, bias| {
            tensors.affine(&format!("{name}.{p}"), &[output, input], output, bias)
        };
        layers.push(LayerWeights {
            query: linear("self_attn.q_proj", width, width, true)?,
            key: linear("self_attn.k_proj", width, kv, true)?,
            value: linear("self_attn.v_proj", width, kv, true)?,
            out: linear("self_attn.o_proj", width, width, false)?,
            up: linear("mlp.up_proj", width, hidden, false)?,
            down: linear("mlp.down_proj", hidden, width, false)?,
            gate: Some(linear("mlp.gate_proj", width, hidden, false)?),
            norm1: tensors.norm(&format!("{name}.input_layernorm"), width, false)?,
            norm2: tensors.norm(&format!("{name}.post_attention_layernorm"), width, false)?,
        });
    }
    Ok(Parameters {
        tokens,
        positions: None,
        segments: None,
        layers,
        norm,
        activation: activation_table(TableKind::Silu, tensors.scale)?,
        rope: rope(theta, cfg.max_position_embeddings, head, tensors.scale),
    })
}

fn rope(theta: f64, positions: usize, head: usize, scale: Scale) -> (Vec<i32>, Vec<i32>) {
    let denom = f64::from(scale.denom_i32());
    let freq: Vec<_> = (0..head / 2)
        .map(|i| libm::pow(theta, -((2 * i) as f64) / head as f64))
        .collect();
    (0..positions)
        .flat_map(|r| freq.iter().map(move |f| r as f64 * f))
        .map(|angle| {
            let (s, c) = libm::sincos(angle);
            ((denom * c).round() as i32, (denom * s).round() as i32)
        })
        .unzip()
}
