//! GPT-2 uses learned positional embeddings and a decoder with pre-LayerNorm.
//! Each layer uses causal attention followed by a GELU MLP. Final logits reuse the token embeddings.

use atlas_trace::Scale;

use super::Model;
use crate::layers::NormalizationKind;
use crate::protocol::Result;
use crate::weights::{
    transpose, Affine, Config, LayerWeights, LoadResult, ModelWeights, Parameters, Tensors,
};

pub(crate) const ACTIVATION: &str = "gelu_new";

pub(super) fn model(weights: &ModelWeights, rows: usize) -> Result<Model<'_>> {
    let kind = NormalizationKind::Layer;
    weights.assemble(rows, kind, None, weights.decoder(kind))
}

pub(crate) fn load(cfg: &Config, tensors: &Tensors<'_>, hidden: usize) -> LoadResult<Parameters> {
    if !cfg.scale_attn_weights || cfg.scale_attn_by_inverse_layer_idx || cfg.reorder_and_upcast_attn
    {
        return Err("unsupported GPT-2 score scaling".into());
    }
    let width = cfg.hidden_size;
    if cfg.tie_word_embeddings == Some(false) {
        return Err("untied GPT-2 logits".into());
    }
    let prefix = if tensors.has("transformer.wte.weight") {
        "transformer."
    } else {
        ""
    };
    let tokens = tensors.embeddings(&format!("{prefix}wte.weight"), &[cfg.vocab_size, width])?;
    let positions = tensors.read(
        &format!("{prefix}wpe.weight"),
        &[cfg.max_position_embeddings, width],
    )?;
    let norm = tensors.norm(&format!("{prefix}ln_f"), width, true)?;
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let name = format!("{prefix}h.{i}");
        let linear = |p: &str, input, output| -> LoadResult<_> {
            let mut a = tensors.affine(&format!("{name}.{p}"), &[input, output], output, true)?;
            a.weights = transpose(&a.weights, input, output);
            Ok(a)
        };
        let qkv = linear("attn.c_attn", width, 3 * width)?;
        let part = |i: usize| Affine {
            weights: qkv.weights[i * width * width..][..width * width].to_vec(),
            bias: qkv.bias[i * width..][..width].to_vec(),
        };
        layers.push(LayerWeights {
            query: part(0),
            key: part(1),
            value: part(2),
            out: linear("attn.c_proj", width, width)?,
            up: linear("mlp.c_fc", width, hidden)?,
            down: linear("mlp.c_proj", hidden, width)?,
            gate: None,
            norm1: tensors.norm(&format!("{name}.ln_1"), width, true)?,
            norm2: tensors.norm(&format!("{name}.ln_2"), width, true)?,
        });
    }
    Ok(Parameters {
        tokens,
        positions: Some(positions),
        segments: None,
        layers,
        norm,
        activation: activation(tensors.scale),
        rope: Default::default(),
    })
}

fn activation(scale: Scale) -> (i32, Vec<i32>) {
    let bound = 8 * scale.denom_i32();
    let denom = f64::from(scale.denom_i32());
    let factor = libm::sqrt(2.0 / std::f64::consts::PI);
    let entries = (-bound..bound)
        .map(|x| {
            let magnitude = f64::from(x.abs());
            let u = magnitude / denom;
            let c = (1.0 - libm::tanh(factor * (u + 0.044715 * u * u * u))) / 2.0;
            libm::floor(0.5 - magnitude * c) as i32
        })
        .collect();
    (-bound, entries)
}
