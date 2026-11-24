//! BERT normalizes the sum of token, position, and segment embeddings before the encoder.
//! Its layers use bidirectional attention, a GELU MLP, and LayerNorm after each residual addition.

use atlas_trace::semantics::tables::TableKind;

use super::{Architecture, Model};
use crate::layers::NormalizationKind;
use crate::protocol::Result;
use crate::weights::{
    activation_table, Config, LayerWeights, LoadResult, ModelWeights, Parameters, Tensors,
};

pub(crate) const ACTIVATION: &str = "gelu";

pub(super) fn model(weights: &ModelWeights, rows: usize) -> Result<Model<'_>> {
    let kind = NormalizationKind::Layer;
    let embedding_norm = weights.normalization(&weights.parameters.norm, kind);
    weights.assemble(rows, kind, None, Architecture::Encoder { embedding_norm })
}

pub(crate) fn load(cfg: &Config, tensors: &Tensors<'_>, hidden: usize) -> LoadResult<Parameters> {
    if cfg.is_decoder {
        return Err("BERT decoder mode is unsupported".into());
    }
    if cfg
        .position_embedding_type
        .as_deref()
        .is_some_and(|x| x != "absolute")
    {
        return Err("unsupported BERT embeddings".into());
    }
    let width = cfg.hidden_size;
    let prefix = if tensors.has("bert.embeddings.word_embeddings.weight") {
        "bert."
    } else {
        ""
    };
    let tokens = tensors.read(
        &format!("{prefix}embeddings.word_embeddings.weight"),
        &[cfg.vocab_size, width],
    )?;
    let positions = tensors.read(
        &format!("{prefix}embeddings.position_embeddings.weight"),
        &[cfg.max_position_embeddings, width],
    )?;
    let segments = tensors.read(
        &format!("{prefix}embeddings.token_type_embeddings.weight"),
        &[cfg.type_vocab_size, width],
    )?;
    let norm = tensors.norm(&format!("{prefix}embeddings.LayerNorm"), width, true)?;
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let name = format!("{prefix}encoder.layer.{i}");
        let linear = |p, input, output| {
            tensors.affine(&format!("{name}.{p}"), &[output, input], output, true)
        };
        layers.push(LayerWeights {
            query: linear("attention.self.query", width, width)?,
            key: linear("attention.self.key", width, width)?,
            value: linear("attention.self.value", width, width)?,
            out: linear("attention.output.dense", width, width)?,
            up: linear("intermediate.dense", width, hidden)?,
            down: linear("output.dense", hidden, width)?,
            gate: None,
            norm1: tensors.norm(&format!("{name}.attention.output.LayerNorm"), width, true)?,
            norm2: tensors.norm(&format!("{name}.output.LayerNorm"), width, true)?,
        });
    }
    Ok(Parameters {
        tokens,
        positions: Some(positions),
        segments: Some(segments),
        layers,
        norm,
        activation: activation_table(TableKind::GeluErf, tensors.scale)?,
        rope: Default::default(),
    })
}
