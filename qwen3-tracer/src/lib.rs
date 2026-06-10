use std::path::Path;

use qwen3_common::{
    LayerRawWitness as CommonLayerRawWitness, LayerShape, LayerWeights as CommonLayerWeights,
};
pub use qwen3_common::{
    LayerRawWitness, LayerWeights, TraceError, TraceLayerRawInput, layer_raw_input_from_trace_dir,
    read_layer_weights,
};
use qwen3_prover::{layer::LayerProverInput, layer_input::layer_prover_input};

pub struct TraceLayerInput {
    pub shape: LayerShape,
    pub hidden_out: Vec<i32>,
    pub weights: CommonLayerWeights,
    pub raw_witness: CommonLayerRawWitness,
    pub input: LayerProverInput,
}

pub fn layer_input_from_trace_dir(
    trace_dir: impl AsRef<Path>,
    q8_cache: impl AsRef<Path>,
    layer: usize,
) -> Result<TraceLayerInput, TraceError> {
    let raw = layer_raw_input_from_trace_dir(trace_dir, q8_cache, layer)?;
    let input = layer_prover_input(raw.shape, raw.weights.clone(), raw.raw_witness.clone())
        .ok_or(TraceError::LayerInput)?;
    Ok(TraceLayerInput {
        shape: raw.shape,
        hidden_out: raw.hidden_out,
        weights: raw.weights,
        raw_witness: raw.raw_witness,
        input,
    })
}
