#![cfg(feature = "safetensors-weights")]

use model_specific_proving::weights::ModelWeights;
use std::path::PathBuf;

fn model(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../models")
        .join(name)
}

#[test]
fn proves_qwen2_with_rotary_grouped_attention_and_gated_mlp() {
    let weights = ModelWeights::from_safetensors(&model("qwen2-tiny"), 14).unwrap();
    let decoder = weights.model(3).unwrap();
    let tokens = [17, 42, 23];
    let (output, proof) = decoder.prove(&tokens, None).unwrap();
    assert_eq!(output.len() % tokens.len(), 0);
    decoder.verify(&tokens, None, &output, &proof).unwrap();
}

#[test]
fn proves_qwen3_with_head_normalization() {
    let weights = ModelWeights::from_safetensors(&model("qwen3-tiny"), 14).unwrap();
    let tokens = [17, 42, 23, 61];
    let decoder = weights.model(tokens.len()).unwrap();
    let (output, proof) = decoder.prove(&tokens, None).unwrap();
    assert_eq!(output.len(), tokens.len() * 1024);
    decoder.verify(&tokens, None, &output, &proof).unwrap();
}

#[test]
fn proves_bert_with_post_normalization() {
    let weights = ModelWeights::from_safetensors(&model("bge-tiny"), 14).unwrap();
    let encoder = weights.model(2).unwrap();
    let tokens = [17, 42];
    let segments = [0, 1];
    let (output, proof) = encoder.prove(&tokens, Some(&segments)).unwrap();
    assert_eq!(encoder.cls(&output).unwrap().len(), 48);
    encoder
        .verify(&tokens, Some(&segments), &output, &proof)
        .unwrap();
}
