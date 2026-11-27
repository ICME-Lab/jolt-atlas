#![cfg(feature = "safetensors-weights")]

use model_specific_proving::{protocol::TopKConfig, weights::ModelWeights};
use std::{io::Cursor, path::PathBuf};

fn model(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../models")
        .join(name)
}

#[test]
fn proves_qwen3_with_head_normalization() {
    let weights = ModelWeights::from_safetensors(&model("qwen3-tiny"), 14).unwrap();
    let tokens = [17, 42, 23, 61];
    let decoder = weights.model(tokens.len()).unwrap();
    let cfg = TopKConfig {
        positions: 0..4,
        k: 4,
        rows_per_block: 4,
        vocab_per_block: 256,
    };
    let mut proof = Cursor::new(Vec::new());
    let output = decoder.prove_top_k_to(&tokens, &cfg, &mut proof).unwrap();
    assert_eq!(output.len(), tokens.len());
    decoder
        .verify_top_k_from(&tokens, &cfg, &output, proof)
        .unwrap();
}

#[test]
fn proves_bert_with_post_normalization() {
    let weights = ModelWeights::from_safetensors(&model("bge-tiny"), 14).unwrap();
    let encoder = weights.model(2).unwrap();
    let tokens = [17, 42];
    let segments = [0, 1];
    let mut proof = Cursor::new(Vec::new());
    let output = encoder
        .prove_to(&tokens, Some(&segments), &mut proof)
        .unwrap();
    assert_eq!(encoder.cls(&output).unwrap().len(), 48);
    encoder
        .verify_from(&tokens, Some(&segments), &output, proof)
        .unwrap();
}
