#![cfg(feature = "safetensors-weights")]

use std::{io::Cursor, path::Path};

use model_specific_proving::{protocol::TopKConfig, weights::ModelWeights};

#[test]
fn proves_gpt2_tiny_with_world_bitz() {
    prove("gpt2-tiny", &[17, 42]);
}

#[test]
fn proves_qwen2_tiny_with_world_bitz() {
    prove("qwen2-tiny", &[17, 42, 23]);
}

fn prove(name: &str, tokens: &[usize]) {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../models")
        .join(name);
    let weights = ModelWeights::from_safetensors(&directory, 14).unwrap();
    let model = weights.model(tokens.len()).unwrap();
    let cfg = TopKConfig {
        positions: 0..tokens.len(),
        k: 4,
        rows_per_block: 2,
        vocab_per_block: 256,
    };
    let mut proof = Cursor::new(Vec::new());
    let mut output = model.prove_top_k_to(tokens, &cfg, &mut proof).unwrap();
    assert_eq!(output.len(), tokens.len());
    model
        .verify_top_k_from(tokens, &cfg, &output, &mut proof)
        .unwrap();
    output[0][0].logit += 1;
    assert!(model
        .verify_top_k_from(tokens, &cfg, &output, proof)
        .is_err());
}
