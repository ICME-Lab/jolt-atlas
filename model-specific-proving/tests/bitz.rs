#![cfg(feature = "safetensors-weights")]

use std::path::Path;

use model_specific_proving::weights::ModelWeights;

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
    let (mut logits, proof) = model.prove(tokens, None).unwrap();
    assert_eq!(logits.len(), tokens.len() * 1000);
    model.verify(tokens, None, &logits, &proof).unwrap();

    logits[0] += 1;
    assert!(model.verify(tokens, None, &logits, &proof).is_err());
}
