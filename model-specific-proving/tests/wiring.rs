#![cfg(all(feature = "akita", feature = "safetensors-weights"))]

use jolt_akita::{AkitaChunkProfile, AkitaScheduleArtifacts, AkitaScheme};
use model_specific_proving::{models::ModelWeights, protocol::AkitaAdviceSetup};
use std::path::PathBuf;

fn model(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../models")
        .join(name)
}

fn setup() -> AkitaAdviceSetup {
    let artifacts =
        AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2));
    AkitaAdviceSetup::new(artifacts).unwrap()
}

#[test]
fn proves_qwen2_with_rotary_grouped_attention_and_gated_mlp() {
    let weights = ModelWeights::from_safetensors(&model("qwen2-tiny"), 14).unwrap();
    let decoder = weights.model(3).unwrap();
    let tokens = [17, 42, 23];
    let setup = setup();
    let (output, proof) = decoder.prove::<AkitaScheme>(&tokens, None, &setup).unwrap();
    assert_eq!(output.len() % tokens.len(), 0);
    decoder
        .verify(&tokens, None, &output, &proof, &setup)
        .unwrap();
}

#[test]
fn proves_bert_with_post_normalization() {
    let weights = ModelWeights::from_safetensors(&model("bge-tiny"), 14).unwrap();
    let encoder = weights.model(2).unwrap();
    let tokens = [17, 42];
    let segments = [0, 1];
    let setup = setup();
    let (output, proof) = encoder
        .prove::<AkitaScheme>(&tokens, Some(&segments), &setup)
        .unwrap();
    assert_eq!(encoder.cls(&output).unwrap().len(), 48);
    encoder
        .verify(&tokens, Some(&segments), &output, &proof, &setup)
        .unwrap();
}
