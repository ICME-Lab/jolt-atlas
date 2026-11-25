#![cfg(all(feature = "akita", feature = "safetensors-weights"))]

use std::path::Path;

use jolt_akita::{AkitaChunkProfile, AkitaScheduleArtifacts, AkitaScheme};
use model_specific_proving::{protocol::AkitaAdviceSetup, weights::ModelWeights};

#[test]
fn proves_gpt2_tiny_from_exported_weights() {
    prove_gpt2(|setup| setup);
}

#[cfg(all(feature = "akita-metal", target_os = "macos"))]
#[test]
fn proves_gpt2_on_metal_and_verifies_on_cpu() {
    prove_gpt2(|setup| setup.with_metal().unwrap());
}

fn prove_gpt2(configure: impl FnOnce(AkitaAdviceSetup) -> AkitaAdviceSetup) {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("../models/gpt2-tiny");
    let tokens = [17, 42];
    let weights = ModelWeights::from_safetensors(&directory, 14).unwrap();
    let model = weights.model(tokens.len()).unwrap();
    let schedules =
        AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2));
    let prover = configure(AkitaAdviceSetup::new(schedules.clone()).unwrap());
    let verifier = AkitaAdviceSetup::new(schedules).unwrap();

    let (mut logits, proof) = model.prove::<AkitaScheme>(&tokens, None, &prover).unwrap();
    assert_eq!(logits.len(), tokens.len() * 1000);
    model
        .verify(&tokens, None, &logits, &proof, &verifier)
        .unwrap();

    logits[0] += 1;
    assert!(model
        .verify(&tokens, None, &logits, &proof, &verifier)
        .is_err());
}
