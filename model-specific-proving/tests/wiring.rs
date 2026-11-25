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
fn proves_qwen3_with_head_normalization() {
    let weights = ModelWeights::from_safetensors(&model("qwen3-tiny"), 14).unwrap();
    let tokens = [17, 42, 23, 61];
    let decoder = weights.model(tokens.len()).unwrap();
    let setup = setup();
    let (output, proof) = decoder.prove::<AkitaScheme>(&tokens, None, &setup).unwrap();
    assert_eq!(output.len(), tokens.len() * 1024);
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

#[cfg(feature = "akita-cluster")]
#[test]
fn proves_model_with_two_akita_worker_processes() {
    use std::{
        io::{BufRead, BufReader},
        process::{Child, Command, Stdio},
    };

    struct Worker(Child);

    impl Drop for Worker {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    let mut workers = Vec::new();
    let mut addresses = Vec::new();
    for _ in 0..2 {
        let mut worker = Worker(
            Command::new(env!("CARGO_BIN_EXE_akita-worker"))
                .arg("127.0.0.1:0")
                .env("RAYON_NUM_THREADS", "2")
                .stdout(Stdio::piped())
                .spawn()
                .unwrap(),
        );
        let mut address = String::new();
        BufReader::new(worker.0.stdout.take().unwrap())
            .read_line(&mut address)
            .unwrap();
        addresses.push(address.trim().parse().unwrap());
        workers.push(worker);
    }

    let weights = ModelWeights::from_safetensors(&model("qwen2-tiny"), 14).unwrap();
    let tokens = [17, 42, 23];
    let decoder = weights.model(tokens.len()).unwrap();
    let prover = setup().with_workers(addresses).unwrap();
    let (output, proof) = decoder
        .prove::<AkitaScheme>(&tokens, None, &prover)
        .unwrap();
    decoder
        .verify(&tokens, None, &output, &proof, &setup())
        .unwrap();
}
