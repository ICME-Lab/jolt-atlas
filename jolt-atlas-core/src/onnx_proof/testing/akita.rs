//! End-to-end Akita proofs for the checked-in and downloaded ONNX models.

use super::zoo::zoo;
use crate::onnx_proof::akita::{prove, verify};
use atlas_onnx_import::fixtures::{synth_inputs, GOLDEN_SEED};

fn roundtrip(name: &str, input_length: usize, outputs: usize) {
    let imported = zoo(name, input_length);
    let inputs = synth_inputs(&imported, GOLDEN_SEED);
    let (proof, io) = prove(&imported.model, &inputs).expect("Akita proof generation failed");
    assert_eq!(io.outputs.len(), outputs);
    verify(&proof, &imported.model, &io).expect("Akita proof verification failed");
}

#[test]
fn bge_tiny() {
    roundtrip("bge-tiny", 16, 1);
}

#[test]
fn gpt2_tiny() {
    roundtrip("gpt2-tiny", 16, 5);
}

#[test]
fn qwen2_tiny() {
    roundtrip("qwen2-tiny", 16, 1);
}

#[test]
#[ignore = "requires scripts/download_bge_small_en_v1_5.py"]
fn bge_small_en_v1_5() {
    roundtrip("bge-small-en-v1.5", 16, 1);
}

#[test]
#[ignore = "requires scripts/download_gpt2.py"]
fn gpt2() {
    roundtrip("gpt2", 16, 25);
}

#[test]
#[ignore = "requires scripts/download_qwen.py"]
fn qwen() {
    roundtrip("qwen", 8, 1);
}
