//! Cross-stage digests of the checked-in zoo.
//!
//! The zoo tests prove and verify with the same build, so a coordinated
//! prover+verifier change passes them unseen. These tests pin, for each
//! checked-in model on its golden inputs under Akita over Fp128, a
//! blake2b-256 hash of the serialized proof and the prover's final transcript
//! state. A behavior-preserving change leaves every pin untouched; a
//! protocol-breaking change re-pins in its own commit, naming the cause.

use atlas_onnx_import::fixtures::{synth_inputs, GOLDEN_SEED};
use atlas_pcs::{akita::AkitaScheme, commitment_scheme::CommitmentScheme};
use atlas_trace::{Model, Tensor};
use atlas_transcript::Blake2bProver;
use blake2::{digest::consts::U32, Blake2b, Digest};

use super::zoo::{op_gauntlet_inputs, zoo};
use crate::onnx_proof::{
    proof_serialization::serialize_proof, prove, AtlasProverPreprocessing, AtlasSharedPreprocessing,
};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Prove `model` on `inputs` under `PCS` and compare against the pinned
/// `(proof hash, final transcript state)`.
fn assert_digest<PCS: CommitmentScheme>(
    name: &str,
    model: Model,
    inputs: &[Tensor<i32>],
    proof: &str,
    transcript: &str,
) {
    let pp = AtlasProverPreprocessing::<PCS::Field, PCS>::new(
        AtlasSharedPreprocessing::preprocess(model),
    );
    let (p, _, debug) = prove::<PCS::Field, Blake2bProver, PCS>(&pp, inputs);
    let bytes = serialize_proof(&p).expect("proof serializes");
    let got_proof = hex(&Blake2b::<U32>::digest(&bytes));
    let debug = debug.expect("test builds carry debug info");
    let got_transcript = hex(debug
        .transcript_history
        .last()
        .expect("the transcript records its construction"));
    assert_eq!(
        (got_proof.as_str(), got_transcript.as_str()),
        (proof, transcript),
        "{name}: proof digest changed. Re-pin only in a commit declared protocol-breaking"
    );
}

#[test]
fn digest_op_gauntlet() {
    let imported = zoo("op_gauntlet", 16);
    let inputs = op_gauntlet_inputs(&imported);
    assert_digest::<AkitaScheme>(
        "op_gauntlet",
        imported.model,
        &inputs,
        "9e41536504e7d32ffc8e65e3a5b54a1a716d26d28b75253a427bef99277c71af",
        "ca8cf0b598775aea959534b881f07bbc83c12336430e9e8bcfd689573431f9bf",
    );
}

#[test]
fn digest_gpt2_tiny() {
    let imported = zoo("gpt2-tiny", 16);
    let inputs = synth_inputs(&imported, GOLDEN_SEED);
    assert_digest::<AkitaScheme>(
        "gpt2-tiny",
        imported.model,
        &inputs,
        "01f4adc0775389cff1151bdeb05c7b41b046f9b3c4426449d2b99bcbf2fbe510",
        "f63308e4992a94850a906a0a5c7e2f4d478e12317a06e88429140b10e33b2f94",
    );
}

#[test]
fn digest_qwen2_tiny() {
    let imported = zoo("qwen2-tiny", 16);
    let inputs = synth_inputs(&imported, GOLDEN_SEED);
    assert_digest::<AkitaScheme>(
        "qwen2-tiny",
        imported.model,
        &inputs,
        "25b83ac935c6dfd9911721c95970fe88d751743be350fccc4b4442a1e5e4ab16",
        "141ef83fa4463dc5f9ca8221acab1c0b25aed49c426760a2ed4c165a5d764f0f",
    );
}

#[test]
fn digest_bge_tiny() {
    let imported = zoo("bge-tiny", 16);
    let inputs = synth_inputs(&imported, GOLDEN_SEED);
    assert_digest::<AkitaScheme>(
        "bge-tiny",
        imported.model,
        &inputs,
        "42ca418893815c4739bbf76fa5abcef9a306184386888c0e04d306250f8f9549",
        "1a4ac4ffdd508082fe79ff802d51ba53ebfe40d884e883f3f0a4844e27728e2a",
    );
}
