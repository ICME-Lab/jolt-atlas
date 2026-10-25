//! Cross-stage digests of the checked-in zoo.
//!
//! The zoo tests prove and verify with the same build, so a coordinated
//! prover+verifier change passes them unseen. These tests pin, for each
//! checked-in model on its golden inputs under HyperKZG (seeded setup), a
//! blake2b-256 hash of the serialized proof and the prover's final transcript
//! state. A behavior-preserving change leaves every pin untouched; a
//! protocol-breaking change re-pins in its own commit, naming the cause.

use ark_bn254::Bn254;
use atlas_onnx_import::fixtures::{synth_inputs, GOLDEN_SEED};
use atlas_trace::{Model, Tensor};
use blake2::{digest::consts::U32, Blake2b, Digest};
use joltworks::{
    poly::commitment::{commitment_scheme::CommitmentScheme, hyperkzg::HyperKZG},
    transcripts::Blake2bProver,
};

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
    assert_digest::<HyperKZG<Bn254>>(
        "op_gauntlet",
        imported.model,
        &inputs,
        "08daab6eca3076cd450743d2e033cf884c0180b36e41d6e4ce787dc2a538c3c8",
        "3471d8739b78f6afe23928799236c3f1c283f194a652895a7ad1ff65fe98ad4e",
    );
}

#[test]
fn digest_gpt2_tiny() {
    let imported = zoo("gpt2-tiny", 16);
    let inputs = synth_inputs(&imported, GOLDEN_SEED);
    assert_digest::<HyperKZG<Bn254>>(
        "gpt2-tiny",
        imported.model,
        &inputs,
        "208ba1fb6ac3d2d4a900f4ca2eb3e04afaf03fb4570e23c09d03725062d5b59d",
        "8d5076327b6d7bd6aca09536712aca96aecc46edf5f1c6133718711a7c3c953e",
    );
}

#[test]
fn digest_qwen2_tiny() {
    let imported = zoo("qwen2-tiny", 16);
    let inputs = synth_inputs(&imported, GOLDEN_SEED);
    assert_digest::<HyperKZG<Bn254>>(
        "qwen2-tiny",
        imported.model,
        &inputs,
        "724b2b0c7e964cc6bb4c3c493ddb1649c7c63bcf35c823491dda616fa349a3c4",
        "a987395ea7d1d1251fa57d40b90650519d0dfb7c6395d393c06544f59b8b5496",
    );
}

#[test]
fn digest_bge_tiny() {
    let imported = zoo("bge-tiny", 16);
    let inputs = synth_inputs(&imported, GOLDEN_SEED);
    assert_digest::<HyperKZG<Bn254>>(
        "bge-tiny",
        imported.model,
        &inputs,
        "e26e5010c8f1794ecf07d2792e6cb5d470ed080bb78737ab0ebb4aac22d56ac2",
        "f9ec5d1cec3774dcd7e7e7a10ba95711b59f20f311497d61971b1109275df27c",
    );
}
