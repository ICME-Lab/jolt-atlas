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
        "6139d1c916384ae22b56de3973e4f3efa3f10e36ec4faf35c596aa11d448f64d",
        "94dff6fdf6d6613ed7af53edaeca27bccb980e9f9c165867f7b737ba968c95d3",
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
        "f4b4662e9705cdb2fb1961b4eb0417e8f4320767a0b168832faada1ce5974856",
        "8d331c6ef89b740060517cd017bf48878474dff60c98eabfb952af224b12bae5",
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
        "c06a9cf252764b24ca3e7822891df02f347df04f111bbd9f4c0b89812f8d4003",
        "f8d1e285eac0e9ff0b5f1513e5f9c94e24ed90151919570e191ac2e500713b24",
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
        "e897d1d7b0adc82f5bc3e6e4eb6cbb2ac2c4ed876026b24461b69a79a68498b2",
        "cd06ad35885d65e370b6a1256812809464d343eebce1379c87df935a9a36283b",
    );
}
