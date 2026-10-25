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
        "e9d9c90c04f8fceb424310e50ad6eb989440905c6a69b3fff948f0be80cc8c8a",
        "9c5136552235aed751d980eba7503f74b3a5217277d9bb88829d0f3ba223d42f",
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
        "dfce3e35fe224f3aaaa2ebe7e18d7bd6149f94eade325f952fb2e07d1b5a3fb3",
        "f7e4a89b3ce0edf93f9978b3034c26d44eb84badfd006a1e916efb345197d71f",
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
        "e2e055992052dd00641312dbd6ef85e126ba702bf9dac73989dfcdfc8ae2884b",
        "07cbe9a66a30b5491099ba3558590136737a3949d719d684990fd05e3215f8fb",
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
        "0831474f9dd616d817e094d7f7a3a580cad51a2004e6d92d2e789ad50a2cd513",
        "de9983ca3adb215f3bd8ebc065d6c35fa8f5241758b7992572063b952e36e350",
    );
}
