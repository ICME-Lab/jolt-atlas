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
        "8f7a91aa7245e41fd7fa5312bdbc977488aef676a6d726fe1ee6efe579616885",
        "b92b7b6d520a82b8614eeb05acb83749798a615cb33a427ec1b93731d9bed0b9",
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
        "48d8fa08366b5bc03c0bd87e58a10f8937a3888d4be327c75f34b72483085e46",
        "f0800348ff51f2be3d5dcae1808b01321208812644a682cbf29e22e0244083d3",
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
        "854aa46a899bdb39883ec817d1402e9ca8e9b4d772d8ddf88550dcdd4464bec1",
        "a595259cef944ca0a45797f7d767758e33176552e5c22f0699695c8b0daffd8f",
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
        "cf31c4bd74a502e9011c3a84282af4ea8c335f78d5f564e266dcf6a864675191",
        "8badde1c6ea879538de234ea1ee9da406adea64e145ec7dc282734a656d91097",
    );
}
