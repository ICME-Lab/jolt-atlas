//! Cross-stage digests of the checked-in zoo.
//!
//! The zoo tests prove and verify with the same build, so a coordinated
//! prover+verifier change passes them unseen. These tests pin, for each
//! checked-in model on its golden inputs under BitZ over Fp128, a
//! blake2b-256 hash of the proof's bytes and the prover's final transcript
//! state. A behavior-preserving change leaves every pin untouched; a
//! protocol-breaking change re-pins in its own commit, naming the cause.

use atlas_onnx_import::fixtures::{synth_inputs, GOLDEN_SEED};
use atlas_pcs::{bitz::BitzScheme, commitment_scheme::CommitmentScheme};
use atlas_trace::{Model, Tensor};
use atlas_transcript::Blake2bProver;
use blake2::{digest::consts::U32, Blake2b, Digest};

use super::zoo::{op_gauntlet_inputs, zoo};
use crate::onnx_proof::{prove, AtlasProverPreprocessing, AtlasSharedPreprocessing};

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
    let got_proof = hex(&Blake2b::<U32>::digest(&p.0));
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
    assert_digest::<BitzScheme>(
        "op_gauntlet",
        imported.model,
        &inputs,
        "53fe85b1854fcc61e6c70af87c8bff8ae27b7b2e82ae9540659e6a0bf901cee4",
        "bea1330d0f9afacc8f0c743a5f2c6e5748d27247e8bf0863935f256d5c067636",
    );
}

#[test]
fn digest_gpt2_tiny() {
    let imported = zoo("gpt2-tiny", 16);
    let inputs = synth_inputs(&imported, GOLDEN_SEED);
    assert_digest::<BitzScheme>(
        "gpt2-tiny",
        imported.model,
        &inputs,
        "fa339e19f7b7f464efeaf9fb740f575a1c38b726b07b00bd148faf6f7c619cba",
        "88f5d8dbb7d9e39019cc0db520e5766617c0430000af2c4c6b23be5619332633",
    );
}

#[test]
fn digest_qwen2_tiny() {
    let imported = zoo("qwen2-tiny", 16);
    let inputs = synth_inputs(&imported, GOLDEN_SEED);
    assert_digest::<BitzScheme>(
        "qwen2-tiny",
        imported.model,
        &inputs,
        "2fd9613cee54f046db004b64ff3385b7924110edd03259099c338c383fc0758e",
        "6c686281278a156a8ee3058e96690cee65c2baaaced572db8f4f6951033d9ecc",
    );
}

#[test]
fn digest_bge_tiny() {
    let imported = zoo("bge-tiny", 16);
    let inputs = synth_inputs(&imported, GOLDEN_SEED);
    assert_digest::<BitzScheme>(
        "bge-tiny",
        imported.model,
        &inputs,
        "aff6d1421a97c0c38dce1db965b90e83644f1f5ff7e28f3d9d50a3c30264186c",
        "f81e0fbbc1ef8ea256736bfee9bb40abd71e241ba1e4d7d643dcd99679715ad3",
    );
}
