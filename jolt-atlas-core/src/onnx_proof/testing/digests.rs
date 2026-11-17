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
        "7fb77ff08c1f86e898c8d2d656dcb2dc7c80ea97362e8edfbacc7130ec4c71f7",
        "133c878b546e5fd32302257a759505b2020d0232f747368120ab7de5d5c44f90",
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
        "c484da45ab6ae6659883074739450efa2a9a9153b6c5b288a9d812359d72c4a0",
        "2ddb49a2eac0d928df5ea52ae3358bd0805b25e2a333fcd4162c33f4cce59143",
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
        "0092b4b937f1195085280e65110b47aeaf8a76aa347df062d57a610bfa072041",
        "b22b29266512f6effabba7f32e3d75f105516587787c493c77a0f0cf2b2eb6b8",
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
        "598cf9799622ec468515ddf97d0da1679499168f749e5f2a1c9198534c4f62f4",
        "0db86f035876238b6e44337c9974d6373d41c458cd187168fb90b26d9f38a9b3",
    );
}
