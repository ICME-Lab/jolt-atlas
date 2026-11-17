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
        "74cf5051dd8ec8b5c5a891107f3afb8a500ffa2ee0427eae58f11ed5a17b8f1d",
        "2e9dd041a67e810d0d78ebbad1a9a80ab93216ac1da1b5dd094a2fe1a8c24ad0",
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
        "e514322efd927d2f3fb6bc85eb3f97fedaa62598b1dd56a855c09d80661accca",
        "30300e046ae3316e2b927a85f3a41b03656f611a0bf0a36859e8a24e2eb9e625",
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
        "99147fda2db0c20fcb910337d9fb799c9c01ff7fb469a4276c29d8423ab4d98f",
        "31b9e314d2046733e09c276d274239004674ceedd6d333368fc6cc522657114a",
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
        "afd541c7bdccf25fc598cac20ddcfe2e3285c3d9b5f9d2dabfd2e85746301c36",
        "fd09a7269c296111822ef1fd6f61438793e7a7cf9338b627f9bfa23fb185a69a",
    );
}
