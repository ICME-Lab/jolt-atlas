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
        "6076aa41a6af2089321454e125b826f741abdd7776cd3c43ce8407e44acafae1",
        "0fb9ab9a0866294b34225d8927397839fd981bb4fafe5a293a02934049ea8566",
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
        "cb898205d76269ed93854798dd8dd8c4ac691ab3c3d14c3cffe48e45a5cf31da",
        "db44b21c10007d83fc0075489a35c25e09267f13ee803afb377de6c3cd111b5d",
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
        "662b86833702211e96067e3be884f17922529c5778282af816720f94612fc92f",
        "0d4b25dcfee4b4d1b0ff5afd31d38adcb3ec9c9aa5c5153411332402a2bdbcfb",
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
        "afb0152a113efdaa08ed11061036315b9f551ed72e1dcf9acce4b4fa017ea64d",
        "5f83bf66d8c89a706a61fd6547b9cab65e9bb1018ac3d1454e598b84cf12e891",
    );
}
