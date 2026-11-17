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
        "660e2de36798089af971f2713465df1b93e86f2b78782eeed1906aea42703ca8",
        "7fabdfd18555d91242e28a0baef1bee582328c27869bf6bccaf73278f3e17127",
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
        "01913f50df58781e1fb01c8466beefbcaa040bcd26e7f9f59917804c4c820894",
        "d6874eb3a5201efe967ce74d3b106aa91f11f81476973c9d47f13750203baf79",
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
        "04a1a6153708c4cb494439a35acf495803cd89975eeedfbaae690bbd9c7629bf",
        "6fcade83e826c52bbe4054c48f70e3918fce456695f1e0f05c7d7e54e004bbde",
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
        "383eb6aa7f2a5ce4420d44615a68568a0f5af3a3320a64860e6502e08564cbb0",
        "9a1a3d5ae51ebfef179105973e305d4e823eff203048210c5c28f675e35eed3f",
    );
}
