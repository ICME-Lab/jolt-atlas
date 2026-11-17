//! Cross-stage digests of the checked-in zoo.
//!
//! The zoo tests prove and verify with the same build, so a coordinated
//! prover+verifier change passes them unseen. These tests pin, for each
//! checked-in model on its golden inputs under Akita over Fp128, a
//! blake2b-256 hash of the proof's bytes and the prover's final transcript
//! state. A behavior-preserving change leaves every pin untouched; a
//! protocol-breaking change re-pins in its own commit, naming the cause.

use atlas_onnx_import::fixtures::{synth_inputs, GOLDEN_SEED};
use atlas_pcs::{akita::AkitaScheme, commitment_scheme::CommitmentScheme};
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
    assert_digest::<AkitaScheme>(
        "op_gauntlet",
        imported.model,
        &inputs,
        "f7e523de87457b3f73b281d36e8a8375ce85ec93ea4086288a2c75ec3b09f0cb",
        "4cc32e3f596cc1a8b4536545523116a9e0197f298c0ee34453ca26843f73da13",
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
        "b030c746addcee6fb3dd8b471889d68ac484547874292ddfb275a125e20a132c",
        "5344f40889b3572faf407eec8b950440ff8b4006a73b049e42385e9c4021a5a9",
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
        "9e5f9cb99d842331fcd1d3bf156254b8b62f73c3d7474652fca487e6d6cc2416",
        "844e2a848173f8a78850a30d62133e9803995f833b3874fa005c7ce4451c8895",
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
        "e0d6bc9584bc2c989d46ec2dfdd7f42e2ca515eea52f955f430640a942cdabc1",
        "32666ecbf2127b74d72104a084a4adeb97bef08e0bd0a11f15bb9a06306beb6d",
    );
}
