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
        "ae99dcf85980ffb6e67109be03882b060dd85d62c14a4c69a58dcf7ea61280fb",
        "63d412d50301a8b4c9db92500d79b0f565abd5eca325b06a1f6ff337ee2a7895",
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
        "28aa9643479c34ba474e18e30d1d72f27becb55a7de08649dfc780554b828889",
        "49a63d80bcab661264f5b49d9c44515f45dc9213b85b6025242a7cfac13b8587",
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
        "d9be6178c05094e050dddc35ee1fa2ab2bbb1e2d17d245cd2a2c4f3d9ed681f3",
        "80bb5c16852323645a3f7619aa760612dbd0236b4ec7a496b5fb924428e0a9c8",
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
        "e7a0b9a00246fc9f7c176611dc6289a0d7e4c27a8afdf0cca56b464ac0f89bc9",
        "245c597f9fe22435b8b77080039d686dd699e5356e549332a91d1f14a1cfd302",
    );
}
