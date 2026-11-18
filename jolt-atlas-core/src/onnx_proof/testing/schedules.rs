//! Pins of the checked-in zoo's verification schedules.
//!
//! A model's schedule ([`Schedule`]) is what the verifier checks for it, in transcript order.
//! These tests pin a blake2b-256 hash of the rendered schedule (its `Display`: one line per
//! step) of each checked-in model, so a change to what is checked, or to its order, shows up
//! as a changed pin in review. `cargo run --release -p jolt-atlas-core --example prove --
//! --model models/<name> --schedule` prints the text a pin hashes.

use atlas_algebra::field::fp128::Fp128;
use atlas_pcs::bitz::BitzScheme;
use atlas_trace::Tensor;
use blake2::{digest::consts::U32, Blake2b, Digest};
use rand::{rngs::StdRng, SeedableRng};

use super::zoo::zoo;
use super::{prove_verify_roundtrip, ModelBuilder, RoundtripOptions};
use crate::onnx_proof::schedule::{OpRelation, RelId, Schedule, Step};

/// Compile `name`'s schedule and compare the hash of its rendering against `pin`.
fn assert_schedule_pin(name: &str, pin: &str) {
    let schedule =
        Schedule::<Fp128>::compile(&zoo(name, 16).model).expect("the model has a schedule");
    let rendered = schedule.to_string();
    assert_eq!(
        rendered.lines().count(),
        schedule.steps().len(),
        "{name}: one line per step"
    );
    let got: String = Blake2b::<U32>::digest(rendered.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(
        got, pin,
        "{name}: the verification schedule changed. Review the change (`--schedule` prints \
         it) and re-pin"
    );
}

#[test]
fn schedule_pin_op_gauntlet() {
    assert_schedule_pin(
        "op_gauntlet",
        "17ef81ef128938c80dbb2c75e3dd6aeaef3f427d0457e2fb523e1054c70dc824",
    );
}

#[test]
fn schedule_pin_gpt2_tiny() {
    assert_schedule_pin(
        "gpt2-tiny",
        "589f3e785b2b18ba61d18597106f76697c7a4108753e16b790207b6c041724e6",
    );
}

#[test]
fn schedule_pin_qwen2_tiny() {
    assert_schedule_pin(
        "qwen2-tiny",
        "82641ac0d8d74257755594f3e046d94ff7a98289dee96e16e2cfbfb6e4452c72",
    );
}

#[test]
fn schedule_pin_bge_tiny() {
    assert_schedule_pin(
        "bge-tiny",
        "2f09db4b9c45f3b1b29a85cf46adf57e77260a92f8a7163f84005a91f2e7aecb",
    );
}

/// A model with no table read has no lookups step and no one-hot step in its schedule, hence
/// nothing in its proof for either, and its proof verifies.
#[test]
fn schedule_of_a_lookup_free_model_has_no_lookup_step() {
    let mut rng = StdRng::seed_from_u64(0x5C4E);
    let mut b = ModelBuilder::new();
    let x = b.input(vec![8]);
    let y = b.input(vec![8]);
    let hidden = b.add(x, y);
    let out = b.add(hidden, y);
    b.mark_output(out);
    let model = b.build();

    let schedule = Schedule::<Fp128>::compile(&model).expect("the model has a schedule");
    // The first member of a sumcheck over `members`.
    let first = |members: &[RelId]| schedule.relation(members[0]).expect("a member");
    let is_read = |relation: &OpRelation<Fp128>| {
        !matches!(
            relation,
            OpRelation::Op(_) | OpRelation::RemainderBucketValue(_)
        )
    };
    let setups = schedule
        .steps()
        .iter()
        .filter(|step| matches!(step, Step::Setup(_)));
    assert_eq!(setups.count(), 0);
    assert!(!schedule.sumchecks().any(|members| is_read(first(members))));
    assert!(schedule
        .sumchecks()
        .any(|members| matches!(first(members), OpRelation::RemainderBucketValue(_))));

    let inputs = [
        Tensor::random_small(&mut rng, &[8]),
        Tensor::random_small(&mut rng, &[8]),
    ];
    prove_verify_roundtrip::<BitzScheme>(model, &inputs, RoundtripOptions::default());
}
