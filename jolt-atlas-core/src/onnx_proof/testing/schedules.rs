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
        "3378335db2ac694d8641b28e9bb91b035fb4d524d999d5559d171c0a0a5ffa89",
    );
}

#[test]
fn schedule_pin_gpt2_tiny() {
    assert_schedule_pin(
        "gpt2-tiny",
        "42bc673b52a256f565335d559eee29bcd07e17e81c4ee04c8fd458e22f977649",
    );
}

#[test]
fn schedule_pin_qwen2_tiny() {
    assert_schedule_pin(
        "qwen2-tiny",
        "b9c9301c44a820ed50e2f24a5ea9c268ea20d983bc3e9903b8d527975fc4138f",
    );
}

#[test]
fn schedule_pin_bge_tiny() {
    assert_schedule_pin(
        "bge-tiny",
        "dedae5954d0a4c20f90c76b2642b1213fa926c5bf10d7b8248f55a4754cf9d3f",
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
