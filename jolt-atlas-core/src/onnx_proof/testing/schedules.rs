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
        "ec0c675df29c6fea4fa3c3bbc34e1f3c9946e0e1f37a4b5f97b183150f1fd1c3",
    );
}

#[test]
fn schedule_pin_gpt2_tiny() {
    assert_schedule_pin(
        "gpt2-tiny",
        "2705bff414a86885bd1c2fed3c9fe013c98e469154423c8f0c653ac71f74adbc",
    );
}

#[test]
fn schedule_pin_qwen2_tiny() {
    assert_schedule_pin(
        "qwen2-tiny",
        "2f80002a082d5c2f6e964924065367c590200614b42bebd0b903ce0bb5e874b6",
    );
}

#[test]
fn schedule_pin_bge_tiny() {
    assert_schedule_pin(
        "bge-tiny",
        "0940eef8d92f069095a99e022128ef5584dd2a15449872e2f33006fd6c40ecfb",
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

/// A node's output claims may not be reduced before a consumer has appended its claim: a
/// schedule that reduces an input first, ahead of the sums that read it, is rejected, whatever
/// wrote it.
#[test]
fn schedule_rejects_a_reduction_before_a_consumer_claim() {
    let mut b = ModelBuilder::new();
    let x = b.input(vec![8]);
    let y = b.input(vec![8]);
    let hidden = b.add(x, y);
    let out = b.add(hidden, y);
    b.mark_output(out);
    let schedule = Schedule::<Fp128>::compile(&b.build()).expect("the model has a schedule");

    let mut steps = schedule.steps().to_vec();
    assert!(schedule.with_steps(steps.clone()).is_ok());
    let reduce = steps
        .iter()
        .rposition(|step| matches!(step, Step::Reduce { .. }))
        .expect("an input's claims are reduced");
    assert!(reduce > 0, "a consumer's steps precede the reduction");
    let step = steps.remove(reduce);
    steps.insert(0, step);
    assert!(schedule.with_steps(steps).is_err());
}

/// The sumchecks of a model's nodes are batched by wave: a chain of sums has fewer sumchecks
/// than nodes, and none outside a batch.
#[test]
fn schedule_batches_node_sumchecks_in_waves() {
    let mut b = ModelBuilder::new();
    let x = b.input(vec![8]);
    let mut last = x;
    let mut squares = 0;
    for _ in 0..6 {
        last = b.mul(last, x);
        squares += 1;
    }
    b.mark_output(last);
    let schedule = Schedule::<Fp128>::compile(&b.build()).expect("the model has a schedule");
    let unbatched = schedule
        .steps()
        .iter()
        .filter(|step| matches!(step, Step::Sumcheck { batched: false, .. }))
        .count();
    assert_eq!(unbatched, 0);
    assert!(schedule.sumchecks().count() < squares);
}

/// A digit-backed node takes its claim at a point of its own and never reduces its
/// consumers' claims; every other node that has a claim to run on reduces them.
#[test]
fn digit_backed_nodes_take_a_point_and_no_reduction() {
    use crate::node_view::ModelView;
    use crate::onnx_proof::range_bucket::is_digit_backed;
    use std::collections::BTreeSet;

    for name in ["op_gauntlet", "gpt2-tiny", "qwen2-tiny", "bge-tiny"] {
        let model = zoo(name, 16).model;
        let schedule = Schedule::<Fp128>::compile(&model).expect("the model has a schedule");
        let digit_backed: BTreeSet<usize> = model
            .views()
            .iter()
            .filter(|node| is_digit_backed(&model, node))
            .map(|node| node.idx)
            .collect();
        assert!(!digit_backed.is_empty(), "{name}: a digit-backed node");
        let mut pointed = BTreeSet::new();
        for step in schedule.steps() {
            match step {
                Step::Point { node, .. } => assert!(pointed.insert(*node), "{name}: n{node}"),
                Step::Reduce { node } => {
                    assert!(!digit_backed.contains(node), "{name}: n{node} is reduced")
                }
                _ => {}
            }
        }
        assert_eq!(pointed, digit_backed, "{name}");
    }
}
