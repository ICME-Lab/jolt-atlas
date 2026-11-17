//! The packed range buckets ([`range_bucket`](crate::onnx_proof::range_bucket)): what they hold
//! and commit for a model.

use atlas_pcs::akita::AkitaScheme;
use atlas_sumcheck::opening_proof::{OpeningId, SumcheckId};
use atlas_trace::Tensor;
use atlas_transcript::Blake2bProver;
use rand::{rngs::StdRng, SeedableRng};

use super::{models::probe_models, prove_verify_roundtrip, ModelBuilder, RoundtripOptions};
use crate::node_view::ModelView;
use crate::onnx_proof::{
    prove,
    range_bucket::{checks_its_output, range_buckets, Remainder},
    schedule::DEFERRED_PROOF_IDX,
    AtlasProverPreprocessing, AtlasSharedPreprocessing, Fp128,
};

/// A model output is a public `i32` tensor, so it is never a bucket member, whatever its values,
/// and the committed polynomials are a function of the model: the prover commits the same list
/// for an execution whose output holds `i32::MAX` as for any other, and both proofs verify.
#[test]
fn model_outputs_are_not_range_checked() {
    let mut b = ModelBuilder::new();
    let x = b.input(vec![4]);
    let y = b.input(vec![4]);
    let hidden = b.add(x, y);
    let out = b.add(hidden, y);
    b.mark_output(out);
    let model = b.build();

    let buckets = range_buckets(&model).expect("bucket layout");
    let members: Vec<usize> = buckets
        .iter()
        .flat_map(|bucket| bucket.nodes.iter().map(|n| n.idx))
        .collect();
    assert_eq!(members, vec![hidden], "the hidden sum only");

    let shared = AtlasSharedPreprocessing::preprocess(model.clone());
    let pp = AtlasProverPreprocessing::<Fp128, AkitaScheme>::new(shared);
    let tensor = |values: [i32; 4]| Tensor::from_slice(&values, &[4]).unwrap();
    for inputs in [
        vec![tensor([1, -2, 3, -4]), tensor([5, 6, -7, 8])],
        vec![tensor([i32::MAX, 0, i32::MIN, -1]), tensor([0, 0, 0, 0])],
    ] {
        let io = model.trace(&inputs).expect("honest execution").io(&model);
        let (proof, _, _) = prove::<Fp128, Blake2bProver, AkitaScheme>(&pp, &inputs);
        assert_eq!(proof.commitments().len(), pp.schedule.committed().len());
        prove_verify_roundtrip::<AkitaScheme>(model.clone(), &inputs, RoundtripOptions::default());
        if inputs[0].data()[0] == i32::MAX {
            assert_eq!(io.outputs[0].data()[0], i32::MAX);
        }
    }
}

/// A model with no table lookup and no range-checked value proves nothing after the node loop: its
/// proof has no post-loop entry (a stray one would fail the drain check).
#[test]
fn lookup_free_model_has_no_post_loop_proof() {
    let mut rng = StdRng::seed_from_u64(0x10F0);
    let mut b = ModelBuilder::new();
    let x = b.input(vec![16]);
    let out = b.neg(x);
    b.mark_output(out);
    let model = b.build();
    let inputs = [Tensor::random_small(&mut rng, &[16])];

    let pp = AtlasProverPreprocessing::<Fp128, AkitaScheme>::new(
        AtlasSharedPreprocessing::preprocess(model.clone()),
    );
    let (proof, _, _) = prove::<Fp128, Blake2bProver, AkitaScheme>(&pp, &inputs);
    assert!(
        proof.proofs().keys().all(|id| id.0 != DEFERRED_PROOF_IDX),
        "a lookup-free model's proof has a post-loop entry"
    );
    prove_verify_roundtrip::<AkitaScheme>(model, &inputs, RoundtripOptions::default());
}

/// The `scalars` probe model puts one-element nodes under the soundness sweeps: every range
/// claim (the three hidden outputs and the fused `Mul`'s remainder) is alone in a one-cell
/// bucket, and every value the digit tamper targets has a digit to move.
#[test]
fn scalars_probe_layout() {
    let probe = probe_models()
        .into_iter()
        .find(|m| m.name == "scalars")
        .expect("the scalars probe model");
    let model = probe.model;
    let trace = model.trace(&probe.inputs).expect("honest execution");
    let checked: Vec<usize> = model
        .views()
        .iter()
        .filter(|n| checks_its_output(n))
        .map(|n| n.idx)
        .filter(|idx| !model.output_indices().contains(idx))
        .collect();
    assert_eq!(
        checked.len(),
        3,
        "a Mul and two Adds; the Sub is the output"
    );

    let buckets = range_buckets(&model).expect("bucket layout");
    assert_eq!(buckets.len(), checked.len() + 1, "one bucket per claim");
    for bucket in &buckets {
        assert_eq!(bucket.log_t, 0);
        assert_eq!(bucket.nodes.len(), 1);
        assert_eq!(bucket.nodes[0].log_t, 0);
    }
    let output_claims = buckets
        .iter()
        .filter(|b| {
            b.nodes[0]
                .claim
                .is_some_and(|c| c.remainder == Remainder::Output)
        })
        .count();
    assert_eq!(output_claims, checked.len());

    for &idx in &checked {
        let out = i64::from(trace.node_outputs[idx].data()[0]);
        assert!(out + (1 << 31) >= 4, "node {idx}: OUT has no upper digit");
    }
}

/// Every value a probe model's nodes leave to be bounded is in exactly one range bucket, once
/// per range claim.
#[test]
fn every_range_claim_is_in_exactly_one_bucket() {
    use crate::onnx_proof::range_bucket::{node_range_claims, RangeClaim};
    let mut families = std::collections::BTreeSet::new();
    for probe in probe_models() {
        let model = probe.model;
        let mut expected: Vec<(usize, RangeClaim, usize)> = Vec::new();
        for node in &model.views() {
            for (claim, width) in node_range_claims(&model, node).expect("bounds") {
                families.insert(claim.remainder);
                expected.push((node.idx, claim, width));
            }
        }
        let mut packed: Vec<(usize, RangeClaim, usize)> = range_buckets(&model)
            .expect("remainder layout")
            .iter()
            .flat_map(|bucket| {
                bucket.nodes.iter().map(|member| {
                    (
                        member.idx,
                        member.claim.expect("a remainder member has a claim"),
                        bucket.width,
                    )
                })
            })
            .collect();
        let key = |m: &(usize, RangeClaim, usize)| (m.0, m.1.remainder, m.1.slack.is_some());
        expected.sort_by_key(key);
        packed.sort_by_key(key);
        assert_eq!(packed, expected, "{}", probe.name);
    }
    assert_eq!(
        families.len(),
        8,
        "the probe models hold every range-claim family: {families:?}"
    );
}

/// A slack claim whose bound is opened at another point than its remainder is rejected.
#[test]
fn range_claim_rejects_a_bound_at_another_point() {
    use crate::onnx_proof::range_bucket::{Bound, RangeClaim, Remainder};
    use atlas_algebra::field::Ring;
    use atlas_algebra::poly::point::OpeningPoint;
    use atlas_sumcheck::opening_proof::VerifierOpeningAccumulator;
    use common::VirtualPoly;

    let (node, divisor) = (5usize, 3usize);
    let claim = RangeClaim {
        remainder: Remainder::RsqrtDiv,
        slack: Some(Bound::Node(divisor)),
    };
    let point = |seed: u64| OpeningPoint::new((0..4).map(|i| Fp128::from_u64(seed + i)).collect());
    let bound_id = OpeningId::new(
        VirtualPoly::NodeOutput(divisor),
        SumcheckId::NodeExecution(node),
    );
    let stage = |bound_point| {
        let mut accumulator = VerifierOpeningAccumulator::<Fp128>::new();
        accumulator.openings.insert(
            Remainder::RsqrtDiv
                .opening_id(node)
                .expect("an advice opening"),
            (point(1), Fp128::from_u64(2)),
        );
        accumulator
            .openings
            .insert(bound_id, (bound_point, Fp128::from_u64(7)));
        accumulator
    };
    let (_, slack) = claim
        .evaluate(node, &stage(point(1)))
        .expect("one common point");
    assert_eq!(slack, Fp128::from_u64(7 - 1 - 2));
    assert!(claim.evaluate(node, &stage(point(100))).is_err());
}

/// `ScalarConstDiv` by a divisor that is not positive is rejected before anything is proven.
#[test]
fn scalar_const_div_rejects_a_non_positive_divisor() {
    use crate::onnx_proof::schedule::Schedule;
    let mut b = ModelBuilder::new();
    let x = b.input(vec![8]);
    let out = b.scalar_const_div(x, -3);
    b.mark_output(out);
    let model = b.build();
    assert!(Schedule::<Fp128>::compile(&model).is_err());
    assert!(range_buckets(&model).is_err());
}
