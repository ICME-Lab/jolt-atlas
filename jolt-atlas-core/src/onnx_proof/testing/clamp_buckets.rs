//! The packed clamp buckets ([`global_clamp`](crate::onnx_proof::global_clamp),
//! [`clamp_split`]): what a bucket commits for a given statement.

use atlas_pcs::akita::AkitaScheme;
use atlas_sumcheck::opening_proof::{OpeningId, SumcheckId};
use atlas_trace::Tensor;
use atlas_transcript::Blake2bProver;
use rand::{rngs::StdRng, SeedableRng};

use super::{models::probe_models, prove_verify_roundtrip, ModelBuilder, RoundtripOptions};
use crate::node_view::{clamp_value_bound, ModelView};
use crate::onnx_proof::{
    clamp_lookups::clamp_intermediate,
    clamp_split,
    deferred_lookups::DEFERRED_PROOF_IDX,
    global_clamp::{clamp_buckets, remainder_buckets},
    prove, AtlasProverPreprocessing, AtlasSharedPreprocessing, Fp128, ProofType,
};

/// A clamp bucket whose nodes are all unsaturated model outputs proves nothing through its
/// columns (each node ties `acc = out` in its own turn), so it commits nothing: it is absent from
/// the committed-polynomial list for its statement, the prover commits exactly that list, and
/// the proof verifies.
#[test]
fn exact_output_bucket_commits_nothing() {
    let mut rng = StdRng::seed_from_u64(0xB0C7);
    let mut b = ModelBuilder::new();
    let x = b.input(vec![256]);
    let c = b.constant(Tensor::random_small(&mut rng, &[256]));
    let out = b.mul(x, c);
    b.mark_output(out);
    let model = b.build();
    let inputs = vec![Tensor::random_small(&mut rng, &[256])];

    let io = model.trace(&inputs).expect("honest execution").io(&model);
    assert!(
        clamp_split::exact_output_verifier(&model, &io, out),
        "the clamped output is unsaturated"
    );
    let buckets = clamp_buckets(&model).expect("bucket layout");
    let bucket = buckets
        .iter()
        .find(|bucket| bucket.nodes.iter().any(|n| n.idx == out))
        .expect("the clamped output has a bucket");
    assert!(bucket.nodes.iter().all(|n| n.idx == out));

    let shared = AtlasSharedPreprocessing::preprocess(model.clone());
    let committed = shared
        .get_models_committed_polynomials(&io)
        .expect("committed list");
    for poly in bucket.committed_polys() {
        assert!(
            !committed.contains(&poly),
            "{poly:?} is committed for a bucket with nothing to check"
        );
    }
    let pp = AtlasProverPreprocessing::<Fp128, AkitaScheme>::new(shared);
    let (proof, _, _) = prove::<Fp128, Blake2bProver, AkitaScheme>(&pp, &inputs);
    assert_eq!(proof.commitments().len(), committed.len());

    prove_verify_roundtrip::<AkitaScheme>(model, &inputs, RoundtripOptions::default());
}

/// A model with no table lookup and no clamped node proves nothing after the node loop: its
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

/// The `saturating` probe model saturates an internal clamped node, so the soundness
/// sweeps run over a bucket that carries dense `OUT` digits and nonempty slack chunks together,
/// and its proof carries the slack chunk check beside the digit range check.
#[test]
fn saturating_probe_saturates() {
    let probe = probe_models()
        .into_iter()
        .find(|m| m.name == "saturating")
        .expect("the saturating probe model");
    let model = probe.model;
    let trace = model.trace(&probe.inputs).expect("honest execution");
    let outputs = model.output_indices();
    let saturating: Vec<usize> = model
        .views()
        .iter()
        .filter(|n| {
            !outputs.contains(&n.idx)
                && clamp_value_bound(&model, n).expect("shape").is_some()
                && clamp_intermediate(n, &trace)
                    .data()
                    .iter()
                    .any(|&acc| acc > i32::MAX as i64 || acc < i32::MIN as i64)
        })
        .map(|n| n.idx)
        .collect();
    assert!(!saturating.is_empty(), "no internal clamped node saturates");

    let buckets = clamp_buckets(&model).expect("shapes");
    let pp = AtlasProverPreprocessing::<Fp128, AkitaScheme>::new(
        AtlasSharedPreprocessing::preprocess(model),
    );
    let (proof, _, _) = prove::<Fp128, Blake2bProver, AkitaScheme>(&pp, &probe.inputs);
    assert!(
        proof
            .proofs()
            .keys()
            .any(|id| id.1 == ProofType::DeferredClampDigitRange),
        "the saturating probe's proof has no digit range check"
    );
    // The slack chunk check ran: some bucket's slack chunks are opened by its Booleanity.
    let slack_opened = buckets.iter().any(|bucket| {
        clamp_split::slack_chunk_polys(bucket)
            .into_iter()
            .all(|poly| {
                proof
                    .opening_claims()
                    .openings()
                    .contains_key(&OpeningId::new(poly, SumcheckId::Booleanity))
            })
    });
    assert!(
        slack_opened,
        "the saturating probe's proof opens no slack chunks"
    );
}

/// The `scalars` probe model puts one-element nodes under the soundness sweeps: every clamped
/// node is alone in a one-cell clamp bucket, the fused `Mul`'s remainder in a one-cell remainder
/// bucket, one internal node saturates, and every value the digit tamper targets has a digit to
/// move.
#[test]
fn scalars_probe_layout() {
    let probe = probe_models()
        .into_iter()
        .find(|m| m.name == "scalars")
        .expect("the scalars probe model");
    let model = probe.model;
    let trace = model.trace(&probe.inputs).expect("honest execution");
    let outputs = model.output_indices();
    let clamped: Vec<usize> = model
        .views()
        .iter()
        .filter(|n| clamp_value_bound(&model, n).expect("shape").is_some())
        .map(|n| n.idx)
        .collect();
    assert_eq!(clamped.len(), 4, "a Mul, two Adds and a Sub");

    let buckets = clamp_buckets(&model).expect("bucket layout");
    assert_eq!(
        buckets.len(),
        clamped.len(),
        "one bucket per one-element node"
    );
    let remainders = remainder_buckets(&model).expect("remainder layout");
    for bucket in buckets.iter().chain(&remainders) {
        assert_eq!(bucket.log_t, 0);
        assert_eq!(bucket.nodes.len(), 1);
        assert_eq!(bucket.nodes[0].log_t, 0);
    }
    assert_eq!(remainders.len(), 1, "the fused Mul");

    let views = model.views();
    let mut saturated = 0;
    for &idx in clamped.iter().filter(|idx| !outputs.contains(idx)) {
        let acc = clamp_intermediate(&views[idx], &trace).data()[0];
        saturated += usize::from(acc > i32::MAX as i64);
        let out = i64::from(trace.node_outputs[idx].data()[0]);
        assert!(out + (1 << 31) >= 4, "node {idx}: OUT has no upper digit");
    }
    assert_eq!(saturated, 1, "exactly the second Add saturates");
}

/// Every remainder a probe model's nodes leave to be bounded is in exactly one remainder
/// bucket, once per range claim.
#[test]
fn every_range_claim_is_in_exactly_one_bucket() {
    use crate::onnx_proof::global_clamp::{node_remainders, range_claims, RangeClaim};
    let mut families = std::collections::BTreeSet::new();
    for probe in probe_models() {
        let model = probe.model;
        let mut expected: Vec<(usize, RangeClaim, usize)> = Vec::new();
        for node in &model.views() {
            for (remainder, bound) in node_remainders(node).expect("bounds") {
                families.insert(remainder);
                for (claim, width) in range_claims(remainder, bound) {
                    expected.push((node.idx, claim, width));
                }
            }
        }
        let mut packed: Vec<(usize, RangeClaim, usize)> = remainder_buckets(&model)
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
        "the probe models hold every remainder family: {families:?}"
    );
}

/// A slack claim whose bound is opened at another point than its remainder is rejected.
#[test]
fn range_claim_rejects_a_bound_at_another_point() {
    use crate::onnx_proof::global_clamp::{Bound, RangeClaim, Remainder};
    use atlas_algebra::field::Ring;
    use atlas_algebra::poly::point::OpeningPoint;
    use atlas_sumcheck::opening_proof::VerifierOpeningAccumulator;
    use common::VirtualPoly;

    let (node, divisor) = (5usize, 3usize);
    let claim = RangeClaim {
        remainder: Remainder::Div,
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
            Remainder::Div.opening_id(node),
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
    use crate::onnx_proof::plan::OpPlan;
    let mut b = ModelBuilder::new();
    let x = b.input(vec![8]);
    let out = b.scalar_const_div(x, -3);
    b.mark_output(out);
    let model = b.build();
    let inputs = vec![Tensor::from_slice(&[1, 2, 3, 4, 5, 6, 7, 8], &[8]).unwrap()];
    let io = model.trace(&inputs).expect("honest execution").io(&model);
    let node = model.views()[out].clone();
    let atlas_trace::Op::ScalarConstDiv(op) = &node.operator else {
        panic!("the output is a ScalarConstDiv");
    };
    assert!(op.plan(&node, &model, &io).is_err());
    assert!(remainder_buckets(&model).is_err());
}
