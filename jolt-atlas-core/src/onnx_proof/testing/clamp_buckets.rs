//! The packed clamp buckets ([`global_clamp`](crate::onnx_proof::global_clamp),
//! [`clamp_split`]): what a bucket commits for a given statement.

use atlas_pcs::akita::AkitaScheme;
use atlas_trace::Tensor;
use atlas_transcript::Blake2bProver;
use rand::{rngs::StdRng, SeedableRng};

use super::{models::probe_models, prove_verify_roundtrip, ModelBuilder, RoundtripOptions};
use crate::node_view::{clamp_value_bound, ModelView};
use crate::onnx_proof::{
    clamp_lookups::{clamp_intermediate, is_scalar},
    clamp_split,
    global_clamp::clamp_buckets,
    prove, AtlasProverPreprocessing, AtlasSharedPreprocessing, Fp128,
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

/// The `saturating` probe model saturates an internal, non-scalar clamped node, so the soundness
/// sweeps run over a clamp bucket with nonempty slack chunks.
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
                && !is_scalar(n)
                && clamp_value_bound(&model, n).expect("shape").is_some()
                && clamp_intermediate(n, &trace)
                    .data()
                    .iter()
                    .any(|&acc| acc > i32::MAX as i64 || acc < i32::MIN as i64)
        })
        .map(|n| n.idx)
        .collect();
    assert!(
        !saturating.is_empty(),
        "no internal non-scalar clamped node saturates"
    );
}
