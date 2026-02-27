use crate::onnx_proof::{
    AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing, ONNXProof, malicious_prover::MaliciousONNXProof,
};
use ark_bn254::{Bn254, Fr};
use atlas_onnx_tracer::{
    model::{test::ModelBuilder,  Model},
    tensor::Tensor,
};
use joltworks::{
    poly::{
        commitment::{ hyperkzg::HyperKZG},
    },
    transcripts::{Blake2bTranscript,},
};
use rand::{rngs::StdRng, SeedableRng};


fn sub_model(rng: &mut StdRng, t: usize) -> Model {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![t]);
    let c = b.constant(Tensor::random_small(rng, &[t]));
    let out = b.sub(i, c);
    b.mark_output(out);
    b.build()
}

#[test]
fn soundness_baseline_sub_honest_prover_verifies() {
    let t = 1 << 12;
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let input = Tensor::<i32>::random_small(&mut rng, &[t]);
    let model = sub_model(&mut rng, t);

    let pp = AtlasSharedPreprocessing::preprocess(model);
    let prover_pp = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(pp);
    let (proof, io, debug_info) =
        MaliciousONNXProof::prove(&prover_pp, &[input]);
    let proof: ONNXProof<Fr, Blake2bTranscript, HyperKZG<Bn254>> = proof.into();
    let verifier_pp = AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_pp);

    let res = proof.verify(&verifier_pp, &io, debug_info);
    assert!(res.is_ok(), "honest prover baseline should verify: {res:?}");
}

#[ignore = "for malicious-sub prover experiments: expects verification failure"]
#[test]
fn soundness_sub_malicious_prover_should_fail_verification() {
    // NOTE:
    // This test is intentionally ignored in normal runs.
    // It is a harness for local adversarial experiments where the Sub prover
    // implementation is temporarily modified to produce invalid proofs.
    //
    // Expected behavior for such experiments:
    // - honest Sub prover: verification succeeds (baseline test above)
    // - malicious Sub prover: verification must fail (this test)
    //
    // Run manually with:
    // cargo test -p jolt-atlas-core soundness_sub_malicious_prover_should_fail_verification -- --ignored --nocapture
    // Tensor length used by the single-node Sub model.
    let t = 1 << 12;
    // Deterministic RNG so the experiment is reproducible.
    let mut rng = StdRng::seed_from_u64(0xBAD5EED);
    // Generate the model input tensor.
    let input = Tensor::<i32>::random_small(&mut rng, &[t]);
    // Build a minimal graph: input - constant.
    let model = sub_model(&mut rng, t);

    // Preprocess model metadata shared by prover/verifier.
    let pp = AtlasSharedPreprocessing::preprocess(model);
    // Create prover preprocessing (includes commitment setup).
    let prover_pp = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(pp);
    // Generate proof and execution IO from the (possibly modified) prover code path.
    let (proof, io, debug_info) =
        MaliciousONNXProof::prove(&prover_pp, &[input]);
    let proof: ONNXProof<Fr, Blake2bTranscript, HyperKZG<Bn254>> = proof.into();
    // Derive verifier preprocessing from prover preprocessing.
    let verifier_pp = AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_pp);

    // Verify the proof against the recorded IO.
    let res = proof.verify(&verifier_pp, &io, debug_info);
    // In malicious-prover experiments, verification is expected to fail.
    assert!(
        res.is_err(),
        "malicious-sub experiment expected verifier rejection, but got: {res:?}"
    );
}
