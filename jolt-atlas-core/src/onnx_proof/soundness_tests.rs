use crate::onnx_proof::{
    AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing, ONNXProof,
};
use ark_bn254::{Bn254, Fr};
use atlas_onnx_tracer::{
    model::{test::ModelBuilder, Model},
    tensor::Tensor,
};
use joltworks::{poly::commitment::hyperkzg::HyperKZG, transcripts::Blake2bTranscript};
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
        ONNXProof::<Fr, Blake2bTranscript, HyperKZG<Bn254>>::prove(&prover_pp, &[input]);
    let verifier_pp = AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_pp);

    let res = proof.verify(&verifier_pp, &io, debug_info);
    assert!(res.is_ok(), "honest prover baseline should verify: {res:?}");
}
