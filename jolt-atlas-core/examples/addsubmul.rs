use atlas_onnx_tracer::{model::Model, tensor::Tensor};
use jolt_atlas_core::{
    onnx_proof::{
        AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing,
        Blake2bTranscript, Bn254, Fr, HyperKZG, ONNXProof,
    },
    utils::logging::setup_tracing,
};
use rand::{rngs::StdRng, SeedableRng};

fn main() {
    let (_guard, _tracing_enabled) = setup_tracing("addsubmul ONNX Proof");
    let working_dir = "../atlas-onnx-tracer/models/test_add_sub_mul/";
    // Create test input vector of size 65536
    // Using small values to avoid overflow
    let mut rng = StdRng::seed_from_u64(0x100);
    // Create tensor with shape [65536]
    let input = Tensor::random_small(&mut rng, &[1 << 16]);
    let model = Model::load(&format!("{working_dir}network.onnx"), &Default::default());

    let pp = AtlasSharedPreprocessing::preprocess(model);
    let prover_preprocessing = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(pp);

    let (proof, io, _debug_info) =
        ONNXProof::<Fr, Blake2bTranscript, HyperKZG<Bn254>>::prove(&prover_preprocessing, &input);

    let verifier_preprocessing =
        AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_preprocessing);

    proof.verify(&verifier_preprocessing, &io, None).unwrap();
}
