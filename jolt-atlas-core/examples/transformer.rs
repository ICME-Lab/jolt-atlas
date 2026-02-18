use atlas_onnx_tracer::{model::Model, tensor::Tensor};
use common::utils::logging::setup_tracing;
use jolt_atlas_core::onnx_proof::{
    AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing,
    Blake2bTranscript, Bn254, Fr, HyperKZG, ONNXProof,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn main() {
    let (_guard, _tracing_enabled) = setup_tracing("addsubmul ONNX Proof");
    let working_dir = "../atlas-onnx-tracer/models/transformer/";
    let mut rng = StdRng::seed_from_u64(0x1096);
    let input_data: Vec<i32> = (0..64 * 64)
        .map(|_| (1 << 7) + rng.gen_range(-50..=50))
        .collect();
    let input = Tensor::new(Some(&input_data), &[1, 64, 64]).unwrap();
    let model = Model::load(&format!("{working_dir}network.onnx"), &Default::default());

    tracing::info!("Loaded model and generated input data");
    let pp = AtlasSharedPreprocessing::preprocess(model);
    let prover_preprocessing = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(pp);

    let (proof, io, _debug_info) =
        ONNXProof::<Fr, Blake2bTranscript, HyperKZG<Bn254>>::prove(&prover_preprocessing, &[input]);

    let verifier_preprocessing =
        AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_preprocessing);

    proof.verify(&verifier_preprocessing, &io, None).unwrap();
}
