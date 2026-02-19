use ark_bn254::{Bn254, Fr};
use atlas_onnx_tracer::{model::Model, tensor::Tensor};
use joltworks::{poly::commitment::hyperkzg::HyperKZG, transcripts::Blake2bTranscript};

use crate::onnx_proof::{
    AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing, ONNXProof,
};

/// Run the prove-and-verify workflow, returning the execution IO.
pub fn unit_test_op(model: Model, inputs: &[Tensor<i32>]) {
    let pp = AtlasSharedPreprocessing::preprocess(model);
    let prover_preprocessing = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(pp);
    let (proof, io, debug_info) =
        ONNXProof::<Fr, Blake2bTranscript, HyperKZG<Bn254>>::prove(&prover_preprocessing, inputs);
    let verifier_preprocessing =
        AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_preprocessing);
    proof
        .verify(&verifier_preprocessing, &io, debug_info)
        .unwrap();
}
