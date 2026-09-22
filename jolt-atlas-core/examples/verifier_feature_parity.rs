//! Produce a deterministic ordinary proof for comparison across feature builds.
//! Run each build into a different directory and compare proof.bin and io.json.
use ark_bn254::{Bn254, Fr};
use ark_serialize::{CanonicalSerialize, Compress};
use atlas_onnx_tracer::{model::test::ModelBuilder, tensor::Tensor};
use jolt_atlas_core::onnx_proof::{
    AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing, ONNXProof,
};
use joltworks::{poly::commitment::hyperkzg::HyperKZG, transcripts::Blake2bTranscript};
use std::{fs, path::PathBuf};

fn main() {
    let directory = PathBuf::from(std::env::args().nth(1).expect("output directory required"));
    assert!(!directory.exists(), "use a fresh output directory");
    let mut builder = ModelBuilder::with_scale(14);
    let input = builder.input(vec![2, 8]);
    let left = builder.sigmoid(input);
    let right = builder.sigmoid(input);
    let sum = builder.add(left, right);
    let output = builder.softmax_last_axis(sum);
    builder.mark_output(output);
    let shared = AtlasSharedPreprocessing::preprocess(builder.build());
    let prover = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(shared);
    let verifier = AtlasVerifierPreprocessing::from(&prover);
    let values: Vec<i32> = (0..16).map(|i| (i - 8) * 127).collect();
    let input = Tensor::new(Some(&values), &[2, 8]).unwrap();
    type Proof = ONNXProof<Fr, Blake2bTranscript, HyperKZG<Bn254>>;
    let (proof, io, _) = Proof::prove(&prover, &[input]);
    proof.verify(&verifier, &io, None).unwrap();
    let mut canonical = Vec::new();
    proof.serialize_compressed(&mut canonical).unwrap();
    let compact = proof.serialize_compact(Compress::Yes).unwrap();
    let decoded = Proof::deserialize_compact(&compact, proof.opening_claims.0.len()).unwrap();
    decoded.verify(&verifier, &io, None).unwrap();
    let mut recovered = Vec::new();
    decoded.serialize_compressed(&mut recovered).unwrap();
    assert_eq!(canonical, recovered);
    let mut changed = io.clone();
    changed.outputs[0].inner[0] += 1;
    assert!(decoded.verify(&verifier, &changed, None).is_err());
    fs::create_dir_all(&directory).unwrap();
    fs::write(directory.join("proof.bin"), canonical).unwrap();
    fs::write(directory.join("compact.bin"), compact).unwrap();
    fs::write(directory.join("io.json"), serde_json::to_vec(&io).unwrap()).unwrap();
}
