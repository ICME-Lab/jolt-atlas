//! Prove/verify timing benchmark for a standalone Cos node.
//!
//! Cos and Sin share the same neural-teleportation proving machinery
//! (`onnx_proof/neural_teleport/`), so timing one is representative of both.
//!
//! ```bash
//! cargo run --release --package jolt-atlas-core --example cos_bench
//! ```
use atlas_onnx_tracer::{model::test::ModelBuilder, tensor::Tensor};
use common::consts::MODEL_SCALE;
use jolt_atlas_core::onnx_proof::{
    AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing,
    Blake2bTranscript, Bn254, Fr, HyperKZG, ONNXProof,
};
use rand::{rngs::StdRng, SeedableRng};
use std::time::{Duration, Instant};

const INPUT_LENS: &[usize] = &[1 << 8, 1 << 16];
const NUM_RUNS: usize = 5;

fn cos_model(input_len: usize) -> atlas_onnx_tracer::model::Model {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![input_len]);
    let res = b.cos(i);
    b.mark_output(res);
    b.build()
}

fn run_once(
    prover_preprocessing: &AtlasProverPreprocessing<Fr, HyperKZG<Bn254>>,
    verifier_preprocessing: &AtlasVerifierPreprocessing<Fr, HyperKZG<Bn254>>,
    input: &Tensor<i32>,
) -> (Duration, Duration) {
    let prove_start = Instant::now();
    let (proof, io, debug_info) = ONNXProof::<Fr, Blake2bTranscript, HyperKZG<Bn254>>::prove(
        prover_preprocessing,
        &[input.clone()],
    );
    let prove_time = prove_start.elapsed();

    let verify_start = Instant::now();
    proof
        .verify(verifier_preprocessing, &io, debug_info)
        .unwrap();
    let verify_time = verify_start.elapsed();

    (prove_time, verify_time)
}

fn mean(durations: &[Duration]) -> Duration {
    durations.iter().sum::<Duration>() / durations.len() as u32
}

fn main() {
    let mut rng = StdRng::seed_from_u64(0xC05);

    println!("MODEL_SCALE = {MODEL_SCALE}");
    println!("runs per input length = {NUM_RUNS} (+1 warmup, discarded)");

    for &input_len in INPUT_LENS {
        let input = Tensor::random_range(&mut rng, &[input_len], -50000..50000);

        let model = cos_model(input_len);
        let pp = AtlasSharedPreprocessing::preprocess(model);
        let prover_preprocessing = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(pp);
        let verifier_preprocessing =
            AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_preprocessing);

        // Warmup run, discarded (page faults, allocator warmup, etc.).
        run_once(&prover_preprocessing, &verifier_preprocessing, &input);

        let mut prove_times = Vec::with_capacity(NUM_RUNS);
        let mut verify_times = Vec::with_capacity(NUM_RUNS);
        for _ in 0..NUM_RUNS {
            let (prove_time, verify_time) =
                run_once(&prover_preprocessing, &verifier_preprocessing, &input);
            prove_times.push(prove_time);
            verify_times.push(verify_time);
        }

        println!("--- input length = {input_len} ---");
        println!("prove times  = {prove_times:?}");
        println!("verify times = {verify_times:?}");
        println!("mean prove time  = {:?}", mean(&prove_times));
        println!("mean verify time = {:?}", mean(&verify_times));
    }
}
