use atlas_onnx_tracer::{model::Model, tensor::Tensor};
use common::utils::logging::setup_tracing;
use jolt_atlas_core::onnx_proof::{
    AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing,
    Blake2bTranscript, Bn254, Fr, HyperKZG, ONNXProof,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{env, process};

const USAGE: &str = "jolt-atlas - prove and verify bundled ONNX fixtures

Usage:
  jolt-atlas prove <model> [--trace | --trace-terminal]
  jolt-atlas --help

Bundled models:
  nanoGPT       small GPT fixture
  microgpt      tiny GPT fixture
  minigpt       small GPT fixture
  transformer   self-attention fixture

Examples:
  jolt-atlas prove nanoGPT
  jolt-atlas prove transformer --trace-terminal
";

fn main() {
    let args: Vec<String> = env::args().collect();

    match args.get(1).map(String::as_str) {
        Some("prove") => {
            let Some(model) = args.get(2).map(String::as_str) else {
                eprintln!("missing model name\n\n{USAGE}");
                process::exit(2);
            };
            if let Err(err) = prove_fixture(model) {
                eprintln!("error: {err}");
                eprintln!("\n{USAGE}");
                process::exit(2);
            }
        }
        Some("-h") | Some("--help") | None => print!("{USAGE}"),
        Some(other) => {
            eprintln!("unknown command: {other}\n\n{USAGE}");
            process::exit(2);
        }
    }
}

fn prove_fixture(model: &str) -> Result<(), String> {
    match model {
        "nanoGPT" | "nanogpt" => prove_gpt_fixture(
            "nanoGPT",
            "atlas-onnx-tracer/models/nanoGPT/network.onnx",
            64,
            1 << 5,
            -20..=20,
            0x1096,
        ),
        "microgpt" => prove_gpt_fixture(
            "microGPT",
            "atlas-onnx-tracer/models/microgpt/network.onnx",
            16,
            0,
            0..=31,
            0x42,
        ),
        "minigpt" => prove_gpt_fixture(
            "miniGPT",
            "atlas-onnx-tracer/models/minigpt/network.onnx",
            32,
            0,
            0..=1023,
            0x44,
        ),
        "transformer" => prove_transformer(),
        other => Err(format!("unsupported bundled model: {other}")),
    }
}

fn prove_gpt_fixture(
    title: &str,
    model_path: &str,
    block_size: usize,
    offset: i32,
    range: std::ops::RangeInclusive<i32>,
    seed: u64,
) -> Result<(), String> {
    let (_guard, _tracing_enabled) = setup_tracing(&format!("{title} ONNX Proof"));
    let mut rng = StdRng::seed_from_u64(seed);
    let input_data: Vec<i32> = (0..block_size)
        .map(|_| offset + rng.gen_range(range.clone()))
        .collect();
    let input = Tensor::new(Some(&input_data), &[1, block_size]).map_err(|err| err.to_string())?;
    prove_and_verify(model_path, vec![input])
}

fn prove_transformer() -> Result<(), String> {
    let (_guard, _tracing_enabled) = setup_tracing("Transformer ONNX Proof");
    let mut rng = StdRng::seed_from_u64(0x1096);
    let input_data: Vec<i32> = (0..64 * 64)
        .map(|_| (1 << 7) + rng.gen_range(-50..=50))
        .collect();
    let input = Tensor::new(Some(&input_data), &[1, 64, 64]).map_err(|err| err.to_string())?;
    prove_and_verify(
        "atlas-onnx-tracer/models/transformer/network.onnx",
        vec![input],
    )
}

fn prove_and_verify(model_path: &str, inputs: Vec<Tensor<i32>>) -> Result<(), String> {
    let model = Model::load(model_path, &Default::default());
    let pp = AtlasSharedPreprocessing::preprocess(model);
    let prover_preprocessing = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(pp);

    let timing = std::time::Instant::now();
    let (proof, io, _debug_info) =
        ONNXProof::<Fr, Blake2bTranscript, HyperKZG<Bn254>>::prove(&prover_preprocessing, &inputs);
    println!("Proof generation took {:.2?}", timing.elapsed());

    let verifier_preprocessing =
        AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_preprocessing);
    proof
        .verify(&verifier_preprocessing, &io, None)
        .map_err(|err| format!("proof verification failed: {err:?}"))?;
    println!("Proof verified successfully!");
    Ok(())
}
