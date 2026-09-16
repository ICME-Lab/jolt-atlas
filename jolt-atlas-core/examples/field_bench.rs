//! Field benchmark: proves one model over BN254 and over the 128-bit Solinas
//! field with the mock commitment scheme, so the IOP cost of the field can be
//! measured without a PCS in the way. BN254 + HyperKZG is included as the
//! production reference.
//!
//! ```bash
//! cargo run --release -p jolt-atlas-core --example field_bench -- [nanoGPT|gpt2|qwen] [runs] [config filter]
//! ```

use atlas_onnx_tracer::{
    model::{Model, RunArgs},
    tensor::Tensor,
};
use jolt_atlas_core::onnx_proof::{
    AkitaScheme, AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing,
    Blake2bTranscript, Bn254, DoryScheme, Fr, HyperKZG, ONNXProof,
};
use joltworks::{
    field::{fp128::Fp128, JoltField},
    poly::commitment::{commitment_scheme::CommitmentScheme, mock::MockCommitScheme},
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::{Duration, Instant};

fn nano_gpt() -> (Model, Vec<Tensor<i32>>) {
    let mut rng = StdRng::seed_from_u64(0x1096);
    let input_data: Vec<i32> = (0..64)
        .map(|_| (1 << 5) + rng.gen_range(-20..=20))
        .collect();
    let input = Tensor::new(Some(&input_data), &[1, 64]).unwrap();
    let model = Model::load(
        "atlas-onnx-tracer/models/nanoGPT/network.onnx",
        &Default::default(),
    );
    (model, vec![input])
}

fn gpt2() -> (Model, Vec<Tensor<i32>>) {
    let seq_len: usize = 16;
    let run_args = RunArgs::new([
        ("batch_size", 1),
        ("sequence_length", seq_len),
        ("past_sequence_length", 0),
    ]);
    let model = Model::load("atlas-onnx-tracer/models/gpt2/network.onnx", &run_args);
    let mut rng = StdRng::seed_from_u64(42);
    let vocab_size: i32 = 50257;
    let input_ids: Vec<i32> = (0..seq_len).map(|_| rng.gen_range(0..vocab_size)).collect();
    let position_ids: Vec<i32> = (0..seq_len as i32).collect();
    let attention_mask: Vec<i32> = vec![1 << run_args.scale; seq_len];
    let inputs = vec![
        Tensor::new(Some(&input_ids), &[1, seq_len]).unwrap(),
        Tensor::new(Some(&position_ids), &[1, seq_len]).unwrap(),
        Tensor::new(Some(&attention_mask), &[1, seq_len]).unwrap(),
    ];
    (model, inputs)
}

/// Qwen: tokenizes the default prompt of the `qwen` example. Requires
/// `common::consts::MODEL_SCALE == 14` and the model download.
fn qwen() -> (Model, Vec<Tensor<i32>>) {
    assert_eq!(
        common::consts::MODEL_SCALE,
        14,
        "Qwen needs MODEL_SCALE == 14"
    );
    let tokenizer =
        tokenizers::Tokenizer::from_file("atlas-onnx-tracer/models/qwen/tokenizer.json")
            .expect("run scripts/download_qwen.py first");
    let token_ids = tokenizer
        .encode("The quick brown fox jumps over the lazy dog", false)
        .expect("tokenization failed")
        .get_ids()
        .to_vec();
    let seq_len = token_ids.len();
    let run_args = RunArgs::new([
        ("batch_size", 1),
        ("sequence_length", seq_len),
        ("past_sequence_length", 0),
    ]);
    let scale = run_args.scale;
    let model = Model::load("atlas-onnx-tracer/models/qwen/network.onnx", &run_args);
    let input_ids: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();
    let position_ids: Vec<i32> = (0..seq_len as i32).map(|i| i << scale).collect();
    let attention_mask: Vec<i32> = vec![1 << scale; seq_len];
    let inputs = vec![
        Tensor::new(Some(&input_ids), &[1, seq_len]).unwrap(),
        Tensor::new(Some(&position_ids), &[1, seq_len]).unwrap(),
        Tensor::new(Some(&attention_mask), &[1, seq_len]).unwrap(),
    ];
    (model, inputs)
}

struct Timing {
    prove: Duration,
    verify: Duration,
    proof_bytes: usize,
}

type RunFn = fn(AtlasSharedPreprocessing, &[Tensor<i32>]) -> Timing;

fn run<F: JoltField, PCS: CommitmentScheme<Field = F>>(
    shared: AtlasSharedPreprocessing,
    inputs: &[Tensor<i32>],
) -> Timing {
    let prover_preprocessing = AtlasProverPreprocessing::<F, PCS>::new(shared);
    let t = Instant::now();
    let (proof, io, _) =
        ONNXProof::<F, Blake2bTranscript, PCS>::prove(&prover_preprocessing, inputs);
    let prove = t.elapsed();
    println!("  prove took {prove:.3?}");
    let verifier_preprocessing = AtlasVerifierPreprocessing::<F, PCS>::from(&prover_preprocessing);
    let t = Instant::now();
    if let Err(e) = proof.verify(&verifier_preprocessing, &io, None) {
        println!("  VERIFICATION FAILED: {e:?}");
    }
    let verify = t.elapsed();
    let mut bytes = Vec::new();
    ark_serialize::CanonicalSerialize::serialize_compressed(&proof, &mut bytes).unwrap();
    Timing {
        prove,
        verify,
        proof_bytes: bytes.len(),
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model_name = args.next().unwrap_or_else(|| "nanoGPT".to_string());
    let runs: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(2);
    let filter: Option<String> = args.next();
    let (model, inputs) = match model_name.as_str() {
        "nanoGPT" => nano_gpt(),
        "gpt2" => gpt2(),
        "qwen" => qwen(),
        other => panic!("unknown model {other}; use nanoGPT, gpt2, or qwen"),
    };
    println!(
        "model = {model_name}, max_num_vars = {}, runs = {runs}",
        model.max_num_vars()
    );
    let shared = AtlasSharedPreprocessing::preprocess(model);

    let configs: [(&str, RunFn); 5] = [
        ("BN254 + HyperKZG", run::<Fr, HyperKZG<Bn254>>),
        ("BN254 + Dory", run::<Fr, DoryScheme>),
        ("BN254 + Mock", run::<Fr, MockCommitScheme<Fr>>),
        ("Fp128 + Mock", run::<Fp128, MockCommitScheme<Fp128>>),
        ("Fp128 + Akita", run::<Fp128, AkitaScheme>),
    ];
    println!(
        "{:<18} {:>4} {:>12} {:>12} {:>12}",
        "config", "run", "prove", "verify", "proof bytes"
    );
    for (label, f) in configs {
        if filter.as_ref().is_some_and(|f| !label.contains(f.as_str())) {
            continue;
        }
        for i in 0..runs {
            let t = f(shared.clone(), &inputs);
            println!(
                "{:<18} {:>4} {:>12.3?} {:>12.3?} {:>12}",
                label, i, t.prove, t.verify, t.proof_bytes
            );
        }
    }
}
