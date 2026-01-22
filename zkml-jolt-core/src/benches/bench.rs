use crate::jolt::JoltSNARK;
use ark_bn254::Fr;
use jolt_core::{poly::commitment::dory::DoryCommitmentScheme, transcripts::KeccakTranscript};
use onnx_tracer::{builder, graph::model::Model, model, tensor::Tensor};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde_json::Value;
use std::{collections::HashMap, fs::File, io::Read, path::PathBuf, time::Instant};

type PCS = DoryCommitmentScheme;

#[derive(Debug, Copy, Clone, clap::ValueEnum)]
pub enum BenchType {
    MLP,
    ArticleClassification,
    SelfAttention,
    ZKComparison,
}

pub fn benchmarks(bench_type: BenchType) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    match bench_type {
        BenchType::MLP => mlp(),
        BenchType::ArticleClassification => article_classification(),
        BenchType::SelfAttention => self_attention(),
        BenchType::ZKComparison => zk_comparison(),
    }
}

fn prove_and_verify<F, const N: usize>(
    model: F,
    input_data: [i32; N],
    shape: [usize; 2],
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: Fn() -> Model + 'static + Copy,
{
    let mut tasks = Vec::new();
    let task = move || {
        let input = Tensor::new(Some(&input_data), &shape).unwrap();
        let preprocessing =
            JoltSNARK::<Fr, PCS, KeccakTranscript>::prover_preprocess(model, 1 << 14);
        let (snark, program_io, _debug_info) =
            JoltSNARK::<Fr, PCS, KeccakTranscript>::prove(&preprocessing, model, &input);
        snark
            .verify(&(&preprocessing).into(), program_io, None)
            .unwrap();
    };
    tasks.push((
        tracing::info_span!("Example_E2E"),
        Box::new(task) as Box<dyn FnOnce()>,
    ));
    tasks
}

fn mlp() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    prove_and_verify(
        || model(&"tests/perceptron_2.onnx".into()),
        [1, 2, 3, 4],
        [1, 4],
    )
}

/// Load vocab.json into HashMap<String, (usize, i32)>
pub fn load_vocab(path: &str) -> Result<HashMap<String, (usize, i32)>, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let json_value: Value = serde_json::from_str(&contents)?;
    let mut vocab = HashMap::new();

    if let Value::Object(map) = json_value {
        for (word, data) in map {
            if let (Some(index), Some(idf)) = (
                data.get("index").and_then(|v| v.as_u64()),
                data.get("idf").and_then(|v| v.as_f64()),
            ) {
                vocab.insert(word, (index as usize, (idf * 1000.0) as i32)); // Scale IDF and convert to i32
            }
        }
    }

    Ok(vocab)
}

pub fn build_input_vector(text: &str, vocab: &HashMap<String, (usize, i32)>) -> Vec<i32> {
    let mut vec = vec![0; 512];

    // Split text into tokens (preserve punctuation as tokens)
    let re = regex::Regex::new(r"\w+|[^\w\s]").unwrap();
    for cap in re.captures_iter(text) {
        let token = cap.get(0).unwrap().as_str().to_lowercase();
        if let Some(&(index, idf)) = vocab.get(&token) {
            if index < 512 {
                vec[index] += idf; // accumulate idf value
            }
        }
    }

    vec
}

fn article_classification() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let mut tasks = Vec::new();
    let task = move || {
        let working_dir: &str = "onnx-tracer/models/article_classification/";

        // Load the vocab mapping from JSON
        let vocab_path = format!("{working_dir}/vocab.json");
        let vocab = load_vocab(&vocab_path).expect("Failed to load vocab");

        // Input text string to classify
        let input_text = "The government plans new trade policies.";

        // Build input vector from the input text (512 features for small MLP)
        let input_vector = build_input_vector(input_text, &vocab);

        let input = Tensor::new(Some(&input_vector), &[1, 512]).unwrap();
        let model_func = || model(&PathBuf::from(format!("{working_dir}network.onnx")));
        let preprocessing =
            JoltSNARK::<Fr, PCS, KeccakTranscript>::prover_preprocess(model_func, 1 << 20);
        let (snark, program_io, _debug_info) =
            JoltSNARK::<Fr, PCS, KeccakTranscript>::prove(&preprocessing, model_func, &input);
        snark
            .verify(&(&preprocessing).into(), program_io, None)
            .unwrap();
    };
    tasks.push((
        tracing::info_span!("ArticleClassification_E2E"),
        Box::new(task) as Box<dyn FnOnce()>,
    ));
    tasks
}

fn self_attention() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let mut tasks = Vec::new();
    let task = move || {
        let model_func = || {
            model(&PathBuf::from(
                "onnx-tracer/models/self_attention/network.onnx",
            ))
        };
        let shape = [1, 64, 64];
        let mut rng = StdRng::seed_from_u64(123456);
        let mut input_data = vec![0i32; shape.iter().product()];
        for input in input_data.iter_mut() {
            *input = rng.gen_range(-256..256);
        }
        let input = Tensor::new(Some(&input_data), &shape).unwrap();
        let preprocessing =
            JoltSNARK::<Fr, PCS, KeccakTranscript>::prover_preprocess(model_func, 1 << 21);
        let (snark, program_io, _debug_info) =
            JoltSNARK::<Fr, PCS, KeccakTranscript>::prove(&preprocessing, model_func, &input);
        snark
            .verify(&(&preprocessing).into(), program_io, None)
            .unwrap();
    };
    tasks.push((
        tracing::info_span!("SelfAttention_E2E"),
        Box::new(task) as Box<dyn FnOnce()>,
    ));
    tasks
}

/// Benchmark comparing ZK vs non-ZK proving overhead.
///
/// Runs the same model with both `prove` (non-ZK) and `prove_full_zk` (ZK with hiding commitments)
/// to measure the overhead of zero-knowledge proving.
///
/// Uses the article classification model (165K parameters) for a realistic benchmark.
fn zk_comparison() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let mut tasks = Vec::new();

    // Combined benchmark task that runs both and compares
    let comparison_task = move || {
        let working_dir: &str = "onnx-tracer/models/article_classification/";

        // Load the vocab mapping from JSON
        let vocab_path = format!("{working_dir}/vocab.json");
        let vocab = load_vocab(&vocab_path).expect("Failed to load vocab");

        // Input text string to classify
        let input_text = "The government plans new trade policies.";

        // Build input vector from the input text (512 features for small MLP)
        let input_vector = build_input_vector(input_text, &vocab);

        let model_func = || model(&PathBuf::from(format!("{working_dir}network.onnx")));

        println!("\n{}", "=".repeat(60));
        println!("  ZK vs Non-ZK Proving Benchmark");
        println!("  Model: Article Classification (165K parameters)");
        println!("  Input: \"{}\"", input_text);
        println!("{}", "=".repeat(60));

        // Preprocessing (shared between both modes)
        println!("\n  Preprocessing...");
        let preprocess_start = Instant::now();
        let preprocessing =
            JoltSNARK::<Fr, PCS, KeccakTranscript>::prover_preprocess(model_func, 1 << 20);
        let preprocess_time = preprocess_start.elapsed();
        println!("    Preprocess time: {:>10.2?}", preprocess_time);

        // Non-ZK proving
        {
            let input = Tensor::new(Some(&input_vector), &[1, 512]).unwrap();

            println!("\n  Non-ZK Mode:");
            let start = Instant::now();
            let (snark, program_io, _debug_info) =
                JoltSNARK::<Fr, PCS, KeccakTranscript>::prove(&preprocessing, model_func, &input);
            let non_zk_prove_time = start.elapsed();
            println!("    Prove time:  {:>12.2?}", non_zk_prove_time);

            let start = Instant::now();
            snark
                .verify(&(&preprocessing).into(), program_io, None)
                .unwrap();
            let non_zk_verify_time = start.elapsed();
            println!("    Verify time: {:>12.2?}", non_zk_verify_time);

            // Full ZK proving
            let input = Tensor::new(Some(&input_vector), &[1, 512]).unwrap();

            println!("\n  Full ZK Mode (Hiding Commitments):");
            let start = Instant::now();
            let (snark_zk, program_io_zk, _debug_info) =
                JoltSNARK::<Fr, PCS, KeccakTranscript>::prove_full_zk(
                    &preprocessing,
                    model_func,
                    &input,
                );
            let zk_prove_time = start.elapsed();
            println!("    Prove time:  {:>12.2?}", zk_prove_time);

            let start = Instant::now();
            snark_zk
                .verify_full_zk(&(&preprocessing).into(), program_io_zk, None)
                .unwrap();
            let zk_verify_time = start.elapsed();
            println!("    Verify time: {:>12.2?}", zk_verify_time);

            // Calculate overhead
            let prove_overhead =
                (zk_prove_time.as_secs_f64() / non_zk_prove_time.as_secs_f64() - 1.0) * 100.0;
            let verify_overhead =
                (zk_verify_time.as_secs_f64() / non_zk_verify_time.as_secs_f64() - 1.0) * 100.0;

            println!("\n  ZK Overhead:");
            println!("    Prove:  {:>+7.1}%", prove_overhead);
            println!("    Verify: {:>+7.1}%", verify_overhead);
            println!("\n{}\n", "=".repeat(60));
        }
    };

    tasks.push((
        tracing::info_span!("ZK_Comparison"),
        Box::new(comparison_task) as Box<dyn FnOnce()>,
    ));

    tasks
}
