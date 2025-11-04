use crate::jolt::JoltSNARK;
use ark_bn254::Fr;
use jolt_core::{poly::commitment::dory::DoryCommitmentScheme, transcripts::KeccakTranscript};
use onnx_tracer::{graph::model::Model, model, tensor::Tensor};
use serde_json::Value;
use std::{collections::HashMap, fs::File, io::Read, path::PathBuf};

type PCS = DoryCommitmentScheme;

#[derive(Debug, Copy, Clone, clap::ValueEnum)]
pub enum BenchType {
    MLP,
    ArticleClassification,
}

pub fn benchmarks(bench_type: BenchType) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    match bench_type {
        BenchType::MLP => mlp(),
        BenchType::ArticleClassification => article_classification(),
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
        || model(&"../tests/perceptron_2.onnx".into()),
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
        let working_dir: &str = "../onnx-tracer/models/article_classification/";

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
