//! Article Classification Example
//!
//! This example demonstrates how to use the Jolt SNARK system with an article classification model.
//! The model classifies text into categories: business, entertainment, politics, sport, tech.
#![allow(clippy::upper_case_acronyms)]

use atlas_onnx_tracer::{model::Model, tensor::Tensor};
use serde_json::Value;
use std::{collections::HashMap, fs::File, io::Read};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Article Classification");
    println!("======================================");

    let working_dir = "atlas-onnx-tracer/models/article_classification/";

    // Load the vocab mapping from JSON
    let vocab_path = format!("{working_dir}/vocab.json");
    let vocab = load_vocab(&vocab_path).expect("Failed to load vocab");
    println!("Loaded vocabulary with {} entries", vocab.len());

    // Test inputs and expected outputs
    let test_cases = [
        ("The government plans new trade policies.", "business"),
        ("The latest computer model has impressive features.", "tech"),
        ("The football match ended in a thrilling draw.", "sport"),
        (
            "The new movie has received rave reviews from critics.",
            "entertainment",
        ),
        ("The stock market saw a significant drop today.", "business"),
    ];

    let classes = ["business", "entertainment", "politics", "sport", "tech"];
    println!("\nTesting model outputs:");
    println!("======================");

    // Test all inputs to verify model accuracy
    let mut correct_predictions = 0;
    for (i, (input_text, expected)) in test_cases.iter().enumerate() {
        let input_vector = build_input_vector(input_text, &vocab);
        let input = Tensor::new(Some(&input_vector), &[1, 512]).unwrap();

        // Load and run model
        let model_instance =
            Model::load(&format!("{working_dir}network.onnx"), &Default::default());
        let result = model_instance.forward(&[input.clone()]);
        let output = result[0].clone();

        // Get prediction
        let (pred_idx, max_val) = output
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        let predicted = classes[pred_idx];
        let is_correct = predicted == *expected;

        if is_correct {
            correct_predictions += 1;
        }

        println!("\nTest {}: '{}'", i + 1, input_text);
        println!(
            "  Expected: {} | Predicted: {} | Confidence: {:.4} | {}",
            expected,
            predicted,
            max_val,
            if is_correct { "CORRECT" } else { "INCORRECT" }
        );
    }

    let accuracy = (correct_predictions as f32 / test_cases.len() as f32) * 100.0;
    println!(
        "\nModel Accuracy: {}/{} ({:.1}%)",
        correct_predictions,
        test_cases.len(),
        accuracy
    );

    println!("\nArticle classification example completed successfully!");

    Ok(())
}

/// Load vocab.json into HashMap<String, (usize, i32)>
fn load_vocab(path: &str) -> Result<HashMap<String, (usize, i32)>, Box<dyn std::error::Error>> {
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

fn build_input_vector(text: &str, vocab: &HashMap<String, (usize, i32)>) -> Vec<i32> {
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
