//! Authorization Model Example
//!
//! This example demonstrates how to use the Jolt SNARK system with a transaction authorization model.
//! The model makes authorization decisions based on transaction features like budget, trust, amount, etc.

use ark_bn254::Fr;
use jolt_core::{poly::commitment::dory::DoryCommitmentScheme, transcripts::KeccakTranscript};
use onnx_tracer::{model, tensor::Tensor};
use serde_json::Value;
use std::{collections::HashMap, fs::File, io::Read, path::PathBuf};
use zkml_jolt_core::jolt::JoltSNARK;

#[allow(clippy::upper_case_acronyms)]
type PCS = DoryCommitmentScheme;

/// Load authorization vocab.json into HashMap<String, usize>
fn load_authorization_vocab(
    path: &str,
) -> Result<HashMap<String, usize>, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let json_value: Value = serde_json::from_str(&contents)?;
    let mut vocab = HashMap::new();

    if let Some(Value::Object(map)) = json_value.get("vocab_mapping") {
        for (feature_key, data) in map {
            if let Some(index) = data.get("index").and_then(|v| v.as_u64()) {
                vocab.insert(feature_key.clone(), index as usize);
            }
        }
    }

    Ok(vocab)
}

/// Transaction features for the authorization model
#[derive(Debug)]
pub struct AuthorizationFeatures {
    pub budget: usize,
    pub trust: usize,
    pub amount: usize,
    pub category: usize,
    pub velocity: usize,
    pub day: usize,
    pub time: usize,
    pub risk: usize,
}

/// Build input vector for authorization model from transaction features
fn build_authorization_vector(
    features: &AuthorizationFeatures,
    vocab: &HashMap<String, usize>,
) -> Vec<i32> {
    let mut vec = vec![0; 64];

    // Map each feature to its one-hot position
    let feature_values = [
        ("budget", features.budget),
        ("trust", features.trust),
        ("amount", features.amount),
        ("category", features.category),
        ("velocity", features.velocity),
        ("day", features.day),
        ("time", features.time),
        ("risk", features.risk),
    ];

    for (feature_type, value) in feature_values {
        let feature_key = format!("{feature_type}_{value}");
        if let Some(&index) = vocab.get(&feature_key) {
            if index < 64 {
                vec[index] = 1; // One-hot encoding
            }
        }
    }

    vec
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Transaction Authorization with Jolt SNARK");
    println!("=========================================");

    let working_dir = "onnx-tracer/models/authorization/";

    // Load the vocab mapping from JSON
    let vocab_path = format!("{working_dir}/vocab.json");
    let vocab = load_authorization_vocab(&vocab_path).expect("Failed to load authorization vocab");
    println!(
        "Loaded authorization vocabulary with {} entries",
        vocab.len()
    );

    // Test authorization scenarios
    // (budget, trust, amount, category, velocity, day, time, risk, expected_decision)
    let test_scenarios = [
        (15, 7, 8, 0, 2, 1, 1, 0, "AUTHORIZED"), // High trust, sufficient budget
        (5, 4, 12, 0, 2, 1, 1, 0, "DENIED"),     // Amount exceeds budget
        (15, 1, 12, 0, 2, 1, 1, 0, "DENIED"),    // Low trust, high amount
        (15, 7, 8, 0, 7, 1, 1, 0, "DENIED"),     // High velocity
        (15, 0, 5, 2, 2, 1, 1, 0, "DENIED"),     // Restricted category for untrusted merchant
        (15, 7, 14, 0, 2, 1, 3, 0, "DENIED"),    // Late night high-value transaction
    ];

    println!("\nTesting authorization model:");
    println!("=============================");

    let mut correct_predictions = 0;

    for (i, &(budget, trust, amount, category, velocity, day, time, risk, expected)) in
        test_scenarios.iter().enumerate()
    {
        // Build input vector from transaction features
        let features = AuthorizationFeatures {
            budget,
            trust,
            amount,
            category,
            velocity,
            day,
            time,
            risk,
        };
        let input_vector = build_authorization_vector(&features, &vocab);
        let input = Tensor::new(Some(&input_vector), &[1, 64]).unwrap();

        // Load and run model
        let model_fn = || model(&PathBuf::from(format!("{working_dir}network.onnx")));
        let model_instance = model_fn();
        let result = model_instance.forward(&[input.clone()]).unwrap();
        let output = result.outputs[0].clone();

        // Get prediction (class 0 = AUTHORIZED, class 1 = DENIED)
        let (pred_idx, max_val) = output
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        let prediction = if pred_idx == 0 {
            "AUTHORIZED"
        } else {
            "DENIED"
        };

        let is_correct = prediction == expected;
        if is_correct {
            correct_predictions += 1;
        }

        println!(
            "\nTest {}: budget={}, trust={}, amount={}, category={}, velocity={}, day={}, time={}, risk={}",
            i + 1,
            budget,
            trust,
            amount,
            category,
            velocity,
            day,
            time,
            risk
        );
        println!(
            "  Expected: {} | Predicted: {} | Confidence: {:.4} | {}",
            expected,
            prediction,
            max_val,
            if is_correct { "CORRECT" } else { "INCORRECT" }
        );
    }

    let accuracy = (correct_predictions as f32 / test_scenarios.len() as f32) * 100.0;
    println!(
        "\nAuthorization Accuracy: {}/{} ({:.1}%)",
        correct_predictions,
        test_scenarios.len(),
        accuracy
    );

    // Generate a proof for the first test case (authorized transaction)
    println!("\nGenerating SNARK proof for authorized transaction:");
    println!("=================================================");

    let proof_scenario = test_scenarios[0];
    let (budget, trust, amount, category, velocity, day, time, risk, _) = proof_scenario;

    let proof_features = AuthorizationFeatures {
        budget,
        trust,
        amount,
        category,
        velocity,
        day,
        time,
        risk,
    };

    println!("Transaction: budget={budget}, trust={trust}, amount={amount}, category={category}, velocity={velocity}, day={day}, time={time}, risk={risk}");

    let proof_input_vector = build_authorization_vector(&proof_features, &vocab);
    let proof_input = Tensor::new(Some(&proof_input_vector), &[1, 64]).unwrap();

    println!("Preprocessing model...");

    let model_fn = || model(&PathBuf::from(format!("{working_dir}network.onnx")));
    let preprocessing =
        JoltSNARK::<Fr, PCS, KeccakTranscript>::prover_preprocess(model_fn, 1 << 14);

    println!("Generating proof...");
    let start_time = std::time::Instant::now();
    let (snark, program_io, _debug_info) =
        JoltSNARK::<Fr, PCS, KeccakTranscript>::prove(&preprocessing, model_fn, &proof_input);
    let prove_time = start_time.elapsed();

    println!("Verifying proof...");
    let start_time = std::time::Instant::now();
    snark.verify(&(&preprocessing).into(), program_io, None)?;
    let verify_time = start_time.elapsed();

    println!("Proof verified successfully!");
    println!("Proving time: {prove_time:?}");
    println!("Verification time: {verify_time:?}");

    println!("\nAuthorization example completed successfully!");

    Ok(())
}
