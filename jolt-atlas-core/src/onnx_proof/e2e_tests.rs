use crate::onnx_proof::{
    proof_serialization::serialize_proof, AtlasProverPreprocessing, AtlasSharedPreprocessing,
    AtlasVerifierPreprocessing, ONNXProof,
};
use ark_bn254::{Bn254, Fr};
use atlas_onnx_tracer::{
    model::{trace::ModelExecutionIO, Model, RunArgs},
    tensor::Tensor,
};
use joltworks::{poly::commitment::hyperkzg::HyperKZG, transcripts::Blake2bTranscript};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde_json::Value;
use std::{collections::HashMap, fs::File, io::Read, time::Instant};

// Fixed-point scale factor: 2^7 = 128
const SCALE: i32 = 128;

/// Configuration for test prove-and-verify workflows.
///
/// Uses a builder pattern — all options default to `false`.
///
/// ```ignore
/// let io = prove_and_verify(dir, &[input], &RunArgs::default(), TestConfig::default());
/// let io = prove_and_verify(dir, &[input], &run_args, TestConfig::new()
///     .print_model()
///     .print_timing()
///     .debug_info());
/// ```
#[derive(Clone, Debug, Default)]
struct TestConfig {
    print_model: bool,
    print_timing: bool,
    debug_info: bool,
    print_proof_size: bool,
}

impl TestConfig {
    fn new() -> Self {
        Self::default()
    }

    fn print_model(mut self) -> Self {
        self.print_model = true;
        self
    }

    fn print_timing(mut self) -> Self {
        self.print_timing = true;
        self
    }

    fn debug_info(mut self) -> Self {
        self.debug_info = true;
        self
    }

    fn print_proof_size(mut self) -> Self {
        self.print_proof_size = true;
        self
    }
}

/// Run the prove-and-verify workflow, returning the execution IO.
fn prove_and_verify(
    model_dir: &str,
    inputs: &[Tensor<i32>],
    run_args: &RunArgs,
    config: TestConfig,
) -> ModelExecutionIO {
    let model = Model::load(&format!("{model_dir}network.onnx"), run_args);
    if config.print_model {
        println!("model: {}", model.pretty_print());
    }

    let pp = AtlasSharedPreprocessing::preprocess(model);
    let prover_preprocessing = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(pp);

    let timing = Instant::now();
    let (proof, io, debug_info) =
        ONNXProof::<Fr, Blake2bTranscript, HyperKZG<Bn254>>::prove(&prover_preprocessing, inputs);
    if config.print_timing {
        println!("Proof generation took {:?}", timing.elapsed());
    }

    if config.print_proof_size {
        let bytes = serialize_proof(&proof).expect("proof serialization failed");
        println!(
            "Proof size: {:.1} kB ({} bytes)",
            bytes.len() as f64 / 1024.0,
            bytes.len()
        );
    }

    let verifier_preprocessing =
        AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_preprocessing);

    let debug_for_verify = if config.debug_info { debug_info } else { None };
    let timing = Instant::now();
    proof
        .verify(&verifier_preprocessing, &io, debug_for_verify)
        .unwrap();
    if config.print_timing {
        println!("Proof verification took {:?}", timing.elapsed());
    }

    io
}

#[ignore = "requires GPT-2 ONNX model download (run scripts/download_gpt2.py first)"]
#[test]
fn test_gpt2() {
    let working_dir = "../atlas-onnx-tracer/models/gpt2/";
    let mut rng = StdRng::seed_from_u64(42);
    let seq_len: usize = 16;
    let vocab_size: i32 = 50257;

    // Input 0: input_ids — random token IDs used as Gather indices
    let input_ids_data: Vec<i32> = (0..seq_len).map(|_| rng.gen_range(0..vocab_size)).collect();
    let input_ids = Tensor::new(Some(&input_ids_data), &[1, seq_len]).unwrap();

    // Input 1: position_ids — sequential positions used as Gather indices
    let position_ids_data: Vec<i32> = (0..seq_len as i32).collect();
    let position_ids = Tensor::new(Some(&position_ids_data), &[1, seq_len]).unwrap();

    // Input 2: attention_mask — all 1s (attend everywhere)
    // The model's Cast handler divides by scale to de-quantize, so we provide
    // the mask in quantized form: 1.0 in fixed-point = 1 << scale.
    let attention_mask_data: Vec<i32> = vec![SCALE; seq_len];
    let attention_mask = Tensor::new(Some(&attention_mask_data), &[1, seq_len]).unwrap();

    // Configure RunArgs for GPT-2
    let run_args = RunArgs::new([
        ("batch_size", 1),
        ("sequence_length", seq_len),
        ("past_sequence_length", 0),
    ])
    .with_pre_rebase_nonlinear(true);

    prove_and_verify(
        working_dir,
        &[input_ids, position_ids, attention_mask],
        &run_args,
        TestConfig::new().print_model().print_timing(),
    );
}

#[test]
fn test_nanoGPT() {
    let working_dir = "../atlas-onnx-tracer/models/nanoGPT/";
    let mut rng = StdRng::seed_from_u64(0x1096);
    let input_data: Vec<i32> = (0..64)
        .map(|_| (1 << 5) + rng.gen_range(-20..=20))
        .collect();
    let input = Tensor::new(Some(&input_data), &[1, 64]).unwrap();
    prove_and_verify(
        working_dir,
        &[input],
        &Default::default(),
        TestConfig::new()
            .print_model()
            .print_timing()
            .print_proof_size(),
    );
}

#[test]
fn test_transformer() {
    let working_dir = "../atlas-onnx-tracer/models/transformer/";
    let mut rng = StdRng::seed_from_u64(0x1096);
    let input_data: Vec<i32> = (0..64 * 64)
        .map(|_| (1 << 7) + rng.gen_range(-50..=50))
        .collect();
    let input = Tensor::new(Some(&input_data), &[1, 64, 64]).unwrap();
    prove_and_verify(
        working_dir,
        &[input],
        &Default::default(),
        TestConfig::default().print_timing().print_proof_size(),
    );
}

#[test]
fn test_minigpt() {
    let working_dir = "../atlas-onnx-tracer/models/minigpt/";
    let mut rng = StdRng::seed_from_u64(0x42);

    // Model hyperparameters (matching the minigpt Python script by @karpathy)
    // vocab_size=1024, n_embd=32, n_head=8, n_layer=2, block_size=32
    let vocab_size: i32 = 1024;
    let block_size = 32;

    // Generate random token IDs in [0, vocab_size)
    let input_data: Vec<i32> = (0..block_size)
        .map(|_| rng.gen_range(0..vocab_size))
        .collect();
    let input = Tensor::new(Some(&input_data), &[1, block_size]).unwrap();
    prove_and_verify(
        working_dir,
        &[input],
        &Default::default(),
        TestConfig::new().print_model().print_timing(),
    );
}

#[test]
fn test_microgpt() {
    let working_dir = "../atlas-onnx-tracer/models/microgpt/";
    let mut rng = StdRng::seed_from_u64(0x42);

    // Model hyperparameters (matching the microGPT Python script by @karpathy)
    // vocab_size=32, n_embd=16, n_head=4, n_layer=1, block_size=16
    let vocab_size: i32 = 32;
    let block_size = 16;

    // Generate random token IDs in [0, vocab_size)
    let input_data: Vec<i32> = (0..block_size)
        .map(|_| rng.gen_range(0..vocab_size))
        .collect();
    let input = Tensor::new(Some(&input_data), &[1, block_size]).unwrap();
    prove_and_verify(
        working_dir,
        &[input],
        &Default::default(),
        TestConfig::new().print_model().print_timing(),
    );
}

#[test]
fn test_layernorm_head() {
    let working_dir = "../atlas-onnx-tracer/models/layernorm_head/";
    let mut rng = StdRng::seed_from_u64(0x8096);
    let input_data: Vec<i32> = (0..16 * 16)
        .map(|_| (1 << 7) + rng.gen_range(-50..=50))
        .collect();
    let input = Tensor::construct(input_data, vec![16, 16]);
    prove_and_verify(
        working_dir,
        &[input],
        &Default::default(),
        TestConfig::default(),
    );
}

#[test]
fn test_multihead_attention() {
    let working_dir = "../atlas-onnx-tracer/models/multihead_attention/";
    let mut rng = StdRng::seed_from_u64(0x1013);
    let input_data: Vec<i32> = (0..16 * 128)
        .map(|_| SCALE + rng.gen_range(-10..=10))
        .collect();
    let input = Tensor::construct(input_data, vec![1, 1, 16, 128]);
    prove_and_verify(
        working_dir,
        &[input],
        &Default::default(),
        TestConfig::new().print_model().debug_info(),
    );
}

#[test]
fn test_self_attention_layer() {
    let working_dir = "../atlas-onnx-tracer/models/self_attention_layer/";
    let mut rng = StdRng::seed_from_u64(0x1003);
    let input_data: Vec<i32> = (0..64 * 64)
        .map(|_| SCALE + rng.gen_range(-10..=10))
        .collect();
    let input = Tensor::construct(input_data, vec![1, 64, 64]);

    prove_and_verify(
        working_dir,
        &[input],
        &Default::default(),
        TestConfig::new().print_model().debug_info(),
    );
}

#[test]
fn test_sum_axes() {
    let working_dir = "../atlas-onnx-tracer/models/sum_axes_test/";
    let mut rng = StdRng::seed_from_u64(0x923);
    let input = Tensor::random_small(&mut rng, &[1, 4, 8]);
    prove_and_verify(
        working_dir,
        &[input],
        &Default::default(),
        TestConfig::new().print_model().debug_info(),
    );
}

#[ignore = "hzkg fails when all coeffs are zero"]
#[test]
fn test_sum_independent() {
    let working_dir = "../atlas-onnx-tracer/models/sum_independent/";
    let mut rng = StdRng::seed_from_u64(0x923);
    let input = Tensor::random_small(&mut rng, &[1, 4, 8]);
    prove_and_verify(
        working_dir,
        &[input],
        &Default::default(),
        TestConfig::new().print_model().debug_info(),
    );
}

#[test]
fn test_sum_operations_e2e() {
    // Test 1D sum along axis 0
    let working_dir = "../atlas-onnx-tracer/models/sum_1d_axis0/";
    let mut rng = StdRng::seed_from_u64(0x923);
    let input = Tensor::random_small(&mut rng, &[8]);
    prove_and_verify(
        working_dir,
        &[input],
        &Default::default(),
        TestConfig::new().print_model().debug_info(),
    );

    // Test 2D sum along axis 0
    let working_dir = "../atlas-onnx-tracer/models/sum_2d_axis0/";
    let mut rng = StdRng::seed_from_u64(0x924);
    let input = Tensor::random_small(&mut rng, &[4, 8]);
    prove_and_verify(
        working_dir,
        &[input],
        &Default::default(),
        TestConfig::new().print_model().debug_info(),
    );

    // Test 2D sum along axis 1
    let working_dir = "../atlas-onnx-tracer/models/sum_2d_axis1/";
    let mut rng = StdRng::seed_from_u64(0x925);
    let input = Tensor::random_small(&mut rng, &[4, 8]);
    prove_and_verify(
        working_dir,
        &[input],
        &Default::default(),
        TestConfig::new().print_model().debug_info(),
    );

    // Test 3D sum along axis 2
    let working_dir = "../atlas-onnx-tracer/models/sum_3d_axis2/";
    let mut rng = StdRng::seed_from_u64(0x926);
    let input = Tensor::random_small(&mut rng, &[1, 4, 8]);
    prove_and_verify(
        working_dir,
        &[input],
        &Default::default(),
        TestConfig::new().print_model().debug_info(),
    );
}

#[ignore = "hzkg fails when all coeffs are zero"]
#[test]
fn test_layernorm_partial_head() {
    let working_dir = "../atlas-onnx-tracer/models/layernorm_partial_head/";
    let input_data = vec![SCALE; 16 * 16];
    let input = Tensor::construct(input_data, vec![16, 16]);
    prove_and_verify(
        working_dir,
        &[input],
        &Default::default(),
        TestConfig::new().print_model(),
    );
}

#[test]
fn test_article_classification() {
    let working_dir = "../atlas-onnx-tracer/models/article_classification/";

    // Load the vocab mapping from JSON
    let vocab_path = format!("{working_dir}/vocab.json");
    let vocab = load_vocab(&vocab_path).expect("Failed to load vocab");
    // Input text string to classify
    let input_text = "The government plans new trade policies.";

    // Build input vector from the input text (512 features for small MLP)
    let input_vector = build_input_vector(input_text, &vocab);
    let input = Tensor::construct(input_vector, vec![1, 512]);

    prove_and_verify(
        working_dir,
        &[input],
        &Default::default(),
        TestConfig::new().print_model().print_timing(),
    );

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
                    vocab.insert(word, (index as usize, (idf * 1000.0) as i32));
                    // Scale IDF and convert to i32
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
}

#[ignore = "hzkg fails when all coeffs are zero"]
#[test]
fn test_add_sub_mul() {
    let working_dir = "../atlas-onnx-tracer/models/test_add_sub_mul/";

    // Create test input vector of size 65536
    // Using small values to avoid overflow
    let mut rng = StdRng::seed_from_u64(0x100);
    // Create tensor with shape [65536]
    let input = Tensor::random_range(&mut rng, &[1 << 16], -(1 << 10)..(1 << 10));

    prove_and_verify(
        working_dir,
        &[input],
        &Default::default(),
        TestConfig::new().print_model().print_timing().debug_info(),
    );
}

#[test]
fn test_rsqrt() {
    let working_dir = "../atlas-onnx-tracer/models/rsqrt/";

    // Create test input vector of size 4
    let mut rng = StdRng::seed_from_u64(0x100);
    let input_vec = (0..4)
        .map(|_| rng.gen_range(1..i32::MAX))
        .collect::<Vec<i32>>();
    let input = Tensor::construct(input_vec, vec![4]);

    prove_and_verify(
        working_dir,
        &[input],
        &Default::default(),
        TestConfig::new().print_model().print_timing(),
    );
}

#[test]
fn test_perceptron() {
    let working_dir = "../atlas-onnx-tracer/models/perceptron/";
    let input = Tensor::construct(vec![1, 2, 3, 4], vec![1, 4]);

    prove_and_verify(
        working_dir,
        &[input],
        &Default::default(),
        TestConfig::new()
            .print_model()
            .print_timing()
            .print_proof_size(),
    );
}

#[ignore = "hzkg fails when all coeffs are zero"]
#[test]
fn test_broadcast() {
    let working_dir = "../atlas-onnx-tracer/models/broadcast/";
    let input = Tensor::construct(vec![1, 2, 3, 4], vec![4]);

    let io = prove_and_verify(
        working_dir,
        &[input],
        &Default::default(),
        TestConfig::new().print_model().print_timing(),
    );

    // Print output for verification
    println!("Output shape: {:?}", io.outputs[0].dims());
    println!("Expected: input [4] broadcasted through operations to shape [2, 5, 4]");
}

#[test]
fn test_reshape() {
    let working_dir = "../atlas-onnx-tracer/models/reshape/";
    let input = Tensor::construct(vec![1, 2, 3, 4], vec![4]);

    let io = prove_and_verify(
        working_dir,
        &[input],
        &Default::default(),
        TestConfig::new().print_model().print_timing(),
    );

    println!("Output shape: {:?}", io.outputs[0].dims());
}

#[test]
fn test_moveaxis() {
    let working_dir = "../atlas-onnx-tracer/models/moveaxis/";
    let input_vector: Vec<i32> = (1..=64).collect();
    let input = Tensor::construct(input_vector, vec![2, 4, 8]);

    let io = prove_and_verify(
        working_dir,
        &[input],
        &Default::default(),
        TestConfig::new().print_model().print_timing(),
    );

    println!("Output shape: {:?}", io.outputs[0].dims());
}

#[test]
fn test_gather() {
    let working_dir = "../atlas-onnx-tracer/models/gather/";
    let mut rng = StdRng::seed_from_u64(0x100);
    // Input values in [0, 8)
    let input = Tensor::random_range(&mut rng, &[1, 64], 0..65);

    prove_and_verify(
        working_dir,
        &[input],
        &Default::default(),
        TestConfig::new().print_model(),
    );
}

#[test]
fn test_tanh() {
    let working_dir = "../atlas-onnx-tracer/models/tanh/";
    let input_vector = vec![10, 40, 70, 100];
    let input = Tensor::new(Some(&input_vector), &[4]).unwrap();

    prove_and_verify(
        working_dir,
        &[input],
        &Default::default(),
        TestConfig::new().print_model().print_timing(),
    );
}

#[test]
fn test_mlp_square() {
    let working_dir = "../atlas-onnx-tracer/models/mlp_square/";
    let input_vector = vec![
        (70.0 * SCALE as f32) as i32,
        (71.0 * SCALE as f32) as i32,
        (72.0 * SCALE as f32) as i32,
        (73.0 * SCALE as f32) as i32,
    ];
    let input = Tensor::new(Some(&input_vector), &[1, 4]).unwrap();

    prove_and_verify(
        working_dir,
        &[input],
        &Default::default(),
        TestConfig::default(),
    );
}

#[test]
fn test_mlp_square_4layer() {
    let working_dir = "../atlas-onnx-tracer/models/mlp_square_4layer/";
    let input_vector = vec![
        (1.0 * SCALE as f32) as i32,
        (2.0 * SCALE as f32) as i32,
        (3.0 * SCALE as f32) as i32,
        (4.0 * SCALE as f32) as i32,
    ];
    let input = Tensor::new(Some(&input_vector), &[1, 4]).unwrap();

    prove_and_verify(
        working_dir,
        &[input],
        &Default::default(),
        TestConfig::default(),
    );
}
