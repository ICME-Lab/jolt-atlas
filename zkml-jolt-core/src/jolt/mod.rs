//! --- Jolt ONNX VM ---

use crate::jolt::{
    bytecode::{BytecodePreprocessing, JoltONNXBytecode},
    dag::{
        jolt_dag::JoltDAG,
        state_manager::{Proofs, StateManager, VerifierState},
    },
    pcs::{Openings, ProverOpeningAccumulator, VerifierOpeningAccumulator},
    precompiles::PrecompilePreprocessing,
    trace::trace,
};
#[cfg(test)]
use jolt_core::poly::commitment::dory::DoryGlobals;
use jolt_core::{
    field::JoltField,
    poly::{commitment::commitment_scheme::CommitmentScheme, opening_proof::OpeningPoint},
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
    zkvm::witness::DTH_ROOT_OF_K,
};
use onnx_tracer::{ProgramIO, graph::model::Model, tensor::Tensor};
use serde::{Deserialize, Serialize};
use std::{cell::RefCell, rc::Rc};

pub mod bytecode;
pub mod dag;
pub mod executor;
pub mod lookup_table;
pub mod memory;
pub mod pcs;
pub mod precompiles;
pub mod r1cs;
pub mod sumcheck;
pub mod trace;
pub mod witness;

#[derive(Debug, Clone)]
pub struct Claims<F: JoltField>(Openings<F>);

/// A SNARK for ONNX model inference
#[derive(Clone, Debug)]
pub struct JoltSNARK<F: JoltField, PCS: CommitmentScheme<Field = F>, FS: Transcript> {
    opening_claims: Claims<F>,
    commitments: Vec<PCS::Commitment>,
    proofs: Proofs<F, PCS, FS>,
    pub trace_length: usize,
    memory_K: usize,
    _bytecode_d: usize,
    twist_sumcheck_switch_index: usize,
}

impl<F, PCS, FS> JoltSNARK<F, PCS, FS>
where
    FS: Transcript,
    PCS: CommitmentScheme<Field = F>,
    F: JoltField,
{
    /// Jolt DAG prover
    #[allow(clippy::type_complexity)]
    #[tracing::instrument(skip_all, name = "Jolt::prove")]
    pub fn prove<ModelFunc>(
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        model: ModelFunc,
        input: &Tensor<i32>,
    ) -> (Self, ProgramIO, Option<ProverDebugInfo<F, FS, PCS>>)
    where
        ModelFunc: Fn() -> Model,
    {
        let (trace, program_io) = trace(model, input, &preprocessing.shared.bytecode);
        let state_manager: StateManager<'_, F, FS, PCS> =
            StateManager::new_prover(preprocessing, trace, program_io.clone());
        let (snark, debug_info) = JoltDAG::prove(state_manager).ok().unwrap();
        (snark, program_io, debug_info)
    }

    #[tracing::instrument(skip_all, name = "Jolt::verify")]
    pub fn verify(
        self,
        preprocessing: &JoltVerifierPreprocessing<F, PCS>,
        program_io: ProgramIO,
        _debug_info: Option<ProverDebugInfo<F, FS, PCS>>,
    ) -> Result<(), ProofVerifyError> {
        #[cfg(test)]
        let T = self.trace_length.next_power_of_two();
        // Need to initialize globals because the verifier computes commitments
        // in `VerifierOpeningAccumulator::append` inside of a `#[cfg(test)]` block
        #[cfg(test)]
        let _guard = DoryGlobals::initialize(DTH_ROOT_OF_K, T);

        let state_manager = self.to_verifier_state_manager(preprocessing, program_io);

        #[cfg(test)]
        {
            if let Some(debug_info) = _debug_info {
                let mut transcript = state_manager.transcript.borrow_mut();
                transcript.compare_to(debug_info.transcript);
                let opening_accumulator = state_manager.get_verifier_accumulator();
                opening_accumulator
                    .borrow_mut()
                    .compare_to(debug_info.opening_accumulator);
            }
        }

        JoltDAG::verify(state_manager).expect("Verification failed");

        Ok(())
    }

    pub fn from_prover_state_manager(mut state_manager: StateManager<'_, F, FS, PCS>) -> Self {
        let prover_state = state_manager.prover_state.as_mut().unwrap();
        let openings = std::mem::take(&mut prover_state.accumulator.borrow_mut().openings);
        let commitments = state_manager.commitments.take();
        let proofs = state_manager.proofs.take();
        let trace_length = prover_state.trace.len();
        let memory_K = state_manager.memory_K;
        let twist_sumcheck_switch_index = state_manager.twist_sumcheck_switch_index;

        Self {
            opening_claims: Claims(openings),
            commitments,
            proofs,
            trace_length,
            memory_K,
            _bytecode_d: prover_state.preprocessing.shared.bytecode.d,
            twist_sumcheck_switch_index,
        }
    }

    pub fn to_verifier_state_manager<'a>(
        self,
        preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
        program_io: ProgramIO,
    ) -> StateManager<'a, F, FS, PCS> {
        let mut opening_accumulator = VerifierOpeningAccumulator::<F>::new();
        // Populate claims in the verifier accumulator
        for (key, (_, claim)) in self.opening_claims.0.iter() {
            opening_accumulator
                .openings_mut()
                .insert(*key, (OpeningPoint::default(), *claim));
        }

        let proofs = Rc::new(RefCell::new(self.proofs));
        let commitments = Rc::new(RefCell::new(self.commitments));
        let transcript = Rc::new(RefCell::new(FS::new(b"Jolt")));

        StateManager {
            transcript,
            proofs,
            commitments,
            program_io,
            memory_K: self.memory_K,
            twist_sumcheck_switch_index: self.twist_sumcheck_switch_index,
            prover_state: None,
            verifier_state: Some(VerifierState {
                preprocessing,
                trace_length: self.trace_length,
                accumulator: Rc::new(RefCell::new(opening_accumulator)),
            }),
        }
    }
}

#[allow(dead_code)]
pub struct ProverDebugInfo<F, ProofTranscript, PCS>
where
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    pub transcript: ProofTranscript,
    pub opening_accumulator: ProverOpeningAccumulator<F>,
    pub prover_setup: PCS::ProverSetup,
}

/// Preprocessing data needed for both prover and verifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoltSharedPreprocessing {
    /// The preprocessed bytecode
    pub bytecode: BytecodePreprocessing,
    /// The preprocessed data for all precompiles
    pub precompiles: PrecompilePreprocessing,
}

/// Preprocessing data needed only for the prover
#[derive(Clone, Serialize, Deserialize)]
pub struct JoltProverPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub generators: PCS::ProverSetup,
    pub shared: JoltSharedPreprocessing,
}

impl<F, PCS> JoltProverPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub fn memory_K(&self) -> usize {
        self.shared.bytecode.memory_K
    }

    pub fn bytecode_d(&self) -> usize {
        self.shared.bytecode.d
    }

    pub fn bytecode(&self) -> &[JoltONNXBytecode] {
        &self.shared.bytecode.bytecode
    }

    pub fn is_precompiles_enabled(&self) -> bool {
        !self.shared.precompiles.is_empty()
    }
}

/// Preprocessing data needed only for the verifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub generators: PCS::VerifierSetup,
    pub shared: JoltSharedPreprocessing,
}

impl<F, PCS> JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn memory_K(&self) -> usize {
        self.shared.bytecode.memory_K
    }

    pub fn is_precompiles_enabled(&self) -> bool {
        !self.shared.precompiles.is_empty()
    }
}

impl<F, PCS, FS> JoltSNARK<F, PCS, FS>
where
    FS: Transcript,
    PCS: CommitmentScheme<Field = F>,
    F: JoltField,
{
    /// Preprocesses the ONNX model to produce shared preprocessing data
    #[tracing::instrument(skip_all, name = "Jolt::preprocess")]
    pub fn shared_preprocess<ModelFunc>(model: ModelFunc) -> JoltSharedPreprocessing
    where
        ModelFunc: Fn() -> Model + Copy,
    {
        let bytecode_preprocessing = BytecodePreprocessing::preprocess(model);
        let precompile_preprocessing = PrecompilePreprocessing::preprocess(&bytecode_preprocessing);
        JoltSharedPreprocessing {
            bytecode: bytecode_preprocessing,
            precompiles: precompile_preprocessing,
        }
    }

    /// Preprocesses the ONNX model to produce prover preprocessing data.
    /// * Preproceses the bytecode
    /// * Sets up commitment key
    #[tracing::instrument(skip_all, name = "Jolt::preprocess")]
    pub fn prover_preprocess<ModelFunc>(
        model: ModelFunc,
        max_trace_length: usize,
    ) -> JoltProverPreprocessing<F, PCS>
    where
        ModelFunc: Fn() -> Model + Copy,
    {
        let shared = Self::shared_preprocess(model);
        let max_T: usize = max_trace_length.next_power_of_two();
        let generators = PCS::setup_prover(DTH_ROOT_OF_K.log_2() + max_T.log_2());
        JoltProverPreprocessing { shared, generators }
    }
}

impl<F, PCS> From<&JoltProverPreprocessing<F, PCS>> for JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn from(preprocessing: &JoltProverPreprocessing<F, PCS>) -> Self {
        let generators = PCS::setup_verifier(&preprocessing.generators);
        JoltVerifierPreprocessing {
            generators,
            shared: preprocessing.shared.clone(),
        }
    }
}

#[cfg(test)]
mod e2e_tests {
    use std::{collections::HashMap, fs::File, io::Read, path::PathBuf};

    use crate::jolt::JoltSNARK;
    use ark_bn254::Fr;
    use jolt_core::{
        poly::commitment::{dory::DoryCommitmentScheme, mock::MockCommitScheme},
        transcripts::KeccakTranscript,
    };
    use log::debug;
    use onnx_tracer::{builder, decode_model, graph::model::Model, model, tensor::Tensor};
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use serde_json::Value;
    use serial_test::serial;

    type PCS0 = DoryCommitmentScheme;
    type _PCS1 = MockCommitScheme<Fr>;

    fn run_snark_test<ModelFunc>(model: ModelFunc, input_data: &[i32], shape: &[usize])
    where
        ModelFunc: Fn() -> Model + Copy,
    {
        let input = Tensor::new(Some(input_data), shape).unwrap();
        let preprocessing =
            JoltSNARK::<Fr, PCS0, KeccakTranscript>::prover_preprocess(model, 1 << 20);
        let (snark, program_io, _debug_info) =
            JoltSNARK::<Fr, PCS0, KeccakTranscript>::prove(&preprocessing, model, &input);
        snark
            .verify(&(&preprocessing).into(), program_io, None)
            .unwrap();
    }

    #[test]
    fn test_self_attention_transformer() {
        let model = model(&PathBuf::from(
            "../onnx-tracer/models/self_attention_transformer/network.onnx",
        ));
        let bytecode = decode_model(model);
        for b in bytecode.iter() {
            debug!("{:#?}", b.opcode)
        }
    }

    #[serial]
    #[test]
    fn test_self_attention_2d_transformer() {
        let mut rng = StdRng::seed_from_u64(12345);
        // use small values to prevent overflow in onnx-tracer
        let input_data: Vec<i32> = (0..256).map(|_| rng.gen_range(0..1)).collect();
        run_snark_test(
            || {
                model(&PathBuf::from(
                    "../onnx-tracer/models/self_attention_2d_transformer/network.onnx",
                ))
            },
            &input_data,
            &[16, 16],
        );
    }

    #[test]
    fn test_self_attention_2d_transformer2() {
        let model = model(&PathBuf::from(
            "../onnx-tracer/models/self_attention_2d_transformer/network.onnx",
        ));
        let bytecode = decode_model(model);
        for b in bytecode.iter() {
            debug!("{:#?}", b.opcode)
        }
    }

    #[serial]
    #[test]
    fn test_layernorm_partial_head() {
        let mut rng = StdRng::seed_from_u64(12345);
        // use small values to prevent overflow in onnx-tracer
        let input_data: Vec<i32> = (0..256).map(|_| rng.gen_range(0..3)).collect();
        run_snark_test(
            || {
                model(&PathBuf::from(
                    "../onnx-tracer/models/layernorm_partial_head/network.onnx",
                ))
            },
            &input_data,
            &[16, 16],
        );
    }

    #[serial]
    #[test]
    fn test_layernorm_head() {
        let mut rng = StdRng::seed_from_u64(123456);
        // use small values to prevent overflow in onnx-tracer
        let input_data: Vec<i32> = (0..256).map(|_| rng.gen_range(0..2)).collect();
        run_snark_test(
            || {
                model(&PathBuf::from(
                    "../onnx-tracer/models/layernorm_head/network.onnx",
                ))
            },
            &input_data,
            &[16, 16],
        );
    }

    /// Load vocab.json into HashMap<String, (usize, i32)>
    pub fn load_vocab(
        path: &str,
    ) -> Result<HashMap<String, (usize, i32)>, Box<dyn std::error::Error>> {
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

    #[test]
    pub fn test_article_classification_output() {
        let working_dir: &str = "../onnx-tracer/models/article_classification/";

        // Load the vocab mapping from JSON
        let vocab_path = format!("{working_dir}/vocab.json",);
        let vocab = load_vocab(&vocab_path).expect("Failed to load vocab");

        // Input text string to classify
        let input_texts = [
            "The government plans new trade policies.",
            "The latest computer model has impressive features.",
            "The football match ended in a thrilling draw.",
            "The new movie has received rave reviews from critics.",
            "The stock market saw a significant drop today.",
        ];

        let expected_classes = ["business", "tech", "sport", "entertainment", "business"];

        let mut predicted_classes = Vec::new();

        for input_text in &input_texts {
            // Build input vector from the input text
            let input_vector = build_input_vector(input_text, &vocab);

            let input = Tensor::new(Some(&input_vector), &[1, 512]).unwrap();

            // Load model
            let model = model(&PathBuf::from(format!("{working_dir}network.onnx")));

            // Run inference
            let result = model.forward(&[input.clone()]).unwrap();
            let output = result.outputs[0].clone();

            // Map index to label
            let classes = ["business", "entertainment", "politics", "sport", "tech"];
            let (pred_idx, max_val) = output
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();

            println!("Input text: '{input_text}'");
            println!("Output: {}", output.show());
            println!("Max value: {max_val} at index: {pred_idx}");
            println!("Predicted class: {}", classes[pred_idx]);

            predicted_classes.push(classes[pred_idx]);
        }
        // Check if predicted classes match expected classes
        for (predicted, expected) in predicted_classes.iter().zip(expected_classes.iter()) {
            assert_eq!(predicted, expected, "Mismatch in predicted class");
        }
    }

    #[serial]
    #[test]
    pub fn test_article_classification() {
        let working_dir: &str = "../onnx-tracer/models/article_classification/";

        // Load the vocab mapping from JSON
        let vocab_path = format!("{working_dir}/vocab.json",);
        let vocab = load_vocab(&vocab_path).expect("Failed to load vocab");

        // Input text string to classify
        let input_text = "The government plans new trade policies.";

        // Build input vector from the input text (512 features for small MLP)
        let input_vector = build_input_vector(input_text, &vocab);

        run_snark_test(
            || model(&PathBuf::from(format!("{working_dir}network.onnx"))),
            &input_vector,
            &[1, 512],
        );
    }

    // ========================================================================================
    // AUTHORIZATION MODEL TESTS
    // ========================================================================================
    //
    // These tests validate the authorization model which performs transaction authorization
    // decisions using an MLP neural network with one-hot encoded features.
    //
    // Model Architecture:
    // - Input Size: 64 (one-hot encoded features)
    // - Architecture: 64 → 32 → 16 → 4
    // - Output: 4 classes (Authorized=0, Denied=1, Padding=2,3)
    // - Features: Budget, Trust, Amount, Category, Velocity, Day, Time (Risk unused)
    //
    // The model follows the same pattern as article classification but for financial
    // transaction authorization with power-of-2 dimensions for SNARK compatibility.
    // ========================================================================================

    /// Load authorization vocab.json into HashMap<String, usize>
    pub fn load_authorization_vocab(
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
    pub fn build_authorization_vector(
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

    #[test]
    pub fn test_authorization_output() {
        let working_dir: &str = "../onnx-tracer/models/authorization/";

        // Load the vocab mapping from JSON
        let vocab_path = format!("{working_dir}/vocab.json");
        let vocab =
            load_authorization_vocab(&vocab_path).expect("Failed to load authorization vocab");

        // Test authorization scenarios
        let test_scenarios = [
            // (budget, trust, amount, category, velocity, day, time, risk, expected_decision)
            (15, 7, 8, 0, 2, 1, 1, 0, "AUTHORIZED"), // High trust, sufficient budget
            (5, 4, 12, 0, 2, 1, 1, 0, "DENIED"),     // Amount exceeds budget
            (15, 1, 12, 0, 2, 1, 1, 0, "DENIED"),    // Low trust, high amount
            (15, 7, 8, 0, 7, 1, 1, 0, "DENIED"),     // High velocity
            (15, 0, 5, 2, 2, 1, 1, 0, "DENIED"),     // Restricted category for untrusted merchant
            (15, 7, 14, 0, 2, 1, 3, 0, "DENIED"),    // Late night high-value transaction
        ];

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

            // Load model
            let model = model(&PathBuf::from(format!("{working_dir}network.onnx")));

            // Run inference
            let result = model.forward(&[input.clone()]).unwrap();
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

            println!(
                "Test {}: budget={}, trust={}, amount={}, category={}, velocity={}, day={}, time={}, risk={}",
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
            println!("Output: {}", output.show());
            println!("Max value: {max_val} at index: {pred_idx}");
            println!("Expected: {expected}, Predicted: {prediction}");

            if prediction == expected {
                correct_predictions += 1;
                println!("✅ CORRECT");
            } else {
                println!("❌ INCORRECT");
            }
            println!();
        }

        let accuracy = correct_predictions as f32 / test_scenarios.len() as f32 * 100.0;
        println!(
            "Authorization Test Accuracy: {}/{} ({:.1}%)",
            correct_predictions,
            test_scenarios.len(),
            accuracy
        );

        // Require at least 80% accuracy
        assert!(
            accuracy >= 80.0,
            "Authorization accuracy too low: {accuracy:.1}%"
        );
    }

    #[serial]
    #[test]
    pub fn test_authorization() {
        let working_dir: &str = "../onnx-tracer/models/authorization/";

        // Load the vocab mapping from JSON
        let vocab_path = format!("{working_dir}/vocab.json");
        let vocab =
            load_authorization_vocab(&vocab_path).expect("Failed to load authorization vocab");

        // Test with a high trust merchant and sufficient budget (should authorize)
        let budget = 15; // High budget
        let trust = 7; // High trust
        let amount = 8; // Moderate amount
        let category = 0; // Safe category
        let velocity = 2; // Low velocity
        let day = 1; // Tuesday
        let time = 1; // Morning
        let risk = 0; // No risk (not used)

        // Build input vector from transaction features (64 features for authorization MLP)
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

        run_snark_test(
            || model(&PathBuf::from(format!("{working_dir}network.onnx"))),
            &input_vector,
            &[1, 64],
        );
    }

    #[test]
    #[serial]
    fn test_triple_matmult_model() {
        run_snark_test(
            builder::triple_matmult_model,
            &[1, 2, 3, 4, 1, 2, 3, 4],
            &[1, 8],
        );
    }

    #[test]
    #[serial]
    fn test_reduce_mean() {
        run_snark_test(
            builder::reduce_mean_model,
            &[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            &[4, 4],
        );
    }

    #[test]
    #[serial]
    fn test_layernorm_prefix() {
        run_snark_test(
            builder::layernorm_prefix_model,
            &[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            &[4, 4],
        );
    }

    #[test]
    #[serial]
    fn test_addsubmuldivdiv() {
        run_snark_test(builder::addsubmuldivdiv_model, &[1, 2, 3, 4], &[1, 4]);
    }

    #[test]
    #[serial]
    fn test_addsubmul_binary() {
        run_snark_test(
            || {
                model(&PathBuf::from(
                    "../onnx-tracer/models/addsubmul1/network.onnx",
                ))
            },
            &[1, 2, 3, 4, 1, 2, 3, 4, 1, 2],
            &[1, 10],
        );
    }

    #[test]
    #[serial]
    fn test_simple_mlp_binary() {
        run_snark_test(
            || {
                model(&PathBuf::from(
                    "../onnx-tracer/models/simple_mlp/network.onnx",
                ))
            },
            &[1, 2, 3, 4, 1, 2, 3, 4],
            &[1, 8],
        );
    }

    #[test]
    #[serial]
    fn test_simple_mlp_small_binary() {
        run_snark_test(
            || {
                model(&PathBuf::from(
                    "../onnx-tracer/models/simple_mlp_small/network.onnx",
                ))
            },
            &[1, 2, 3, 4],
            &[1, 4],
        );
    }

    #[test]
    #[serial]
    fn test_rsqrt_binary() {
        run_snark_test(
            || model(&PathBuf::from("../onnx-tracer/models/rsqrt/network.onnx")),
            &[512],
            &[1, 1],
        );
    }

    #[test]
    #[serial]
    fn test_simple_mlp_small() {
        run_snark_test(builder::simple_mlp_small_model, &[1, 2, 3, 4], &[1, 4]);
    }

    #[test]
    #[serial]
    fn test_perceptron_binary() {
        run_snark_test(
            || model(&PathBuf::from("../tests/perceptron.onnx")),
            &[1, 2, 3, 4, 1, 2, 3, 4, 1, 2],
            &[1, 10],
        );
    }

    #[test]
    #[serial]
    fn test_perceptron_2_binary() {
        run_snark_test(
            || model(&PathBuf::from("../tests/perceptron_2.onnx")),
            &[1, 2, 3, 4],
            &[1, 4],
        );
    }

    #[test]
    #[serial]
    fn test_addsubmuldiv() {
        run_snark_test(builder::addsubmuldiv_model, &[1, 2, 3, 4], &[1, 4]);
    }

    #[test]
    #[serial]
    fn test_addsubmuldivadd() {
        run_snark_test(builder::addsubmuldivadd_model, &[1, 2, 3, 4], &[1, 4]);
    }

    #[test]
    #[serial]
    fn test_tiny_mlp_head() {
        run_snark_test(builder::tiny_mlp_head_model, &[1, 2, 3, 4], &[1, 4]);
    }

    #[test]
    #[serial]
    fn test_dual_matmult_model() {
        run_snark_test(builder::dual_matmult_model, &[1, 2, 3, 4], &[1, 4]);
    }

    #[test]
    #[serial]
    fn test_neg_dual_matmult_model() {
        run_snark_test(builder::dual_matmult_model, &[-1, -2, -3, -4], &[1, 4]);
    }

    #[test]
    #[serial]
    fn test_addsubmulconst() {
        run_snark_test(builder::addsubmulconst_model, &[1, 2, 3, 4], &[1, 4]);
    }

    #[test]
    #[serial]
    fn test_addsubmul() {
        run_snark_test(builder::addsubmul_model, &[1, 2, 3, 4], &[1, 4]);
    }

    #[test]
    #[serial]
    fn test_add() {
        run_snark_test(builder::add_model, &[3, 4, 5, 0], &[1, 4]);
    }

    #[test]
    #[serial]
    fn test_scalar() {
        run_snark_test(builder::scalar_addsubmul_model, &[10], &[1, 1]);
    }

    #[test]
    #[serial]
    fn test_relu() {
        run_snark_test(builder::relu_model, &[-3, -2, 0, 1], &[1, 4]);
    }

    #[test]
    #[serial]
    fn test_rsqrt() {
        run_snark_test(builder::rsqrt_model, &[-3, -2, 0, 1], &[1, 4]);
    }
}
