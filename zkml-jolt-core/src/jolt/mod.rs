pub mod bytecode;
pub mod execution_trace;
pub mod instruction;
pub mod instruction_lookups;
pub mod precompiles;
pub mod r1cs;
pub mod tensor_heap;

use crate::jolt::{
    bytecode::{BytecodePreprocessing, BytecodeProof},
    execution_trace::JoltONNXCycle,
    instruction::{
        VirtualInstructionSequence, argmax::ArgMaxInstruction, div::DIVInstruction,
        rebase_scale::REBASEInstruction, sigmoid::SigmoidInstruction,
    },
    instruction_lookups::LookupsProof,
    precompiles::{PrecompilePreprocessing, PrecompileProof},
    r1cs::{
        constraints::{JoltONNXConstraints, R1CSConstraints},
        spartan::UniformSpartanProof,
    },
    tensor_heap::TensorHeapTwistProof,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use execution_trace::WORD_SIZE;
use jolt_core::{
    field::JoltField,
    poly::{
        commitment::{commitment_scheme::CommitmentScheme, dory::DoryGlobals},
        opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator},
    },
    utils::{errors::ProofVerifyError, transcript::Transcript},
};
use onnx_tracer::{
    ProgramIO,
    constants::MAX_TENSOR_SIZE,
    trace_types::{ONNXInstr, ONNXOpcode},
};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

#[derive(Clone, Serialize, Deserialize)]
pub struct JoltProverPreprocessing<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub shared: JoltSharedPreprocessing,
    field: F::SmallValueLookupTables,
    _p: PhantomData<(ProofTranscript, PCS)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoltSharedPreprocessing {
    pub bytecode: BytecodePreprocessing,
    pub precompiles: PrecompilePreprocessing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoltVerifierPreprocessing<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub shared: JoltSharedPreprocessing,
    _p: PhantomData<(F, PCS, ProofTranscript)>,
}

impl<F, PCS, ProofTranscript> From<&JoltProverPreprocessing<F, PCS, ProofTranscript>>
    for JoltVerifierPreprocessing<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    fn from(preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>) -> Self {
        JoltVerifierPreprocessing {
            shared: preprocessing.shared.clone(),
            _p: PhantomData,
        }
    }
}

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltSNARK<F, PCS, ProofTranscript>
where
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    F: JoltField,
{
    pub trace_length: usize,
    bytecode: BytecodeProof<F, ProofTranscript>,
    instruction_lookups: LookupsProof<WORD_SIZE, F, PCS, ProofTranscript>,
    tensor_heap: TensorHeapTwistProof<F, ProofTranscript>,
    r1cs: UniformSpartanProof<F, ProofTranscript>,
    precompile: Option<PrecompileProof<F, ProofTranscript>>,
    _p: PhantomData<PCS>,
}

impl<F, PCS, ProofTranscript> JoltSNARK<F, PCS, ProofTranscript>
where
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    F: JoltField,
{
    #[tracing::instrument(skip_all, name = "Jolt::preprocess")]
    pub fn shared_preprocess(bytecode: Vec<ONNXInstr>) -> JoltSharedPreprocessing {
        let bytecode: Vec<ONNXInstr> = bytecode
            .into_iter()
            .flat_map(|instr| match instr.opcode {
                ONNXOpcode::Div => DIVInstruction::<32>::virtual_sequence(instr),
                ONNXOpcode::ArgMax => ArgMaxInstruction::<32>::virtual_sequence(instr),
                ONNXOpcode::RebaseScale(_) => REBASEInstruction::<32>::virtual_sequence(instr),
                ONNXOpcode::Sigmoid => SigmoidInstruction::<32>::virtual_sequence(instr),
                _ => vec![instr],
            })
            .collect();
        let bytecode_preprocessing = BytecodePreprocessing::preprocess(bytecode.clone());
        let precompile_preprocessing = PrecompilePreprocessing::preprocess(&bytecode);
        JoltSharedPreprocessing {
            bytecode: bytecode_preprocessing,
            precompiles: precompile_preprocessing,
        }
    }

    #[tracing::instrument(skip_all, name = "Jolt::preprocess")]
    pub fn prover_preprocess(
        bytecode: Vec<ONNXInstr>,
    ) -> JoltProverPreprocessing<F, PCS, ProofTranscript> {
        let small_value_lookup_tables = F::compute_lookup_tables();
        F::initialize_lookup_tables(small_value_lookup_tables.clone());
        let shared = Self::shared_preprocess(bytecode);
        JoltProverPreprocessing {
            shared,
            field: small_value_lookup_tables,
            _p: PhantomData,
        }
    }

    #[tracing::instrument(skip_all, name = "Jolt::prove")]
    pub fn prove(
        mut preprocessing: JoltProverPreprocessing<F, PCS, ProofTranscript>,
        mut trace: Vec<JoltONNXCycle>,
        program_output: &ProgramIO,
    ) -> Self {
        let trace_length = trace.len();
        println!("Trace length: {trace_length}");
        F::initialize_lookup_tables(std::mem::take(&mut preprocessing.field));
        // Pad to next power of two, adding one for the extra NoOp we add below
        let padded_trace_length = (trace_length + 1).next_power_of_two();
        let last_node = trace.last().unwrap();
        // Add Output node, which maps the final tensor output to the reserved output address
        trace.push(JoltONNXCycle::output_node(
            last_node,
            program_output.output.clone(),
        ));

        // HACK(Forpee): Not sure if this is correct. RV pushes a jump instr:
        // ```
        // // Final JALR sets NextUnexpandedPC = 0
        // trace.push(RV32IMCycle::last_jalr(last_address + 4 * (padding - 1)));
        // ```
        // Pad with noOps
        trace.resize(padded_trace_length, JoltONNXCycle::no_op());

        let tensor_heap_addresses: Vec<usize> = trace
            .iter()
            .map(|cycle| cycle.td_write().0.last().unwrap() + 1)
            .collect();
        let tensor_heap_K = tensor_heap_addresses
            .iter()
            .max()
            .unwrap()
            .next_power_of_two();
        let K = [
            preprocessing.shared.bytecode.code_size,
            tensor_heap_K,
            1 << 16, // K for instruction lookups Shout
        ]
        .into_iter()
        .max()
        .unwrap();
        println!("T = {padded_trace_length}, K = {K}");
        let _guard = DoryGlobals::initialize(K, padded_trace_length);
        let mut transcript = ProofTranscript::new(b"Jolt transcript");
        let mut opening_accumulator: ProverOpeningAccumulator<F, PCS, ProofTranscript> =
            ProverOpeningAccumulator::new();
        let constraint_builder = JoltONNXConstraints::construct_constraints(padded_trace_length);
        let spartan_key = UniformSpartanProof::<F, ProofTranscript>::setup(
            &constraint_builder,
            padded_trace_length,
        );
        transcript.append_scalar(&spartan_key.vk_digest);
        let r1cs_snark = UniformSpartanProof::prove::<PCS>(
            &preprocessing,
            &constraint_builder,
            &spartan_key,
            &trace,
            &mut transcript,
        )
        .ok()
        .unwrap();
        let instruction_lookups_snark: LookupsProof<WORD_SIZE, F, PCS, ProofTranscript> =
            LookupsProof::prove(
                &preprocessing,
                &trace,
                &mut opening_accumulator,
                &mut transcript,
            );
        let bytecode_snark =
            BytecodeProof::prove(&preprocessing.shared.bytecode, &trace, &mut transcript);
        let tensor_heap_snark = TensorHeapTwistProof::prove(
            &preprocessing,
            &trace,
            tensor_heap_K,
            &mut opening_accumulator,
            &mut transcript,
            program_output,
        );
        let precompiles_proof =
            PrecompileProof::prove(&preprocessing.shared.precompiles, &trace, &mut transcript);
        JoltSNARK {
            trace_length: trace_length + 1,
            r1cs: r1cs_snark,
            tensor_heap: tensor_heap_snark,
            instruction_lookups: instruction_lookups_snark,
            bytecode: bytecode_snark,
            precompile: precompiles_proof,
            _p: PhantomData,
        }
    }

    #[tracing::instrument(skip_all)]
    pub fn verify(
        &self,
        preprocessing: JoltVerifierPreprocessing<F, PCS, ProofTranscript>,
        program_output: ProgramIO,
    ) -> Result<(), ProofVerifyError> {
        let mut transcript = ProofTranscript::new(b"Jolt transcript");
        let mut opening_accumulator: VerifierOpeningAccumulator<F, PCS, ProofTranscript> =
            VerifierOpeningAccumulator::new();
        // Regenerate the uniform Spartan key
        let padded_trace_length = self.trace_length.next_power_of_two();
        let r1cs_builder = JoltONNXConstraints::construct_constraints(padded_trace_length);
        let spartan_key =
            UniformSpartanProof::<F, ProofTranscript>::setup(&r1cs_builder, padded_trace_length);
        transcript.append_scalar(&spartan_key.vk_digest);
        self.r1cs
            .verify::<PCS>(&spartan_key, &mut transcript)
            .map_err(|e| ProofVerifyError::SpartanError(e.to_string()))?;
        self.instruction_lookups
            .verify(&mut opening_accumulator, &mut transcript)?;
        self.bytecode.verify(
            &preprocessing.shared.bytecode,
            padded_trace_length,
            &mut transcript,
        )?;
        self.tensor_heap.verify(
            padded_trace_length * MAX_TENSOR_SIZE,
            &mut opening_accumulator,
            &mut transcript,
            program_output,
        )?;

        // Check if we have precompiles to verify
        if !preprocessing.shared.precompiles.is_empty() {
            if let Some(precompile_proof) = &self.precompile {
                precompile_proof.verify(&preprocessing.shared.precompiles, &mut transcript)?;
            } else {
                return Err(ProofVerifyError::InternalError);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod e2e_tests {
    use crate::{
        jolt::{JoltProverPreprocessing, JoltSNARK, execution_trace::jolt_execution_trace},
        program::ONNXProgram,
    };
    use ark_bn254::Fr;
    use jolt_core::{
        poly::commitment::dory::DoryCommitmentScheme, utils::transcript::KeccakTranscript,
    };
    use log::{debug, info};
    use onnx_tracer::{
        builder, constants::MAX_TENSOR_SIZE, graph::model::Model, logger::init_logger, model,
        tensor::Tensor,
    };
    use rand::{Rng, thread_rng};
    use serde_json::Value;
    use serial_test::serial;
    use std::{collections::HashMap, fs::File, io::Read};

    type PCS = DoryCommitmentScheme<KeccakTranscript>;

    struct ZKMLTestHelper;

    impl ZKMLTestHelper {
        fn prove_and_verify<F>(
            model_fn: F,
            input: &Tensor<i32>,
            expected_output: Option<u64>,
        ) -> onnx_tracer::trace_types::ONNXCycle
        where
            F: Fn() -> Model,
        {
            let model = model_fn();
            let program_bytecode = onnx_tracer::decode_model(model.clone());
            let pp: JoltProverPreprocessing<Fr, PCS, KeccakTranscript> =
                JoltSNARK::prover_preprocess(program_bytecode);
            let (raw_trace, program_output) = onnx_tracer::execution_trace(model, input);
            // Verify expected output if provided
            if let Some(expected) = expected_output {
                assert_eq!(
                    expected,
                    raw_trace.last().unwrap().ts1_vals()[0],
                    "Output mismatch for input: {input:?}",
                );
            }

            let execution_trace = jolt_execution_trace(raw_trace.clone());
            let snark: JoltSNARK<Fr, PCS, KeccakTranscript> =
                JoltSNARK::prove(pp.clone(), execution_trace, &program_output);

            snark.verify((&pp).into(), program_output).unwrap();
            raw_trace.into_iter().last().unwrap()
        }

        fn prove_and_verify_simple<F>(model_fn: F, input: &Tensor<i32>)
        where
            F: Fn() -> Model,
        {
            Self::prove_and_verify(model_fn, input, None);
        }

        fn test_inference<F>(
            model_fn: F,
            test_cases: &[(Vec<i32>, Vec<usize>, i32)], // (input, shape, expected)
        ) where
            F: Fn() -> Model,
        {
            Self::test_inference_with(model_fn, test_cases);
        }

        fn test_inference_with<F>(
            model_fn: F,
            test_cases: &[(Vec<i32>, Vec<usize>, i32)], // (input, shape, expected)
        ) where
            F: Fn() -> Model,
        {
            let mut model = model_fn();
            for (input_data, shape, expected) in test_cases {
                let input = Tensor::new(Some(input_data), shape).unwrap();
                let result = model.forward(&[input]).unwrap();
                assert_eq!(
                    result.outputs[0].inner[0], *expected,
                    "Inference failed for input: {input_data:?}",
                );
            }
            model.clear_execution_trace();
        }
    }

    struct ModelTestConfig {
        _name: String,
        input_data: Vec<i32>,
        input_shape: Vec<usize>,
        expected_output: Option<u64>,
    }

    impl ModelTestConfig {
        fn new(name: &str, input_data: Vec<i32>, input_shape: Vec<usize>) -> Self {
            Self {
                _name: name.to_string(),
                input_data,
                input_shape,
                expected_output: None,
            }
        }

        fn with_expected_output(mut self, expected: u64) -> Self {
            self.expected_output = Some(expected);
            self
        }

        fn to_tensor(&self) -> Tensor<i32> {
            Tensor::new(Some(&self.input_data), &self.input_shape).unwrap()
        }
    }

    // TODO: MLP model requires larger tensor size
    // To run this test, increase `onnx_tracer::constants::MAX_TENSOR_SIZE` to 1024
    // This temporary workaround maintains performance for smaller models
    // Issue tracked at: https://github.com/ICME-Lab/jolt-atlas/issues/21
    #[ignore]
    #[serial]
    #[test]
    fn test_perceptron_2() {
        let config = ModelTestConfig::new("perceptron_2", vec![1, 2, 3, 4], vec![1, 4]);

        let model_fn = || model(&"../tests/perceptron_2.onnx".into());

        ZKMLTestHelper::prove_and_verify_simple(model_fn, &config.to_tensor());
    }

    #[serial]
    #[test]
    fn test_perceptron() {
        let config = ModelTestConfig::new(
            "perceptron",
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            vec![1, 10],
        );

        let model_fn = || model(&"../tests/perceptron.onnx".into());

        ZKMLTestHelper::prove_and_verify_simple(model_fn, &config.to_tensor());
    }

    #[serial]
    #[test]
    fn test_custom_multiclass0() {
        let config = ModelTestConfig::new(
            "multiclass0",
            vec![8, 14, 30, 29, 0, 0, 0, 0], // "this university grants scholarships"
            vec![1, 8],
        )
        .with_expected_output(1); // class -> 1: education

        ZKMLTestHelper::prove_and_verify(
            builder::multiclass0,
            &config.to_tensor(),
            config.expected_output,
        );
    }

    #[test]
    #[serial]
    fn test_sentiment0() {
        let config = ModelTestConfig::new(
            "sentiment0",
            vec![3, 4, 5, 0, 0], // [This, is, great, 0, 0]
            vec![1, 5],
        );

        ZKMLTestHelper::prove_and_verify_simple(builder::sentiment0, &config.to_tensor());
    }

    #[test]
    #[serial]
    fn test_custom_select() {
        let config = ModelTestConfig::new(
            "sentiment_select",
            vec![3, 4, 5, 0, 0], // [This, is, great, 0, 0]
            vec![1, 5],
        );

        ZKMLTestHelper::prove_and_verify_simple(builder::sentiment_select, &config.to_tensor());
    }

    #[serial]
    #[test]
    fn test_argmax_e2e() {
        let config = ModelTestConfig::new("argmax", vec![10, 20, 30, 50, 50], vec![5]);

        ZKMLTestHelper::prove_and_verify_simple(builder::argmax_model, &config.to_tensor());
    }

    #[serial]
    #[test]
    fn test_rebase_scale_e2e() {
        let config = ModelTestConfig::new("rebase_scale", vec![10, 20, 30, 40, 50], vec![5]);

        ZKMLTestHelper::prove_and_verify_simple(builder::rebase_scale_model, &config.to_tensor());
    }

    fn test_arithmetic_model<F>(model_fn: F, test_name: &str)
    where
        F: Fn() -> Model,
    {
        let config = ModelTestConfig::new(test_name, vec![10, 20, 30, 40], vec![1, 4]);

        ZKMLTestHelper::prove_and_verify_simple(model_fn, &config.to_tensor());
    }

    #[serial]
    #[test]
    fn test_add() {
        // tests a model with 3 (2^s - 1) nodes
        test_arithmetic_model(builder::custom_add_model, "add");
    }

    #[serial]
    #[test]
    fn test_addmul() {
        // tests a model with 4 (2^s) nodes
        test_arithmetic_model(builder::custom_addmul_model, "addmul");
    }

    #[serial]
    #[test]
    fn test_addsubmuldivdiv() {
        test_arithmetic_model(builder::custom_addsubmuldivdiv_model, "addsubmuldivdiv");
    }

    #[serial]
    #[test]
    fn test_custom_addsubmuldiv15() {
        // tests a model with 15 (2^s - 1) nodes finishing with a virtual instruction
        test_arithmetic_model(builder::custom_addsubmuldiv15_model, "addsubmuldiv15");
    }

    #[serial]
    #[test]
    fn test_addsubmuldiv() {
        // tests a model with 16 (2^s) nodes finishing with a virtual instruction
        test_arithmetic_model(builder::custom_addsubmuldiv_model, "addsubmuldiv");
    }

    #[serial]
    #[test]
    fn test_custom_addsubmulconst() {
        test_arithmetic_model(builder::custom_addsubmulconst_model, "addsubmulconst");
    }

    #[serial]
    #[test]
    fn test_custom_addsubmul() {
        let config = ModelTestConfig::new("addsubmul", vec![-10, -20, -30, -40], vec![1, 4]);
        ZKMLTestHelper::prove_and_verify_simple(
            builder::custom_addsubmul_model,
            &config.to_tensor(),
        );
    }

    #[serial]
    #[test]
    fn test_scalar_addsubmul() {
        let config = ModelTestConfig::new("scalar_addsubmul", vec![60], vec![1]);

        ZKMLTestHelper::prove_and_verify_simple(
            builder::scalar_addsubmul_model,
            &config.to_tensor(),
        );
    }

    // #[serial]
    // #[test]
    // fn test_altered_input_and_output() {
    //     let model = builder::custom_add_model();
    //     let program_bytecode = onnx_tracer::decode_model(model.clone());
    //     let pp: JoltProverPreprocessing<Fr, PCS, KeccakTranscript> =
    //         JoltSNARK::prover_preprocess(program_bytecode);

    //     let input = Tensor::new(Some(&[10, 20, 30, 40]), &[1, 4]).unwrap();

    //     let (raw_trace, program_output) = onnx_tracer::execution_trace(model, &input);

    //     let execution_trace = jolt_execution_trace(raw_trace.clone());
    //     let snark: JoltSNARK<Fr, PCS, KeccakTranscript> =
    //         JoltSNARK::prove(pp.clone(), execution_trace, &program_output);

    //     // Verify with correct input and output
    //     snark.verify((&pp).into(), program_output.clone()).unwrap();

    //     // Alter input and assert verification error
    //     let mut altered_input = program_output.clone();
    //     altered_input.input[0] += 1; // alter input
    //     assert!(snark.verify((&pp).into(), altered_input).is_err());

    //     // Alter output and assert verification error
    //     let mut altered_output = program_output.clone();
    //     altered_output.output[0] += 1; // alter output
    //     assert!(snark.verify((&pp).into(), altered_output).is_err());
    // }

    #[serial]
    #[test]
    fn test_simple_matmult() {
        // Test matrix multiplication: [1, 4] × [3, 4] → [1, 3] (ONNX implicitly transposes B to [4, 3])
        // Input: [1, 2, 3, 4]
        // Weight matrix stored as [3, 4]: [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]]
        // But ONNX uses it as transposed [4, 3]: [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        // Expected output: [1*1 + 2*4 + 3*7 + 4*10, 1*2 + 2*5 + 3*8 + 4*11, 1*3 + 2*6 + 3*9 + 4*12]
        //                = [1 + 8 + 21 + 40, 2 + 10 + 24 + 44, 3 + 12 + 27 + 48]
        //                = [70, 80, 90]
        let config = ModelTestConfig::new("simple_matmult", vec![1, 2, 3, 4], vec![1, 4]);

        ZKMLTestHelper::prove_and_verify_simple(builder::simple_matmult_model, &config.to_tensor());
    }

    #[serial]
    #[test]
    fn test_tiny_mlp_head_model() {
        let config = ModelTestConfig::new("tiny_mlp_head_model", vec![1, 2, 3, 4], vec![1, 4]);

        ZKMLTestHelper::prove_and_verify_simple(builder::tiny_mlp_head_model, &config.to_tensor());
    }

    #[test]
    fn test_multiclass_inference() {
        let test_cases = vec![
            (vec![1, 2, 3, 4, 0, 0, 0, 0], vec![1, 8], 2), // "cheap flights to rome" -> travel
            (vec![5, 6, 7, 8, 9, 0, 0, 0], vec![1, 8], 3), // "box office hits this weekend" -> entertainment
            (vec![10, 11, 12, 13, 0, 0, 0, 0], vec![1, 8], 0), // "quarterly earnings beat guidance" -> business
            (vec![14, 15, 16, 0, 0, 0, 0, 0], vec![1, 8], 1), // "university admissions tips" -> education
            (vec![21, 22, 23, 24, 0, 0, 0, 0], vec![1, 8], 3), // "new streaming series announced" -> entertainment
            (vec![8, 14, 24, 29, 0, 0, 0, 0], vec![1, 8], 1), // "this university announced scholarships" -> education
            (vec![29, 28, 0, 0, 0, 0, 0, 0], vec![1, 8], 1),  // "scholarships news" -> education
        ];

        ZKMLTestHelper::test_inference(builder::multiclass0, &test_cases);
    }

    #[test]
    fn test_sentiment0_inference() {
        let test_cases = vec![
            (vec![1, 2, 3, 0, 0], vec![1, 5], 1), // "I love this" -> positive
            (vec![1, 10, 3, 0, 0], vec![1, 5], 0), // "I hate this" -> negative
            (vec![3, 4, 5, 0, 0], vec![1, 5], 1), // "This is great" -> positive
            (vec![3, 4, 11, 0, 0], vec![1, 5], 0), // "This is bad" -> negative
        ];

        ZKMLTestHelper::test_inference(builder::sentiment0, &test_cases);
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
                    data.get("idf").and_then(|v| v.as_i64()),
                ) {
                    vocab.insert(word, (index as usize, idf as i32));
                }
            }
        }

        Ok(vocab)
    }

    /// Tokenize and convert text to vector of length 1000
    pub fn build_input_vector(text: &str, vocab: &HashMap<String, (usize, i32)>) -> Vec<i32> {
        let mut vec = vec![0; 1000];

        // Split text into tokens (preserve punctuation as tokens)
        let re = regex::Regex::new(r"\w+|[^\w\s]").unwrap();
        for cap in re.captures_iter(text) {
            let token = cap.get(0).unwrap().as_str().to_lowercase();
            if let Some(&(index, idf)) = vocab.get(&token) {
                if index < 1000 {
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

        let expected_classes = ["politics", "tech", "sport", "entertainment", "business"];

        let mut predicted_classes = Vec::new();

        for input_text in &input_texts {
            // Build input vector from the input text
            let input_vector = build_input_vector(input_text, &vocab);

            // Prepare ONNX program
            let text_classification = ONNXProgram {
                model_path: format!("{working_dir}network.onnx").into(),
                inputs: Tensor::new(Some(&input_vector), &[1, 1000]).unwrap(),
            };

            // Decode to program bytecode (for EZKL use)
            let _program_bytecode = text_classification.decode();

            // Load model
            let model = model(&text_classification.model_path);

            // Run inference
            let result = model
                .forward(&[text_classification.inputs.clone()])
                .unwrap();
            let output = result.outputs[0].clone();

            // Map index to label
            let classes = ["business", "entertainment", "politics", "sport", "tech"];
            let (pred_idx, _) = output
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();

            debug!("Output: {}", output.show());
            debug!("Predicted class: {}", classes[pred_idx]);

            predicted_classes.push(classes[pred_idx]);
        }
        // Check if predicted classes match expected classes
        for (predicted, expected) in predicted_classes.iter().zip(expected_classes.iter()) {
            assert_eq!(predicted, expected, "Mismatch in predicted class");
        }
    }

    #[test]
    fn test_medium_classification() {
        let mut input_vector = vec![846, 3, 195, 4, 374, 14, 259];
        input_vector.resize(100, 0); // Resize to match the input shape

        let text_classification = ONNXProgram {
            model_path: "../onnx-tracer/models/medium_text_classification/network.onnx".into(),
            inputs: Tensor::new(Some(&input_vector), &[1, 100]).unwrap(), // Example input
        };
        let program_bytecode = text_classification.decode();
        debug!("Program code: {program_bytecode:#?}",);
        text_classification.trace();
    }

    #[test]
    fn test_medium_classification_output() {
        let mut input_vector = vec![197, 10, 862, 8, 23, 53, 2, 319, 34, 122, 100, 53, 33];
        input_vector.resize(100, 0); // Resize to match the input shape

        let text_classification = ONNXProgram {
            model_path: "../onnx-tracer/models/medium_text_classification/network.onnx".into(),
            inputs: Tensor::new(Some(&input_vector), &[1, 100]).unwrap(), // Example input
        };
        let program_bytecode = text_classification.decode();
        debug!("Program code: {program_bytecode:#?}",);
        let model = model(&text_classification.model_path);

        let result = model
            .forward(&[text_classification.inputs.clone()])
            .unwrap();
        let output = result.outputs[0].clone();
        debug!("Output: {output:#?}",);
    }

    #[should_panic]
    #[test]
    fn test_subgraph() {
        let subgraph_program = ONNXProgram {
            model_path: "../onnx-tracer/models/subgraph/network.onnx".into(),
            inputs: Tensor::new(Some(&[1, 2, 3, 4]), &[1, 4]).unwrap(), // Example input
        };
        let program_bytecode = subgraph_program.decode();

        debug!("Program decoded");
        debug!("Program code: {program_bytecode:#?}",);

        // Test that the addresses of a subgraph are monotonically increasing
        let mut i = 0;
        for instr in program_bytecode {
            assert!(instr.address > i);
            i = instr.address;
        }

        subgraph_program.trace();
    }

    #[serial]
    #[test]
    fn test_sentiment_select() {
        let config = ModelTestConfig::new(
            "sentiment_select",
            vec![3, 4, 5, 0, 0], // [This, is, great, 0, 0]
            vec![1, 5],
        );

        let model_fn = || model(&"../onnx-tracer/models/sentiment_select/network.onnx".into());

        ZKMLTestHelper::prove_and_verify_simple(model_fn, &config.to_tensor());
    }

    #[serial]
    #[test]
    fn test_addsubmuladd() {
        let config = ModelTestConfig::new(
            "addsubmuladd",
            vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            vec![1, 10],
        );

        let model_fn = || model(&"../onnx-tracer/models/addsubmuladd/network.onnx".into());

        ZKMLTestHelper::prove_and_verify_simple(model_fn, &config.to_tensor());
    }

    #[ignore]
    #[test]
    fn test_multiclass0() {
        init_logger();
        let input_vector = [1, 2, 3, 4, 5, 6, 7, 8];

        let multiclass0 = ONNXProgram {
            model_path: "../onnx-tracer/models/multiclass0/network.onnx".into(),
            inputs: Tensor::new(Some(&input_vector), &[1, 8]).unwrap(), // Example input
        };
        let program_bytecode = multiclass0.decode();
        info!("Program code: {program_bytecode:#?}");
    }

    #[serial]
    #[test]
    fn test_sigmoid_e2e() {
        let mut rng = thread_rng();
        let mut v = vec![0; MAX_TENSOR_SIZE];
        for i in 0..MAX_TENSOR_SIZE {
            v[i] = rng.gen_range(-128..=128);
        }
        // Simple case: input tensor with both positive and negative values
        let config = ModelTestConfig::new("sigmoid", v.to_vec(), vec![1, MAX_TENSOR_SIZE]);

        let model_fn = builder::sigmoid_model;

        ZKMLTestHelper::prove_and_verify_simple(model_fn, &config.to_tensor());
    }

    #[serial]
    #[test]
    fn test_proof_serialize_deserialize_roundtrip() {
        use crate::jolt::JoltProverPreprocessing;
        use crate::jolt::JoltSNARK;
        use ark_bn254::Fr;
        use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
        use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
        use jolt_core::utils::transcript::KeccakTranscript;
        use onnx_tracer::builder;
        use onnx_tracer::tensor::Tensor;

        type PCS = DoryCommitmentScheme<KeccakTranscript>;

        // Simple test model and input
        let model_fn = builder::custom_add_model;
        let input = Tensor::new(Some(&[1, 2, 3, 4]), &[1, 4]).unwrap();

        // Build program
        let model = model_fn();
        let program_bytecode = onnx_tracer::decode_model(model.clone());
        let pp: JoltProverPreprocessing<Fr, PCS, KeccakTranscript> =
            JoltSNARK::prover_preprocess(program_bytecode);

        // Generate execution trace
        let (raw_trace, program_output) = onnx_tracer::execution_trace(model, &input);
        let execution_trace = crate::jolt::execution_trace::jolt_execution_trace(raw_trace.clone());

        // Create SNARK proof
        let snark: JoltSNARK<Fr, PCS, KeccakTranscript> =
            JoltSNARK::prove(pp.clone(), execution_trace, &program_output);

        // Serialize proof
        let mut serialized_proof = Vec::new();
        snark.serialize_compressed(&mut serialized_proof).unwrap();

        // Deserialize proof
        let deserialized_snark =
            JoltSNARK::<Fr, PCS, KeccakTranscript>::deserialize_compressed(&*serialized_proof)
                .unwrap();

        // Verify the deserialized proof
        deserialized_snark
            .verify((&pp).into(), program_output)
            .expect("Deserialized proof verification failed");
    }
}
