use crate::onnx_proof::ops::{OperatorProver, OperatorVerifier};
use atlas_onnx_tracer::{
    model::{
        trace::{LayerData, ModelExecutionIO, Trace},
        Model,
    },
    tensor::Tensor,
};
use common::VirtualPolynomial;
use joltworks::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, Openings, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator,
        },
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub mod lookup_tables;
pub mod op_lookups;
pub mod ops;
pub mod range_checking;
pub mod witness;

/// Prover state that owns all data needed during proving.
/// Created once before the proving loop and passed to operator handlers.
pub struct Prover<F: JoltField, T: Transcript> {
    pub trace: Trace,
    pub preprocessing: AtlasSharedPreprocessing,
    pub accumulator: ProverOpeningAccumulator<F>,
    pub transcript: T,
}

impl<F: JoltField, T: Transcript> Prover<F, T> {
    /// Create a new prover with the given preprocessing and trace
    pub fn new(preprocessing: AtlasSharedPreprocessing, trace: Trace) -> Self {
        let max_T = preprocessing.model.max_T();
        Self {
            trace,
            preprocessing,
            accumulator: ProverOpeningAccumulator::new(max_T),
            transcript: T::new(b"ONNXProof"),
        }
    }
}

/// Verifier state that owns all data needed during verification.
/// Created once before the verification loop and passed to operator handlers.
pub struct Verifier<'a, F: JoltField, T: Transcript> {
    pub preprocessing: &'a AtlasSharedPreprocessing,
    pub accumulator: VerifierOpeningAccumulator<F>,
    pub transcript: T,
    pub proofs: &'a BTreeMap<ProofId, SumcheckInstanceProof<F, T>>,
    pub io: &'a ModelExecutionIO,
}

impl<'a, F: JoltField, T: Transcript> Verifier<'a, F, T> {
    /// Create a new verifier with the given preprocessing, proofs, and IO
    pub fn new(
        preprocessing: &'a AtlasSharedPreprocessing,
        proofs: &'a BTreeMap<ProofId, SumcheckInstanceProof<F, T>>,
        io: &'a ModelExecutionIO,
    ) -> Self {
        let max_T = preprocessing.model.max_T();
        Self {
            preprocessing,
            accumulator: VerifierOpeningAccumulator::new(max_T),
            transcript: T::new(b"ONNXProof"),
            proofs,
            io,
        }
    }
}

/* ---------- Prover Logic ---------- */

#[derive(Debug, Clone)]
pub struct ONNXProof<F: JoltField, T: Transcript, PCS: CommitmentScheme<Field = F>> {
    pub opening_claims: Claims<F>,
    pub proofs: BTreeMap<ProofId, SumcheckInstanceProof<F, T>>,
    pub commitments: Vec<PCS::Commitment>,
}

impl<F: JoltField, T: Transcript, PCS: CommitmentScheme<Field = F>> ONNXProof<F, T, PCS> {
    pub fn prove(pp: &AtlasSharedPreprocessing, input: &Tensor<i32>) -> (Self, ModelExecutionIO) {
        // Generate trace and io
        let trace = pp.model.trace(&[input.clone()]); // TODO: Allow for multiple inputs
        let io = Trace::io(&trace, &pp.model);

        // Initialize prover state
        let mut prover: Prover<F, T> = Prover::new(pp.clone(), trace);
        let mut proofs = BTreeMap::new();

        // Evaluate output MLE at random point τ
        let output_index = pp.model.outputs()[0];
        let output_computation_node = &pp.model[output_index];
        let LayerData {
            operands: _,
            output,
        } = Trace::layer_data(&prover.trace, output_computation_node);
        let r_node_output = prover
            .transcript
            .challenge_vector_optimized::<F>(output.len().log_2());
        let output_claim = MultilinearPolynomial::from(output.clone()).evaluate(&r_node_output);
        prover.accumulator.append_virtual(
            &mut prover.transcript,
            VirtualPolynomial::NodeOutput(output_computation_node.idx),
            SumcheckId::Execution,
            r_node_output.clone().into(),
            output_claim,
        );

        // Iterate over computation graph in reverse topological order
        // Prove each operation using sum-check and virtual polynomials
        for (_, computation_node) in pp.model.graph.nodes.iter().rev() {
            let new_proofs = OperatorProver::prove(computation_node, &mut prover);
            for (proof_id, proof) in new_proofs {
                proofs.insert(proof_id, proof);
            }
        }

        (
            Self {
                proofs,
                opening_claims: Claims(prover.accumulator.take()),
                commitments: vec![],
            },
            io,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ProofId(pub usize, pub ProofType);

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ProofType {
    Execution,
    RaOneHotChecks,
    SoftmaxDivSumMax,
    SoftmaxExponentiation,
    RangeCheck,
}

#[derive(Debug, Clone)]
pub struct Claims<F: JoltField>(pub Openings<F>);

/* ---------- Verifier Logic ---------- */

impl<F: JoltField, T: Transcript, PCS: CommitmentScheme<Field = F>> ONNXProof<F, T, PCS> {
    pub fn verify(
        &self,
        pp: &AtlasSharedPreprocessing,
        io: &ModelExecutionIO,
    ) -> Result<(), ProofVerifyError> {
        // Initialize verifier state
        let mut verifier: Verifier<F, T> = Verifier::new(pp, &self.proofs, io);
        // Populate claims in the verifier accumulator
        for (key, (_, claim)) in &self.opening_claims.0 {
            verifier
                .accumulator
                .openings
                .insert(*key, (OpeningPoint::default(), *claim));
        }

        // Evaluate output MLE at random point τ
        let output_index = pp.model.outputs()[0];
        let output_computation_node = &pp.model[output_index];
        let r_node_output = verifier
            .transcript
            .challenge_vector_optimized::<F>(output_computation_node.num_output_elements().log_2());
        let expected_output_claim =
            MultilinearPolynomial::from(io.outputs[0].clone()).evaluate(&r_node_output);
        verifier.accumulator.append_virtual(
            &mut verifier.transcript,
            VirtualPolynomial::NodeOutput(output_computation_node.idx),
            SumcheckId::Execution,
            r_node_output.clone().into(),
        );
        let output_claim = verifier
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(output_computation_node.idx),
                SumcheckId::Execution,
            )
            .1;
        if expected_output_claim != output_claim {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Expected output claim does not match actual output claim".to_string(),
            ));
        }

        // Iterate over computation graph in reverse topological order
        // Verify each operation using dispatch
        for (_, computation_node) in pp.model.graph.nodes.iter().rev() {
            let res = OperatorVerifier::verify(computation_node, &mut verifier);
            #[cfg(test)]
            {
                if let Err(e) = &res {
                    println!("Verification failed at node {computation_node:#?}: {e:?}",);
                }
            }
            res?;
        }
        Ok(())
    }
}

/* ---------- Preprocessing ---------- */

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AtlasSharedPreprocessing {
    pub model: Model,
}

impl AtlasSharedPreprocessing {
    pub fn preprocess(model: Model) -> Self {
        Self { model }
    }
}

#[derive(Clone)]
pub struct AtlasProverPreprocessing<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    pub generators: PCS::ProverSetup,
    pub shared: AtlasSharedPreprocessing,
}

impl<F, PCS> AtlasProverPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    #[tracing::instrument(skip_all, name = "AtlasProverPreprocessing::gen")]
    pub fn new(shared: AtlasSharedPreprocessing) -> AtlasProverPreprocessing<F, PCS> {
        use common::consts::ONEHOT_CHUNK_THRESHOLD_LOG_T;
        let max_T: usize = shared.model.max_T();
        let max_log_T = max_T.log_2();
        // Use the maximum possible log_k_chunk for generator setup
        let max_log_k_chunk = if max_log_T < ONEHOT_CHUNK_THRESHOLD_LOG_T {
            4
        } else {
            8
        };
        let generators = PCS::setup_prover(max_log_k_chunk + max_log_T);
        AtlasProverPreprocessing { generators, shared }
    }
}

#[derive(Debug, Clone)]
pub struct AtlasVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub generators: PCS::VerifierSetup,
    pub shared: AtlasSharedPreprocessing,
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> From<&AtlasProverPreprocessing<F, PCS>>
    for AtlasVerifierPreprocessing<F, PCS>
{
    fn from(prover_preprocessing: &AtlasProverPreprocessing<F, PCS>) -> Self {
        let generators = PCS::setup_verifier(&prover_preprocessing.generators);
        Self {
            generators,
            shared: prover_preprocessing.shared.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::onnx_proof::{AtlasSharedPreprocessing, ONNXProof};
    use ark_bn254::Fr;
    use atlas_onnx_tracer::{model::Model, tensor::Tensor};
    use joltworks::{poly::commitment::dory::DoryCommitmentScheme, transcripts::Blake2bTranscript};
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use serde_json::Value;
    use std::{collections::HashMap, fs::File, io::Read, time::Instant};

    // Fixed-point scale factor: 2^7 = 128
    const SCALE: i32 = 128;

    #[test]
    fn test_self_attention_layer() {
        let working_dir = "../atlas-onnx-tracer/models/self_attention_layer/";
        let mut _rng = StdRng::seed_from_u64(0x1003);
        let input_data = vec![SCALE; 64 * 64];
        // let input_data: Vec<i32> = (0..64 * 64)
        //     .map(|_| SCALE + rng.gen_range(-1..=1))
        //     .collect(); // TODO(Forpee): Investigate bug - Einsum mk,kn->mn node failing
        let input = Tensor::construct(input_data, vec![1, 64, 64]);

        // Load the model
        let model = Model::load(&format!("{working_dir}network.onnx"), &Default::default());
        let pp = AtlasSharedPreprocessing::preprocess(model);
        let (proof, io) =
            ONNXProof::<Fr, Blake2bTranscript, DoryCommitmentScheme>::prove(&pp, &input);

        // verify proof
        proof.verify(&pp, &io).unwrap();
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

        // Load the model
        let model = Model::load(&format!("{working_dir}network.onnx"), &Default::default());
        println!("model: {}", model.pretty_print());

        let pp = AtlasSharedPreprocessing::preprocess(model);
        let timing = Instant::now();
        let (proof, io) = ONNXProof::<Fr, Blake2bTranscript, DoryCommitmentScheme>::prove(
            &pp,
            &Tensor::construct(input_vector, vec![1, 512]),
        );
        println!("Proof generation took {:?}", timing.elapsed());

        // verify proof
        proof.verify(&pp, &io).unwrap();

        /// Load vocab.json into HashMap<String, (usize, i32)>
        fn load_vocab(
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

    #[test]
    fn test_add_sub_mul() {
        let working_dir = "../atlas-onnx-tracer/models/test_add_sub_mul/";

        // Create test input vector of size 65536
        // Using small values to avoid overflow
        let mut rng = StdRng::seed_from_u64(0x100);
        // Create tensor with shape [65536]
        let input = Tensor::random_small(&mut rng, &[1 << 16]);

        // Load the model
        let model = Model::load(&format!("{working_dir}network.onnx"), &Default::default());
        println!("model: {}", model.pretty_print());

        let pp = AtlasSharedPreprocessing::preprocess(model);
        let (proof, io) =
            ONNXProof::<Fr, Blake2bTranscript, DoryCommitmentScheme>::prove(&pp, &input);

        // verify proof
        proof.verify(&pp, &io).unwrap();
    }

    #[test]
    fn test_rsqrt() {
        let working_dir = "../atlas-onnx-tracer/models/rsqrt/";

        // Create test input vector of size 4
        let mut rng = StdRng::seed_from_u64(0x100);
        let input_vec = (0..4)
            .map(|_| {
                // Generate random positive non-zero values
                rng.gen_range(1..i32::MAX)
            })
            .collect::<Vec<i32>>();

        let input = Tensor::construct(input_vec, vec![4]);

        // Load the model
        let model = Model::load(&format!("{working_dir}network.onnx"), &Default::default());
        println!("model: {}", model.pretty_print());

        let pp = AtlasSharedPreprocessing::preprocess(model);
        let timing = Instant::now();
        let (proof, io) =
            ONNXProof::<Fr, Blake2bTranscript, DoryCommitmentScheme>::prove(&pp, &input);
        println!("Proof generation took {:?}", timing.elapsed());

        // verify proof
        proof.verify(&pp, &io).unwrap();
    }

    #[test]
    fn test_perceptron() {
        let working_dir = "../atlas-onnx-tracer/models/perceptron/";

        // Create test input vector of size [4]
        // Using simple values to test broadcasting
        let input_vector = vec![1, 2, 3, 4];

        let input = Tensor::construct(input_vector, vec![1, 4]);

        // Load the model
        let model = Model::load(&format!("{working_dir}network.onnx"), &Default::default());
        println!("model: {}", model.pretty_print());

        let pp = AtlasSharedPreprocessing::preprocess(model);
        let timing = Instant::now();
        let (proof, io) =
            ONNXProof::<Fr, Blake2bTranscript, DoryCommitmentScheme>::prove(&pp, &input);
        println!("Proof generation took {:?}", timing.elapsed());

        // verify proof
        proof.verify(&pp, &io).unwrap();
    }

    #[test]
    fn test_broadcast() {
        let working_dir = "../atlas-onnx-tracer/models/broadcast/";

        // Create test input vector of size [4]
        // Using simple values to test broadcasting
        let input_vector = vec![1, 2, 3, 4];

        let input = Tensor::construct(input_vector, vec![4]);

        // Load the model
        let model = Model::load(&format!("{working_dir}network.onnx"), &Default::default());
        println!("model: {}", model.pretty_print());

        let pp = AtlasSharedPreprocessing::preprocess(model);
        let timing = Instant::now();
        let (proof, io) =
            ONNXProof::<Fr, Blake2bTranscript, DoryCommitmentScheme>::prove(&pp, &input);
        println!("Proof generation took {:?}", timing.elapsed());

        // verify proof
        proof.verify(&pp, &io).unwrap();

        // Print output for verification
        println!("Output shape: {:?}", io.outputs[0].dims());
        println!("Expected: input [4] broadcasted through operations to shape [2, 5, 4]");
    }

    #[test]
    fn test_reshape() {
        let working_dir = "../atlas-onnx-tracer/models/reshape/";

        // Create test input vector of size [4]
        // Using simple values to test reshaping
        let input_vector = vec![1, 2, 3, 4];

        let input = Tensor::construct(input_vector, vec![4]);

        // Load the model
        let model = Model::load(&format!("{working_dir}network.onnx"), &Default::default());
        println!("model: {}", model.pretty_print());

        let pp = AtlasSharedPreprocessing::preprocess(model);
        let timing = Instant::now();
        let (proof, io) =
            ONNXProof::<Fr, Blake2bTranscript, DoryCommitmentScheme>::prove(&pp, &input);
        println!("Proof generation took {:?}", timing.elapsed());

        // verify proof
        proof.verify(&pp, &io).unwrap();

        // Print output for verification
        println!("Output shape: {:?}", io.outputs[0].dims());
    }

    #[test]
    fn test_moveaxis() {
        let working_dir = "../atlas-onnx-tracer/models/moveaxis/";

        // Create test input vector of size [2, 4, 8]
        // Using simple values to test moveaxis
        let input_vector: Vec<i32> = (1..=64).collect();

        let input = Tensor::construct(input_vector, vec![2, 4, 8]);
        // Load the model
        let model = Model::load(&format!("{working_dir}network.onnx"), &Default::default());
        println!("model: {}", model.pretty_print());

        let pp = AtlasSharedPreprocessing::preprocess(model);
        let timing = Instant::now();
        let (proof, io) =
            ONNXProof::<Fr, Blake2bTranscript, DoryCommitmentScheme>::prove(&pp, &input);
        println!("Proof generation took {:?}", timing.elapsed());

        // verify proof
        proof.verify(&pp, &io).unwrap();

        // Print output for verification
        println!("Output shape: {:?}", io.outputs[0].dims());
    }

    #[test]
    fn test_mlp_square() {
        // Create test input vector [1, 4]
        // Using simple values for testing
        let input_vector = vec![
            (70.0 * SCALE as f32) as i32,
            (71.0 * SCALE as f32) as i32,
            (72.0 * SCALE as f32) as i32,
            (73.0 * SCALE as f32) as i32,
        ];
        let working_dir = "../atlas-onnx-tracer/models/mlp_square/";

        let input = Tensor::new(Some(&input_vector), &[1, 4]).unwrap();

        // Load the model
        let model = Model::load(&format!("{working_dir}network.onnx"), &Default::default());

        let pp = AtlasSharedPreprocessing::preprocess(model);
        let (proof, io) =
            ONNXProof::<Fr, Blake2bTranscript, DoryCommitmentScheme>::prove(&pp, &input);

        // verify proof
        proof.verify(&pp, &io).unwrap();
    }

    #[test]
    fn test_mlp_square_4layer() {
        // Create test input vector [1, 4]
        // Using simple values for testing
        let input_vector = vec![
            (1.0 * SCALE as f32) as i32,
            (2.0 * SCALE as f32) as i32,
            (3.0 * SCALE as f32) as i32,
            (4.0 * SCALE as f32) as i32,
        ];
        let working_dir = "../atlas-onnx-tracer/models/mlp_square_4layer/";

        let input = Tensor::new(Some(&input_vector), &[1, 4]).unwrap();

        // Load the model
        let model = Model::load(&format!("{working_dir}network.onnx"), &Default::default());

        let pp = AtlasSharedPreprocessing::preprocess(model);
        let (proof, io) =
            ONNXProof::<Fr, Blake2bTranscript, DoryCommitmentScheme>::prove(&pp, &input);

        // verify proof
        proof.verify(&pp, &io).unwrap();
    }
}
