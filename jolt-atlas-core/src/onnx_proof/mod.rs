use crate::onnx_proof::{
    ops::{NodeCommittedPolynomials, OperatorProver, OperatorVerifier},
    witness::WitnessGenerator,
};
use atlas_onnx_tracer::{
    model::{
        trace::{LayerData, ModelExecutionIO, Trace},
        Model,
    },
    ops::Operator,
    tensor::Tensor,
};
use common::{CommittedPolynomial, VirtualPolynomial};
use joltworks::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, Openings, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator,
        },
        rlc_polynomial::build_materialized_rlc,
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub mod lookup_tables;
pub mod neural_teleport;
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
        Self {
            trace,
            preprocessing,
            accumulator: ProverOpeningAccumulator::new(),
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
        Self {
            preprocessing,
            accumulator: VerifierOpeningAccumulator::new(),
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
    reduced_opening_proof: Option<ReducedOpeningProof<F, T, PCS>>,
}

#[derive(Debug, Clone)]
pub struct ReducedOpeningProof<F: JoltField, T: Transcript, PCS: CommitmentScheme<Field = F>> {
    pub sumcheck_proof: SumcheckInstanceProof<F, T>,
    pub sumcheck_claims: Vec<F>,
    joint_opening_proof: PCS::Proof,
}

impl<F: JoltField, T: Transcript, PCS: CommitmentScheme<Field = F>> ONNXProof<F, T, PCS> {
    pub fn prove(
        pp: &AtlasProverPreprocessing<F, PCS>,
        input: &Tensor<i32>,
    ) -> (Self, ModelExecutionIO, Option<ProverDebugInfo<F, T>>) {
        // Generate trace and io
        let trace = pp.model().trace(&[input.clone()]); // TODO: Allow for multiple inputs
        let io = Trace::io(&trace, pp.model());

        // Initialize prover state
        let mut prover: Prover<F, T> = Prover::new(pp.shared.clone(), trace);
        let mut proofs = BTreeMap::new();

        let poly_map = Self::polynomial_map(pp.model(), &prover.trace);
        let commitments = Self::commit_to_polynomials(&poly_map, &pp.generators);
        for commitment in commitments.iter() {
            prover.transcript.append_serializable(commitment);
        }

        // Evaluate output MLE at random point τ
        let output_index = pp.model().outputs()[0];
        let output_computation_node = &pp.model()[output_index];
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
        for (_, computation_node) in pp.model().graph.nodes.iter().rev() {
            let new_proofs = OperatorProver::prove(computation_node, &mut prover);
            for (proof_id, proof) in new_proofs {
                proofs.insert(proof_id, proof);
            }
        }

        let reduced_opening_proof = if poly_map.is_empty() {
            None
        } else {
            prover.accumulator.prepare_for_sumcheck(&poly_map);

            // Run sumcheck
            let (accumulator_sumcheck_proof, r_sumcheck_acc) = prover
                .accumulator
                .prove_batch_opening_sumcheck(&mut prover.transcript);

            // Finalize sumcheck (uses claims cached via cache_openings, derives gamma, cleans up)
            let state = prover
                .accumulator
                .finalize_batch_opening_sumcheck(r_sumcheck_acc.clone(), &mut prover.transcript);
            let sumcheck_claims: Vec<F> = state.sumcheck_claims.clone();
            // Build RLC
            let rlc = build_materialized_rlc(&state.gamma_powers, &poly_map);
            // Create joint opening proof
            let joint_opening_proof = PCS::prove(
                &pp.generators,
                &rlc,
                &state.r_sumcheck,
                None,
                &mut prover.transcript,
            );
            Some(ReducedOpeningProof {
                sumcheck_proof: accumulator_sumcheck_proof,
                sumcheck_claims,
                joint_opening_proof,
            })
        };
        #[cfg(test)]
        let debug_info = Some(ProverDebugInfo {
            transcript: prover.transcript.clone(),
            opening_accumulator: prover.accumulator.clone(),
        });
        #[cfg(not(test))]
        let debug_info = None;
        (
            Self {
                proofs,
                opening_claims: Claims(prover.accumulator.take()),
                commitments,
                reduced_opening_proof,
            },
            io,
            debug_info,
        )
    }

    fn polynomial_map(
        model: &Model,
        trace: &Trace,
    ) -> BTreeMap<CommittedPolynomial, MultilinearPolynomial<F>> {
        let mut poly_map = BTreeMap::new();
        for (_, node) in model.graph.nodes.iter() {
            let node_polys = NodeCommittedPolynomials::get_committed_polynomials::<F, T>(node);
            for committed_poly in node_polys {
                let witness_poly = committed_poly.generate_witness(model, trace);
                poly_map.insert(committed_poly, witness_poly);
            }
        }
        poly_map
    }

    fn commit_to_polynomials(
        poly_map: &BTreeMap<CommittedPolynomial, MultilinearPolynomial<F>>,
        pcs: &PCS::ProverSetup,
    ) -> Vec<PCS::Commitment> {
        poly_map
            .values()
            .map(|poly| PCS::commit(poly, pcs).0)
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ProofId(pub usize, pub ProofType);

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ProofType {
    Execution,
    NeuralTeleport,
    RaOneHotChecks,
    RaHammingWeight,
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
        pp: &AtlasVerifierPreprocessing<F, PCS>,
        io: &ModelExecutionIO,
        _debug_info: Option<ProverDebugInfo<F, T>>,
    ) -> Result<(), ProofVerifyError> {
        // Initialize verifier state
        let mut verifier: Verifier<F, T> = Verifier::new(&pp.shared, &self.proofs, io);
        #[cfg(test)]
        {
            if let Some(debug_info) = _debug_info {
                verifier.transcript.compare_to(debug_info.transcript);
                verifier
                    .accumulator
                    .compare_to(debug_info.opening_accumulator);
            }
        }
        // Populate claims in the verifier accumulator
        for (key, (_, claim)) in &self.opening_claims.0 {
            verifier
                .accumulator
                .openings
                .insert(*key, (OpeningPoint::default(), *claim));
        }

        for commitment in self.commitments.iter() {
            verifier.transcript.append_serializable(commitment);
        }

        // Evaluate output MLE at random point τ
        let output_index = pp.model().outputs()[0];
        let output_computation_node = &pp.model()[output_index];
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
        for (_, computation_node) in pp.model().graph.nodes.iter().rev() {
            let res = OperatorVerifier::verify(computation_node, &mut verifier);
            #[cfg(test)]
            {
                if let Err(e) = &res {
                    println!("Verification failed at node {computation_node:#?}: {e:?}",);
                }
            }
            res?;
        }

        if let Some(reduced_opening_proof) = &self.reduced_opening_proof {
            // Prepare - populate sumcheck claims
            verifier
                .accumulator
                .prepare_for_sumcheck(&reduced_opening_proof.sumcheck_claims);

            // Verify sumcheck
            let r_sumcheck = verifier.accumulator.verify_batch_opening_sumcheck(
                &reduced_opening_proof.sumcheck_proof,
                &mut verifier.transcript,
            )?;

            // Finalize and store state in accumulator for Stage 8
            let verifier_state = verifier.accumulator.finalize_batch_opening_sumcheck(
                r_sumcheck,
                &reduced_opening_proof.sumcheck_claims,
                &mut verifier.transcript,
            );

            // Compute joint commitment
            let joint_commitment =
                PCS::combine_commitments(&self.commitments, &verifier_state.gamma_powers);

            // Verify joint opening
            verifier.accumulator.verify_joint_opening::<_, PCS>(
                &pp.generators,
                &reduced_opening_proof.joint_opening_proof,
                &joint_commitment,
                &verifier_state,
                &mut verifier.transcript,
            )?;
        } else {
            let committed_polys = pp.shared.get_models_committed_polynomials::<F, T>();
            if !committed_polys.is_empty() {
                return Err(ProofVerifyError::MissingReductionProof);
            }
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

    pub fn get_models_committed_polynomials<F: JoltField, T: Transcript>(
        &self,
    ) -> Vec<CommittedPolynomial> {
        let mut polys = vec![];
        for (_, node) in self.model.graph.nodes.iter() {
            let node_polys = NodeCommittedPolynomials::get_committed_polynomials::<F, T>(node);
            polys.extend(node_polys);
        }
        polys
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
        let small_lookups_log_kt = shared
            .model
            .graph
            .nodes
            .iter()
            .map(|node| match &node.1.operator {
                Operator::Tanh(inner) => inner.log_table + node.1.num_output_elements().log_2(),
                Operator::Gather(_inner) => {
                    todo!()
                }
                _ => 0,
            })
            .max()
            .unwrap_or(0);
        let generators = PCS::setup_prover((max_log_k_chunk + max_log_T).max(small_lookups_log_kt));
        AtlasProverPreprocessing { generators, shared }
    }

    pub fn model(&self) -> &Model {
        &self.shared.model
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

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> AtlasVerifierPreprocessing<F, PCS> {
    pub fn model(&self) -> &Model {
        &self.shared.model
    }
}

#[allow(dead_code)]
pub struct ProverDebugInfo<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    pub(crate) transcript: ProofTranscript,
    pub(crate) opening_accumulator: ProverOpeningAccumulator<F>,
}

#[cfg(test)]
mod tests {
    use crate::onnx_proof::{
        AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing, ONNXProof,
    };
    use ark_bn254::{Bn254, Fr};
    use atlas_onnx_tracer::{
        model::{trace::ModelExecutionIO, Model},
        tensor::Tensor,
    };
    use joltworks::{poly::commitment::hyperkzg::HyperKZG, transcripts::Blake2bTranscript};
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use serde_json::Value;
    use std::{collections::HashMap, fs::File, io::Read, time::Instant};

    // Fixed-point scale factor: 2^7 = 128
    const SCALE: i32 = 128;

    /// Helper function to run the common prove-and-verify workflow.
    /// Returns the proof IO for additional assertions if needed.
    fn prove_and_verify(
        model_dir: &str,
        input: &Tensor<i32>,
        print_model: bool,
        print_timing: bool,
    ) -> ModelExecutionIO {
        prove_and_verify_with_debug(model_dir, input, print_model, print_timing, false).0
    }

    /// Helper function with debug info support.
    fn prove_and_verify_with_debug(
        model_dir: &str,
        input: &Tensor<i32>,
        print_model: bool,
        print_timing: bool,
        use_debug_info: bool,
    ) -> (ModelExecutionIO, ()) {
        let model = Model::load(&format!("{model_dir}network.onnx"), &Default::default());
        if print_model {
            println!("model: {}", model.pretty_print());
        }

        let pp = AtlasSharedPreprocessing::preprocess(model);
        let prover_preprocessing = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(pp);

        let timing = Instant::now();
        let (proof, io, debug_info) = ONNXProof::<Fr, Blake2bTranscript, HyperKZG<Bn254>>::prove(
            &prover_preprocessing,
            input,
        );
        if print_timing {
            println!("Proof generation took {:?}", timing.elapsed());
        }

        let verifier_preprocessing =
            AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_preprocessing);

        let debug_for_verify = if use_debug_info { debug_info } else { None };
        proof
            .verify(&verifier_preprocessing, &io, debug_for_verify)
            .unwrap();

        (io, ())
    }

    #[test]
    fn test_self_attention_layer() {
        let working_dir = "../atlas-onnx-tracer/models/self_attention_layer/";
        let mut _rng = StdRng::seed_from_u64(0x1003);
        let input_data = vec![SCALE; 64 * 64];
        // let input_data: Vec<i32> = (0..64 * 64)
        //     .map(|_| SCALE + _rng.gen_range(-1..=1))
        //     .collect(); // TODO(Forpee): Investigate bug - Einsum mk,kn->mn node failing
        let input = Tensor::construct(input_data, vec![1, 64, 64]);

        prove_and_verify(working_dir, &input, true, false);
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

        prove_and_verify(working_dir, &input, true, true);

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

        prove_and_verify_with_debug(working_dir, &input, true, false, true);
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

        prove_and_verify(working_dir, &input, true, true);
    }

    #[test]
    fn test_perceptron() {
        let working_dir = "../atlas-onnx-tracer/models/perceptron/";
        let input = Tensor::construct(vec![1, 2, 3, 4], vec![1, 4]);

        prove_and_verify(working_dir, &input, true, true);
    }

    #[test]
    fn test_broadcast() {
        let working_dir = "../atlas-onnx-tracer/models/broadcast/";
        let input = Tensor::construct(vec![1, 2, 3, 4], vec![4]);

        let io = prove_and_verify(working_dir, &input, true, true);

        // Print output for verification
        println!("Output shape: {:?}", io.outputs[0].dims());
        println!("Expected: input [4] broadcasted through operations to shape [2, 5, 4]");
    }

    #[test]
    fn test_reshape() {
        let working_dir = "../atlas-onnx-tracer/models/reshape/";
        let input = Tensor::construct(vec![1, 2, 3, 4], vec![4]);

        let io = prove_and_verify(working_dir, &input, true, true);

        println!("Output shape: {:?}", io.outputs[0].dims());
    }

    #[test]
    fn test_moveaxis() {
        let working_dir = "../atlas-onnx-tracer/models/moveaxis/";
        let input_vector: Vec<i32> = (1..=64).collect();
        let input = Tensor::construct(input_vector, vec![2, 4, 8]);

        let io = prove_and_verify(working_dir, &input, true, true);

        println!("Output shape: {:?}", io.outputs[0].dims());
    }

    #[test]
    fn test_gather() {
        let working_dir = "../atlas-onnx-tracer/models/gather/";
        let mut rng = StdRng::seed_from_u64(0x100);
        // Input values in [0, 8)
        let input = Tensor::random_range(&mut rng, &[1, 64], 0..65);

        prove_and_verify(working_dir, &input, true, false);
    }

    #[test]
    fn test_tanh() {
        let working_dir = "../atlas-onnx-tracer/models/tanh/";
        let input_vector = vec![10, 40, 70, 100, 130];
        let input = Tensor::new(Some(&input_vector), &[5]).unwrap();

        prove_and_verify(working_dir, &input, true, true);
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

        prove_and_verify(working_dir, &input, false, false);
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

        prove_and_verify(working_dir, &input, false, false);
    }
}
