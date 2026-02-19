//! Proofs for ONNX neural network computations.
//!
//! This module implements the core proving and verification logic for ONNX neural networks.
//! It provides:
//! - [`ONNXProof`]: The main proof structure containing all sumcheck proofs and commitments
//! - [`Prover`] and [`Verifier`]: State management for proving and verification
//! - Preprocessing for model setup and commitment scheme initialization
//! - Operator-specific proving logic for neural network operations

use crate::onnx_proof::{
    ops::{NodeCommittedPolynomials, OperatorProver, OperatorVerifier},
    witness::WitnessGenerator,
};
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, Read, SerializationError, Valid, Validate,
    Write,
};
use atlas_onnx_tracer::{
    model::{
        trace::{LayerData, ModelExecutionIO, Trace},
        Model,
    },
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
            VerifierOpeningAccumulator, VirtualOperandClaims,
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
pub mod proof_serialization;
pub mod range_checking;
pub mod witness;

pub use ark_bn254::{Bn254, Fr};
pub use joltworks::{poly::commitment::hyperkzg::HyperKZG, transcripts::Blake2bTranscript};

/// Prover state that owns all data needed during proving.
/// Created once before the proving loop and passed to operator handlers.
pub struct Prover<F: JoltField, T: Transcript> {
    /// Execution trace of the neural network model.
    pub trace: Trace,
    /// Shared preprocessing data (model structure).
    pub preprocessing: AtlasSharedPreprocessing,
    /// Opening accumulator for batching polynomial openings.
    pub accumulator: ProverOpeningAccumulator<F>,
    /// Interactive proof transcript.
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
    /// Shared preprocessing data (model structure).
    pub preprocessing: &'a AtlasSharedPreprocessing,
    /// Opening accumulator for batching polynomial openings.
    pub accumulator: VerifierOpeningAccumulator<F>,
    /// Interactive proof transcript.
    pub transcript: T,
    /// Map of proof IDs to sumcheck proofs.
    pub proofs: &'a BTreeMap<ProofId, SumcheckInstanceProof<F, T>>,
    /// Model execution inputs and outputs.
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

/// Complete ZK proof for an ONNX neural network computation.
///
/// Contains all sumcheck proofs, polynomial commitments, and opening proofs
/// needed to verify the correct execution of a neural network.
#[derive(Debug, Clone)]
pub struct ONNXProof<F: JoltField, T: Transcript, PCS: CommitmentScheme<Field = F>> {
    /// Opening claims for committed polynomials.
    pub opening_claims: Claims<F>,
    /// Map of proof IDs to sumcheck instance proofs.
    pub proofs: BTreeMap<ProofId, SumcheckInstanceProof<F, T>>,
    /// Claims for virtual polynomial operands.
    pub virtual_operand_claims: VirtualOperandClaims<F>,
    /// Polynomial commitments for witness polynomials.
    pub commitments: Vec<PCS::Commitment>,
    /// Batched opening proof using reduction sum-check protocol to reduce all polynomial openings to the same point.
    reduced_opening_proof: Option<ReducedOpeningProof<F, T, PCS>>,
}

/// Batched polynomial opening proof using sumcheck reduction.
///
/// Reduces multiple polynomial openings to a single joint opening using sumcheck.
#[derive(Debug, Clone)]
pub struct ReducedOpeningProof<F: JoltField, T: Transcript, PCS: CommitmentScheme<Field = F>> {
    /// Sumcheck proof for batching multiple openings.
    pub sumcheck_proof: SumcheckInstanceProof<F, T>,
    /// Evaluation claims at the sumcheck point.
    pub sumcheck_claims: Vec<F>,
    /// Joint opening proof for the batched polynomial.
    joint_opening_proof: PCS::Proof,
}

impl<F: JoltField, T: Transcript, PCS: CommitmentScheme<Field = F>> ONNXProof<F, T, PCS> {
    /// Generate a proof for an ONNX neural network computation.
    ///
    /// Executes the model with the given inputs, generates a trace, and produces
    /// sumcheck proofs for each operation. Returns the proof, execution IO, and
    /// optional debug information.
    #[tracing::instrument(skip_all, name = "ONNXProof::prove")]
    pub fn prove(
        pp: &AtlasProverPreprocessing<F, PCS>,
        inputs: &[Tensor<i32>],
    ) -> (Self, ModelExecutionIO, Option<ProverDebugInfo<F, T>>) {
        // Generate trace and io
        let trace = pp.model().trace(inputs);
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
        let span = tracing::span!(tracing::Level::INFO, "IOP Proving Portion");
        let _guard = span.enter();
        for (_, computation_node) in pp.model().graph.nodes.iter().rev() {
            let new_proofs = OperatorProver::prove(computation_node, &mut prover);
            for (proof_id, proof) in new_proofs {
                proofs.insert(proof_id, proof);
            }
        }
        drop(_guard);
        drop(span);

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

        let (opening_claims, virtual_operand_claims) = prover.accumulator.take();
        (
            Self {
                proofs,
                opening_claims: Claims(opening_claims),
                virtual_operand_claims,
                commitments,
                reduced_opening_proof,
            },
            io,
            debug_info,
        )
    }

    #[tracing::instrument(skip_all)]
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

    #[tracing::instrument(skip_all)]
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

/// Unique identifier for a sumcheck proof instance.
///
/// Combines the node index with the proof type to uniquely identify each proof.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, CanonicalSerialize, CanonicalDeserialize,
)]
pub struct ProofId(pub usize, pub ProofType);

/// Type of sumcheck proof for different operations in the neural network.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ProofType {
    /// Execution sumcheck for basic operations.
    Execution,
    /// Neural teleportation for tanh approximation.
    NeuralTeleport,
    /// Read-address one-hot encoding checks.
    RaOneHotChecks,
    /// Hamming weight check for read-addresses.
    RaHammingWeight,
    /// Softmax division by sum of max.
    SoftmaxDivSumMax,
    /// Softmax exponentiation read-raf checking.
    SoftmaxExponentiationReadRaf,
    /// Softmax exponentiation read-address one-hot encoding checks.
    SoftmaxExponentiationRaOneHot,
    /// Range-checking for remainders.
    RangeCheck,
}

impl CanonicalSerialize for ProofType {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        let tag: u8 = match self {
            Self::Execution => 0,
            Self::NeuralTeleport => 1,
            Self::RaOneHotChecks => 2,
            Self::RaHammingWeight => 3,
            Self::SoftmaxDivSumMax => 4,
            Self::SoftmaxExponentiationReadRaf => 5,
            Self::SoftmaxExponentiationRaOneHot => 6,
            Self::RangeCheck => 7,
        };
        tag.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        0u8.serialized_size(compress)
    }
}

impl Valid for ProofType {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for ProofType {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let tag = u8::deserialize_with_mode(reader, compress, validate)?;
        match tag {
            0 => Ok(Self::Execution),
            1 => Ok(Self::NeuralTeleport),
            2 => Ok(Self::RaOneHotChecks),
            3 => Ok(Self::RaHammingWeight),
            4 => Ok(Self::SoftmaxDivSumMax),
            5 => Ok(Self::SoftmaxExponentiationReadRaf),
            6 => Ok(Self::SoftmaxExponentiationRaOneHot),
            7 => Ok(Self::RangeCheck),
            _ => Err(SerializationError::InvalidData),
        }
    }
}

/// Wrapper for polynomial opening claims.
#[derive(Debug, Clone)]
pub struct Claims<F: JoltField>(pub Openings<F>);

/* ---------- Verifier Logic ---------- */

impl<F: JoltField, T: Transcript, PCS: CommitmentScheme<Field = F>> ONNXProof<F, T, PCS> {
    /// Verify a proof for an ONNX neural network computation.
    ///
    /// Checks all sumcheck proofs, validates opening claims, and verifies that the
    /// computation produces the expected output.
    #[tracing::instrument(skip_all, name = "ONNXProof::verify")]
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
        verifier.accumulator.virtual_operand_claims = self.virtual_operand_claims.clone();

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
            let reduction_res = verifier.accumulator.verify_batch_opening_sumcheck(
                &reduced_opening_proof.sumcheck_proof,
                &mut verifier.transcript,
            );
            #[cfg(test)]
            {
                if let Err(e) = &reduction_res {
                    println!("Opening reduction via sumcheck failed: {e:?}");
                }
            }
            let r_sumcheck = reduction_res?;

            // Finalize and store state in accumulator
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

/// Shared preprocessing data for both prover and verifier.
///
/// Contains the ONNX model structure that is used by both parties.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AtlasSharedPreprocessing {
    /// The ONNX neural network model.
    pub model: Model,
}

impl AtlasSharedPreprocessing {
    /// Preprocess an ONNX model for proving and verification.
    #[tracing::instrument(skip_all, name = "AtlasSharedPreprocessing::preprocess")]
    pub fn preprocess(model: Model) -> Self {
        Self { model }
    }

    /// Get all committed polynomials required by the model's operations.
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

/// Prover-specific preprocessing with commitment scheme generators.
///
/// Contains the polynomial commitment scheme setup for the prover.
#[derive(Clone)]
pub struct AtlasProverPreprocessing<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    /// Polynomial commitment scheme prover setup (SRS generators).
    pub generators: PCS::ProverSetup,
    /// Shared preprocessing data.
    pub shared: AtlasSharedPreprocessing,
}

impl<F, PCS> AtlasProverPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    /// Create new prover preprocessing from shared preprocessing.
    ///
    /// Generates the polynomial commitment scheme setup (SRS) based on the
    /// maximum number of variables in the model.
    #[tracing::instrument(skip_all, name = "AtlasProverPreprocessing::gen")]
    pub fn new(shared: AtlasSharedPreprocessing) -> AtlasProverPreprocessing<F, PCS> {
        let model = &shared.model;
        let max_num_vars = model.max_num_vars();
        tracing::info!("Prover preprocessing: max_num_vars = {max_num_vars}");
        let generators = PCS::setup_prover(max_num_vars);
        AtlasProverPreprocessing { generators, shared }
    }

    /// Get a reference to the ONNX model.
    pub fn model(&self) -> &Model {
        &self.shared.model
    }
}

/// Verifier-specific preprocessing with commitment scheme verification key.
///
/// Contains the polynomial commitment scheme setup for the verifier.
#[derive(Debug, Clone)]
pub struct AtlasVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    /// Polynomial commitment scheme verifier setup (verification key).
    pub generators: PCS::VerifierSetup,
    /// Shared preprocessing data.
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
    /// Get a reference to the ONNX model.
    pub fn model(&self) -> &Model {
        &self.shared.model
    }
}

/// Debug information from the prover for testing and verification.
///
/// Contains the transcript and opening accumulator state for comparing
/// prover and verifier execution in tests.
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
        let (proof, io, debug_info) = ONNXProof::<Fr, Blake2bTranscript, HyperKZG<Bn254>>::prove(
            &prover_preprocessing,
            inputs,
        );
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
            TestConfig::default(),
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
            TestConfig::new().print_model().print_timing(),
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
}
