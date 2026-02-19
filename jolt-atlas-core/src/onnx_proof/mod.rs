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

#[cfg(test)]
mod e2e_tests;

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
