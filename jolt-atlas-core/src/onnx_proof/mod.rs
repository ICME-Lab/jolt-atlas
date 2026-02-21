//! Proofs for ONNX neural network computations.
//!
//! This module implements the core proving and verification logic for ONNX neural networks.
//! It provides:
//! - [`ONNXProof`]: The main proof structure containing all sumcheck proofs and commitments
//! - [`ONNXProof::prove`]: Generate a proof for an ONNX model execution
//! - [`ONNXProof::verify`]: Verify a proof against expected IO
//! - Preprocessing structs for model setup and commitment scheme initialization
//!
//! Internal implementation details (helper methods, state structs) live in
//! [`prover`], [`verifier`], and [`types`] submodules.

use atlas_onnx_tracer::{
    model::trace::{ModelExecutionIO, Trace},
    tensor::Tensor,
};
use joltworks::{
    field::JoltField,
    poly::{commitment::commitment_scheme::CommitmentScheme, opening_proof::VirtualOperandClaims},
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};
use std::collections::BTreeMap;

// ── Submodules ───────────────────────────────────────────────────────────

pub mod neural_teleport;
pub mod op_lookups;
pub mod ops;
pub mod proof_serialization;
pub mod range_checking;
pub mod witness;

mod preprocessing;
mod prover;
mod types;
mod verifier;

#[cfg(test)]
mod e2e_tests;

// ── Re-exports ───────────────────────────────────────────────────────────

pub use preprocessing::{
    AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing,
};
pub use prover::Prover;
pub use types::{Claims, ProofId, ProofType, ProverDebugInfo};
pub use verifier::Verifier;

pub use ark_bn254::{Bn254, Fr};
pub use joltworks::{poly::commitment::hyperkzg::HyperKZG, transcripts::Blake2bTranscript};

// ── Core proof structures ────────────────────────────────────────────────

/// Proof for an ONNX neural network computation.
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

// ── Public API: prove & verify ───────────────────────────────────────────

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

        // Commit to witness polynomials and append commitments to transcript
        let (poly_map, commitments) = Self::commit_witness_polynomials(
            pp.model(),
            &prover.trace,
            &pp.generators,
            &mut prover.transcript,
        );

        // Evaluate output MLE at random point τ
        Self::output_claim(&mut prover);

        // IOP portion
        Self::iop(pp.model().nodes(), &mut prover, &mut proofs);

        // Reduction sum-check + PCS::prove
        let reduced_opening_proof =
            Self::prove_reduced_openings(&mut prover, &poly_map, &pp.generators);
        Self::finalize_proof(prover, io, commitments, proofs, reduced_opening_proof)
    }

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

        // Populate claims and commitments in the verifier accumulator
        self.populate_accumulator(&mut verifier);

        // Verify output MLE at random point τ
        Self::verify_output_claim(pp.model(), io, &mut verifier)?;

        // Verify each operation in reverse topological order
        Self::verify_iop(pp.model(), &mut verifier)?;

        // Verify reduced opening proof
        self.verify_reduced_openings(pp, &mut verifier)
    }
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
