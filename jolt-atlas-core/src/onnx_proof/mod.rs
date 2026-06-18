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
    poly::commitment::commitment_scheme::CommitmentScheme,
    subprotocols::{evaluation_reduction::EvalReductionProof, sumcheck::SumcheckInstanceProof},
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

#[cfg(test)]
mod malicious_prover;
mod preprocessing;
mod prover;
mod types;
mod verifier;
#[cfg(feature = "zk")]
pub mod zk;

#[cfg(test)]
mod e2e_tests;
#[cfg(test)]
mod soundness_tests;

// ── Re-exports ───────────────────────────────────────────────────────────

pub use preprocessing::{
    AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing,
};
pub use prover::Prover;
pub use types::{Claims, ProofId, ProofType, ProverDebugInfo};
pub use verifier::Verifier;

pub use ark_bn254::{Bn254, Fr};
pub use joltworks::{poly::commitment::hyperkzg::HyperKZG, transcripts::Blake2bTranscript};

// ── Public-input transcript binding (soundness, issue #230) ──────────────

/// Bind the model's public input tensors into the Fiat-Shamir transcript
/// *before* any challenge is derived.
///
/// The input tensors are supplied by the prover through [`ModelExecutionIO`]
/// and are not otherwise committed. If they are absent from the transcript,
/// every challenge (hence every sumcheck/opening point) is independent of the
/// inputs. A malicious prover can then run the honest prover on a real input,
/// read the (input-independent) opening points, and swap `io.inputs` for any
/// `Input'` that matches the baked evaluation claims at those points,
/// producing an accepted proof of the false statement `out = Model(Input')`.
/// See issue #230 (the adaptive forge that defeats the per-node evaluation
/// check alone); #244 closed the naive instantiation via the per-node binding.
///
/// Absorbing the inputs here forces the opening points to depend on the input,
/// so any change to `io.inputs` diverges the verifier's transcript from the
/// proof and the IOP rejects before the evaluation check is even reached. The
/// prover and verifier must call this at the same position (first append, right
/// after the transcript is created) so their transcripts stay in lockstep.
///
/// Outputs are already bound (the output claim is checked against `io.outputs`
/// at a transcript-derived point and the IOP ties the output to the trace), so
/// only inputs need binding here.
pub(crate) fn append_inputs_to_transcript<T: Transcript>(
    transcript: &mut T,
    io: &ModelExecutionIO,
) {
    transcript.append_message(b"model_inputs");
    transcript.append_u64(io.inputs.len() as u64);
    transcript.append_u64(io.input_indices.len() as u64);

    // Bind all inputs (and all indices) even if a malformed `ModelExecutionIO` is provided.
    for (i, tensor) in io.inputs.iter().enumerate() {
        let node_idx = io.input_indices.get(i).copied().unwrap_or(usize::MAX);
        transcript.append_u64(node_idx as u64);

        let dims = tensor.dims();
        transcript.append_u64(dims.len() as u64);
        for d in dims {
            transcript.append_u64(*d as u64);
        }

        // Encode values explicitly in LE for cross-platform determinism.
        let mut bytes = Vec::with_capacity(tensor.inner.len() * 4);
        for v in &tensor.inner {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        transcript.append_bytes(&bytes);
    }

    // Ensure any extra indices also perturb the transcript.
    for node_idx in io.input_indices.iter().skip(io.inputs.len()) {
        transcript.append_u64(*node_idx as u64);
        transcript.append_message(b"missing_tensor");
    }
}

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
    /// Polynomial commitments for witness polynomials.
    pub commitments: Vec<PCS::Commitment>,
    /// Evaluation reduction proofs h polynomials for each opening claim.
    pub eval_reduction_proofs: BTreeMap<usize, EvalReductionProof<F>>,
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
        // TODO: Deduplicate shared preprocessing, which is present in both AtlasProverPreprocessing and Prover state
        let mut prover: Prover<F, T> = Prover::new(pp.shared.clone(), trace);
        let mut proofs = BTreeMap::new();

        // Bind public inputs into the transcript before any challenge (issue #230).
        append_inputs_to_transcript(&mut prover.transcript, &io);

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
        let mut eval_reduction_proofs = BTreeMap::new();
        Self::iop(
            pp.model().nodes(),
            &mut prover,
            &mut proofs,
            &mut eval_reduction_proofs,
        );

        // Reduction sum-check + PCS::prove
        let reduced_opening_proof =
            Self::prove_reduced_openings(&mut prover, &poly_map, &pp.generators);
        Self::finalize_proof(
            prover,
            io,
            commitments,
            proofs,
            eval_reduction_proofs,
            reduced_opening_proof,
        )
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

        // Bind public inputs into the transcript before any challenge (issue #230).
        // Must mirror the prover's first append, after `compare_to` installs the
        // expected state history and before any commitment is absorbed.
        append_inputs_to_transcript(&mut verifier.transcript, io);

        // Populate claims and commitments in the verifier accumulator.
        self.populate_accumulator(&mut verifier);

        // Verify output MLE at random point τ
        Self::verify_output_claim(pp.model(), &mut verifier)?;

        // Verify each operation in reverse topological order
        self.verify_iop(pp.model(), &mut verifier)?;

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
