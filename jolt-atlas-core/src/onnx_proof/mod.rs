//! Proofs for ONNX neural network computations.
//!
//! This module implements the core proving and verification logic for ONNX neural networks.
//! It provides:
//! - [`ONNXProof`]: The main proof structure containing all sumcheck proofs and commitments,
//!   owned by `jolt-atlas-verifier` together with its wire format and re-exported here
//! - [`prove`]: Generate a proof for an ONNX model execution
//! - [`verify`]: Verify a proof against expected IO
//! - Preprocessing structs for model setup and commitment scheme initialization
//!
//! Internal implementation details (helper functions, state structs) live in
//! [`prover`], [`verifier`], and [`types`] submodules.

use atlas_trace::{trace::ModelExecutionIO, Tensor};

use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_pcs::commitment_scheme::CommitmentScheme;
use atlas_transcript::{ProvingTranscript, VerifyingTranscript};
use std::collections::BTreeMap;

// ── Submodules ───────────────────────────────────────────────────────────

pub mod clamp_lookups;
pub mod clamp_split;
pub mod deferred_lookups;
pub mod fused_rebase;
pub mod global_clamp;
pub mod neural_teleport;
pub mod op_lookups;
pub mod ops;
pub mod range_checking;
pub mod relation;
pub mod witness;

mod preprocessing;
mod prover;
mod types;

#[cfg(test)]
pub(crate) mod testing;

// ── Re-exports ───────────────────────────────────────────────────────────

pub use jolt_atlas_verifier::onnx_proof::{
    append_inputs_to_transcript, trace_view, verifier, verify_with, Verifier,
};
pub use jolt_atlas_verifier::{
    proof_serialization, Claims, ONNXProof, ProofId, ProofType, ReducedOpeningProof,
};
pub use preprocessing::{
    AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing,
};
pub use prover::Prover;
pub use types::ProverDebugInfo;

pub use atlas_algebra::field::fp128::Fp128;
pub use atlas_pcs::akita::AkitaScheme;
pub use atlas_transcript::{Blake2bProver, Blake2bVerifier};

// ── Core proof structures ────────────────────────────────────────────────

// ── Public API: prove & verify ───────────────────────────────────────────

/// Generate a proof for an ONNX neural network computation.
///
/// Executes the model with the given inputs, generates a trace, and produces
/// sumcheck proofs for each operation. Returns the proof, execution IO, and
/// optional debug information.
#[tracing::instrument(skip_all, name = "ONNXProof::prove")]
#[allow(clippy::type_complexity)]
pub fn prove<F: JoltField, T: ProvingTranscript, PCS: CommitmentScheme<Field = F>>(
    pp: &AtlasProverPreprocessing<F, PCS>,
    inputs: &[Tensor<i32>],
) -> (
    ONNXProof<F, PCS>,
    ModelExecutionIO,
    Option<ProverDebugInfo<F>>,
) {
    let trace = pp.model().trace(inputs).expect("model execution failed");
    prove_from_trace::<F, T, PCS>(pp, trace)
}

/// Prove from an already-executed trace. The public IO is derived from
/// the trace, so this is the prover's *claimed* statement; the soundness
/// sweeps tamper with the trace before calling it.
#[allow(clippy::type_complexity)]
pub(crate) fn prove_from_trace<
    F: JoltField,
    T: ProvingTranscript,
    PCS: CommitmentScheme<Field = F>,
>(
    pp: &AtlasProverPreprocessing<F, PCS>,
    trace: atlas_trace::Trace,
) -> (
    ONNXProof<F, PCS>,
    ModelExecutionIO,
    Option<ProverDebugInfo<F>>,
) {
    let io = trace.io(pp.model());

    // Initialize prover state
    // TODO: Deduplicate shared preprocessing, which is present in both AtlasProverPreprocessing and Prover state
    let mut prover: Prover<F, T> = Prover::new(pp.shared.clone(), trace);
    let mut proofs = BTreeMap::new();

    // Bind public inputs into the transcript before any challenge (issue #230).
    append_inputs_to_transcript(&mut prover.transcript, &io);

    // Commit to witness polynomials and append commitments to transcript
    let (poly_map, commitments, hints) = prover::commit_witness_polynomials::<F, T, PCS>(
        pp.model(),
        &prover.trace,
        &pp.generators,
        &mut prover.transcript,
    );

    // Evaluate output MLE at random point τ
    prover::output_claim(&mut prover);

    // IOP portion
    let mut eval_reduction_proofs = BTreeMap::new();
    prover::iop(
        pp.model(),
        &mut prover,
        &mut proofs,
        &mut eval_reduction_proofs,
    );

    // Reduction sum-check + PCS::prove
    let reduced_opening_proof =
        prover::prove_reduced_openings(&mut prover, &poly_map, hints, &pp.generators);
    prover::finalize_proof(
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
/// [`jolt_atlas_verifier::onnx_proof::verify`], plus the test-only lockstep comparison: with
/// the prover's `debug_info`, the verifier checks its transcript and opening accumulator against
/// the prover's step by step.
pub fn verify<F: JoltField, T: VerifyingTranscript, PCS: CommitmentScheme<Field = F>>(
    proof: &ONNXProof<F, PCS>,
    pp: &AtlasVerifierPreprocessing<F, PCS>,
    io: &ModelExecutionIO,
    _debug_info: Option<ProverDebugInfo<F>>,
) -> Result<(), ProofVerifyError> {
    verify_with::<F, T, PCS>(proof, pp, io, |_verifier| {
        #[cfg(test)]
        if let Some(debug_info) = _debug_info {
            _verifier
                .transcript
                .compare_to(debug_info.transcript_history);
            _verifier
                .accumulator
                .compare_to(debug_info.opening_accumulator);
        }
    })
}
