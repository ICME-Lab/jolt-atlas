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

use joltworks::{
    field::JoltField,
    poly::commitment::commitment_scheme::CommitmentScheme,
    transcripts::{ProvingTranscript, Transcript, VerifyingTranscript},
    utils::errors::ProofVerifyError,
};
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
pub mod trace_view;
pub mod witness;

mod preprocessing;
mod prover;
mod types;
mod verifier;

#[cfg(test)]
pub(crate) mod testing;

// ── Re-exports ───────────────────────────────────────────────────────────

pub use jolt_atlas_verifier::{
    proof_serialization, Claims, ONNXProof, ProofId, ProofType, ReducedOpeningProof,
};
pub use preprocessing::{
    AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing,
};
pub use prover::Prover;
pub use types::ProverDebugInfo;
pub use verifier::Verifier;

pub use joltworks::field::fp128::Fp128;
pub use joltworks::{
    poly::commitment::akita::AkitaScheme,
    transcripts::{Blake2bProver, Blake2bVerifier},
};

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
        let mut bytes = Vec::with_capacity(tensor.data().len() * 4);
        for v in tensor.data() {
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
/// Checks all sumcheck proofs, validates opening claims, and verifies that the
/// computation produces the expected output.
#[tracing::instrument(skip_all, name = "ONNXProof::verify")]
pub fn verify<F: JoltField, T: VerifyingTranscript, PCS: CommitmentScheme<Field = F>>(
    proof: &ONNXProof<F, PCS>,
    pp: &AtlasVerifierPreprocessing<F, PCS>,
    io: &ModelExecutionIO,
    _debug_info: Option<ProverDebugInfo<F>>,
) -> Result<(), ProofVerifyError> {
    // Initialize verifier state
    let mut verifier: Verifier<F, T> = Verifier::new(&pp.shared, proof.proofs(), io, proof.narg());
    #[cfg(test)]
    {
        if let Some(debug_info) = _debug_info {
            verifier
                .transcript
                .compare_to(debug_info.transcript_history);
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
    verifier::populate_accumulator(proof, &mut verifier);

    // Verify output MLE at random point τ
    verifier::verify_output_claim(pp.model(), &mut verifier)?;

    // Verify each operation in reverse topological order
    verifier::verify_iop(proof, pp.model(), &mut verifier)?;

    // Verify reduced opening proof
    verifier::verify_reduced_openings(proof, pp, &mut verifier)?;

    // Every proof-of-work nonce in the NARG string must have been read.
    verifier.transcript.finish()
}
