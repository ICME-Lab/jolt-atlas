//! Verification of ONNX neural network proofs.
//!
//! [`verify`] (and [`ONNXProof::verify`]) checks a proof against the model's preprocessing and
//! public IO. The per-op verifiers, the relations they are derived from, and the shared lookup
//! machinery live in the submodules; the prover in `jolt-atlas-core` builds on the same
//! modules, so both sides read one definition of every relation, parameter set and layout.

use atlas_trace::trace::ModelExecutionIO;

use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_pcs::commitment_scheme::CommitmentScheme;
use atlas_transcript::{Transcript, VerifyingTranscript};

// ── Submodules ───────────────────────────────────────────────────────────

pub mod fused_rebase;
pub mod lookup;
pub mod ops;
pub mod preprocessing;
pub mod range_bucket;
pub mod range_check;
pub mod relation;
pub mod schedule;
pub mod verifier;

// ── Re-exports ───────────────────────────────────────────────────────────

pub use crate::ONNXProof;
pub use preprocessing::{AtlasSharedPreprocessing, AtlasVerifierPreprocessing};
pub use verifier::Verifier;

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
pub fn append_inputs_to_transcript<T: Transcript>(transcript: &mut T, io: &ModelExecutionIO) {
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

// ── Verification ─────────────────────────────────────────────────────────

/// Verify a proof for an ONNX neural network computation.
///
/// Reads the proof front to back through the transcript: the commitments, then what each
/// step of the model's schedule reads, then the opening reduction and the joint opening
/// proof, and requires the bytes to end there.
pub fn verify<F: JoltField, T: VerifyingTranscript, PCS: CommitmentScheme<Field = F>>(
    proof: &ONNXProof,
    pp: &AtlasVerifierPreprocessing<F, PCS>,
    io: &ModelExecutionIO,
) -> Result<(), ProofVerifyError> {
    verify_with::<F, T, PCS>(proof, pp, io, |_| {})
}

/// [`verify`], with `setup` run on the fresh verifier state before anything is absorbed (the
/// prover's test-only lockstep comparison installs its expected transcript history here).
#[tracing::instrument(skip_all, name = "ONNXProof::verify")]
pub fn verify_with<F: JoltField, T: VerifyingTranscript, PCS: CommitmentScheme<Field = F>>(
    proof: &ONNXProof,
    pp: &AtlasVerifierPreprocessing<F, PCS>,
    io: &ModelExecutionIO,
    setup: impl FnOnce(&mut Verifier<'_, F, T>),
) -> Result<(), ProofVerifyError> {
    let mut verifier: Verifier<F, T> = Verifier::new(io, &proof.0);
    setup(&mut verifier);

    // Every node's operands are earlier nodes, the layout ops have the dims their ties hold
    // for, and the public IO has the model's shape.
    crate::node_view::check_model_shape(pp.model())?;
    verifier::check_io_shape(pp.model(), io)?;

    // Bind public inputs into the transcript before any challenge (issue #230).
    // Must mirror the prover's first append, after `setup` installs any expected
    // state history and before any commitment is read.
    append_inputs_to_transcript(&mut verifier.transcript, io);

    // Read the commitments: the schedule says how many.
    let commitments = verifier::read_commitments(pp, &mut verifier)?;

    // Verify output MLE at random point τ
    verifier::verify_output_claim(pp.model(), &mut verifier)?;

    // Run the model's schedule: each node in reverse topological order, then the model-wide
    // sumchecks.
    verifier.run(&pp.schedule)?;

    // Reduce the committed openings to one point and verify the joint opening proof.
    verifier::verify_reduced_openings(&commitments, pp, &mut verifier)?;

    // The proof was read to its end: nothing follows the last value the verifier needs.
    verifier.transcript.finish()
}

impl ONNXProof {
    /// Verify this proof against the model's preprocessing and public IO; see [`verify`].
    pub fn verify<F: JoltField, T: VerifyingTranscript, PCS: CommitmentScheme<Field = F>>(
        &self,
        pp: &AtlasVerifierPreprocessing<F, PCS>,
        io: &ModelExecutionIO,
    ) -> Result<(), ProofVerifyError> {
        verify::<F, T, PCS>(self, pp, io)
    }
}
