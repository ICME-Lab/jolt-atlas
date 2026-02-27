use atlas_onnx_tracer::{
    model::trace::{ModelExecutionIO, Trace},
    tensor::Tensor,
};
use joltworks::{
    field::JoltField,
    poly::commitment::commitment_scheme::CommitmentScheme,
    transcripts::Transcript,
};

// ── Re-exports ───────────────────────────────────────────────────────────

// pub use preprocessing::{
//     AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing,
// };
// pub use prover::Prover;
// pub use types::{Claims, ProofId, ProofType, ProverDebugInfo};
// pub use verifier::Verifier;

pub use ark_bn254::{Bn254, Fr};
pub use joltworks::{poly::commitment::hyperkzg::HyperKZG, transcripts::Blake2bTranscript};

use crate::onnx_proof::{AtlasProverPreprocessing, ONNXProof, Prover, ProverDebugInfo};
use std::collections::BTreeMap;

/// Test helper namespace for constructing malicious prover experiments.
pub struct MaliciousONNXProof;

// ── Public API: prove & verify ───────────────────────────────────────────

impl MaliciousONNXProof {
    /// Generate a proof for an ONNX neural network computation.
    ///
    /// Executes the model with the given inputs, generates a trace, and produces
    /// sumcheck proofs for each operation. Returns the proof, execution IO, and
    /// optional debug information.
    #[tracing::instrument(skip_all, name = "ONNXProof::prove")]
    pub fn prove<F: JoltField, T: Transcript, PCS: CommitmentScheme<Field = F>>(
        pp: &AtlasProverPreprocessing<F, PCS>,
        inputs: &[Tensor<i32>],
    ) -> (ONNXProof<F, T, PCS>, ModelExecutionIO, Option<ProverDebugInfo<F, T>>) {
        // Generate trace and io
        let trace = pp.model().trace(inputs);
        let io = Trace::io(&trace, pp.model());

        // Initialize prover state
        let mut prover: Prover<F, T> = Prover::new(pp.shared.clone(), trace);
        let mut proofs = BTreeMap::new();

        // Commit to witness polynomials and append commitments to transcript
        let (poly_map, commitments) = ONNXProof::<F, T, PCS>::commit_witness_polynomials(
            pp.model(),
            &prover.trace,
            &pp.generators,
            &mut prover.transcript,
        );

        // Evaluate output MLE at random point τ
        ONNXProof::<F, T, PCS>::output_claim(&mut prover);

        // IOP portion
        ONNXProof::<F, T, PCS>::iop(pp.model().nodes(), &mut prover, &mut proofs);

        // Reduction sum-check + PCS::prove
        let reduced_opening_proof =
            ONNXProof::<F, T, PCS>::prove_reduced_openings(&mut prover, &poly_map, &pp.generators);
        ONNXProof::<F, T, PCS>::finalize_proof(prover, io, commitments, proofs, reduced_opening_proof)
    }
}
