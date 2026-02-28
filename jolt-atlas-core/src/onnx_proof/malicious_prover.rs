use atlas_onnx_tracer::{
    model::{trace::{ModelExecutionIO, Trace}, Model},
    node::ComputationNode,
    ops::Operator,
    tensor::Tensor,
};
use joltworks::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme, opening_proof::ProverOpeningAccumulator,
        unipoly::CompressedUniPoly,
    },
    subprotocols::{sumcheck::SumcheckInstanceProof, sumcheck_prover::SumcheckInstanceProver},
    transcripts::{AppendToTranscript, Transcript},
};

pub use ark_bn254::{Bn254, Fr};
pub use joltworks::{poly::commitment::hyperkzg::HyperKZG, transcripts::Blake2bTranscript};

use crate::onnx_proof::{
    ops::OperatorProver, AtlasProverPreprocessing, ONNXProof, ProofId, Prover, ProverDebugInfo,
};
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
    ) -> (
        ONNXProof<F, T, PCS>,
        ModelExecutionIO,
        Option<ProverDebugInfo<F, T>>,
    ) {
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
        Self::iop(pp.model().nodes(), &mut prover, &mut proofs);

        // Reduction sum-check + PCS::prove
        let reduced_opening_proof =
            ONNXProof::<F, T, PCS>::prove_reduced_openings(&mut prover, &poly_map, &pp.generators);
        ONNXProof::<F, T, PCS>::finalize_proof(
            prover,
            io,
            commitments,
            proofs,
            reduced_opening_proof,
        )
    }

    /// Same as `prove`, but mutates the trace so that the first `Sub` node output
    /// is forced to zero before proof generation.
    pub fn prove_with_sub_trace_tamper_zero<
        F: JoltField,
        T: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        pp: &AtlasProverPreprocessing<F, PCS>,
        inputs: &[Tensor<i32>],
    ) -> (
        ONNXProof<F, T, PCS>,
        ModelExecutionIO,
        Option<ProverDebugInfo<F, T>>,
    ) {
        let mut trace = pp.model().trace(inputs);
        Self::tamper_first_sub_output_to_zero(&mut trace, pp.model());
        let io = Trace::io(&trace, pp.model());

        let mut prover: Prover<F, T> = Prover::new(pp.shared.clone(), trace);
        let mut proofs = BTreeMap::new();

        let (poly_map, commitments) = ONNXProof::<F, T, PCS>::commit_witness_polynomials(
            pp.model(),
            &prover.trace,
            &pp.generators,
            &mut prover.transcript,
        );
        ONNXProof::<F, T, PCS>::output_claim(&mut prover);
        Self::iop(pp.model().nodes(), &mut prover, &mut proofs);
        let reduced_opening_proof =
            ONNXProof::<F, T, PCS>::prove_reduced_openings(&mut prover, &poly_map, &pp.generators);
        ONNXProof::<F, T, PCS>::finalize_proof(
            prover,
            io,
            commitments,
            proofs,
            reduced_opening_proof,
        )
    }

    fn tamper_first_sub_output_to_zero(trace: &mut Trace, model: &Model) {
        let sub_node = model
            .graph
            .nodes
            .values()
            .find(|node| matches!(node.operator, Operator::Sub(_)))
            .expect("model should contain a Sub node");
        let output = trace
            .node_outputs
            .get_mut(&sub_node.idx)
            .expect("trace should contain output for Sub node");
        output.inner.fill(0);
    }

    /// Iterate over computation graph in reverse topological order
    /// Prove each operation using sum-check and virtual polynomials
    #[tracing::instrument(skip_all, name = "ONNXProof::iop")]
    pub(super) fn iop<F: JoltField, T: Transcript>(
        computation_nodes: &BTreeMap<usize, ComputationNode>,
        prover: &mut Prover<F, T>,
        proofs: &mut BTreeMap<ProofId, SumcheckInstanceProof<F, T>>,
    ) {
        for (_, computation_node) in computation_nodes.iter().rev() {
            proofs.extend(OperatorProver::prove(computation_node, prover));
        }
    }
}

/// Variant of `Sumcheck::prove` that also returns the final claim and leaves
/// opening caching to the caller (for adversarial experiments).
pub fn malicious_sumcheck_prove<F: JoltField, ProofTranscript: Transcript>(
    sumcheck_instance: &mut dyn SumcheckInstanceProver<F, ProofTranscript>,
    opening_accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut ProofTranscript,
) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F::Challenge>, F) {
    let num_rounds = sumcheck_instance.num_rounds();

    // Append input claims to transcript
    let input_claim = sumcheck_instance.input_claim(opening_accumulator);
    transcript.append_scalar(&input_claim);
    let mut previous_claim = input_claim;
    let mut r_sumcheck: Vec<F::Challenge> = Vec::with_capacity(num_rounds);
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
    for round in 0..num_rounds {
        let univariate_poly = sumcheck_instance.compute_message(round, previous_claim);
        // append the prover's message to the transcript
        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        let r_j = transcript.challenge_scalar_optimized::<F>();
        r_sumcheck.push(r_j);

        // Cache claim for this round
        previous_claim = univariate_poly.evaluate(&r_j);
        sumcheck_instance.ingest_challenge(r_j, round);
        compressed_polys.push(compressed_poly);
    }

    let final_claim = previous_claim;

    // Allow the sumcheck instance to perform any end-of-protocol work (e.g. flushing
    // delayed bindings) after the final challenge has been ingested and before we cache
    // openings.
    sumcheck_instance.finalize();

    // Deliberately do not call `cache_openings` here. The caller controls how
    // openings and operand claims are cached for attack experiments.
    (SumcheckInstanceProof::new(compressed_polys), r_sumcheck, final_claim)
}
