//! Module for constructing malicious provers for ONNX proof experiments. Contains a variant of the main prover that generates proofs for tampered traces, as well as a variant of the sumcheck prover that allows the caller to control how openings are cached (for attack experiments).
use atlas_onnx_tracer::{
    model::{
        trace::{ModelExecutionIO, Trace},
        Model,
    },
    node::ComputationNode,
    ops::Operator,
    tensor::Tensor,
};
use joltworks::{
    field::JoltField,
    poly::{commitment::commitment_scheme::CommitmentScheme, unipoly::UniPoly},
    subprotocols::{
        evaluation_reduction::{EvalReductionProof, ReducedInstance},
        sumcheck::SumcheckInstanceProof,
    },
    transcripts::Transcript,
};

use crate::onnx_proof::{
    ops::{malicious_sub::malicious_sub_prove, OperatorProver},
    AtlasProverPreprocessing, ONNXProof, ProofId, Prover, ProverDebugInfo,
};
use std::collections::BTreeMap;

/// Test helper namespace for constructing malicious prover experiments.
pub struct MaliciousONNXProof;

type ProverOutput<F, T, PCS> = (
    ONNXProof<F, T, PCS>,
    ModelExecutionIO,
    Option<ProverDebugInfo<F, T>>,
);

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
    ) -> ProverOutput<F, T, PCS> {
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
        let mut eval_reduction_proofs = BTreeMap::new();
        Self::iop(
            pp.model().nodes(),
            &mut prover,
            &mut proofs,
            &mut eval_reduction_proofs,
        );

        // Reduction sum-check + PCS::prove
        let reduced_opening_proof =
            ONNXProof::<F, T, PCS>::prove_reduced_openings(&mut prover, &poly_map, &pp.generators);
        ONNXProof::<F, T, PCS>::finalize_proof(
            prover,
            io,
            commitments,
            proofs,
            eval_reduction_proofs,
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
    ) -> ProverOutput<F, T, PCS> {
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

        let mut eval_reduction_proofs = BTreeMap::new();
        Self::iop(
            pp.model().nodes(),
            &mut prover,
            &mut proofs,
            &mut eval_reduction_proofs,
        );
        let reduced_opening_proof =
            ONNXProof::<F, T, PCS>::prove_reduced_openings(&mut prover, &poly_map, &pp.generators);
        ONNXProof::<F, T, PCS>::finalize_proof(
            prover,
            io,
            commitments,
            proofs,
            eval_reduction_proofs,
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

    /// Iterate over computation graph in reverse topological order.
    /// Uses the malicious Sub prover for Sub nodes; honest provers for all others.
    #[tracing::instrument(skip_all, name = "ONNXProof::iop")]
    pub(super) fn iop<F: JoltField, T: Transcript>(
        computation_nodes: &BTreeMap<usize, ComputationNode>,
        prover: &mut Prover<F, T>,
        proofs: &mut BTreeMap<ProofId, SumcheckInstanceProof<F, T>>,
        eval_reduction_proofs: &mut BTreeMap<usize, EvalReductionProof<F>>,
    ) {
        for (_, computation_node) in computation_nodes.iter().rev() {
            eval_reduction_proofs.insert(
                computation_node.idx,
                malicious_eval_reduction_prove(computation_node, prover),
            );
            if matches!(computation_node.operator, Operator::Sub(_)) {
                proofs.extend(malicious_sub_prove(computation_node, prover));
            } else {
                let (_, execution_proofs) = OperatorProver::prove(computation_node, prover);
                proofs.extend(execution_proofs);
            }
        }
    }
}

/// Simulate the evaluation reduction protocol, only we just consider the case where num_use = 1 for simplicity.
pub fn malicious_eval_reduction_prove<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
) -> EvalReductionProof<F> {
    // Recover existing opening for this node
    let existing_opening = prover.accumulator.get_node_openings(node.idx);
    assert!(
        existing_opening.len() == 1,
        "Expected exactly one existing opening for node {}",
        node.idx
    );
    let opening = existing_opening[0].clone();

    let h = UniPoly::from_coeff(vec![opening.1]);

    let reduced = ReducedInstance {
        r: opening.0.clone().into(),
        claim: opening.1,
    };

    prover
        .accumulator
        .reduced_evaluations
        .insert(node.idx, reduced);

    EvalReductionProof { h }
}
