//! Prover state and internal proving helpers for [`ONNXProof`].
//!
//! The public entry point is [`ONNXProof::prove`] (defined in the parent module).
//! This file houses the [`Prover`] struct and the private helper methods that
//! `prove` delegates to.

use super::{
    types::{Claims, ProofId, ProverDebugInfo},
    AtlasSharedPreprocessing, ONNXProof, ReducedOpeningProof,
};
use crate::onnx_proof::{
    ops::{NodeCommittedPolynomials, OperatorProver},
    witness::WitnessGenerator,
};
use atlas_onnx_tracer::{
    model::{
        trace::{LayerData, ModelExecutionIO, Trace},
        Model,
    },
    node::ComputationNode,
};
use common::{CommittedPolynomial, VirtualPolynomial};
use joltworks::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{ProverOpeningAccumulator, SumcheckId},
        rlc_polynomial::build_materialized_rlc,
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    utils::math::Math,
};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Prover state
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Internal prover helpers on ONNXProof
// ---------------------------------------------------------------------------

impl<F: JoltField, T: Transcript, PCS: CommitmentScheme<Field = F>> ONNXProof<F, T, PCS> {
    /// Build the witness polynomial map, commit to each polynomial, and append
    /// all commitments to the transcript.
    #[tracing::instrument(skip_all, name = "ONNXProof::commit_witness_polynomials")]
    pub(super) fn commit_witness_polynomials(
        model: &Model,
        trace: &Trace,
        generators: &PCS::ProverSetup,
        transcript: &mut T,
    ) -> (
        BTreeMap<CommittedPolynomial, MultilinearPolynomial<F>>,
        Vec<PCS::Commitment>,
    ) {
        let poly_map = Self::polynomial_map(model, trace);
        let commitments = Self::commit_to_polynomials(&poly_map, generators);
        for commitment in &commitments {
            transcript.append_serializable(commitment);
        }
        (poly_map, commitments)
    }

    pub(super) fn output_claim(prover: &mut Prover<F, T>) {
        // Construct output polynomial
        let output_index = prover.preprocessing.model().outputs()[0];
        let output_computation_node = &prover.preprocessing.model()[output_index];
        let LayerData {
            operands: _,
            output,
        } = Trace::layer_data(&prover.trace, output_computation_node);

        // Sample challenge from verifier
        let r_node_output = prover
            .transcript
            .challenge_vector_optimized::<F>(output.len().log_2());

        // Evaluate output polynomial at r_node_output
        let output_claim = MultilinearPolynomial::from(output.clone()).evaluate(&r_node_output);

        // send claim to verifier
        prover.transcript.append_scalar(&output_claim);
        prover.accumulator.append_virtual(
            &mut prover.transcript,
            VirtualPolynomial::NodeOutput(output_computation_node.idx),
            SumcheckId::Execution,
            r_node_output.clone().into(),
            output_claim,
        );
    }

    /// Iterate over computation graph in reverse topological order
    /// Prove each operation using sum-check and virtual polynomials
    #[tracing::instrument(skip_all, name = "ONNXProof::iop")]
    pub(super) fn iop(
        computation_nodes: &BTreeMap<usize, ComputationNode>,
        prover: &mut Prover<F, T>,
        proofs: &mut BTreeMap<ProofId, SumcheckInstanceProof<F, T>>,
    ) {
        for (_, computation_node) in computation_nodes.iter().rev() {
            let new_proofs = OperatorProver::prove(computation_node, prover);
            for (proof_id, proof) in new_proofs {
                proofs.insert(proof_id, proof);
            }
        }
    }

    #[tracing::instrument(skip_all, name = "ONNXProof::prove_reduced_openings")]
    pub(super) fn prove_reduced_openings(
        prover: &mut Prover<F, T>,
        poly_map: &BTreeMap<CommittedPolynomial, MultilinearPolynomial<F>>,
        generators: &PCS::ProverSetup,
    ) -> Option<ReducedOpeningProof<F, T, PCS>> {
        if poly_map.is_empty() {
            None
        } else {
            prover.accumulator.prepare_for_sumcheck(poly_map);

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
            let rlc = build_materialized_rlc(&state.gamma_powers, poly_map);
            // Create joint opening proof
            let joint_opening_proof = PCS::prove(
                generators,
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
        }
    }

    pub(super) fn finalize_proof(
        mut prover: Prover<F, T>,
        io: ModelExecutionIO,
        commitments: Vec<PCS::Commitment>,
        proofs: BTreeMap<ProofId, SumcheckInstanceProof<F, T>>,
        reduced_opening_proof: Option<ReducedOpeningProof<F, T, PCS>>,
    ) -> (Self, ModelExecutionIO, Option<ProverDebugInfo<F, T>>) {
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
    pub(super) fn polynomial_map(
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
