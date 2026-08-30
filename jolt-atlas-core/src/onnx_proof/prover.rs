//! Prover state and internal proving helpers for [`ONNXProof`].
//!
//! The public entry point is [`ONNXProof::prove`] (defined in the parent module).
//! This file houses the [`Prover`] struct and the private helper methods that
//! `prove` delegates to.

use super::{
    types::{Claims, ProofId, ProverDebugInfo},
    AtlasSharedPreprocessing, ONNXProof, ReducedOpeningProof,
};
use crate::{onnx_proof::ops::OperatorProver, utils::opening_access::AccOpeningAccessor};
use atlas_onnx_tracer::{
    model::{
        trace::{LayerData, ModelExecutionIO, Trace},
        Model,
    },
    node::ComputationNode,
};
use common::{parallel::ParallelFlagGuard, CommittedPoly, VirtualPoly};
use joltworks::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{OpeningId, ProverOpeningAccumulator, SumcheckId},
    },
    subprotocols::{evaluation_reduction::EvalReductionProof, sumcheck::SumcheckInstanceProof},
    transcripts::Transcript,
    utils::math::Math,
};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
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
    /// Lookups registered during the node loop, proven afterwards in one batch
    /// (see [`deferred_lookups`](crate::onnx_proof::deferred_lookups)).
    pub deferred: Vec<crate::onnx_proof::deferred_lookups::ProverLookupJob>,
    /// Sumcheck instances built during the node loop and proven afterwards,
    /// one batched sumcheck per [`DeferredBatch`](crate::onnx_proof::deferred_lookups::DeferredBatch).
    pub deferred_batches: BTreeMap<
        crate::onnx_proof::deferred_lookups::DeferredBatch,
        Vec<Box<dyn joltworks::subprotocols::sumcheck_prover::SumcheckInstanceProver<F, T>>>,
    >,
    /// One-hot checks to build and prove right after their parent batch.
    pub deferred_onehots: Vec<(
        crate::onnx_proof::deferred_lookups::DeferredOneHot,
        Vec<usize>,
    )>,
}

impl<F: JoltField, T: Transcript> Prover<F, T> {
    /// Create a new prover with the given preprocessing and trace
    pub fn new(preprocessing: AtlasSharedPreprocessing, trace: Trace) -> Self {
        Self {
            trace,
            preprocessing,
            accumulator: ProverOpeningAccumulator::new(),
            transcript: T::new(b"ONNXProof"),
            deferred: Vec::new(),
            deferred_batches: BTreeMap::new(),
            deferred_onehots: Vec::new(),
        }
    }

    /// Queue a node-loop-built sumcheck instance for the deferred batch `kind`.
    pub fn defer(
        &mut self,
        kind: crate::onnx_proof::deferred_lookups::DeferredBatch,
        instance: Box<dyn joltworks::subprotocols::sumcheck_prover::SumcheckInstanceProver<F, T>>,
    ) {
        self.deferred_batches
            .entry(kind)
            .or_default()
            .push(instance);
    }
}

// ---------------------------------------------------------------------------
// Internal prover helpers on ONNXProof
// ---------------------------------------------------------------------------

impl<F: JoltField, T: Transcript, PCS: CommitmentScheme<Field = F>> ONNXProof<F, T, PCS> {
    /// Build the witness polynomial map, commit to each polynomial, and append
    /// all commitments to the transcript.
    #[tracing::instrument(skip_all, name = "ONNXProof::commit_witness_polynomials")]
    #[allow(clippy::type_complexity)]
    pub(super) fn commit_witness_polynomials(
        model: &Model,
        trace: &Trace,
        generators: &PCS::ProverSetup,
        transcript: &mut T,
    ) -> (
        BTreeMap<CommittedPoly, MultilinearPolynomial<F>>,
        Vec<PCS::Commitment>,
        Vec<PCS::OpeningProofHint>,
    ) {
        let poly_map = Self::polynomial_map(model, trace);
        let (commitments, hints) = Self::commit_to_polynomials_with_hints(&poly_map, generators);
        for commitment in &commitments {
            transcript.append_serializable(commitment);
        }
        (poly_map, commitments, hints)
    }

    pub(crate) fn output_claim(prover: &mut Prover<F, T>) {
        // Construct output polynomial
        let output_index = prover.preprocessing.model().outputs()[0];
        let output_computation_node = &prover.preprocessing.model()[output_index];
        let LayerData {
            operands: _,
            output,
        } = Trace::layer_data(&prover.trace, output_computation_node);

        // Pad output tensor to power of 2 with 0 for MLE constructions
        let output = output.padded_next_power_of_two();

        // Sample challenge from verifier
        let r_node_output = prover
            .transcript
            .challenge_vector_optimized::<F>(output.len().log_2());

        // Evaluate output polynomial at r_node_output
        let output_claim = MultilinearPolynomial::from(output.clone()).evaluate(&r_node_output);

        let output_opening_id = OpeningId::new(
            VirtualPoly::NodeOutput(output_computation_node.idx),
            // We add a NodeOutput claim for the model output. We use idx+1 as the execution,
            // since some nodes add a claim during their own execution that would otherwise collide with the output
            SumcheckId::NodeExecution(output_computation_node.idx + 1),
        );

        let mut provider =
            AccOpeningAccessor::new(&mut prover.accumulator, output_computation_node)
                .into_provider(&mut prover.transcript, r_node_output.into());

        // append_virtual handles both transcript append and insertion into openings
        provider.append_custom(output_opening_id, output_claim);
    }

    /// Iterate over computation graph in reverse topological order
    /// Prove each operation using sum-check and virtual polynomials
    #[tracing::instrument(skip_all, name = "ONNXProof::iop")]
    pub(crate) fn iop(
        computation_nodes: &BTreeMap<usize, ComputationNode>,
        prover: &mut Prover<F, T>,
        proofs: &mut BTreeMap<ProofId, SumcheckInstanceProof<F, T>>,
        eval_reduction_proofs: &mut BTreeMap<usize, EvalReductionProof<F>>,
    ) {
        for (_, node) in computation_nodes.iter().rev() {
            let (eval_reduction_proof, execution_proofs) = OperatorProver::prove(node, prover);
            eval_reduction_proofs.insert(node.idx, eval_reduction_proof);
            proofs.extend(execution_proofs);
        }
        crate::onnx_proof::deferred_lookups::prove_all(prover, proofs);
    }

    #[tracing::instrument(skip_all, name = "ONNXProof::prove_reduced_openings")]
    pub(super) fn prove_reduced_openings(
        prover: &mut Prover<F, T>,
        poly_map: &BTreeMap<CommittedPoly, MultilinearPolynomial<F>>,
        hints: Vec<PCS::OpeningProofHint>,
        generators: &PCS::ProverSetup,
    ) -> Option<ReducedOpeningProof<F, T, PCS>> {
        if poly_map.is_empty() {
            return None;
        }
        // Committed polynomials nobody opened (e.g. the clamp chunks of an
        // output node whose public output is unsaturated) stay out of the
        // joint opening; the verifier selects the same subset from its openings.
        let opened = prover.accumulator.opened_polynomials();
        if opened.is_empty() {
            // Nothing was opened (every committed polynomial is unused, e.g. a
            // model whose only clamped node is an exact output): no joint opening.
            return None;
        }
        let (poly_map, hints): (BTreeMap<CommittedPoly, MultilinearPolynomial<F>>, Vec<_>) =
            if opened.len() == poly_map.len() {
                (poly_map.clone(), hints)
            } else {
                poly_map
                    .iter()
                    .zip(hints)
                    .filter(|((p, _), _)| opened.contains(p))
                    .map(|((p, poly), h)| ((*p, poly.clone()), h))
                    .unzip()
            };
        let poly_map = &poly_map;
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
        // Joint opening of the RLC `Σ γ_i · poly_i` (materialized or not, per PCS)
        let joint_opening_proof = PCS::prove_rlc(
            generators,
            poly_map,
            &state.gamma_powers,
            hints,
            &state.r_sumcheck,
            &mut prover.transcript,
        );
        Some(ReducedOpeningProof {
            sumcheck_proof: accumulator_sumcheck_proof,
            sumcheck_claims,
            joint_opening_proof,
        })
    }

    pub(super) fn finalize_proof(
        mut prover: Prover<F, T>,
        io: ModelExecutionIO,
        commitments: Vec<PCS::Commitment>,
        proofs: BTreeMap<ProofId, SumcheckInstanceProof<F, T>>,
        eval_reduction_proofs: BTreeMap<usize, EvalReductionProof<F>>,
        reduced_opening_proof: Option<ReducedOpeningProof<F, T, PCS>>,
    ) -> (Self, ModelExecutionIO, Option<ProverDebugInfo<F, T>>) {
        #[cfg(test)]
        let debug_info = Some(ProverDebugInfo {
            transcript: prover.transcript.clone(),
            opening_accumulator: prover.accumulator.clone(),
        });
        #[cfg(not(test))]
        let debug_info = None;
        let opening_claims = prover.accumulator.take();
        (
            Self {
                proofs,
                opening_claims: Claims(opening_claims),
                commitments,
                eval_reduction_proofs,
                reduced_opening_proof,
            },
            io,
            debug_info,
        )
    }

    #[tracing::instrument(skip_all, name = "ONNXProof::polynomial_map")]
    pub(super) fn polynomial_map(
        model: &Model,
        trace: &Trace,
    ) -> BTreeMap<CommittedPoly, MultilinearPolynomial<F>> {
        // Walk nodes *sequentially* with inner (per-tensor) parallelism ENABLED,
        // mirroring `Model::trace`. Witness generation re-executes each fused
        // node's accumulation (e.g. `einsum_intermediate`) to recover its
        // pre-clamp value, and that cost is dominated by a few large einsum nodes
        // (attention, MLP, logits). The previous per-node `par_iter` disabled
        // inner parallelism to avoid nested-rayon fragmentation, which pinned
        // each of those whales to a single thread — the reason this phase ran
        // several times slower than the identical einsums do inside `trace`.
        // Giving each node the whole thread pool for its accumulation and one-hot
        // build parallelizes the whales; the many small nodes are cheap enough
        // that losing node-level parallelism for them is negligible.
        // `generate_node_witnesses` still computes each node's accumulation once
        // and slices every committed-poly chunk from it.
        let mut polys: BTreeMap<CommittedPoly, MultilinearPolynomial<F>> = model
            .graph
            .nodes
            .values()
            .flat_map(|node| {
                crate::onnx_proof::witness::generate_node_witnesses::<F, T>(node, model, trace)
            })
            .collect();
        // Packed clamp buckets (model-level, not per node).
        polys.extend(crate::onnx_proof::global_clamp::bucket_witnesses::<F>(
            model, trace,
        ));
        polys
    }

    #[tracing::instrument(skip_all)]
    #[cfg_attr(not(feature = "zk"), allow(dead_code))]
    pub(super) fn commit_to_polynomials(
        poly_map: &BTreeMap<CommittedPoly, MultilinearPolynomial<F>>,
        pcs: &PCS::ProverSetup,
    ) -> Vec<PCS::Commitment> {
        Self::commit_to_polynomials_with_hints(poly_map, pcs).0
    }

    /// Commit to every polynomial (in `poly_map` order), keeping the per-polynomial
    /// opening hints so the joint opening can be assembled from them.
    #[tracing::instrument(skip_all)]
    pub(super) fn commit_to_polynomials_with_hints(
        poly_map: &BTreeMap<CommittedPoly, MultilinearPolynomial<F>>,
        pcs: &PCS::ProverSetup,
    ) -> (Vec<PCS::Commitment>, Vec<PCS::OpeningProofHint>) {
        // Rayon jobs were overly fragmented, and the resulting context switching degraded performance,
        // so the parallelism granularity is now limited to the polynomial level.
        let _guard = ParallelFlagGuard::disabled();
        poly_map
            .values()
            .collect::<Vec<_>>()
            .par_iter()
            .map(|poly| PCS::commit(poly, pcs))
            .unzip()
    }
}
