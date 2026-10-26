//! Prover state and internal proving helpers for [`ONNXProof`].
//!
//! The public entry point is [`prove`](super::prove) (defined in the parent module).
//! This file houses the [`Prover`] struct and the private helper functions that
//! `prove` delegates to.

use super::{
    types::ProverDebugInfo, AtlasSharedPreprocessing, Claims, ONNXProof, ProofId,
    ReducedOpeningProof,
};
use crate::node_view::ModelView;
use crate::onnx_proof::trace_view::TraceView;
use crate::{onnx_proof::ops::OperatorProver, utils::opening_access::AccOpeningAccessor};
use atlas_algebra::{
    field::JoltField,
    poly::multilinear_polynomial::{EvaluateMle, MultilinearPolynomial},
    utils::math::Math,
};
use atlas_pcs::commitment_scheme::CommitmentScheme;
use atlas_sumcheck::{
    evaluation_reduction::EvalReductionProof,
    opening_proof::{OpeningId, ProverOpeningAccumulator, SumcheckId},
    sumcheck::SumcheckInstanceProof,
};
use atlas_trace::{trace::LayerData, trace::ModelExecutionIO, Model, Trace};
use atlas_transcript::{AppendToTranscript, ProvingTranscript, Transcript};
use common::{CommittedPoly, VirtualPoly};
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
        Vec<Box<dyn atlas_sumcheck::sumcheck_prover::SumcheckInstanceProver<F, T>>>,
    >,
    /// One-hot checks to build and prove right after their parent batch.
    pub deferred_onehots: Vec<(
        crate::onnx_proof::deferred_lookups::DeferredOneHot,
        Vec<usize>,
    )>,
}

impl<F: JoltField, T: Transcript> Prover<F, T> {
    /// Create a new prover with the given preprocessing and trace
    pub fn new(preprocessing: AtlasSharedPreprocessing, trace: Trace) -> Self
    where
        T: ProvingTranscript,
    {
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
        instance: Box<dyn atlas_sumcheck::sumcheck_prover::SumcheckInstanceProver<F, T>>,
    ) {
        self.deferred_batches
            .entry(kind)
            .or_default()
            .push(instance);
    }
}

// ---------------------------------------------------------------------------
// Internal proving helpers
// ---------------------------------------------------------------------------

/// Commitments to a proof's witness polynomials: the scheme's batch commitment plus one
/// commitment per polynomial, in `poly_map` order.
pub(super) struct WitnessCommitments<PCS: CommitmentScheme> {
    pub(super) batch: PCS::BatchCommitment,
    pub(super) polys: Vec<PCS::Commitment>,
}

/// Build the witness polynomial map, commit to every polynomial, and append the commitments to
/// the transcript (the batch commitment first).
#[tracing::instrument(skip_all, name = "ONNXProof::commit_witness_polynomials")]
#[allow(clippy::type_complexity)]
pub(super) fn commit_witness_polynomials<
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
>(
    model: &Model,
    trace: &Trace,
    generators: &PCS::ProverSetup,
    transcript: &mut T,
) -> (
    BTreeMap<CommittedPoly, MultilinearPolynomial<F>>,
    WitnessCommitments<PCS>,
    Vec<PCS::OpeningProofHint>,
) {
    let poly_map = polynomial_map::<F>(model, trace);
    let polys: Vec<&MultilinearPolynomial<F>> = poly_map.values().collect();
    let (batch, per_poly) = PCS::batch_commit_all(&polys, generators);
    let (commitments, hints): (Vec<_>, Vec<_>) = per_poly.into_iter().unzip();
    batch.append_to_transcript(transcript);
    for commitment in &commitments {
        transcript.append_serializable(commitment);
    }
    let commitments = WitnessCommitments {
        batch,
        polys: commitments,
    };
    (poly_map, commitments, hints)
}

pub(crate) fn output_claim<F: JoltField, T: Transcript>(prover: &mut Prover<F, T>) {
    // Construct output polynomial
    let output_index = prover.preprocessing.model().output_indices()[0];
    let output_computation_node = &prover.preprocessing.model().view(output_index);
    let LayerData {
        operands: _,
        output,
    } = TraceView::layer_data(&prover.trace, output_computation_node);

    // Sample challenge from verifier over the power-of-two padded output domain
    let r_node_output = prover.transcript.challenge_vector_optimized::<F>(
        output_computation_node
            .pow2_padded_num_output_elements()
            .log_2(),
    );

    // Evaluate output polynomial at r_node_output
    let output_claim = output.evaluate_mle(&r_node_output);

    let output_opening_id = OpeningId::new(
        VirtualPoly::NodeOutput(output_computation_node.idx),
        // We add a NodeOutput claim for the model output. We use idx+1 as the execution,
        // since some nodes add a claim during their own execution that would otherwise collide with the output
        SumcheckId::NodeExecution(output_computation_node.idx + 1),
    );

    let mut provider = AccOpeningAccessor::new(&mut prover.accumulator, output_computation_node)
        .into_provider(&mut prover.transcript, r_node_output.into());

    // append_virtual handles both transcript append and insertion into openings
    provider.append_custom(output_opening_id, output_claim);
}

/// Iterate over computation graph in reverse topological order
/// Prove each operation using sum-check and virtual polynomials
#[tracing::instrument(skip_all, name = "ONNXProof::iop")]
pub(crate) fn iop<F: JoltField, T: Transcript>(
    model: &Model,
    prover: &mut Prover<F, T>,
    proofs: &mut BTreeMap<ProofId, SumcheckInstanceProof<F>>,
    eval_reduction_proofs: &mut BTreeMap<usize, EvalReductionProof<F>>,
) {
    for node in model.views().iter().rev() {
        let (eval_reduction_proof, execution_proofs) = OperatorProver::prove(node, prover);
        eval_reduction_proofs.insert(node.idx, eval_reduction_proof);
        proofs.extend(execution_proofs);
    }
    crate::onnx_proof::deferred_lookups::prove_all(prover, proofs);
}

#[tracing::instrument(skip_all, name = "ONNXProof::prove_reduced_openings")]
pub(super) fn prove_reduced_openings<
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
>(
    prover: &mut Prover<F, T>,
    poly_map: &BTreeMap<CommittedPoly, MultilinearPolynomial<F>>,
    hints: Vec<PCS::OpeningProofHint>,
    generators: &PCS::ProverSetup,
) -> Option<ReducedOpeningProof<F, PCS>> {
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
    prover
        .accumulator
        .prepare_for_sumcheck(poly_map, &mut prover.transcript);

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
    debug_assert!(state.polynomials.iter().eq(poly_map.keys()));
    let joint_opening_proof = PCS::prove_rlc(
        generators,
        poly_map,
        &state.poly_coeffs,
        hints,
        &state.r_sumcheck,
        &mut prover.transcript,
    );
    Some(ReducedOpeningProof::from_parts(
        accumulator_sumcheck_proof,
        sumcheck_claims,
        joint_opening_proof,
    ))
}

#[allow(clippy::type_complexity)]
pub(super) fn finalize_proof<
    F: JoltField,
    T: ProvingTranscript,
    PCS: CommitmentScheme<Field = F>,
>(
    mut prover: Prover<F, T>,
    io: ModelExecutionIO,
    commitments: WitnessCommitments<PCS>,
    proofs: BTreeMap<ProofId, SumcheckInstanceProof<F>>,
    eval_reduction_proofs: BTreeMap<usize, EvalReductionProof<F>>,
    reduced_opening_proof: Option<ReducedOpeningProof<F, PCS>>,
) -> (
    ONNXProof<F, PCS>,
    ModelExecutionIO,
    Option<ProverDebugInfo<F>>,
) {
    #[cfg(test)]
    let debug_info = Some(ProverDebugInfo {
        transcript_history: prover.transcript.state_history().to_vec(),
        opening_accumulator: prover.accumulator.clone(),
    });
    #[cfg(not(test))]
    let debug_info = None;
    let opening_claims = prover.accumulator.take();
    let narg = prover.transcript.into_narg();
    (
        ONNXProof::from_parts(
            Claims::from_parts(opening_claims),
            proofs,
            commitments.batch,
            commitments.polys,
            eval_reduction_proofs,
            reduced_opening_proof,
            narg,
        ),
        io,
        debug_info,
    )
}

#[tracing::instrument(skip_all, name = "ONNXProof::polynomial_map")]
pub(super) fn polynomial_map<F: JoltField>(
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
        .views()
        .iter()
        .flat_map(|node| {
            crate::onnx_proof::witness::generate_node_witnesses::<F>(node, model, trace)
        })
        .collect();
    // Packed clamp buckets (model-level, not per node).
    polys.extend(crate::onnx_proof::global_clamp::bucket_witnesses::<F>(
        model, trace,
    ));
    polys
}
