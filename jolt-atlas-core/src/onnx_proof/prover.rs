//! Prover state and internal proving helpers for [`ONNXProof`].
//!
//! The public entry point is [`prove`](super::prove) (defined in the parent module).
//! This file houses the [`Prover`] struct and the private helper functions that
//! `prove` delegates to.

use super::{
    types::ProverDebugInfo,
    verifier::{seed_opening_transcript, OPENING_TRANSCRIPT_LABEL},
    AtlasSharedPreprocessing, ONNXProof,
};
use crate::node_view::ModelView;
use crate::onnx_proof::trace_view::TraceView;
use crate::onnx_proof::witness::WitnessGenerator;
use crate::onnx_proof::{
    ops::{prove_schedule, Addresses},
    range_bucket::BucketWitnesses,
    schedule::Schedule,
};
use crate::utils::opening_access::AccOpeningAccessor;
use crate::utils::opening_access::IntoProverProvider;
use atlas_algebra::{
    field::JoltField,
    poly::multilinear::{EvaluateMle, MultilinearPolynomial},
    utils::math::Math,
};
use atlas_pcs::commitment_scheme::CommitmentScheme;
use atlas_sumcheck::opening_proof::{prover::ProverOpeningAccumulator, OpeningId, SumcheckId};
use atlas_trace::{trace::LayerData, trace::ModelExecutionIO, Model, Trace};
use atlas_transcript::ProvingTranscript;
use common::{CommittedPoly, VirtualPoly};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Prover state
// ---------------------------------------------------------------------------

/// Prover state that owns all data needed during proving.
/// Created once before the proving loop and passed to operator handlers.
pub struct Prover<F: JoltField, T: ProvingTranscript> {
    /// Execution trace of the neural network model.
    pub trace: Trace,
    /// Shared preprocessing data (model structure).
    pub preprocessing: AtlasSharedPreprocessing,
    /// The public IO of the execution: the statement the schedule is built for.
    pub io: ModelExecutionIO,
    /// Opening accumulator for batching polynomial openings.
    pub accumulator: ProverOpeningAccumulator<F>,
    /// Interactive proof transcript.
    pub transcript: T,
    /// The read addresses each lookup left for its one-hot checks, by lookup and node.
    pub lookup_addresses: BTreeMap<(Addresses, usize), Vec<usize>>,
    /// The packed buckets' witnesses, built with the prover (before the commitment) and read by
    /// the commitment and the buckets' sumcheck steps.
    pub buckets: BucketWitnesses,
}

impl<F: JoltField, T: ProvingTranscript> Prover<F, T> {
    /// Create a new prover with the given preprocessing and trace
    pub fn new(preprocessing: AtlasSharedPreprocessing, trace: Trace) -> Self
    where
        T: ProvingTranscript,
    {
        Self {
            io: trace.io(preprocessing.model()),
            buckets: BucketWitnesses::build(preprocessing.model(), &trace),
            trace,
            preprocessing,
            accumulator: ProverOpeningAccumulator::new(),
            transcript: T::new(b"ONNXProof"),
            lookup_addresses: BTreeMap::new(),
        }
    }

    /// Leave a lookup's read addresses for its one-hot checks.
    pub fn leave_addresses(&mut self, key: Addresses, node_idx: usize, indices: Vec<usize>) {
        let previous = self.lookup_addresses.insert((key, node_idx), indices);
        assert!(
            previous.is_none(),
            "node {node_idx}: the {key:?} addresses were left twice"
        );
    }

    /// Take the read addresses a lookup left for this one-hot check.
    pub fn take_addresses(&mut self, key: Addresses, node_idx: usize) -> Vec<usize> {
        self.lookup_addresses
            .remove(&(key, node_idx))
            .unwrap_or_else(|| {
                panic!("node {node_idx}: the {key:?} addresses were not left by their lookup")
            })
    }
}

// ---------------------------------------------------------------------------
// Internal proving helpers
// ---------------------------------------------------------------------------

/// Build the witness polynomial map, commit to every polynomial, and send the commitments
/// (the batch commitment first, then one per polynomial in `poly_map` order).
#[tracing::instrument(skip_all, name = "ONNXProof::commit_witness_polynomials")]
#[allow(clippy::type_complexity)]
pub(super) fn commit_witness_polynomials<
    F: JoltField,
    T: ProvingTranscript,
    PCS: CommitmentScheme<Field = F>,
>(
    schedule: &Schedule<F>,
    model: &Model,
    trace: &Trace,
    buckets: &BucketWitnesses,
    generators: &PCS::ProverSetup,
    transcript: &mut T,
) -> (
    BTreeMap<CommittedPoly, MultilinearPolynomial<F>>,
    Vec<PCS::OpeningProofHint>,
) {
    let poly_map = polynomial_map::<F>(schedule, model, trace, buckets);
    let polys: Vec<&MultilinearPolynomial<F>> = poly_map.values().collect();
    let (batch, per_poly) = PCS::batch_commit_all(&polys, generators);
    let (commitments, hints): (Vec<PCS::Commitment>, Vec<_>) = per_poly.into_iter().unzip();
    transcript.send_serializable(&batch);
    for commitment in &commitments {
        transcript.send_serializable(commitment);
    }
    (poly_map, hints)
}

pub(crate) fn output_claim<F: JoltField, T: ProvingTranscript>(prover: &mut Prover<F, T>) {
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

    // Bind the output opening at the point and send its claim.
    provider.append_custom(output_opening_id, output_claim);
}

/// Walk the model's schedule: each node in reverse topological order, then the model-wide
/// sumchecks.
#[tracing::instrument(skip_all, name = "ONNXProof::iop")]
pub(crate) fn iop<F: JoltField, T: ProvingTranscript>(
    schedule: &Schedule<F>,
    prover: &mut Prover<F, T>,
) {
    prove_schedule(schedule, prover);
}

/// Reduce the committed openings to one point and send the joint opening proof there. A
/// proof that opens no committed polynomial carries neither.
#[tracing::instrument(skip_all, name = "ONNXProof::prove_reduced_openings")]
pub(super) fn prove_reduced_openings<
    F: JoltField,
    T: ProvingTranscript,
    PCS: CommitmentScheme<Field = F>,
>(
    prover: &mut Prover<F, T>,
    poly_map: &BTreeMap<CommittedPoly, MultilinearPolynomial<F>>,
    hints: Vec<PCS::OpeningProofHint>,
    generators: &PCS::ProverSetup,
) {
    // Committed polynomials nobody opened stay out of the joint opening; the
    // verifier selects the same subset from its openings.
    let opened = prover.accumulator.opened_polynomials();
    if opened.is_empty() {
        // Nothing was opened (every committed polynomial is unused): no joint
        // opening.
        return;
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
    let r_sumcheck_acc = prover
        .accumulator
        .prove_batch_opening_sumcheck(&mut prover.transcript);

    // Finalize sumcheck (uses the claims the groups sent, derives gamma, cleans up)
    let state = prover
        .accumulator
        .finalize_batch_opening_sumcheck(r_sumcheck_acc.clone(), &mut prover.transcript);
    // Joint opening of the RLC `Σ γ_i · poly_i` (materialized or not, per PCS), on the
    // opening's own transcript; the proof is the last message of the prover's.
    debug_assert!(state.polynomials.iter().eq(poly_map.keys()));
    let mut opening_transcript = T::new(OPENING_TRANSCRIPT_LABEL);
    seed_opening_transcript::<F, _, _>(&mut prover.transcript, &mut opening_transcript);
    let joint_opening_proof = PCS::prove_rlc(
        generators,
        poly_map,
        &state.poly_coeffs,
        hints,
        &state.r_sumcheck,
        &mut opening_transcript,
    );
    prover.transcript.send_serializable(&joint_opening_proof);
}

/// The proof: the bytes the prover's transcript wrote.
pub(super) fn finalize_proof<F: JoltField, T: ProvingTranscript>(
    prover: Prover<F, T>,
    io: ModelExecutionIO,
) -> (ONNXProof, ModelExecutionIO, Option<ProverDebugInfo<F>>) {
    #[cfg(test)]
    let debug_info = Some(ProverDebugInfo {
        transcript_history: prover.transcript.state_history().to_vec(),
        opening_accumulator: prover.accumulator.clone(),
    });
    #[cfg(not(test))]
    let debug_info = None;
    (ONNXProof(prover.transcript.into_narg()), io, debug_info)
}

/// The witness of every polynomial the proof commits: the ones the schedule opens, the
/// packed range buckets' digit columns from `buckets` and every other from the trace.
#[tracing::instrument(skip_all, name = "ONNXProof::polynomial_map")]
pub(super) fn polynomial_map<F: JoltField>(
    schedule: &Schedule<F>,
    model: &Model,
    trace: &Trace,
    buckets: &BucketWitnesses,
) -> BTreeMap<CommittedPoly, MultilinearPolynomial<F>> {
    // Polynomials are built one after another, each with the whole thread pool: the cost is
    // dominated by a few large ones, which a parallel walk would pin to a single thread each.
    let mut polys: BTreeMap<CommittedPoly, MultilinearPolynomial<F>> = schedule
        .committed()
        .into_iter()
        .filter(|poly| !matches!(poly, CommittedPoly::GlobalRemainderDigit(..)))
        .map(|poly| {
            let witness = poly.generate_witness(model, trace);
            (poly, witness)
        })
        .collect();
    polys.extend(buckets.committed_polys::<F>(model));
    polys
}
