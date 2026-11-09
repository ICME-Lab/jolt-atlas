//! Verifier state and internal verification helpers for [`ONNXProof`].
//!
//! The public entry point is [`verify`](super::verify) (defined in the parent module).
//! This file houses the [`Verifier`] struct and the helper functions that `verify`
//! delegates to.

use super::{
    ops::eval_reduction::NodeEvalReduction,
    plan::{Plan, Step},
    AtlasSharedPreprocessing, AtlasVerifierPreprocessing,
};
use crate::node_view::ComputationNode;
use crate::node_view::ModelView;
use crate::{onnx_proof::ops::OperatorVerifier, utils::opening_access::AccOpeningAccessor};
use crate::{ONNXProof, ProofId};
use atlas_algebra::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_point::OpeningPoint,
    },
    utils::{errors::ProofVerifyError, math::Math},
};
use atlas_pcs::commitment_scheme::CommitmentScheme;
use atlas_sumcheck::relation::{verify_relation, verify_relation_batch, DynRelationVerifier};
use atlas_sumcheck::{
    evaluation_reduction::EvalReductionProof,
    opening_proof::{OpeningId, SumcheckId, VerifierOpeningAccumulator},
    sumcheck::SumcheckInstanceProof,
};
use atlas_trace::{trace::ModelExecutionIO, Model};
use atlas_transcript::{AppendToTranscript, Transcript, VerifyingTranscript};
use common::{CommittedPoly, VirtualPoly};
use std::collections::{BTreeMap, BTreeSet};

// ---------------------------------------------------------------------------
// Verifier state
// ---------------------------------------------------------------------------

/// Verifier state that owns all data needed during verification.
/// Created once before the verification loop and passed to operator handlers.
pub struct Verifier<'a, F: JoltField, T: Transcript> {
    /// Shared preprocessing data (model structure).
    pub preprocessing: &'a AtlasSharedPreprocessing,
    /// Opening accumulator for batching polynomial openings.
    pub accumulator: VerifierOpeningAccumulator<F>,
    /// Interactive proof transcript.
    pub transcript: T,
    /// The proof's sumcheck proofs, each taken exactly once.
    pub proofs: ProofLedger<'a, ProofId, SumcheckInstanceProof<F>>,
    /// The proof's evaluation-reduction proofs by node, each taken exactly once.
    pub eval_reductions: ProofLedger<'a, usize, EvalReductionProof<F>>,
    /// Model execution inputs and outputs.
    pub io: &'a ModelExecutionIO,
    /// Lookups registered during the node loop, verified afterwards in one
    /// batch (see [`deferred_lookups`](crate::onnx_proof::deferred_lookups)).
    pub deferred: Vec<crate::onnx_proof::deferred_lookups::VerifierLookupJob>,
    /// Verifier instances built during the node loop, verified afterwards per batch.
    pub deferred_batches: BTreeMap<
        crate::onnx_proof::deferred_lookups::DeferredBatch,
        Vec<Box<dyn DynRelationVerifier<F, T>>>,
    >,
    /// One-hot checks to verify right after their parent batch.
    pub deferred_onehots: Vec<crate::onnx_proof::deferred_lookups::DeferredOneHot>,
}

impl<'a, F: JoltField, T: Transcript> Verifier<'a, F, T> {
    /// Create a new verifier with the given preprocessing, proofs, IO, and the
    /// proof's transcript NARG string.
    pub fn new(
        preprocessing: &'a AtlasSharedPreprocessing,
        proofs: &'a BTreeMap<ProofId, SumcheckInstanceProof<F>>,
        eval_reductions: &'a BTreeMap<usize, EvalReductionProof<F>>,
        io: &'a ModelExecutionIO,
        narg: &[u8],
    ) -> Self
    where
        T: VerifyingTranscript,
    {
        Self {
            preprocessing,
            accumulator: VerifierOpeningAccumulator::new(),
            transcript: T::new(b"ONNXProof", narg.to_vec()),
            proofs: ProofLedger::new(proofs),
            eval_reductions: ProofLedger::new(eval_reductions),
            io,
            deferred: Vec::new(),
            deferred_batches: BTreeMap::new(),
            deferred_onehots: Vec::new(),
        }
    }

    /// Queue a node-loop-built verifier instance for the deferred batch `kind`.
    pub fn defer(
        &mut self,
        kind: crate::onnx_proof::deferred_lookups::DeferredBatch,
        instance: Box<dyn DynRelationVerifier<F, T>>,
    ) {
        self.deferred_batches
            .entry(kind)
            .or_default()
            .push(instance);
    }
}

impl<F: JoltField, T: Transcript> Verifier<'_, F, T> {
    /// The drain check: the proof was consumed exactly. `Err` if a sumcheck proof or an
    /// evaluation-reduction proof was never taken, or an opening claim was never bound to a
    /// point.
    pub fn check_consumed(&self) -> Result<(), ProofVerifyError> {
        if let Some(id) = self.proofs.first_unconsumed() {
            return Err(ProofVerifyError::UnconsumedProof(format!("{id:?}")));
        }
        if let Some(node) = self.eval_reductions.first_unconsumed() {
            return Err(ProofVerifyError::UnconsumedEvalReduction(*node));
        }
        if let Some(id) = self.accumulator.first_unbound_claim() {
            return Err(ProofVerifyError::UnboundClaim(format!("{id:?}")));
        }
        Ok(())
    }
}

impl<F: JoltField, T: Transcript> Verifier<'_, F, T> {
    /// The interpreter: run `node`'s plan, step by step, in transcript order.
    #[tracing::instrument(skip_all, name = "Verifier::run_plan", fields(node = node.idx))]
    pub fn run_plan(&mut self, node: &ComputationNode, plan: Plan) -> Result<(), ProofVerifyError> {
        for step in plan.0 {
            match step {
                Step::Reduce => NodeEvalReduction::verify(self, node)?,
                Step::Relation { proof, relation } => {
                    let relations =
                        relation.verifiers(&mut self.accumulator, &mut self.transcript)?;
                    let [relation] = relations.as_slice() else {
                        return Err(ProofVerifyError::Malformed(format!(
                            "plan: node {}'s {proof:?} step is not a single relation",
                            node.idx
                        )));
                    };
                    let proof = self.proofs.take(ProofId(node.idx, proof))?;
                    verify_relation(
                        proof,
                        relation.as_ref(),
                        &mut self.accumulator,
                        &mut self.transcript,
                    )?;
                }
                Step::Batch { proof, relations } => {
                    let mut instances = Vec::new();
                    for relation in relations {
                        instances.extend(
                            relation.verifiers(&mut self.accumulator, &mut self.transcript)?,
                        );
                    }
                    let proof = self.proofs.take(ProofId(node.idx, proof))?;
                    verify_relation_batch(
                        proof,
                        instances.iter().map(|instance| instance.as_ref()).collect(),
                        &mut self.accumulator,
                        &mut self.transcript,
                    )?;
                }
                Step::Tie { relation } => {
                    relation.verify(&mut self.accumulator, &mut self.transcript)?
                }
            }
        }
        Ok(())
    }
}

/// A keyed part of the proof with its consumption record: every entry is taken exactly once.
pub struct ProofLedger<'a, K, V> {
    entries: &'a BTreeMap<K, V>,
    consumed: BTreeSet<K>,
}

impl<'a, K: Ord + Clone + std::fmt::Debug, V> ProofLedger<'a, K, V> {
    fn new(entries: &'a BTreeMap<K, V>) -> Self {
        Self {
            entries,
            consumed: BTreeSet::new(),
        }
    }

    /// Take the entry under `key`. `Err` if the proof carries none, or it was already taken.
    pub fn take(&mut self, key: K) -> Result<&'a V, ProofVerifyError> {
        let entry = self
            .entries
            .get(&key)
            .ok_or_else(|| ProofVerifyError::Malformed(format!("missing proof entry {key:?}")))?;
        if !self.consumed.insert(key.clone()) {
            return Err(ProofVerifyError::Malformed(format!(
                "proof entry {key:?} taken twice"
            )));
        }
        Ok(entry)
    }

    /// Whether the proof carries an entry under `key`.
    pub fn contains(&self, key: &K) -> bool {
        self.entries.contains_key(key)
    }

    fn first_unconsumed(&self) -> Option<&'a K> {
        self.entries.keys().find(|key| !self.consumed.contains(key))
    }
}

// ---------------------------------------------------------------------------
// Internal verification helpers
// ---------------------------------------------------------------------------

/// Populate the verifier accumulator with opening claims and
/// commitments from the proof.
pub fn populate_accumulator<F: JoltField, T: Transcript, PCS: CommitmentScheme<Field = F>>(
    proof: &ONNXProof<F, PCS>,
    verifier: &mut Verifier<'_, F, T>,
) {
    // Load all opening claims from the proof (NodeOutput+Execution claims
    // are now stored directly in opening_claims alongside everything else).
    for (key, (_, claim)) in proof.opening_claims().openings() {
        verifier
            .accumulator
            .openings
            .insert(*key, (OpeningPoint::default(), *claim));
    }

    proof
        .batch_commitment()
        .append_to_transcript(&mut verifier.transcript);
    for commitment in proof.commitments() {
        verifier.transcript.append_serializable(commitment);
    }
}

/// The public IO names the model's inputs and outputs, in order, with one tensor of the node's
/// dims for each.
pub fn check_io_shape(model: &Model, io: &ModelExecutionIO) -> Result<(), ProofVerifyError> {
    let malformed = |what: &str| Err(ProofVerifyError::Malformed(format!("public IO: {what}")));
    if io.input_indices != model.input_indices() {
        return malformed("input indices are not the model's");
    }
    if io.output_indices != model.output_indices() {
        return malformed("output indices are not the model's");
    }
    if io.inputs.len() != io.input_indices.len() || io.outputs.len() != io.output_indices.len() {
        return malformed("tensor count does not match the index count");
    }
    let tensors = io.inputs.iter().chain(&io.outputs);
    let indices = io.input_indices.iter().chain(&io.output_indices);
    for (tensor, &idx) in tensors.zip(indices) {
        if tensor.dims() != model.view(idx).output_dims.as_slice() {
            return malformed("a tensor does not have its node's dims");
        }
    }
    Ok(())
}

/// Verify that the output MLE evaluates correctly at the random challenge point τ.
pub fn verify_output_claim<F: JoltField, T: Transcript>(
    model: &Model,
    verifier: &mut Verifier<'_, F, T>,
) -> Result<(), ProofVerifyError> {
    let (Some(&output_index), Some(output)) =
        (model.output_indices().first(), verifier.io.outputs.first())
    else {
        return Err(ProofVerifyError::Malformed(
            "the model has no output".to_string(),
        ));
    };
    let output_computation_node = &model.view(output_index);
    let r_node_output = verifier.transcript.challenge_vector_optimized::<F>(
        output_computation_node
            .pow2_padded_num_output_elements()
            .log_2(),
    );
    let expected_output_claim =
        MultilinearPolynomial::from(output.padded_pow2()).evaluate(&r_node_output);

    let output_opening_id = OpeningId::new(
        VirtualPoly::NodeOutput(output_computation_node.idx),
        // We add a NodeOutput claim for the model output. We use idx+1 as the execution,
        // since some nodes add a claim during their own execution that would otherwise collide with the output claim.
        SumcheckId::NodeExecution(output_computation_node.idx + 1),
    );

    let mut provider = AccOpeningAccessor::new(&mut verifier.accumulator, output_computation_node)
        .into_provider(&mut verifier.transcript, r_node_output.into());

    // append_virtual now handles both transcript append and opening point update.
    // The claim was loaded from opening_claims in populate_accumulator.
    provider.append_custom(output_opening_id)?;
    // Read the prover's claimed value and compare against IO.
    let output_claim = provider.get_custom(output_opening_id)?.1;
    if expected_output_claim != output_claim {
        return Err(ProofVerifyError::InvalidOpeningProof(
            "Expected output claim does not match actual output claim".to_string(),
        ));
    }
    Ok(())
}

/// Iterate over computation graph in reverse topological order and verify each operation.
#[tracing::instrument(skip_all, name = "ONNXProof::verify_iop")]
pub fn verify_iop<F: JoltField, T: Transcript>(
    model: &Model,
    verifier: &mut Verifier<'_, F, T>,
) -> Result<(), ProofVerifyError> {
    for node in model.views().iter().rev() {
        OperatorVerifier::verify(node, verifier)
            .inspect_err(|e| tracing::debug!(node = node.idx, ?e, "verification failed"))?;
    }
    crate::onnx_proof::deferred_lookups::verify_all(verifier)
}

/// Verify the reduced opening proof (sumcheck reduction + PCS verification).
#[tracing::instrument(skip_all, name = "ONNXProof::verify_reduced_openings")]
pub fn verify_reduced_openings<F: JoltField, T: Transcript, PCS: CommitmentScheme<Field = F>>(
    proof: &ONNXProof<F, PCS>,
    pp: &AtlasVerifierPreprocessing<F, PCS>,
    verifier: &mut Verifier<'_, F, T>,
) -> Result<(), ProofVerifyError> {
    if let Some(reduced_opening_proof) = proof.reduced_opening_proof() {
        verifier.accumulator.prepare_for_sumcheck(
            reduced_opening_proof.sumcheck_claims(),
            &mut verifier.transcript,
        )?;

        let reduction_res = verifier.accumulator.verify_batch_opening_sumcheck(
            reduced_opening_proof.sumcheck_proof(),
            &mut verifier.transcript,
        );
        let r_sumcheck = reduction_res
            .inspect_err(|e| tracing::debug!(?e, "opening reduction via sumcheck failed"))?;

        let verifier_state = verifier.accumulator.finalize_batch_opening_sumcheck(
            r_sumcheck,
            reduced_opening_proof.sumcheck_claims(),
            &mut verifier.transcript,
        );

        // Commitments are in sorted `CommittedPoly` order (the prover's
        // `poly_map`); the joint opening covers the opened subset.
        let mut committed = pp.shared.get_models_committed_polynomials()?;
        committed.sort();
        if committed.len() != proof.commitments().len() {
            return Err(ProofVerifyError::InvalidOpeningProof(format!(
                "expected {} commitments, found {}",
                committed.len(),
                proof.commitments().len()
            )));
        }
        let by_poly: BTreeMap<CommittedPoly, &PCS::Commitment> = committed
            .iter()
            .copied()
            .zip(proof.commitments().iter())
            .collect();
        let selected: Vec<&PCS::Commitment> = verifier_state
            .polynomials
            .iter()
            .map(|p| {
                by_poly.get(p).copied().ok_or_else(|| {
                    ProofVerifyError::InvalidOpeningProof(format!(
                        "opening of uncommitted polynomial {p:?}"
                    ))
                })
            })
            .collect::<Result<_, _>>()?;
        let joint_claim = verifier.accumulator.compute_joint_claim(&verifier_state);
        let num_vars = verifier.accumulator.opened_num_vars(&verifier_state);
        PCS::verify_rlc(
            reduced_opening_proof.joint_opening_proof(),
            &pp.generators,
            &mut verifier.transcript,
            &verifier_state.r_sumcheck,
            &joint_claim,
            &selected,
            &verifier_state.poly_coeffs,
            &num_vars,
            proof.batch_commitment(),
        )?;
    } else if verifier.accumulator.sumchecks_keys().next().is_some() {
        // Some committed polynomial was opened: a joint opening is required.
        return Err(ProofVerifyError::MissingReductionProof);
    }
    Ok(())
}
