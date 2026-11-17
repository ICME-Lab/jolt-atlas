//! Verifier state and internal verification helpers for [`ONNXProof`].
//!
//! The public entry point is [`verify`](super::verify) (defined in the parent module).
//! This file houses the [`Verifier`] struct and the helper functions that `verify`
//! delegates to.

use super::{
    ops::eval_reduction::NodeEvalReduction,
    schedule::{Schedule, Step},
    AtlasVerifierPreprocessing,
};
use crate::node_view::ModelView;
use crate::utils::opening_access::AccOpeningAccessor;
use crate::{ONNXProof, ProofId};
use atlas_algebra::{
    field::JoltField,
    poly::{
        multilinear::{MultilinearPolynomial, PolynomialEvaluation},
        point::OpeningPoint,
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
/// Created once before the schedule runs.
pub struct Verifier<'a, F: JoltField, T: Transcript> {
    /// Opening accumulator for batching polynomial openings.
    pub(crate) accumulator: VerifierOpeningAccumulator<F>,
    /// Interactive proof transcript.
    pub(crate) transcript: T,
    /// The proof's sumcheck proofs, each taken exactly once.
    pub(crate) proofs: ProofLedger<'a, ProofId, SumcheckInstanceProof<F>>,
    /// The proof's evaluation-reduction proofs by node, each taken exactly once.
    pub(crate) eval_reductions: ProofLedger<'a, usize, EvalReductionProof<F>>,
    /// Model execution inputs and outputs.
    pub(crate) io: &'a ModelExecutionIO,
}

impl<'a, F: JoltField, T: Transcript> Verifier<'a, F, T> {
    /// Create a new verifier with the given proofs, IO, and the proof's transcript NARG
    /// string.
    pub fn new(
        proofs: &'a BTreeMap<ProofId, SumcheckInstanceProof<F>>,
        eval_reductions: &'a BTreeMap<usize, EvalReductionProof<F>>,
        io: &'a ModelExecutionIO,
        narg: &[u8],
    ) -> Self
    where
        T: VerifyingTranscript,
    {
        Self {
            accumulator: VerifierOpeningAccumulator::new(),
            transcript: T::new(b"ONNXProof", narg.to_vec()),
            proofs: ProofLedger::new(proofs),
            eval_reductions: ProofLedger::new(eval_reductions),
            io,
        }
    }

    /// The opening accumulator: the claims bound so far.
    pub fn accumulator(&self) -> &VerifierOpeningAccumulator<F> {
        &self.accumulator
    }

    /// The opening accumulator, for tests that stage claims by hand.
    #[cfg(feature = "test-util")]
    pub fn accumulator_mut(&mut self) -> &mut VerifierOpeningAccumulator<F> {
        &mut self.accumulator
    }

    /// The transcript, for tests that draw a point by hand.
    #[cfg(feature = "test-util")]
    pub fn transcript_mut(&mut self) -> &mut T {
        &mut self.transcript
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
    /// The interpreter: run `schedule`, step by step, in transcript order. A relation set up
    /// by a `Setup` step keeps its verifiers until the `Sumcheck` step that names it.
    #[tracing::instrument(skip_all, name = "Verifier::run")]
    pub fn run(&mut self, schedule: &Schedule<F>) -> Result<(), ProofVerifyError> {
        let mut set_up: Vec<Option<Verifiers<F, T>>> =
            schedule.relations().iter().map(|_| None).collect();
        for step in schedule.steps() {
            match step {
                Step::Reduce { node } => NodeEvalReduction::verify(self, *node)
                    .inspect_err(|e| tracing::debug!(node, ?e, "evaluation reduction failed"))?,
                Step::Open(open) => open.verify(&mut self.accumulator, &mut self.transcript)?,
                Step::Setup(id) => {
                    let verifiers = schedule
                        .relation(*id)?
                        .verifiers(&mut self.accumulator, &mut self.transcript)?;
                    if let Some(slot) = set_up.get_mut(id.0) {
                        *slot = Some(verifiers);
                    }
                }
                Step::Sumcheck {
                    proof: proof_id,
                    members,
                    batched,
                } => {
                    let mut instances = Vec::new();
                    for id in members {
                        match set_up.get_mut(id.0).and_then(Option::take) {
                            Some(verifiers) => instances.extend(verifiers),
                            None => instances.extend(
                                schedule
                                    .relation(*id)?
                                    .verifiers(&mut self.accumulator, &mut self.transcript)?,
                            ),
                        }
                    }
                    let proof = self.proofs.take(proof_id.clone())?;
                    let verified = if *batched {
                        verify_relation_batch(
                            proof,
                            instances.iter().map(|instance| instance.as_ref()).collect(),
                            &mut self.accumulator,
                            &mut self.transcript,
                        )
                    } else {
                        let [relation] = instances.as_slice() else {
                            return Err(ProofVerifyError::Malformed(format!(
                                "schedule: the sumcheck {proof_id:?} is not a single relation"
                            )));
                        };
                        verify_relation(
                            proof,
                            relation.as_ref(),
                            &mut self.accumulator,
                            &mut self.transcript,
                        )
                    };
                    verified.inspect_err(|e| tracing::debug!(?proof_id, ?e, "sumcheck failed"))?;
                }
                Step::Tie(id) => super::relation::verify_tie(
                    schedule.tie(*id)?,
                    &mut self.accumulator,
                    &mut self.transcript,
                )?,
            }
        }
        Ok(())
    }
}

/// The verifiers of one relation, set up.
type Verifiers<F, T> = Vec<Box<dyn DynRelationVerifier<F, T>>>;

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

/// Verify the reduced opening proof (sumcheck reduction + PCS verification).
#[tracing::instrument(skip_all, name = "ONNXProof::verify_reduced_openings")]
pub fn verify_reduced_openings<F: JoltField, T: Transcript, PCS: CommitmentScheme<Field = F>>(
    proof: &ONNXProof<F, PCS>,
    pp: &AtlasVerifierPreprocessing<F, PCS>,
    schedule: &Schedule<F>,
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

        // The proof commits the polynomials the schedule opens, in sorted `CommittedPoly`
        // order (the prover's `poly_map`); the joint opening covers the opened subset.
        let committed = schedule.committed();
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
