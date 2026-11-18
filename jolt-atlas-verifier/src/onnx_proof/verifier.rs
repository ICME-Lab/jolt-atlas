//! Verifier state and internal verification helpers for [`ONNXProof`](crate::ONNXProof).
//!
//! The public entry point is [`verify`](super::verify) (defined in the parent module).
//! This file houses the [`Verifier`] struct, its interpreter [`Verifier::run`] (the one loop
//! over the steps of the model's [`Schedule`]) and the helper functions that `verify`
//! delegates to.

use super::{
    ops::{constant::with_public_input, eval_reduction::NodeEvalReduction},
    range_check,
    relation::{verify_tie, OpRelKind},
    schedule::{Schedule, Step},
    AtlasVerifierPreprocessing,
};
use crate::node_view::ModelView;
use crate::utils::opening_access::AccOpeningAccessor;
use atlas_algebra::{
    field::{IntoOpening, JoltField},
    poly::multilinear::{MultilinearPolynomial, PolynomialEvaluation},
    utils::{errors::ProofVerifyError, math::Math},
};
use atlas_pcs::commitment_scheme::CommitmentScheme;
use atlas_sumcheck::evaluation_reduction::ReducedInstance;
use atlas_sumcheck::opening_proof::{OpeningId, SumcheckId, VerifierOpeningAccumulator};
use atlas_sumcheck::relation::{verify_relation, verify_relation_batch, DynRelationVerifier};
use atlas_trace::{trace::ModelExecutionIO, Model};
use atlas_transcript::{Transcript, VerifyingTranscript};
use common::{CommittedPoly, VirtualPoly};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Verifier state
// ---------------------------------------------------------------------------

/// Verifier state that owns all data needed during verification.
/// Created once before the schedule runs. The proof is the transcript's NARG string: the
/// verifier reaches it only through the transcript's reads.
pub struct Verifier<'a, F: JoltField, T: VerifyingTranscript> {
    /// Opening accumulator for batching polynomial openings.
    pub(crate) accumulator: VerifierOpeningAccumulator<F>,
    /// Interactive proof transcript.
    pub(crate) transcript: T,
    /// Model execution inputs and outputs.
    pub(crate) io: &'a ModelExecutionIO,
}

impl<'a, F: JoltField, T: VerifyingTranscript> Verifier<'a, F, T> {
    /// Create a new verifier with the given IO and the proof's transcript NARG string.
    pub fn new(io: &'a ModelExecutionIO, narg: &[u8]) -> Self {
        Self {
            accumulator: VerifierOpeningAccumulator::new(),
            transcript: T::new(b"ONNXProof", narg.to_vec()),
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

impl<F: JoltField, T: VerifyingTranscript> Verifier<'_, F, T> {
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
                Step::Point { node, vars } => {
                    let point: Vec<F> = self
                        .transcript
                        .challenge_vector_optimized::<F>(*vars)
                        .into_opening();
                    let claim = self.transcript.read_scalar()?;
                    self.accumulator
                        .reduced_evaluations
                        .insert(*node, ReducedInstance { r: point, claim });
                }
                Step::Open(open) => open.verify(&mut self.accumulator, &mut self.transcript)?,
                Step::Setup(id) => {
                    let verifiers = schedule
                        .relation(*id)?
                        .verifiers(&mut self.accumulator, &mut self.transcript)?;
                    if let Some(slot) = set_up.get_mut(id.0) {
                        *slot = Some(verifiers);
                    }
                }
                Step::Sumcheck { members, batched } => {
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
                    let verified = if *batched {
                        verify_relation_batch(
                            instances.iter().map(|instance| instance.as_ref()).collect(),
                            &mut self.accumulator,
                            &mut self.transcript,
                        )
                    } else {
                        let [relation] = instances.as_slice() else {
                            return Err(ProofVerifyError::Malformed(
                                "schedule: an unbatched sumcheck is not a single relation".into(),
                            ));
                        };
                        verify_relation(
                            relation.as_ref(),
                            &mut self.accumulator,
                            &mut self.transcript,
                        )
                    };
                    verified.inspect_err(|e| tracing::debug!(?members, ?e, "sumcheck failed"))?;
                }
                Step::Tie(id) => {
                    let relation = schedule.tie(*id)?;
                    // An input's tie takes its public tensor from the IO: the one step that
                    // reads it.
                    let with_input;
                    let relation = if relation.kind == OpRelKind::Input {
                        with_input = with_public_input(relation, self.io)?;
                        &with_input
                    } else {
                        relation
                    };
                    verify_tie(relation, &mut self.accumulator, &mut self.transcript)?;
                }
            }
        }
        Ok(())
    }
}

/// The verifiers of one relation, set up.
type Verifiers<F, T> = Vec<Box<dyn DynRelationVerifier<F, T>>>;

// ---------------------------------------------------------------------------
// Internal verification helpers
// ---------------------------------------------------------------------------

/// The commitments a proof opens with: the scheme's batch commitment, then one commitment per
/// polynomial the schedule opens, in sorted `CommittedPoly` order (the prover's `poly_map`).
pub struct Commitments<PCS: CommitmentScheme> {
    batch: PCS::BatchCommitment,
    polys: Vec<PCS::Commitment>,
}

/// The bits the commitment must bound a committed polynomial's coefficients by, where the
/// protocol relies on a bound: a one-hot polynomial is boolean (with its Hamming-weight check
/// that makes it one-hot) and a digit column holds base-[`range_check::DIGIT_BASE`] digits.
/// The quotient and shift columns are bounded by relations of their own.
fn committed_bits(poly: &CommittedPoly) -> Option<u32> {
    match poly {
        CommittedPoly::NodeOutputRaD(..)
        | CommittedPoly::CosRaD(..)
        | CommittedPoly::SinRaD(..)
        | CommittedPoly::TrigDownscaleRaD(..)
        | CommittedPoly::GatherRa(..)
        | CommittedPoly::GatherRaD(..)
        | CommittedPoly::SoftmaxAddrRaD(..)
        | CommittedPoly::ActivationClampRaD(..)
        | CommittedPoly::ActivationSmallRaD(..)
        | CommittedPoly::SoftmaxClampRaD(..) => Some(1),
        CommittedPoly::GlobalRemainderDigit(..) => Some(range_check::DIGIT_BASE.ilog2()),
        CommittedPoly::RsqrtQuotient(..)
        | CommittedPoly::TrigPeriodQuotient(..)
        | CommittedPoly::SoftmaxShift(..) => None,
    }
}

/// Read the commitments from the transcript: the batch commitment, then as many commitments
/// as the schedule opens committed polynomials. `Err` on a failed read, or if a polynomial the
/// protocol relies on being bounded ([`committed_bits`]) is not committed within that bound.
pub fn read_commitments<F: JoltField, T: VerifyingTranscript, PCS: CommitmentScheme<Field = F>>(
    pp: &AtlasVerifierPreprocessing<F, PCS>,
    verifier: &mut Verifier<'_, F, T>,
) -> Result<Commitments<PCS>, ProofVerifyError> {
    let batch = verifier.transcript.read_serializable()?;
    let committed = pp.schedule.committed();
    let polys: Vec<PCS::Commitment> = committed
        .iter()
        .map(|_| verifier.transcript.read_serializable())
        .collect::<Result<_, _>>()?;
    for (poly, commitment) in committed.iter().zip(&polys) {
        if let Some(bound) = committed_bits(poly) {
            let bits = PCS::coefficient_bits(&batch, commitment);
            if bits.is_none_or(|bits| bits > bound) {
                return Err(ProofVerifyError::InvalidOpeningProof(format!(
                    "{poly:?} is not committed within {bound} bits"
                )));
            }
        }
    }
    Ok(Commitments { batch, polys })
}

/// The transcript label of the joint opening's own transcript.
pub const OPENING_TRANSCRIPT_LABEL: &[u8] = b"ONNXProof/joint_opening";

/// Seed the joint opening's own transcript from the proof's transcript: two field elements
/// drawn there, which bind everything absorbed so far, absorbed here. Prover and verifier run
/// it at the same state of the proof's transcript, on a fresh `opening_transcript`.
///
/// The commitment scheme draws its challenges while it builds its proof, so the prover cannot
/// send the finished proof before them on one transcript. The scheme runs on this transcript
/// instead; the proof's transcript carries the opening proof as its last message. The scheme
/// sends nothing through its transcript: everything it sends is in its proof.
pub fn seed_opening_transcript<F: JoltField, T: Transcript, O: Transcript>(
    transcript: &mut T,
    opening_transcript: &mut O,
) {
    let seed: Vec<F> = transcript.challenge_vector(2);
    opening_transcript.append_message(b"joint_opening_seed");
    opening_transcript.append_scalars(&seed);
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
pub fn verify_output_claim<F: JoltField, T: VerifyingTranscript>(
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

    // Bind the output opening at the point: its claim is read from the transcript.
    provider.append_custom(output_opening_id)?;
    // Compare the prover's claimed value against the IO.
    let output_claim = provider.get_custom(output_opening_id)?.1;
    if expected_output_claim != output_claim {
        return Err(ProofVerifyError::InvalidOpeningProof(
            "Expected output claim does not match actual output claim".to_string(),
        ));
    }
    Ok(())
}

/// Reduce the committed openings to one point and verify the joint opening proof there. A
/// schedule that opens no committed polynomial has neither in its proof.
#[tracing::instrument(skip_all, name = "ONNXProof::verify_reduced_openings")]
pub fn verify_reduced_openings<
    F: JoltField,
    T: VerifyingTranscript,
    PCS: CommitmentScheme<Field = F>,
>(
    commitments: &Commitments<PCS>,
    pp: &AtlasVerifierPreprocessing<F, PCS>,
    verifier: &mut Verifier<'_, F, T>,
) -> Result<(), ProofVerifyError> {
    if !verifier.accumulator.has_committed_openings() {
        return Ok(());
    }
    verifier
        .accumulator
        .prepare_for_sumcheck(&mut verifier.transcript);

    let reduction_res = verifier
        .accumulator
        .verify_batch_opening_sumcheck(&mut verifier.transcript);
    let r_sumcheck = reduction_res
        .inspect_err(|e| tracing::debug!(?e, "opening reduction via sumcheck failed"))?;

    let verifier_state = verifier
        .accumulator
        .finalize_batch_opening_sumcheck(r_sumcheck, &mut verifier.transcript)?;

    // The joint opening covers the opened subset of the committed polynomials.
    let by_poly: BTreeMap<CommittedPoly, &PCS::Commitment> = pp
        .schedule
        .committed()
        .into_iter()
        .zip(commitments.polys.iter())
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

    let mut opening_transcript = T::new(OPENING_TRANSCRIPT_LABEL, Vec::new());
    seed_opening_transcript::<F, _, _>(&mut verifier.transcript, &mut opening_transcript);
    let joint_opening_proof: PCS::Proof = verifier.transcript.read_serializable()?;
    PCS::verify_rlc(
        &joint_opening_proof,
        &pp.generators,
        &mut opening_transcript,
        &verifier_state.r_sumcheck,
        &joint_claim,
        &selected,
        &verifier_state.poly_coeffs,
        &num_vars,
        &commitments.batch,
    )
}
