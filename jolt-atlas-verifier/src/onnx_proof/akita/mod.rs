//! CPU verification of Atlas model proofs with Akita commitments.

mod plan;
mod wire;
pub use plan::{CommitmentGroup, CommitmentPlan, WitnessShape};
use wire::{decode_commitment, decode_opening_proof};

use super::{
    append_inputs_to_transcript, append_model_tensors, schedule::Schedule, verifier, ONNXProof,
    Verifier,
};
use crate::node_view::check_model_shape;
use atlas_algebra::{
    field::akita::AkitaField,
    serialize::{Canonical, Error},
    utils::errors::ProofVerifyError,
};
use atlas_pcs::akita::{
    protocol_tag, schedule_artifacts, verifier_transcript, verify_opening, AkitaScheduleArtifacts,
    Keccak,
};
use atlas_sumcheck::{opening_proof::SumcheckId, relation::verify_relation};
use atlas_trace::{trace::ModelExecutionIO, Model};
use atlas_transcript::{Blake2bVerifier, Transcript, VerifyingTranscript};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

/// Domain separator for the Atlas transcript in Akita CPU proofs.
const TRANSCRIPT_TAG: &[u8] = b"Atlas/Akita/CPU/v1";
/// Domain separator for an individual Akita opening.
pub const OPENING_TAG: &[u8] = b"Atlas/Akita/opening/v1";

/// Public output openings use a consumer index outside the model graph.
pub const OUTPUT_SUMCHECK: SumcheckId = SumcheckId::NodeExecution(usize::MAX);

type WitnessShapes = BTreeMap<common::CommittedPoly, WitnessShape>;

/// Public preprocessing for repeated Akita proofs of the same model.
pub struct Preprocessing<'a> {
    model: &'a Model,
    model_hash: blake3::Hash,
    schedule: Schedule<AkitaField>,
    artifacts: Arc<AkitaScheduleArtifacts>,
    plans: Mutex<BTreeMap<WitnessShapes, Arc<CommitmentPlan>>>,
}

impl<'a> Preprocessing<'a> {
    /// Load the bundled catalogs and compile the verification schedule once.
    pub fn new(model: &'a Model) -> Result<Self, ProofVerifyError> {
        check_akita_model(model)?;
        let mut model_hash = blake3::Hasher::new();
        bincode::serde::encode_into_std_write(model, &mut model_hash, bincode::config::standard())
            .map_err(opening_error)?;
        Ok(Self {
            model,
            model_hash: model_hash.finalize(),
            schedule: Schedule::compile(model)?,
            artifacts: schedule_artifacts().map_err(opening_error)?,
            plans: Mutex::default(),
        })
    }

    /// The model bound to this preprocessing.
    pub fn model(&self) -> &Model {
        self.model
    }

    /// The compiled model schedule.
    pub fn schedule(&self) -> &Schedule<AkitaField> {
        &self.schedule
    }

    /// Bind the model and public IO before absorbing any commitments.
    pub fn bind_statement(&self, io: &ModelExecutionIO, transcript: &mut impl Transcript) {
        transcript.append_message(TRANSCRIPT_TAG);
        transcript.append_bytes(self.model_hash.as_bytes());
        append_inputs_to_transcript(transcript, io);
        append_model_tensors(
            transcript,
            b"model_outputs",
            &io.output_indices,
            &io.outputs,
        );
    }

    /// Reuse the commitment plan for an observed set of original witness shapes.
    pub fn commitment_plan(
        &self,
        shapes: WitnessShapes,
    ) -> Result<Arc<CommitmentPlan>, ProofVerifyError> {
        let mut plans = self.plans.lock().map_err(opening_error)?;
        if let Some(plan) = plans.get(&shapes) {
            return Ok(plan.clone());
        }
        let plan = Arc::new(CommitmentPlan::new(&shapes, self.artifacts.clone())?);
        plans.insert(shapes, plan.clone());
        Ok(plan)
    }

    /// Verify using this model's preprocessing and cached public setups.
    pub fn verify(&self, proof: &ONNXProof, io: &ModelExecutionIO) -> Result<(), ProofVerifyError> {
        verify_preprocessed(proof, self, io)
    }
}

/// Bind a Jolt Akita transcript to the current state of the Atlas transcript.
pub fn opening_seed(transcript: &mut impl Transcript) -> [u8; 32] {
    let mut seed = [0; 32];
    for chunk in seed.as_chunks_mut::<16>().0 {
        chunk.copy_from_slice(&transcript.challenge_u128().to_le_bytes());
    }
    seed
}

/// Convert an Akita or proof transport error at the opening boundary.
pub fn opening_error(error: impl std::fmt::Display) -> ProofVerifyError {
    ProofVerifyError::InvalidOpeningProof(error.to_string())
}

/// A byte message whose outer Atlas transcript frame carries its length.
pub struct AkitaMessage(pub Vec<u8>);

impl Canonical for AkitaMessage {
    fn encode<W: std::io::Write>(&self, mut writer: W) -> Result<(), Error> {
        writer.write_all(&self.0)?;
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        self.0.len()
    }

    fn decode<R: std::io::Read>(mut reader: R) -> Result<Self, Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        Ok(Self(bytes))
    }
}

/// Validate the model entry point shared by the Akita prover and verifier.
pub fn check_akita_model(model: &Model) -> Result<(), ProofVerifyError> {
    check_model_shape(model)?;
    if model.outputs().is_empty() {
        return Err(ProofVerifyError::Malformed(
            "the model has no output".into(),
        ));
    }
    Ok(())
}

/// Verify an Akita CPU proof of a model execution without executing the model.
pub fn verify(
    proof: &ONNXProof,
    model: &Model,
    io: &ModelExecutionIO,
) -> Result<(), ProofVerifyError> {
    Preprocessing::new(model)?.verify(proof, io)
}

fn verify_preprocessed(
    proof: &ONNXProof,
    preprocessing: &Preprocessing<'_>,
    io: &ModelExecutionIO,
) -> Result<(), ProofVerifyError> {
    let model = preprocessing.model();
    let schedule = preprocessing.schedule();
    verifier::check_io_shape(model, io)?;
    let mut verifier = Verifier::<AkitaField, Blake2bVerifier>::new(io, &proof.0);
    preprocessing.bind_statement(io, &mut verifier.transcript);
    // Absorb the commitments before drawing challenges. Their expected shapes
    // come from the model opening requests, which are collected by the IOP.
    let AkitaMessage(commitment_bytes) = verifier.transcript.read_serializable()?;
    let mut checked = BTreeMap::new();
    for (position, (&index, output)) in io.output_indices.iter().zip(&io.outputs).enumerate() {
        if let Some(first) = checked.insert(index, output.data()) {
            if first != output.data() {
                return Err(ProofVerifyError::Malformed(
                    "public tensors for the same output node differ".into(),
                ));
            }
        } else {
            verifier::verify_output_claim_at(
                model,
                &mut verifier,
                index,
                position,
                OUTPUT_SUMCHECK,
            )?;
        }
    }
    verifier.run(schedule)?;

    let mut arities = BTreeMap::new();
    for (id, (point, _)) in verifier.accumulator.openings() {
        if let Some(poly) = id.committed_poly() {
            if arities
                .insert(poly, point.r.len())
                .is_some_and(|n| n != point.r.len())
            {
                return Err(opening_error("inconsistent original commitment arity"));
            }
        }
    }
    if !arities.keys().copied().eq(schedule.committed()) {
        return Err(opening_error(
            "model opening requests differ from the committed schedule",
        ));
    }
    let shapes = arities
        .into_iter()
        .map(|(poly, num_vars)| Ok((poly, WitnessShape::for_model(model, poly, num_vars)?)))
        .collect::<Result<_, ProofVerifyError>>()?;
    let plan = preprocessing.commitment_plan(shapes)?;

    let mut encoded = commitment_bytes.as_slice();
    let mut commitments = Vec::with_capacity(plan.groups.len());
    for group in &plan.groups {
        let (commitment, used) = decode_commitment(encoded)?;
        encoded = encoded
            .get(used..)
            .ok_or_else(|| opening_error("invalid Akita commitment length"))?;
        group.validate_commitment(&commitment)?;
        commitments.push(commitment);
    }
    if !encoded.is_empty() {
        return Err(opening_error("trailing Akita commitments"));
    }

    for check in plan.coefficient_checks()? {
        let setup = check.draw(&mut verifier.transcript);
        verify_relation(
            &check.into_verifier(setup),
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;
    }
    if plan.groups.is_empty() {
        return verifier.transcript.finish();
    }

    verifier
        .accumulator
        .prepare_for_akita_sumcheck(&plan.padding, &mut verifier.transcript)?;
    let point = verifier
        .accumulator
        .verify_batch_opening_sumcheck(&mut verifier.transcript)?;
    let state = verifier
        .accumulator
        .finalize_batch_opening_sumcheck(point, &mut verifier.transcript)?;
    let mut evals = BTreeMap::new();
    for (group, commitment) in plan.groups.iter().zip(&commitments) {
        let values = verifier.transcript.read_scalars(group.polynomials.len())?;
        let seed = opening_seed(&mut verifier.transcript);
        let AkitaMessage(bytes) = verifier.transcript.read_serializable()?;
        let proof = decode_opening_proof(&bytes)?;
        let setup = group.verifier_setup()?;
        let tag = protocol_tag(OPENING_TAG);
        let mut opening = verifier_transcript::<Keccak>(&tag, &seed, &[]);
        verify_opening(
            &setup,
            group.layout,
            commitment,
            group.point(&state.r_sumcheck)?,
            &values,
            &proof,
            &mut opening,
        )
        .map_err(opening_error)?;
        opening.check_eof().map_err(opening_error)?;
        let scale = group.zero_extension_scale(&state.r_sumcheck)?;
        evals.extend(
            group
                .polynomials
                .iter()
                .copied()
                .zip(values.into_iter().map(|value| value * scale)),
        );
    }
    let joint = state.polynomials.iter().zip(&state.poly_coeffs).try_fold(
        AkitaField::default(),
        |joint, (poly, &coeff)| {
            let eval = evals
                .get(poly)
                .ok_or_else(|| opening_error("no Akita opening for a reduced claim"))?;
            Ok(joint + coeff * *eval)
        },
    )?;
    if joint != verifier.accumulator.compute_joint_claim(&state) {
        return Err(opening_error(
            "Akita evaluations do not match the Atlas opening reduction",
        ));
    }
    verifier.transcript.finish()
}
