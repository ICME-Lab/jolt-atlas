//! Shared PCS opening-reduction helpers.
//!
//! This module contains the core Stage 7/8 logic used by the ONNX prover:
//! reduce many committed-polynomial openings with a batched sumcheck, build a
//! materialized random linear combination, and prove or verify one PCS opening.
//! Crates with custom IOP flows, such as the Qwen layer prover, should feed
//! their opening requests into JoltWorks' opening accumulators and call these
//! helpers instead of reimplementing the reduction.

use crate::onnx_proof::ReducedOpeningProof;
use common::CommittedPoly;
use joltworks::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator},
        rlc_polynomial::build_materialized_rlc,
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};
use std::collections::BTreeMap;

/// Prove the standard core batched opening reduction and joint PCS opening.
#[tracing::instrument(skip_all, name = "core::prove_reduced_openings")]
pub fn prove_reduced_openings<F, T, PCS>(
    accumulator: &mut ProverOpeningAccumulator<F>,
    poly_map: &BTreeMap<CommittedPoly, MultilinearPolynomial<F>>,
    setup: &PCS::ProverSetup,
    transcript: &mut T,
) -> Option<ReducedOpeningProof<F, T, PCS>>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    if poly_map.is_empty() {
        return None;
    }

    accumulator.prepare_for_sumcheck(poly_map);
    let (sumcheck_proof, r_sumcheck) = accumulator.prove_batch_opening_sumcheck(transcript);
    let state = accumulator.finalize_batch_opening_sumcheck(r_sumcheck, transcript);
    let sumcheck_claims = state.sumcheck_claims.clone();
    let rlc = build_materialized_rlc(&state.gamma_powers, poly_map);
    let joint_opening_proof = PCS::prove(setup, &rlc, &state.r_sumcheck, None, transcript);

    Some(ReducedOpeningProof {
        sumcheck_proof,
        sumcheck_claims,
        joint_opening_proof,
    })
}

/// Verify the standard core batched opening reduction and joint PCS opening.
#[tracing::instrument(skip_all, name = "core::verify_reduced_openings")]
pub fn verify_reduced_openings<F, T, PCS>(
    accumulator: &mut VerifierOpeningAccumulator<F>,
    commitments: &[PCS::Commitment],
    proof: &ReducedOpeningProof<F, T, PCS>,
    setup: &PCS::VerifierSetup,
    transcript: &mut T,
) -> Result<(), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    accumulator.prepare_for_sumcheck(&proof.sumcheck_claims);
    let r_sumcheck =
        accumulator.verify_batch_opening_sumcheck(&proof.sumcheck_proof, transcript)?;
    let state =
        accumulator.finalize_batch_opening_sumcheck(r_sumcheck, &proof.sumcheck_claims, transcript);
    let joint_commitment = PCS::combine_commitments(commitments, &state.gamma_powers);
    accumulator.verify_joint_opening::<_, PCS>(
        setup,
        &proof.joint_opening_proof,
        &joint_commitment,
        &state,
        transcript,
    )
}
