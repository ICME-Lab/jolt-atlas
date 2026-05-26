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
        rlc_polynomial::build_materialized_rlc_ordered,
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};
use std::{collections::BTreeMap, time::Instant};

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

    let t0 = Instant::now();
    accumulator.prepare_for_sumcheck(poly_map);
    eprintln!(
        "timing: opening_reduction.prepare_for_sumcheck {:.3}s",
        t0.elapsed().as_secs_f64()
    );
    let t0 = Instant::now();
    let (sumcheck_proof, r_sumcheck) = accumulator.prove_batch_opening_sumcheck(transcript);
    eprintln!(
        "timing: opening_reduction.sumcheck {:.3}s",
        t0.elapsed().as_secs_f64()
    );
    let t0 = Instant::now();
    let state = accumulator.finalize_batch_opening_sumcheck(r_sumcheck, transcript);
    eprintln!(
        "timing: opening_reduction.finalize_sumcheck {:.3}s",
        t0.elapsed().as_secs_f64()
    );
    let sumcheck_claims = state.sumcheck_claims.clone();
    let ordered_polys = state
        .polynomials
        .iter()
        .map(|poly| {
            (
                *poly,
                poly_map.get(poly).expect("missing reduced opening poly"),
            )
        })
        .collect::<Vec<_>>();
    let t0 = Instant::now();
    let rlc = build_materialized_rlc_ordered(&state.gamma_powers, &ordered_polys);
    eprintln!(
        "timing: opening_reduction.build_materialized_rlc {:.3}s",
        t0.elapsed().as_secs_f64()
    );
    let t0 = Instant::now();
    let joint_opening_proof = PCS::prove(setup, &rlc, &state.r_sumcheck, None, transcript);
    eprintln!(
        "timing: opening_reduction.pcs_prove {:.3}s",
        t0.elapsed().as_secs_f64()
    );

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
