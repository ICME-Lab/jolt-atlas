use common::CommittedPoly;
use joltworks::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::{
            BIG_ENDIAN, OpeningId, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator,
        },
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};
use std::collections::BTreeMap;

use super::commitments::{LayerCommitments, LayerPolynomialMap};
use crate::{Claim, CommittedOpeningClaim, ProverError};

// PCS openings for one layer. This is intentionally a thin adapter:
// layer ops emit opening claims, then this module feeds them into the shared
// core opening-reduction implementation.

#[derive(Debug, Clone)]
pub struct LayerOpeningReductionProof<F: JoltField, T: Transcript, PCS: CommitmentScheme<Field = F>>
{
    // Opening requests are exactly the Claim values returned by prove_xxx.
    // Individual op provers stay pure: they do not see a PCS accumulator and
    // they do not mutate layer-level opening state. The outer layer wrapper is
    // responsible for reducing these requests and producing a PCS proof.
    pub opening_claims: Vec<crate::Claim<F>>,
    pub committed_opening_claims: Vec<CommittedOpeningClaim<F>>,
    pub reduced_opening: jolt_atlas_core::onnx_proof::ReducedOpeningProof<F, T, PCS>,
}
pub fn prove_layer_openings<F, T, PCS>(
    polynomials: &LayerPolynomialMap<F>,
    opening_claims: Vec<crate::Claim<F>>,
    committed_opening_claims: Vec<CommittedOpeningClaim<F>>,
    setup: &PCS::ProverSetup,
    transcript: &mut T,
) -> crate::error::Result<LayerOpeningReductionProof<F, T, PCS>>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    let mut accumulator = ProverOpeningAccumulator::new();
    append_claims_to_prover_accumulator(
        polynomials,
        &opening_claims,
        &mut accumulator,
        transcript,
    )?;
    append_committed_claims_to_prover_accumulator(
        polynomials,
        &committed_opening_claims,
        &mut accumulator,
        transcript,
    )?;
    let reduced_opening = jolt_atlas_core::opening_reduction::prove_reduced_openings::<F, T, PCS>(
        &mut accumulator,
        &polynomials.core_poly_map_for_openings(&opening_claims, &committed_opening_claims)?,
        setup,
        transcript,
    )
    .ok_or(crate::ProverError::MissingOpening)?;
    Ok(LayerOpeningReductionProof {
        opening_claims,
        committed_opening_claims,
        reduced_opening,
    })
}

pub(crate) fn prove_layer_claim_openings<F, T, PCS>(
    polynomials: &LayerPolynomialMap<F>,
    mut opening_claims: Vec<Claim<F>>,
    committed_opening_claims: Vec<CommittedOpeningClaim<F>>,
    hidden_out: Claim<F>,
    setup: &PCS::ProverSetup,
    transcript: &mut T,
) -> crate::error::Result<LayerOpeningReductionProof<F, T, PCS>>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    // `hidden_out` is the starting claim of the reverse-flow IOP, so it is not
    // returned by `LayerClaims::opening_claims()`.  It still has to be opened
    // against the externally supplied hidden_out commitment.
    opening_claims.push(hidden_out);
    let missing = polynomials.missing_opening_claims(&opening_claims);
    if !missing.is_empty() {
        return Err(ProverError::MissingCommittedPolynomials(missing));
    }
    prove_layer_openings::<F, T, PCS>(
        polynomials,
        opening_claims,
        committed_opening_claims,
        setup,
        transcript,
    )
}

pub fn verify_layer_openings<F, T, PCS>(
    commitments: &LayerCommitments<PCS::Commitment>,
    proof: &LayerOpeningReductionProof<F, T, PCS>,
    setup: &PCS::VerifierSetup,
    transcript: &mut T,
) -> std::result::Result<(), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    let mut accumulator = VerifierOpeningAccumulator::new();
    append_claims_to_verifier_accumulator(
        commitments,
        &proof.opening_claims,
        &mut accumulator,
        transcript,
    )?;
    append_committed_claims_to_verifier_accumulator(
        &proof.committed_opening_claims,
        &mut accumulator,
        transcript,
    )?;
    let core_commitments = commitments_as_core_vec(
        commitments,
        &proof.opening_claims,
        &proof.committed_opening_claims,
    )?;
    jolt_atlas_core::opening_reduction::verify_reduced_openings::<F, T, PCS>(
        &mut accumulator,
        &core_commitments,
        &proof.reduced_opening,
        setup,
        transcript,
    )
}

fn append_committed_claims_to_prover_accumulator<F, T>(
    polynomials: &LayerPolynomialMap<F>,
    claims: &[CommittedOpeningClaim<F>],
    accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
) -> crate::error::Result<()>
where
    F: JoltField,
    T: Transcript,
{
    for claim in claims {
        if polynomials.entry_for_committed_poly(claim.poly).is_none() {
            return Err(crate::ProverError::MissingCommittedPolynomials(vec![
                format!("{:?}", claim.poly),
            ]));
        }
        if claim.sparse {
            let log_k = polynomials.one_hot_log_k(claim.poly).ok_or_else(|| {
                crate::ProverError::MissingCommittedPolynomials(vec![format!("{:?}", claim.poly)])
            })?;
            let (r_address, r_cycle) = claim.point.split_at(log_k);
            accumulator.append_sparse(
                transcript,
                vec![claim.poly],
                claim.sumcheck,
                r_address.to_vec(),
                r_cycle.to_vec(),
                vec![claim.value],
            );
        } else {
            accumulator.append_dense(
                transcript,
                claim.opening_id(),
                claim.point.clone(),
                claim.value,
            );
        }
    }
    Ok(())
}

fn append_committed_claims_to_verifier_accumulator<F, T>(
    claims: &[CommittedOpeningClaim<F>],
    accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut T,
) -> std::result::Result<(), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    for claim in claims {
        let opening_id = claim.opening_id();
        accumulator.openings.insert(
            opening_id,
            (
                OpeningPoint::<BIG_ENDIAN, F>::new(claim.point.clone()),
                claim.value,
            ),
        );
        if claim.sparse {
            let log_k = sparse_log_k(&claim.poly).ok_or_else(|| {
                ProofVerifyError::InvalidOpeningProof(format!(
                    "missing sparse log_k for {:?}",
                    claim.poly
                ))
            })?;
            let (_r_address, _r_cycle) = claim.point.split_at(log_k);
            accumulator.append_sparse(
                transcript,
                vec![claim.poly],
                claim.sumcheck,
                claim.point.clone(),
            );
        } else {
            accumulator.append_dense(transcript, opening_id, claim.point.clone());
        }
    }
    Ok(())
}

fn append_claims_to_prover_accumulator<F, T>(
    polynomials: &LayerPolynomialMap<F>,
    claims: &[crate::Claim<F>],
    accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
) -> crate::error::Result<()>
where
    F: JoltField,
    T: Transcript,
{
    for (idx, claim) in claims.iter().enumerate() {
        let Some(poly) = polynomials.committed_poly_for_tensor(&claim.tensor.0) else {
            return Err(crate::ProverError::MissingCommittedPolynomials(vec![
                claim.tensor.0.clone(),
            ]));
        };
        let Some(entry) = polynomials
            .entries
            .iter()
            .find(|entry| entry.committed_poly == poly)
        else {
            return Err(crate::ProverError::MissingCommittedPolynomials(vec![
                format!("{poly:?}"),
            ]));
        };
        let expected = claim.domain_shape.numel();
        let actual = entry.poly.len();
        if actual != expected {
            return Err(crate::ProverError::CommittedPolynomialDomainMismatch {
                tensor: claim.tensor.0.clone(),
                domain_shape: claim.domain_shape.0.clone(),
                expected,
                actual,
            });
        }
        let sumcheck = SumcheckId::NodeExecution(idx);
        match &entry.poly {
            MultilinearPolynomial::OneHot(one_hot) => {
                let log_k = one_hot.K.trailing_zeros() as usize;
                let (r_address, r_cycle) = claim.point.split_at(log_k);
                accumulator.append_sparse(
                    transcript,
                    vec![poly],
                    sumcheck,
                    r_address.to_vec(),
                    r_cycle.to_vec(),
                    vec![claim.value],
                );
            }
            _ => {
                accumulator.append_dense(
                    transcript,
                    OpeningId::new(poly, sumcheck),
                    claim.point.clone(),
                    claim.value,
                );
            }
        }
    }
    Ok(())
}

fn append_claims_to_verifier_accumulator<F, T>(
    commitments: &LayerCommitments<impl Clone>,
    claims: &[crate::Claim<F>],
    accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut T,
) -> std::result::Result<(), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    for (idx, claim) in claims.iter().enumerate() {
        let Some(poly_idx) = commitments
            .entries
            .iter()
            .position(|entry| entry.name == claim.tensor.0)
        else {
            return Err(ProofVerifyError::InvalidOpeningProof(format!(
                "missing commitment for tensor {}",
                claim.tensor.0
            )));
        };
        let poly = commitments.entries[poly_idx].committed_poly;
        let sumcheck = SumcheckId::NodeExecution(idx);
        let opening_id = OpeningId::new(poly, sumcheck);
        accumulator
            .openings
            .insert(opening_id, (OpeningPoint::default(), claim.value));
        accumulator.append_dense(transcript, opening_id, claim.point.clone());
    }
    Ok(())
}

fn sparse_log_k(poly: &CommittedPoly) -> Option<usize> {
    match poly {
        CommittedPoly::QwenRoundRaD(_, _)
        | CommittedPoly::QwenSiluBaseRaD(_)
        | CommittedPoly::QwenSiluSlopeRaD(_)
        | CommittedPoly::QwenSiluRoundRaD(_)
        | CommittedPoly::QwenSoftmaxExpRaD(_)
        | CommittedPoly::QwenSoftmaxInputRemainderRaD(_) => {
            Some(joltworks::config::OneHotConfig::default().log_k_chunk as usize)
        }
        _ => None,
    }
}

fn commitments_as_core_vec<F: JoltField, C: Clone>(
    commitments: &LayerCommitments<C>,
    claims: &[crate::Claim<F>],
    committed_claims: &[CommittedOpeningClaim<F>],
) -> std::result::Result<Vec<C>, ProofVerifyError> {
    let mut out = BTreeMap::new();
    for (idx, claim) in claims.iter().enumerate() {
        let Some(poly_idx) = commitments
            .entries
            .iter()
            .position(|entry| entry.name == claim.tensor.0)
        else {
            return Err(ProofVerifyError::InvalidOpeningProof(format!(
                "missing commitment for tensor {}",
                claim.tensor.0
            )));
        };
        let poly = commitments.entries[poly_idx].committed_poly;
        out.insert(
            OpeningId::new(poly, SumcheckId::NodeExecution(idx)),
            commitments.entries[poly_idx].commitment.clone(),
        );
    }
    for claim in committed_claims {
        let Some(entry) = commitments
            .entries
            .iter()
            .find(|entry| entry.committed_poly == claim.poly)
        else {
            return Err(ProofVerifyError::InvalidOpeningProof(format!(
                "missing commitment for {:?}",
                claim.poly
            )));
        };
        out.insert(claim.opening_id(), entry.commitment.clone());
    }
    Ok(out.into_values().collect())
}
