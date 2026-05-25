use common::CommittedPoly;
use joltworks::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::{
            OpeningId, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator,
        },
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};
use std::collections::BTreeMap;

use super::commitments::{LayerCommitments, LayerPolynomialMap};
use crate::{Claim, ProverError};

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
    pub reduced_opening: jolt_atlas_core::onnx_proof::ReducedOpeningProof<F, T, PCS>,
}
pub fn prove_layer_openings<F, T, PCS>(
    polynomials: &LayerPolynomialMap<F>,
    opening_claims: Vec<crate::Claim<F>>,
    setup: &PCS::ProverSetup,
    transcript: &mut T,
) -> crate::error::Result<LayerOpeningReductionProof<F, T, PCS>>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    let poly_map = polynomials.core_poly_map_for_claims(&opening_claims)?;
    let mut accumulator = ProverOpeningAccumulator::new();
    append_claims_to_prover_accumulator(
        polynomials,
        &opening_claims,
        &mut accumulator,
        transcript,
    )?;
    let reduced_opening = jolt_atlas_core::opening_reduction::prove_reduced_openings::<F, T, PCS>(
        &mut accumulator,
        &poly_map,
        setup,
        transcript,
    )
    .ok_or(crate::ProverError::MissingOpening)?;
    Ok(LayerOpeningReductionProof {
        opening_claims,
        reduced_opening,
    })
}

pub(crate) fn prove_layer_claim_openings<F, T, PCS>(
    polynomials: &LayerPolynomialMap<F>,
    mut opening_claims: Vec<Claim<F>>,
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
    prove_layer_openings::<F, T, PCS>(polynomials, opening_claims, setup, transcript)
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
    let core_commitments = commitments_as_core_vec(commitments, &proof.opening_claims)?;
    jolt_atlas_core::opening_reduction::verify_reduced_openings::<F, T, PCS>(
        &mut accumulator,
        &core_commitments,
        &proof.reduced_opening,
        setup,
        transcript,
    )
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
        let entry = &polynomials.entries[match poly {
            CommittedPoly::QwenLayerTensor(idx) => idx,
            _ => unreachable!("qwen layer openings only use QwenLayerTensor ids"),
        }];
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
        let poly = CommittedPoly::QwenLayerTensor(poly_idx);
        let sumcheck = SumcheckId::NodeExecution(idx);
        let opening_id = OpeningId::new(poly, sumcheck);
        accumulator
            .openings
            .insert(opening_id, (OpeningPoint::default(), claim.value));
        if is_one_hot_tensor_name(&claim.tensor.0) {
            accumulator.append_sparse(transcript, vec![poly], sumcheck, claim.point.clone());
        } else {
            accumulator.append_dense(transcript, opening_id, claim.point.clone());
        }
    }
    Ok(())
}

fn is_one_hot_tensor_name(name: &str) -> bool {
    name.ends_with("_round_ra")
        || matches!(
            name,
            "softmax_ra" | "softmax_input_remainder_ra" | "silu_ra"
        )
}

fn commitments_as_core_vec<F: JoltField, C: Clone>(
    commitments: &LayerCommitments<C>,
    claims: &[crate::Claim<F>],
) -> std::result::Result<Vec<C>, ProofVerifyError> {
    let mut out = BTreeMap::new();
    for claim in claims {
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
        out.insert(
            CommittedPoly::QwenLayerTensor(poly_idx),
            commitments.entries[poly_idx].commitment.clone(),
        );
    }
    Ok(out.into_values().collect())
}
