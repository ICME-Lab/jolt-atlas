use joltworks::{
    field::JoltField, poly::commitment::commitment_scheme::CommitmentScheme,
    transcripts::Transcript, utils::errors::ProofVerifyError,
};

use super::{
    claims::{draw_hidden_out_point, point_matches_claim},
    commitments::{HiddenStateCommitments, absorb_layer_commitments},
    iop::verify_layer_iop,
    openings::verify_layer_openings,
    tensors::LayerTensorIds,
    types::{LayerClaims, LayerProof, LayerShape, LayerWeights},
};

// Complete layer verifier. This mirrors `prover.rs`: bind the same context,
// verify the IOP, then verify the PCS opening reduction.

pub fn verify_layer<F, T, PCS>(
    layer: usize,
    proof: &LayerProof<F, T, PCS>,
    hidden_state_commitments: HiddenStateCommitments<PCS::Commitment>,
    pcs_setup: &PCS::VerifierSetup,
    weights: &LayerWeights,
    shape: &LayerShape,
    tensors: &LayerTensorIds,
    transcript: &mut T,
) -> std::result::Result<LayerClaims<F>, ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    let commitments = proof
        .commitments
        .with_hidden_state_commitments(hidden_state_commitments);
    absorb_layer_commitments(transcript, layer, shape, &commitments);
    let hidden_out_point = draw_hidden_out_point::<F, T>(transcript, shape);
    if !point_matches_claim(&proof.hidden_out, &hidden_out_point) {
        return Err(ProofVerifyError::InvalidOpeningProof(
            "hidden_out claim point is not transcript-derived".to_string(),
        ));
    }
    let hidden_out = proof.hidden_out.clone();
    let hidden_out_for_opening = hidden_out.clone();
    let claims = verify_layer_iop(
        hidden_out,
        &proof.iop_proof,
        weights,
        shape,
        tensors,
        transcript,
    )?;
    let mut expected_openings = claims.tensor_opening_requests();
    expected_openings.push(hidden_out_for_opening);
    if expected_openings != proof.opening_reduction.tensor_opening_requests {
        return Err(ProofVerifyError::InvalidOpeningProof(
            "layer opening claims do not match verified IOP claims".to_string(),
        ));
    }
    let expected_pcs_requests = proof.iop_proof.pcs_opening_requests();
    if expected_pcs_requests != proof.opening_reduction.pcs_opening_requests {
        return Err(ProofVerifyError::InvalidOpeningProof(
            "layer PCS opening requests do not match verified IOP requests".to_string(),
        ));
    }
    verify_layer_openings::<F, T, PCS>(
        &commitments,
        &proof.opening_reduction,
        pcs_setup,
        transcript,
    )?;
    Ok(claims)
}
