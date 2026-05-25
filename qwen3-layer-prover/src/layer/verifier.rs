use joltworks::{
    field::JoltField, poly::commitment::commitment_scheme::CommitmentScheme,
    transcripts::Transcript, utils::errors::ProofVerifyError,
};

use super::{
    claims::{draw_hidden_out_point, point_matches_claim},
    commitments::{HiddenStateCommitments, absorb_layer_commitments},
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
    let _ = (pcs_setup, weights, tensors);
    verify_layer_openings::<F, T, PCS>(
        vec![proof.hidden_out.clone()],
        &proof.opening_reduction,
        pcs_setup,
        transcript,
    )?;
    todo!("wire verifier after the Claim + Poly layer IOP migration is complete")
}
