use joltworks::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::PolynomialEvaluation,
    },
    transcripts::Transcript,
};
use std::time::Instant;

use crate::{
    claim::{Claim, Poly},
    error::Result,
    proof::ProveResult,
};

use super::{
    commitments::{HiddenStateCommitments, LayerCommitments, absorb_layer_commitments},
    iop::prove_layer_iop,
    openings::prove_layer_openings,
    polys::LayerPolys,
    tensors::LayerTensorIds,
    types::{LayerClaims, LayerIopRoundStat, LayerProof, LayerShape},
};

// Complete layer prover.
//
// Responsibilities:
// - accepts the committed hidden_out polynomial and committed layer polynomials
// - binds caller-supplied hidden-state and RA/lookup commitments
// - delegates algebraic constraints to `iop`
// - delegates PCS opening reduction to `openings`
//
// Important invariant: layer-local commitments are absorbed before the IOP
// starts. Hidden-state commitments are owned by the caller and carried by the
// boundary claim/poly.

pub fn prove_layer<F, T, PCS>(
    layer: usize,
    hidden_out: Poly<F, PCS::Commitment>,
    layer_polys: LayerPolys<F, PCS::Commitment>,
    commitments: LayerCommitments<PCS::Commitment>,
    shape: &LayerShape,
    pcs_setup: &PCS::ProverSetup,
    transcript: &mut T,
) -> Result<ProveResult<LayerClaims<F, PCS::Commitment>, LayerProof<F, T, PCS>>>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    let tensors = LayerTensorIds::default();
    let hidden_state_commitments = HiddenStateCommitments {
        hidden_out: hidden_out.commitment.clone().ok_or_else(|| {
            crate::ProverError::InvalidInput("hidden_out commitment is required".to_string())
        })?,
        hidden_in: layer_polys.hidden_in.commitment.clone().ok_or_else(|| {
            crate::ProverError::InvalidInput("hidden_in commitment is required".to_string())
        })?,
    };
    let transcript_commitments =
        commitments.with_hidden_state_commitments(hidden_state_commitments);

    // Bind the layer-local commitments before any layer-internal challenges.
    // The starting hidden_out evaluation point is derived only after this.
    absorb_layer_commitments(transcript, layer, shape, &transcript_commitments);
    let hidden_out_point =
        transcript.challenge_vector::<F>(shape.hidden_shape().padded_power_of_two().point_len());
    let hidden_out_value = hidden_out.data.evaluate(&hidden_out_point);
    let hidden_out = Claim::new(hidden_out, hidden_out_point, hidden_out_value);

    let t0 = Instant::now();
    let iop = prove_layer_iop(hidden_out.clone(), layer_polys, shape, &tensors, transcript)?;
    eprintln!("timing: prove_layer.iop {:.3}s", t0.elapsed().as_secs_f64());

    // 4. Reduce PCS-backed claims to the opening proof. `LayerClaims` is kept
    // structured here because the field names determine which
    // CommittedPoly/SumcheckId pair each opening belongs to.
    let t0 = Instant::now();
    let opening_reduction = prove_layer_openings::<F, T, PCS, PCS::Commitment>(
        &hidden_out,
        &iop.claims,
        pcs_setup,
        transcript,
    )?;
    eprintln!(
        "timing: prove_layer.openings {:.3}s",
        t0.elapsed().as_secs_f64()
    );

    Ok(ProveResult::new(
        iop.claims,
        LayerProof {
            commitments,
            iop_proof: iop.proof,
            opening_reduction,
        },
    ))
}

pub fn prove_layer_iop_round_stats<F, T, C>(
    hidden_out: Poly<F, C>,
    layer_polys: LayerPolys<F, C>,
    shape: &LayerShape,
    transcript: &mut T,
) -> Result<Vec<LayerIopRoundStat>>
where
    F: JoltField,
    T: Transcript,
    C: Clone,
{
    let tensors = LayerTensorIds::default();
    let hidden_out_point =
        transcript.challenge_vector::<F>(shape.hidden_shape().padded_power_of_two().point_len());
    let hidden_out_value = hidden_out.data.evaluate(&hidden_out_point);
    let hidden_out = Claim::new(hidden_out, hidden_out_point, hidden_out_value);
    let iop = prove_layer_iop(hidden_out, layer_polys, shape, &tensors, transcript)?;
    Ok(iop.proof.round_stats())
}
