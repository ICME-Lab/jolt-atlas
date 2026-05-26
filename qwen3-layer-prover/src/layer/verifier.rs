use common::CommittedPoly;
use joltworks::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{OpeningId, SumcheckId},
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

use crate::claim::{Claim, Poly};

use super::{
    claims::draw_hidden_out_point,
    commitments::{HiddenStateCommitments, absorb_layer_commitments},
    iop::verify_layer_iop,
    openings::verify_layer_openings,
    polys::LayerPolys,
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
    transcript: &mut T,
) -> std::result::Result<LayerClaims<F, PCS::Commitment>, ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    let tensors = LayerTensorIds::default();
    let commitments = proof
        .commitments
        .with_hidden_state_commitments(hidden_state_commitments.clone());
    absorb_layer_commitments(transcript, layer, shape, &commitments);
    let hidden_out_point = draw_hidden_out_point::<F, T>(transcript, shape);
    let hidden_out_value = hidden_out_opening_value(&proof.opening_reduction.opening_claims)?;
    let hidden_out = Claim::new(
        Poly::new(
            MultilinearPolynomial::from(vec![
                F::zero();
                shape.hidden_shape().padded_power_of_two().numel()
            ]),
            Some(hidden_state_commitments.hidden_out.clone()),
        ),
        hidden_out_point,
        hidden_out_value,
    );
    let polys = LayerPolys::for_verifier(
        hidden_state_commitments.hidden_in,
        &commitments,
        weights,
        shape,
        &tensors,
    );
    let claims = verify_layer_iop(
        hidden_out.clone(),
        &proof.iop_proof,
        polys,
        weights,
        shape,
        &tensors,
        transcript,
    )
    .map_err(|err| ProofVerifyError::InvalidOpeningProof(err.to_string()))?;

    verify_direct_eval_claims(&claims)?;
    verify_layer_openings::<F, T, PCS>(
        &hidden_out,
        &claims,
        &proof.opening_reduction,
        pcs_setup,
        transcript,
    )?;
    Ok(claims)
}

fn hidden_out_opening_value<F: JoltField>(
    openings: &joltworks::poly::opening_proof::Openings<F>,
) -> std::result::Result<F, ProofVerifyError> {
    let opening_id = OpeningId::new(
        CommittedPoly::QwenLayerTensor(0),
        SumcheckId::NodeExecution(0),
    );
    openings
        .get(&opening_id)
        .map(|(_, value)| *value)
        .ok_or_else(|| {
            ProofVerifyError::InvalidOpeningProof("missing hidden_out opening claim".to_string())
        })
}

fn verify_direct_eval_claims<F, C>(
    claims: &LayerClaims<F, C>,
) -> std::result::Result<(), ProofVerifyError>
where
    F: JoltField,
{
    for claim in &claims.direct_eval_claims {
        let value = claim.poly.data.evaluate(&claim.point);
        if value != claim.value {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "direct evaluation claim mismatch".to_string(),
            ));
        }
    }
    Ok(())
}
