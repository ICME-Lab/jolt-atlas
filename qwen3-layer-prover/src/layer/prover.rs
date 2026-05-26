use joltworks::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::PolynomialEvaluation,
    },
    transcripts::Transcript,
};

use crate::{
    claim::{Claim, Poly},
    error::Result,
    proof::ProveResult,
};

use super::{
    claims::claim_hidden_out_after_commitments,
    commitments::{HiddenStateCommitments, absorb_layer_commitments, commit_layer_ra_polys},
    iop::{prove_layer_iop, verify_layer_iop},
    openings::prove_layer_openings,
    polys::{LayerPolys, hidden_state_poly},
    tensors::LayerTensorIds,
    types::{LayerClaims, LayerProof, LayerShape, LayerWeights},
    witness::LayerWitness,
};

// Complete layer prover.
//
// Responsibilities:
// - accepts the boundary hidden_out claim and hidden_in polynomial
// - commits only layer-local RA/lookup polynomials
// - delegates algebraic constraints to `iop`
// - delegates PCS opening reduction to `openings`
//
// Important invariant: layer-local commitments are absorbed before the IOP
// starts. Hidden-state commitments are owned by the caller and carried by the
// boundary claim/poly.

pub fn prove_layer<F, T, PCS>(
    layer: usize,
    hidden_out: Poly<F, PCS::Commitment>,
    hidden_in: Poly<F, PCS::Commitment>,
    witness: &LayerWitness,
    weights: &LayerWeights,
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
        hidden_in: hidden_in.commitment.clone().ok_or_else(|| {
            crate::ProverError::InvalidInput("hidden_in commitment is required".to_string())
        })?,
    };

    // 1. Assemble the exact polynomials consumed by the IOP.  At this point
    // RA polys exist but have no commitment attached yet.
    let mut layer_polys =
        LayerPolys::from_witness_with_boundary(hidden_in, witness, weights, shape, &tensors);

    // 2. Commit the layer-local RA/lookup polynomials.
    let commitments = commit_layer_ra_polys::<F, PCS>(&mut layer_polys, &tensors, pcs_setup);
    let transcript_commitments =
        commitments.with_hidden_state_commitments(hidden_state_commitments);

    // 3. Bind the layer-local commitments before any layer-internal
    // challenges. The starting hidden_out claim is provided by the caller.
    absorb_layer_commitments(transcript, layer, shape, &transcript_commitments);
    let hidden_out_point =
        transcript.challenge_vector::<F>(shape.hidden_shape().padded_power_of_two().point_len());
    let hidden_out_value = hidden_out.data.evaluate(&hidden_out_point);
    let hidden_out = Claim::new(hidden_out, hidden_out_point, hidden_out_value);

    let iop = prove_layer_iop(hidden_out.clone(), layer_polys, shape, &tensors, transcript)?;

    // 4. Reduce PCS-backed claims to the opening proof. `LayerClaims` is kept
    // structured here because the field names determine which
    // CommittedPoly/SumcheckId pair each opening belongs to.
    let opening_reduction = prove_layer_openings::<F, T, PCS, PCS::Commitment>(
        &hidden_out,
        &iop.claims,
        pcs_setup,
        transcript,
    )?;

    Ok(ProveResult::new(
        iop.claims,
        LayerProof {
            commitments,
            iop_proof: iop.proof,
            opening_reduction,
        },
    ))
}

/// Smoke-test entry for the layer IOP only.
///
/// This stops before PCS commitment/opening reduction. It checks that a
/// trace-derived witness satisfies the reverse layer claim flow and all
/// op-level sumchecks.
pub fn prove_layer_iop_only_from_witness<F, T>(
    hidden_out: &[i32],
    witness: &super::witness::LayerWitness,
    weights: &LayerWeights,
    shape: &LayerShape,
    tensors: &LayerTensorIds,
    transcript: &mut T,
) -> Result<LayerClaims<F>>
where
    F: JoltField,
    T: Transcript,
{
    let layer_polys = LayerPolys::from_witness(witness, weights, shape, tensors);
    let hidden_out_claim = claim_hidden_out_after_commitments(transcript, hidden_out, shape);
    let hidden_out_claim = Claim::new(
        hidden_state_poly(hidden_out, shape),
        hidden_out_claim.point,
        hidden_out_claim.value,
    );
    Ok(prove_layer_iop(hidden_out_claim, layer_polys, shape, tensors, transcript)?.claims)
}

/// Smoke-test the layer IOP by proving and immediately verifying it.
///
/// This intentionally stays before PCS.  The verifier receives the same
/// materialized polynomials so we can test the reverse claim flow and op-level
/// sumchecks against a real trace before wiring the opening proof.
pub fn prove_and_verify_layer_iop_only_from_witness<F, T>(
    hidden_out: &[i32],
    witness: &super::witness::LayerWitness,
    weights: &LayerWeights,
    shape: &LayerShape,
    tensors: &LayerTensorIds,
    prover_transcript: &mut T,
    verifier_transcript: &mut T,
) -> Result<LayerClaims<F>>
where
    F: JoltField,
    T: Transcript,
{
    let prover_polys = LayerPolys::from_witness(witness, weights, shape, tensors);
    let prover_hidden_out =
        claim_hidden_out_after_commitments(prover_transcript, hidden_out, shape);
    let prover_hidden_out = Claim::new(
        hidden_state_poly(hidden_out, shape),
        prover_hidden_out.point,
        prover_hidden_out.value,
    );
    let proved = prove_layer_iop(
        prover_hidden_out,
        prover_polys,
        shape,
        tensors,
        prover_transcript,
    )?;

    let verifier_polys = LayerPolys::from_witness(witness, weights, shape, tensors);
    let verifier_hidden_out =
        claim_hidden_out_after_commitments(verifier_transcript, hidden_out, shape);
    let verifier_hidden_out = Claim::new(
        hidden_state_poly(hidden_out, shape),
        verifier_hidden_out.point,
        verifier_hidden_out.value,
    );
    let verified = verify_layer_iop(
        verifier_hidden_out,
        &proved.proof,
        verifier_polys,
        weights,
        shape,
        tensors,
        verifier_transcript,
    )?;

    if !same_claim(&proved.claims.hidden_in_a, &verified.hidden_in_a)
        || !same_claim(&proved.claims.hidden_in_b, &verified.hidden_in_b)
    {
        return Err(crate::ProverError::InvalidInput(
            "IOP verifier claims differ from prover claims".to_string(),
        ));
    }

    Ok(verified)
}

fn same_claim<F: JoltField, C1, C2>(lhs: &Claim<F, C1>, rhs: &Claim<F, C2>) -> bool {
    lhs.point == rhs.point && lhs.value == rhs.value
}
