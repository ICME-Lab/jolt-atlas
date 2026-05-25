use std::path::Path;

use joltworks::{
    field::JoltField, poly::commitment::commitment_scheme::CommitmentScheme,
    transcripts::Transcript,
};

use crate::{claim::Claim, error::Result, proof::ProveResult};

use super::{
    claims::claim_hidden_out_after_commitments,
    commitments::{HiddenStateCommitments, absorb_layer_commitments, commit_layer_witness_polys},
    iop::{prove_layer_iop, verify_layer_iop},
    openings::prove_layer_openings,
    polys::LayerPolys,
    tensors::LayerTensorIds,
    types::{LayerClaims, LayerProof, LayerShape, LayerWeights},
    witness::build_layer_witness_from_trace_dir,
};

// Complete layer prover.
//
// Responsibilities:
// - owns trace -> witness materialization
// - owns polynomial commitments and transcript binding
// - delegates algebraic constraints to `iop`
// - delegates PCS opening reduction to `openings`
//
// Important invariant: commitments are absorbed before the IOP starts.

pub fn prove_layer<F, T, PCS>(
    trace_dir: impl AsRef<Path>,
    layer: usize,
    hidden_state_commitments: HiddenStateCommitments<PCS::Commitment>,
    weights: &LayerWeights,
    shape: &LayerShape,
    tensors: &LayerTensorIds,
    pcs_setup: &PCS::ProverSetup,
    transcript: &mut T,
) -> Result<ProveResult<LayerClaims<F>, LayerProof<F, T, PCS>>>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    // 1. Materialize exactly one layer.  This function does not reason about
    // neighbouring layers; hidden_in/hidden_out are supplied as commitments.
    let traced = build_layer_witness_from_trace_dir(trace_dir, layer, weights, shape)
        .map_err(|err| crate::ProverError::InvalidInput(err.to_string()))?;
    let layer_polys =
        LayerPolys::from_witness(&traced.hidden_out, &traced.witness, weights, shape, tensors);

    // 2. Build the layer polynomial set and attach commitments.  The two
    // hidden-state polynomials use the commitments supplied by the caller.
    let committed = commit_layer_witness_polys::<F, PCS>(
        &traced.hidden_out,
        &traced.witness,
        hidden_state_commitments,
        weights,
        shape,
        tensors,
        pcs_setup,
    );

    // 3. Bind commitments before sampling the first evaluation point.  The
    // hidden_out claim is derived here; it is not caller-controlled.
    absorb_layer_commitments(transcript, layer, shape, &committed.commitments);
    let hidden_out = claim_hidden_out_after_commitments(transcript, &traced.hidden_out, shape);

    // 4. Prove the layer equations.  IOP code returns claims to be opened, but
    // does not know about PCS commitment state.
    let hidden_out = Claim::new(
        layer_polys.hidden_out.clone(),
        hidden_out.point,
        hidden_out.value,
    );

    let iop = prove_layer_iop(hidden_out.clone(), layer_polys, shape, tensors, transcript)?;

    // 5. Reduce all requested openings to the PCS proof for this layer.
    let mut opening_claims = iop.claims.boundary_claims();
    opening_claims.extend(iop.claims.direct_eval_claims.clone());
    opening_claims.extend(iop.claims.pcs_claims());
    let opening_reduction =
        prove_layer_openings::<F, T, PCS>(opening_claims, pcs_setup, transcript)?;

    Ok(ProveResult::new(
        iop.claims,
        LayerProof {
            hidden_out,
            commitments: committed.commitments,
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
    let layer_polys = LayerPolys::from_witness(hidden_out, witness, weights, shape, tensors);
    let hidden_out_claim = claim_hidden_out_after_commitments(transcript, hidden_out, shape);
    let hidden_out_claim = Claim::new(
        layer_polys.hidden_out.clone(),
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
    let prover_polys = LayerPolys::from_witness(hidden_out, witness, weights, shape, tensors);
    let prover_hidden_out = claim_hidden_out_after_commitments(prover_transcript, hidden_out, shape);
    let prover_hidden_out = Claim::new(
        prover_polys.hidden_out.clone(),
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

    let verifier_polys = LayerPolys::from_witness(hidden_out, witness, weights, shape, tensors);
    let verifier_hidden_out =
        claim_hidden_out_after_commitments(verifier_transcript, hidden_out, shape);
    let verifier_hidden_out = Claim::new(
        verifier_polys.hidden_out.clone(),
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
