use std::path::Path;

use joltworks::{
    field::JoltField, poly::commitment::commitment_scheme::CommitmentScheme,
    transcripts::Transcript,
};

use crate::{claim::Claim, error::Result, proof::ProveResult};

use super::{
    claims::claim_hidden_out_after_commitments,
    commitments::{HiddenStateCommitments, absorb_layer_commitments, commit_layer_witness_polys},
    iop::prove_layer_iop,
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
    let traced = build_layer_witness_from_trace_dir(trace_dir, layer, weights, shape)?;
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
