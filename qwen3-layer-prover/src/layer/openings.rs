use std::marker::PhantomData;

use joltworks::{
    field::JoltField, poly::commitment::commitment_scheme::CommitmentScheme,
    transcripts::Transcript,
};

use crate::{claim::Claim, error::Result};

/// PCS proof for the claims left unresolved by the layer IOP.
///
/// The IOP no longer returns tensor-shaped requests or PCS-specific request
/// structs. It returns ordinary `Claim`s. This module is the only place that is
/// allowed to inspect `claim.poly.commitment` and decide whether a claim is
/// opened by PCS or checked by direct evaluation.
#[derive(Debug, Clone)]
pub struct LayerOpeningReductionProof<F, T, PCS>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    pub(crate) _marker: PhantomData<(F, T, PCS)>,
}

pub(crate) fn prove_layer_openings<F, T, PCS>(
    claims: Vec<Claim<F>>,
    setup: &PCS::ProverSetup,
    transcript: &mut T,
) -> Result<LayerOpeningReductionProof<F, T, PCS>>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    let _ = (claims, setup, transcript);
    todo!(
        "reduce Claim values with attached commitments using the existing core opening-reduction path"
    )
}

pub(crate) fn verify_layer_openings<F, T, PCS>(
    claims: Vec<Claim<F>>,
    proof: &LayerOpeningReductionProof<F, T, PCS>,
    setup: &PCS::VerifierSetup,
    transcript: &mut T,
) -> std::result::Result<(), joltworks::utils::errors::ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    let _ = (claims, proof, setup, transcript);
    todo!("verify the same Claim-based opening reduction")
}
