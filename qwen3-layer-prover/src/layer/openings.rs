use std::marker::PhantomData;

use common::CommittedPoly;
use joltworks::{
    field::JoltField,
    poly::{commitment::commitment_scheme::CommitmentScheme, opening_proof::SumcheckId},
    transcripts::Transcript,
};

use crate::{claim::Claim, error::Result};

use super::{
    tensors::round_site,
    types::LayerClaims,
};

/// PCS proof for the claims left unresolved by the layer IOP.
///
/// The layer IOP returns structured `LayerClaims`. We keep that structure until
/// this module so the field name can determine the corresponding
/// `CommittedPoly` and `SumcheckId`; `Claim` itself stays a pure polynomial
/// evaluation claim.
#[derive(Debug, Clone)]
pub struct LayerOpeningReductionProof<F, T, PCS>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    pub(crate) _marker: PhantomData<(F, T, PCS)>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct LayerOpeningClaim<F: JoltField, C> {
    claim: Claim<F, C>,
    committed_poly: CommittedPoly,
    sumcheck: SumcheckId,
}

pub(crate) fn prove_layer_openings<F, T, PCS, C>(
    claims: &LayerClaims<F, C>,
    setup: &PCS::ProverSetup,
    transcript: &mut T,
) -> Result<LayerOpeningReductionProof<F, T, PCS>>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
    C: Clone,
{
    let opening_claims = layer_opening_claims(claims);
    let _ = (opening_claims, setup, transcript);
    todo!(
        "reduce Claim values with attached commitments using the existing core opening-reduction path"
    )
}

pub(crate) fn verify_layer_openings<F, T, PCS, C>(
    hidden_out: &Claim<F, C>,
    claims: Option<&LayerClaims<F, C>>,
    proof: &LayerOpeningReductionProof<F, T, PCS>,
    setup: &PCS::VerifierSetup,
    transcript: &mut T,
) -> std::result::Result<(), joltworks::utils::errors::ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
    C: Clone,
{
    let opening_claims = claims.map(layer_opening_claims);
    let _ = (hidden_out, opening_claims, proof, setup, transcript);
    todo!("verify the same Claim-based opening reduction")
}

fn layer_opening_claims<F: JoltField, C: Clone>(
    claims: &LayerClaims<F, C>,
) -> Vec<LayerOpeningClaim<F, C>> {
    let mut out = Vec::new();

    push_round_claims(
        &mut out,
        &claims.down_proj_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::DOWN_PROJ, d),
    );
    push_round_claims(
        &mut out,
        &claims.silu_up_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::SILU_UP, d),
    );
    push_round_claims(
        &mut out,
        &claims.silu_gate_round_ra,
        CommittedPoly::QwenSiluRoundRaD,
    );
    push_lookup_claims(&mut out, &claims.silu_ra, |d| {
        let split = claims.silu_ra.len() / 2;
        if d < split {
            CommittedPoly::QwenSiluBaseRaD(d)
        } else {
            CommittedPoly::QwenSiluSlopeRaD(d - split)
        }
    });
    push_round_claims(
        &mut out,
        &claims.silu_round_ra,
        CommittedPoly::QwenSiluOutputRoundRaD,
    );
    push_round_claims(
        &mut out,
        &claims.gate_proj_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::GATE_PROJ, d),
    );
    push_round_claims(
        &mut out,
        &claims.up_proj_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::UP_PROJ, d),
    );
    push_round_claims(
        &mut out,
        &claims.rms_norm_mlp_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::RMS_NORM_MLP, d),
    );
    push_round_claims(
        &mut out,
        &claims.rms_norm_mlp_norm_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::RMS_NORM_MLP_INTERNAL, d),
    );
    push_round_claims(
        &mut out,
        &claims.o_proj_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::O_PROJ, d),
    );
    push_round_claims(
        &mut out,
        &claims.pv_matmul_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::CONTEXT, d),
    );
    push_round_claims(
        &mut out,
        &claims.softmax_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::SOFTMAX_OUTPUT, d),
    );
    push_round_claims(
        &mut out,
        &claims.softmax_floor_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::SOFTMAX_FLOOR, d),
    );
    push_round_claims(
        &mut out,
        &claims.softmax_exp_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::SOFTMAX_EXP, d),
    );
    push_lookup_claims(
        &mut out,
        &claims.softmax_input_frac_ra,
        CommittedPoly::QwenSoftmaxInputFracRaD,
    );
    push_lookup_claims(
        &mut out,
        &claims.softmax_ra,
        CommittedPoly::QwenSoftmaxExpRaD,
    );
    push_round_claims(
        &mut out,
        &claims.qk_score_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::QK_SCORE_SCALE, d),
    );
    push_round_claims(
        &mut out,
        &claims.qk_score_dot_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::QK_SCORE_DOT, d),
    );
    push_round_claims(
        &mut out,
        &claims.q_rope_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::Q_ROPE, d),
    );
    push_round_claims(
        &mut out,
        &claims.k_rope_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::K_ROPE, d),
    );
    push_round_claims(
        &mut out,
        &claims.q_norm_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::Q_NORM, d),
    );
    push_round_claims(
        &mut out,
        &claims.q_norm_norm_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::Q_NORM_INTERNAL, d),
    );
    push_round_claims(
        &mut out,
        &claims.k_norm_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::K_NORM, d),
    );
    push_round_claims(
        &mut out,
        &claims.k_norm_norm_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::K_NORM_INTERNAL, d),
    );
    push_round_claims(
        &mut out,
        &claims.q_proj_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::Q_PROJ, d),
    );
    push_round_claims(
        &mut out,
        &claims.k_proj_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::K_PROJ, d),
    );
    push_round_claims(
        &mut out,
        &claims.v_proj_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::V_PROJ, d),
    );
    push_round_claims(
        &mut out,
        &claims.rms_norm_atten_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::RMS_NORM_ATTEN, d),
    );
    push_round_claims(
        &mut out,
        &claims.rms_norm_atten_norm_round_ra,
        |d| CommittedPoly::QwenRoundRaD(round_site::RMS_NORM_ATTEN_INTERNAL, d),
    );

    out
}

fn push_round_claims<F, C>(
    out: &mut Vec<LayerOpeningClaim<F, C>>,
    claims: &[Claim<F, C>],
    committed_poly: impl Fn(usize) -> CommittedPoly,
) where
    F: JoltField,
    C: Clone,
{
    push_shout_claims(out, claims, committed_poly);
}

fn push_lookup_claims<F, C>(
    out: &mut Vec<LayerOpeningClaim<F, C>>,
    claims: &[Claim<F, C>],
    committed_poly: impl Fn(usize) -> CommittedPoly,
) where
    F: JoltField,
    C: Clone,
{
    push_shout_claims(out, claims, committed_poly);
}

fn push_shout_claims<F, C>(
    out: &mut Vec<LayerOpeningClaim<F, C>>,
    claims: &[Claim<F, C>],
    committed_poly: impl Fn(usize) -> CommittedPoly,
) where
    F: JoltField,
    C: Clone,
{
    if claims.is_empty() {
        return;
    }

    let chunk_count = infer_shout_chunk_count(claims.len());
    let sumchecks = match claims.len() / chunk_count {
        1 => [SumcheckId::HammingWeight].as_slice(),
        3 => [
            SumcheckId::RaVirtualization,
            SumcheckId::HammingWeight,
            SumcheckId::Booleanity,
        ]
        .as_slice(),
        _ => panic!("unexpected SHOUT opening claim count: {}", claims.len()),
    };

    for (group, sumcheck) in sumchecks.iter().enumerate() {
        for d in 0..chunk_count {
            let claim = claims[group * chunk_count + d].clone();
            out.push(LayerOpeningClaim {
                claim,
                committed_poly: committed_poly(d),
                sumcheck: *sumcheck,
            });
        }
    }
}

fn infer_shout_chunk_count(claim_count: usize) -> usize {
    if claim_count % 3 == 0 {
        claim_count / 3
    } else {
        claim_count
    }
}
