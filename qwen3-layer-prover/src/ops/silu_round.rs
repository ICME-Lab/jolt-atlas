use joltworks::{field::JoltField, transcripts::Transcript, utils::errors::ProofVerifyError};

use crate::{
    claim::Claim,
    error::Result,
    ops::{
        round::{
            ROUND_FRAC_BITS, RoundParams, RoundProof, RoundWitness, prove_round, verify_round,
        },
        silu::{SiluParams, SiluProof, SiluWitness, prove_silu, verify_silu},
    },
};

// Design note for future us:
//
// The MLP path uses:
//     silu = SiLU(gate_proj, frac_bits, ra)
//
// Keep this internally as two simple protocols, `round` then `silu`, but expose
// it as one op to the layer prover so the layer graph does not have to carry the
// intermediate `silu` accumulator node explicitly.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SiluRoundParams {
    pub round: RoundParams,
    pub silu: SiluParams,
}

impl SiluRoundParams {
    pub fn new(round: RoundParams, silu: SiluParams) -> Self {
        Self { round, silu }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SiluRoundWitness {
    pub min_n: i64,
    pub max_n: i64,
    pub gate_proj: Vec<i32>,
    pub silu_acc: Vec<i64>,
    pub silu_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    pub silu_ra: Vec<u8>,
    pub silu: Vec<i32>,
    pub silu_out_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
}

#[derive(Debug, Clone)]
pub struct SiluRoundProof<F: JoltField, T: Transcript> {
    pub round: RoundProof<F, T>,
    pub silu: SiluProof<F, T>,
}

pub fn prove_silu_round<F, T>(
    silu_round_claim: Claim<F>,
    witness: &SiluRoundWitness,
    params: &SiluRoundParams,
    transcript: &mut T,
) -> Result<(
    SiluRoundProof<F, T>,
    Claim<F>,
    [Claim<F>; ROUND_FRAC_BITS],
    Claim<F>,
    [Claim<F>; ROUND_FRAC_BITS],
)>
where
    F: JoltField,
    T: Transcript,
{
    let round_witness = RoundWitness {
        input: witness.silu_acc.clone(),
        output: witness.silu.clone(),
        frac_bits: witness.silu_out_frac_bits.clone(),
    };
    let (round_proof, silu_claim, silu_round_frac_bits) = prove_round(
        vec![silu_round_claim],
        &round_witness,
        &params.round,
        transcript,
    )?;

    let silu_witness = SiluWitness {
        min_n: witness.min_n,
        max_n: witness.max_n,
        gate_proj_round: witness.gate_proj.clone(),
        frac_bits: witness.silu_frac_bits.clone(),
        ra: witness.silu_ra.clone(),
        output: witness.silu_acc.clone(),
    };
    let (silu_proof, gate_proj_round, silu_frac_bits, silu_ra) =
        prove_silu(vec![silu_claim], &silu_witness, &params.silu, transcript)?;

    Ok((
        SiluRoundProof {
            round: round_proof,
            silu: silu_proof,
        },
        gate_proj_round,
        silu_frac_bits,
        silu_ra,
        silu_round_frac_bits,
    ))
}

pub fn verify_silu_round<F, T>(
    silu_round_claim: Claim<F>,
    proof: &SiluRoundProof<F, T>,
    params: &SiluRoundParams,
    transcript: &mut T,
) -> std::result::Result<
    (
        Claim<F>,
        [Claim<F>; ROUND_FRAC_BITS],
        Claim<F>,
        [Claim<F>; ROUND_FRAC_BITS],
    ),
    ProofVerifyError,
>
where
    F: JoltField,
    T: Transcript,
{
    let (silu_claim, silu_round_frac_bits) = verify_round(
        vec![silu_round_claim],
        &proof.round,
        &params.round,
        transcript,
    )?;
    let (gate_proj_round, silu_frac_bits, silu_ra) =
        verify_silu(vec![silu_claim], &proof.silu, &params.silu, transcript)?;

    Ok((
        gate_proj_round,
        silu_frac_bits,
        silu_ra,
        silu_round_frac_bits,
    ))
}
