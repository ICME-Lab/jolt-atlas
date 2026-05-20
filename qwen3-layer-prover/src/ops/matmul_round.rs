use joltworks::{field::JoltField, transcripts::Transcript, utils::errors::ProofVerifyError};

use crate::{
    claim::Claim,
    error::Result,
    ops::{
        matmul::{MatMulParams, MatMulProof, prove_matmul, verify_matmul},
        round::{
            ROUND_FRAC_BITS, RoundParams, RoundProof, RoundWitness, prove_round, verify_round,
        },
    },
};

// Design note for future us:
//
// Many Qwen fixed-point linear layers appear as:
//     y_round = round(input @ W)
//
// Keep the proof internally as two simple protocols, `round` then `matmul`,
// but expose them as one op to the layer prover. This removes noisy intermediate
// accumulator claims from the layer graph without making the sumcheck formula
// artificially complex.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatMulRoundParams {
    pub round: RoundParams,
    pub matmul: MatMulParams,
}

impl MatMulRoundParams {
    pub fn new(round: RoundParams, matmul: MatMulParams) -> Self {
        Self { round, matmul }
    }
}

#[derive(Debug, Clone, Default)]
pub struct MatMulRoundWitness {
    pub input: Vec<i32>,
    pub acc: Vec<i64>,
    pub output: Vec<i32>,
    pub frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
}

#[derive(Debug, Clone)]
pub struct MatMulRoundProof<F: JoltField, T: Transcript> {
    pub round: RoundProof<F, T>,
    pub matmul: MatMulProof<F, T>,
}

pub fn prove_matmul_round<F, T>(
    y_round_claim: Claim<F>,
    witness: &MatMulRoundWitness,
    w: &[i32],
    params: &MatMulRoundParams,
    transcript: &mut T,
) -> Result<(
    MatMulRoundProof<F, T>,
    Claim<F>,
    Claim<F>,
)>
where
    F: JoltField,
    T: Transcript,
{
    let round_witness = RoundWitness::from_input_output(witness.acc.clone(), witness.output.clone());
    let (round_proof, acc_claim, round_ra) = prove_round(
        vec![y_round_claim],
        &round_witness,
        &params.round,
        transcript,
    )?;

    let matmul_result = prove_matmul(acc_claim, &witness.input, w, &params.matmul, transcript)?;
    let input = matmul_result.claims.input;

    Ok((
        MatMulRoundProof {
            round: round_proof,
            matmul: matmul_result.proof,
        },
        input,
        round_ra,
    ))
}

pub fn verify_matmul_round<F, T>(
    y_round_claim: Claim<F>,
    proof: &MatMulRoundProof<F, T>,
    w: &[i32],
    params: &MatMulRoundParams,
    transcript: &mut T,
) -> std::result::Result<(Claim<F>, Claim<F>), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    let (acc_claim, round_ra) =
        verify_round(vec![y_round_claim], &proof.round, &params.round, transcript)?;
    let matmul_claims = verify_matmul(acc_claim, &proof.matmul, w, &params.matmul, transcript)?;

    Ok((matmul_claims.input, round_ra))
}
