use joltworks::{field::JoltField, transcripts::Transcript, utils::errors::ProofVerifyError};

use crate::{
    claim::Claim,
    error::Result,
    ops::{
        hadamard_mul::{
            HadamardMulParams, HadamardMulProof, prove_hadamard_mul, verify_hadamard_mul,
        },
        round::{
            ROUND_FRAC_BITS, RoundParams, RoundProof, RoundWitness, prove_round, verify_round,
        },
    },
};

// Design note for future us:
//
// Elementwise fixed-point multiplication produces an accumulator with twice the
// fractional precision:
//     acc = lhs * rhs
//
// The layer graph wants the rebased tensor, not this accumulator.  Keep the
// proof internally as `round` then `hadamard_mul`, but expose it as one op so
// the layer prover sees only:
//     output = lhs * rhs
//
// This mirrors `MatMulRound`: simple protocols stay reusable, while accumulator
// claims do not leak into the layer-level wiring.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HadamardRoundParams {
    pub round: RoundParams,
    pub hadamard: HadamardMulParams,
}

impl HadamardRoundParams {
    pub fn new(round: RoundParams, hadamard: HadamardMulParams) -> Self {
        Self { round, hadamard }
    }
}

#[derive(Debug, Clone, Default)]
pub struct HadamardRoundWitness {
    pub lhs: Vec<i32>,
    pub rhs: Vec<i32>,
    pub acc: Vec<i64>,
    pub output: Vec<i32>,
    pub frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
}

#[derive(Debug, Clone)]
pub struct HadamardRoundProof<F: JoltField, T: Transcript> {
    pub round: RoundProof<F, T>,
    pub hadamard: HadamardMulProof<F, T>,
}

pub fn prove_hadamard_round<F, T>(
    y_claim: Claim<F>,
    witness: &HadamardRoundWitness,
    params: &HadamardRoundParams,
    transcript: &mut T,
) -> Result<(
    HadamardRoundProof<F, T>,
    Claim<F>,
    Claim<F>,
    [Claim<F>; ROUND_FRAC_BITS],
)>
where
    F: JoltField,
    T: Transcript,
{
    let round_witness = RoundWitness {
        input: witness.acc.clone(),
        output: witness.output.clone(),
        frac_bits: witness.frac_bits.clone(),
    };
    let (round_proof, acc_claim, frac_bits) =
        prove_round(vec![y_claim], &round_witness, &params.round, transcript)?;

    let (hadamard_proof, lhs, rhs) = prove_hadamard_mul(
        acc_claim,
        &witness.lhs,
        &witness.rhs,
        &params.hadamard,
        transcript,
    )?;

    Ok((
        HadamardRoundProof {
            round: round_proof,
            hadamard: hadamard_proof,
        },
        lhs,
        rhs,
        frac_bits,
    ))
}

pub fn verify_hadamard_round<F, T>(
    y_claim: Claim<F>,
    proof: &HadamardRoundProof<F, T>,
    params: &HadamardRoundParams,
    transcript: &mut T,
) -> std::result::Result<(Claim<F>, Claim<F>, [Claim<F>; ROUND_FRAC_BITS]), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    let (acc_claim, frac_bits) =
        verify_round(vec![y_claim], &proof.round, &params.round, transcript)?;
    let (lhs, rhs) = verify_hadamard_mul(acc_claim, &proof.hadamard, &params.hadamard, transcript)?;

    Ok((lhs, rhs, frac_bits))
}
