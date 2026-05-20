use common::VirtualPoly;
use joltworks::{
    field::JoltField,
    poly::opening_proof::{
        BIG_ENDIAN, OpeningId, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
        VerifierOpeningAccumulator,
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

use crate::{
    claim::{Claim, Shape, TensorId},
    error::Result,
    ops::{
        hadamard_mul::{
            HadamardMulParams, HadamardRoundRelationProof, prove_hadamard_round_relation,
            verify_hadamard_round_relation,
        },
        round::{
            ROUND_FRAC_BITS, ROUND_LUT_LEN, RoundLookupProof, RoundParams, RoundWitness,
            padded_lookup_indices, prove_round_lookup, round_lut_table, verify_round_lookup,
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
// The hadamard relation directly includes the fixed-point round relation:
//     lhs * rhs + round_bit * 2^8 - rem = output * 2^8
//
// This removes the separate round-relation sumcheck.  The SHOUT lookup proving
// `round_bit = ROUND_LUT_Q8[rem]` remains.

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
    pub hadamard: HadamardRoundRelationProof<F, T>,
    pub(crate) round_lookup: RoundLookupProof<F, T>,
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
    Claim<F>,
)>
where
    F: JoltField,
    T: Transcript,
{
    let round_witness = RoundWitness::from_input_output(witness.acc.clone(), witness.output.clone());
    let (hadamard_proof, hadamard_claims) = prove_hadamard_round_relation(
        y_claim,
        &witness.lhs,
        &witness.rhs,
        &round_witness.remainder,
        &round_witness.round_bit,
        &params.hadamard,
        transcript,
    )?;
    let mut round_accumulator = ProverOpeningAccumulator::new();
    let round_point = OpeningPoint::<BIG_ENDIAN, F>::new(hadamard_claims.round_point.clone());
    round_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundLut, SumcheckId::NodeExecution(0)),
        (round_point.clone(), hadamard_claims.round_bit_opening),
    );
    round_accumulator.openings.insert(
        OpeningId::new(
            VirtualPoly::QwenRoundRemainder,
            SumcheckId::NodeExecution(0),
        ),
        (round_point, hadamard_claims.remainder_opening),
    );
    let round_lookup = prove_round_lookup(
        hadamard_claims.round_point,
        hadamard_claims.round_bit_opening,
        hadamard_claims.remainder_opening,
        padded_lookup_indices(&round_witness.remainder, &params.round.shape),
        round_lut_table(),
        &mut round_accumulator,
        transcript,
    )?;
    let round_ra = Claim {
        tensor: TensorId::new(format!("{}_round_ra", params.round.input_tensor.0)),
        logical_shape: Shape::new(vec![
            params.round.shape.padded_power_of_two().numel(),
            ROUND_LUT_LEN,
        ]),
        domain_shape: Shape::new(vec![
            params.round.shape.padded_power_of_two().numel(),
            ROUND_LUT_LEN,
        ]),
        point: round_lookup.ra_point.clone(),
        value: round_lookup.ra_opening,
    };

    Ok((
        HadamardRoundProof {
            hadamard: hadamard_proof,
            round_lookup,
        },
        hadamard_claims.lhs,
        hadamard_claims.rhs,
        round_ra,
    ))
}

pub fn verify_hadamard_round<F, T>(
    y_claim: Claim<F>,
    proof: &HadamardRoundProof<F, T>,
    params: &HadamardRoundParams,
    transcript: &mut T,
) -> std::result::Result<(Claim<F>, Claim<F>, Claim<F>), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    let hadamard_claims =
        verify_hadamard_round_relation(y_claim, &proof.hadamard, &params.hadamard, transcript)?;
    let round_lookup = verify_round_lookup(
        params.round.shape.padded_power_of_two().numel(),
        hadamard_claims.round_point,
        hadamard_claims.round_bit_opening,
        hadamard_claims.remainder_opening,
        proof.round_lookup.ra_opening,
        &proof.round_lookup.committed_openings,
        &proof.round_lookup.read_raf,
        &proof.round_lookup.ra_onehot,
        &mut VerifierOpeningAccumulator::new(),
        transcript,
    )?;
    let round_ra = Claim {
        tensor: TensorId::new(format!("{}_round_ra", params.round.input_tensor.0)),
        logical_shape: Shape::new(vec![
            params.round.shape.padded_power_of_two().numel(),
            ROUND_LUT_LEN,
        ]),
        domain_shape: Shape::new(vec![
            params.round.shape.padded_power_of_two().numel(),
            ROUND_LUT_LEN,
        ]),
        point: round_lookup.ra_point,
        value: proof.round_lookup.ra_opening,
    };

    Ok((hadamard_claims.lhs, hadamard_claims.rhs, round_ra))
}
