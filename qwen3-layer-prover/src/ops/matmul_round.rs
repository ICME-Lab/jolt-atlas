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
        matmul::{
            MatMulParams, MatMulRoundRelationProof, prove_matmul_round_relation,
            verify_matmul_round_relation,
        },
        round::{
            ROUND_FRAC_BITS, ROUND_LUT_LEN, RoundLookupProof, RoundParams, RoundWitness,
            padded_lookup_indices, prove_round_lookup, round_lut_table, verify_round_lookup,
        },
    },
};

// Design note for future us:
//
// Many Qwen fixed-point linear layers appear as:
//     y_round = round(input @ W)
//
// The matmul relation directly includes the fixed-point round relation:
//
//   sum_k A[m,k] W[k,n] + round_bit[m,n] * 2^8 - rem[m,n] = Y[m,n] * 2^8
//
// This removes the separate round-relation sumcheck.  The SHOUT lookup proving
// `round_bit = ROUND_LUT_Q8[rem]` remains, because that is the range/lookup
// argument for the low 8-bit remainder.

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
    pub matmul: MatMulRoundRelationProof<F, T>,
    pub(crate) round_lookup: RoundLookupProof<F, T>,
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
    let matmul_result = prove_matmul_round_relation(
        y_round_claim,
        &witness.input,
        w,
        &round_witness.remainder,
        &round_witness.round_bit,
        &params.matmul,
        transcript,
    )?;
    let mut round_accumulator = ProverOpeningAccumulator::new();
    let round_point = OpeningPoint::<BIG_ENDIAN, F>::new(matmul_result.claims.round_point.clone());
    round_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundLut, SumcheckId::NodeExecution(0)),
        (round_point.clone(), matmul_result.claims.round_bit_opening),
    );
    round_accumulator.openings.insert(
        OpeningId::new(
            VirtualPoly::QwenRoundRemainder,
            SumcheckId::NodeExecution(0),
        ),
        (round_point, matmul_result.claims.remainder_opening),
    );
    let round_lookup = prove_round_lookup(
        matmul_result.claims.round_point.clone(),
        matmul_result.claims.round_bit_opening,
        matmul_result.claims.remainder_opening,
        padded_lookup_indices(&round_witness.remainder, &params.round.shape),
        round_lut_table(),
        &mut round_accumulator,
        transcript,
    )?;
    let input = matmul_result.claims.input;
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
        MatMulRoundProof {
            matmul: matmul_result.proof,
            round_lookup,
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
    let matmul_claims = verify_matmul_round_relation(
        y_round_claim,
        &proof.matmul,
        w,
        &params.matmul,
        transcript,
    )?;
    let round_lookup = verify_round_lookup(
        params.round.shape.padded_power_of_two().numel(),
        matmul_claims.round_point,
        matmul_claims.round_bit_opening,
        matmul_claims.remainder_opening,
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

    Ok((matmul_claims.input, round_ra))
}
