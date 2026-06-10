use ark_bn254::Fr;
use ark_ff::One;
use itertools::Itertools;
use joltworks::transcripts::Transcript;
use qwen3_common::ops::matmul::{
    MatMulOutput, MatMulRoundingBits, MatMulVerifierInput, MatMulVerifierOutput,
    draw_matmul_rounding_bit_challenges,
};

use qwen3_common::{
    BitOpeningClaims, EvalClaim, FRAC_BITS, SCALE, append_eval_claim, verify_sumcheck_rounds,
};

use crate::utils::{bits_to_rem, eq_point_eval, eval_i32_matrix_at_point};

pub fn verify_matmul<Tr>(
    claim: EvalClaim,
    input: MatMulVerifierInput,
    proof: &MatMulOutput,
    transcript: &mut Tr,
) -> Option<MatMulVerifierOutput>
where
    Tr: Transcript,
{
    append_eval_claim(transcript, &claim);
    let params = input.params;
    let (row_point, col_point) = claim.point.split_at(params.output_shape.row_vars());
    let rem = EvalClaim::new(proof.rem, claim.point.clone());
    let msb = EvalClaim::new(proof.msb, claim.point.clone());
    append_eval_claim(transcript, &rem);
    append_eval_claim(transcript, &msb);

    // Product sumcheck:
    //
    //   256 * out(row, col) + rem(row, col) - 256 * msb(row, col)
    //     = Σ_k lhs(row, k) * rhs(k, col)
    //
    // It reduces the output claim to lhs(row, k*) and rhs(k*, col).
    let inner_point = verify_k_reduction_sumcheck(
        Fr::from(SCALE) * claim.value + rem.value - Fr::from(SCALE) * msb.value,
        &params,
        proof,
        transcript,
    )?;

    // Rounding-bit sumcheck:
    //
    //   gamma_rem * rem(row, col) + gamma_msb * msb(row, col)
    //     = Σ_x eq((row, col), x)
    //         * (gamma_rem * Σ_j 2^j bit_j(x)
    //            + gamma_msb * bit_7(x)
    //            + Σ_j beta_j bit_j(x)(bit_j(x)-1))
    //
    // This creates the opening claims for the eight output rounding bits.
    let rounding_bit_point = verify_output_rounding_sumcheck(
        claim.point.clone(),
        rem.value,
        msb.value,
        &proof.rounding_bits,
        transcript,
    )?;

    let lhs = EvalClaim::new(proof.lhs, [row_point, &inner_point].concat());
    let rhs = EvalClaim::new(proof.rhs, [&inner_point, col_point].concat());
    let public_evals = build_public_matmul_evals(&input.weight, &rhs, &params)?;
    (rhs == public_evals.rhs).then_some(())?;
    Some(MatMulVerifierOutput {
        lhs,
        rhs,
        rounding_bits: bit_opening_claims(&rounding_bit_point, proof.rounding_bits.bits),
    })
}

struct MatMulPublicEvals {
    rhs: EvalClaim,
}

fn build_public_matmul_evals(
    weight: &[i32],
    rhs: &EvalClaim,
    params: &qwen3_common::ops::matmul::MatMulParams,
) -> Option<MatMulPublicEvals> {
    Some(MatMulPublicEvals {
        rhs: EvalClaim::new(
            eval_i32_matrix_at_point(weight, params.rhs_shape(), &rhs.point)?,
            rhs.point.clone(),
        ),
    })
}

fn verify_k_reduction_sumcheck<Tr>(
    product_claim: Fr,
    params: &qwen3_common::ops::matmul::MatMulParams,
    proof: &MatMulOutput,
    transcript: &mut Tr,
) -> Option<Vec<Fr>>
where
    Tr: Transcript,
{
    let rounds = verify_sumcheck_rounds(
        product_claim,
        &proof.k_reduction.rounds,
        params.inner.ilog2() as usize,
        transcript,
    )?;
    (rounds.final_claim == proof.lhs * proof.rhs).then_some(rounds.point)
}

fn verify_output_rounding_sumcheck<Tr>(
    point: Vec<Fr>,
    rem: Fr,
    msb: Fr,
    proof: &MatMulRoundingBits,
    transcript: &mut Tr,
) -> Option<Vec<Fr>>
where
    Tr: Transcript,
{
    let ([rem_gamma, msb_gamma], booleanity_challenges) =
        draw_matmul_rounding_bit_challenges(transcript)?;
    let claim = rem_gamma * rem + msb_gamma * msb;
    let rounds = verify_sumcheck_rounds(claim, &proof.rounds, point.len(), transcript)?;
    let input_evals = rounding_bits_input_evals(proof)?;
    let public_evals = build_public_rounding_bits_evals(
        &point,
        &rounds.challenges,
        rem_gamma,
        msb_gamma,
        booleanity_challenges,
    )?;
    (rounds.final_claim == rounding_bits_relation(&input_evals, &public_evals))
        .then_some(rounds.point)
}

struct RoundingBitsInputEvals {
    bits: [Fr; FRAC_BITS],
}

struct RoundingBitsPublicEvals {
    eq: Fr,
    rem_gamma: Fr,
    msb_gamma: Fr,
    booleanity: [Fr; FRAC_BITS],
}

fn rounding_bits_relation(input: &RoundingBitsInputEvals, public: &RoundingBitsPublicEvals) -> Fr {
    let rounding =
        public.rem_gamma * bits_to_rem(&input.bits) + public.msb_gamma * input.bits[FRAC_BITS - 1];
    let booleanity = input
        .bits
        .into_iter()
        .map(|bit| bit * (bit - Fr::one()))
        .zip_eq(public.booleanity)
        .map(|(check, challenge)| challenge * check)
        .sum::<Fr>();
    public.eq * (rounding + booleanity)
}

fn rounding_bits_input_evals(proof: &MatMulRoundingBits) -> Option<RoundingBitsInputEvals> {
    Some(RoundingBitsInputEvals { bits: proof.bits })
}

fn build_public_rounding_bits_evals(
    claim_point: &[Fr],
    point: &[<Fr as joltworks::field::JoltField>::Challenge],
    rem_gamma: Fr,
    msb_gamma: Fr,
    booleanity: [Fr; FRAC_BITS],
) -> Option<RoundingBitsPublicEvals> {
    Some(RoundingBitsPublicEvals {
        eq: eq_point_eval(claim_point, point)?,
        rem_gamma,
        msb_gamma,
        booleanity,
    })
}

fn bit_opening_claims(point: &[Fr], values: [Fr; FRAC_BITS]) -> BitOpeningClaims {
    values.map(|value| EvalClaim::new(value, point.to_vec()))
}
