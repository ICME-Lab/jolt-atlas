use ark_bn254::Fr;
use ark_ff::One;
use itertools::Itertools;
use joltworks::transcripts::Transcript;
use qwen3_common::ops::mul::{
    MulInputEvals, MulOutput, MulPublicEvals, MulVerifierOutput, draw_mul_booleanity_challenges,
};

use qwen3_common::{
    BitOpeningClaims, EvalClaim, FRAC_BITS, append_eval_claim, verify_sumcheck_rounds,
};

use crate::utils::eq_point_eval;

pub fn verify_mul<Tr>(
    claim: EvalClaim,
    proof: &MulOutput,
    transcript: &mut Tr,
) -> Option<MulVerifierOutput>
where
    Tr: Transcript,
{
    append_eval_claim(transcript, &claim);
    let booleanity_challenges = draw_mul_booleanity_challenges(transcript)?;
    // One sumcheck proves the rounded multiplication relation at claim.point:
    //
    //   out(x) = (lhs(x) * rhs(x) - rem(x) + 256 * msb(x)) / 256
    //   rem(x) = Σ_j 2^j * bit_j(x)
    //   bit_j(x) * (bit_j(x) - 1) = 0
    //
    // The final point becomes the opening point for lhs, rhs, and all rounding
    // bits.
    let rounds = verify_sumcheck_rounds(claim.value, &proof.rounds, claim.point.len(), transcript)?;
    let input_evals = mul_input_evals(proof)?;
    let public_evals =
        build_public_mul_evals(&claim.point, &rounds.challenges, booleanity_challenges)?;
    (rounds.final_claim == mul_relation(&input_evals, &public_evals)).then_some(())?;

    let point = rounds.point;
    Some(MulVerifierOutput {
        lhs: EvalClaim::new(input_evals.lhs, point.clone()),
        rhs: EvalClaim::new(input_evals.rhs, point.clone()),
        bits: bit_opening_claims(&point, input_evals.bits),
        point,
        input_evals,
    })
}

fn mul_relation(input: &MulInputEvals, public: &MulPublicEvals) -> Fr {
    let remainder = input
        .bits
        .into_iter()
        .enumerate()
        .map(|(bit, value)| Fr::from(1_u64 << bit) * value)
        .sum::<Fr>();
    let rounding = Fr::from(256_u64) * input.bits[FRAC_BITS - 1];
    let main = (input.lhs * input.rhs - remainder + rounding) / Fr::from(256_u64);
    let booleanity = input
        .bits
        .into_iter()
        .map(|bit| bit * (bit - Fr::one()))
        .zip_eq(public.booleanity)
        .map(|(check, challenge)| challenge * check)
        .sum::<Fr>();
    public.eq * (main + booleanity)
}

fn mul_input_evals(proof: &MulOutput) -> Option<MulInputEvals> {
    Some(MulInputEvals {
        lhs: proof.lhs,
        rhs: proof.rhs,
        bits: proof.bits,
    })
}

fn build_public_mul_evals(
    claim_point: &[Fr],
    point: &[<Fr as joltworks::field::JoltField>::Challenge],
    booleanity: [Fr; FRAC_BITS],
) -> Option<MulPublicEvals> {
    Some(MulPublicEvals {
        eq: eq_point_eval(claim_point, point)?,
        booleanity,
    })
}

fn bit_opening_claims(point: &[Fr], values: [Fr; FRAC_BITS]) -> BitOpeningClaims {
    values.map(|value| EvalClaim::new(value, point.to_vec()))
}
