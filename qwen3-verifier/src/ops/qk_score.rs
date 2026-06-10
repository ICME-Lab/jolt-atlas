use ark_bn254::Fr;
use ark_ff::One;
use itertools::Itertools;
use joltworks::transcripts::Transcript;
use qwen3_common::ops::qk_score::{
    QkScoreDotClaims, QkScoreDotOutput, QkScoreDotRoundingBits, QkScoreOutput, QkScoreParams,
    QkScoreScaleOutput, QkScoreVerifierOutput, draw_qk_score_dot_bit_booleanity_challenges,
    draw_qk_score_dot_challenges, draw_qk_score_scale_challenges, qk_score_inv_sqrt_q8,
};

use qwen3_common::{
    BitOpeningClaims, EvalClaim, FRAC_BITS, append_eval_claim, verify_sumcheck_rounds,
};

use crate::utils::{bits_to_rem, combine_eq_points, eq_point_eval, log2};

const QWEN3_GQA_GROUP_SIZE: usize = 2;

pub fn verify_qk_score<Tr>(
    claim: EvalClaim,
    params: QkScoreParams,
    proof: &QkScoreOutput,
    transcript: &mut Tr,
) -> Option<QkScoreVerifierOutput>
where
    Tr: Transcript,
{
    append_eval_claim(transcript, &claim);
    validate_params(params).then_some(())?;
    // Scale sumcheck first converts the rounded QK score claim into a dot
    // product claim:
    //
    //   score(q, h, kpos)
    //     = round(inv_sqrt_q8 * dot(q, h, kpos) / 256)
    //
    // The returned dot claim is then used as the target for the dot-product
    // sumcheck below.
    let scale = verify_qk_score_scale(claim, &params, &proof.scale, transcript)?;
    let (r_q, r_h, r_kpos) = split_score_point(&scale.dot.point, &params);

    append_eval_claim(transcript, &scale.dot);
    let rem = EvalClaim::new(proof.dot.rem, scale.dot.point.clone());
    let msb = EvalClaim::new(proof.dot.msb, scale.dot.point.clone());
    append_eval_claim(transcript, &rem);
    append_eval_claim(transcript, &msb);
    draw_qk_score_dot_challenges(transcript)?;

    // Dot product sumcheck:
    //
    //   256 * dot(q, h, kpos) + rem - 256 * msb
    //     = Σ_{h_kv,d} eq_gqa(h, h_kv) * q(q, h, d) * k(kpos, h_kv, d)
    //
    // It reduces the dot claim to q(q, h*, d*) and k(kpos, h_kv*, d*).
    let inner_point = verify_k_reduction_sumcheck(
        Fr::from(256_u64) * scale.dot.value + rem.value - Fr::from(256_u64) * msb.value,
        r_h,
        &params,
        &proof.dot,
        transcript,
    )?;
    // Rounding bits for the dot value are checked separately from the score
    // scaling bits, because they live at the dot-claim point.
    let dot_remainder_point = verify_output_rounding_sumcheck(
        scale.dot.point.clone(),
        rem.value,
        msb.value,
        &proof.dot.rounding_bits,
        transcript,
    )?;

    let (r_head, r_d) = inner_point.split_at(head_vars(params));
    let (_, q_head_point) = combine_eq_points(r_h, r_head)?;
    let q = EvalClaim::new(proof.dot.q, [r_q, q_head_point.as_slice(), r_d].concat());
    let k = EvalClaim::new(
        proof.dot.k,
        [r_kpos, kv_head_point(r_head, &params)?, r_d].concat(),
    );
    Some(QkScoreVerifierOutput {
        dot: QkScoreDotClaims { q, k },
        score_remainder_bits: scale.score_remainder_bits,
        dot_remainder_bits: bit_opening_claims(
            &dot_remainder_point,
            proof.dot.rounding_bits.dot_remainder_bits,
        ),
    })
}

struct QkScoreScaleVerifierOutput {
    dot: EvalClaim,
    score_remainder_bits: BitOpeningClaims,
}

fn verify_qk_score_scale<Tr>(
    claim: EvalClaim,
    params: &QkScoreParams,
    proof: &QkScoreScaleOutput,
    transcript: &mut Tr,
) -> Option<QkScoreScaleVerifierOutput>
where
    Tr: Transcript,
{
    validate_params(*params).then_some(())?;
    (claim.point.len() == score_vars(*params)).then_some(())?;
    let booleanity_challenges = draw_qk_score_scale_challenges(transcript)?;
    let sumcheck =
        verify_sumcheck_rounds(claim.value, &proof.rounds, score_vars(*params), transcript)?;
    let point = sumcheck.point;
    let final_relation = qk_score_scale_final_relation(
        proof.dot,
        proof.score_remainder_bits,
        qk_score_inv_sqrt_q8(params.head_dim),
        &booleanity_challenges,
    );
    let final_check = eq_point_eval(&claim.point, &sumcheck.challenges)? * final_relation;
    (sumcheck.final_claim == final_check).then_some(())?;
    Some(QkScoreScaleVerifierOutput {
        dot: EvalClaim::new(proof.dot, point.clone()),
        score_remainder_bits: bit_opening_claims(&point, proof.score_remainder_bits),
    })
}

fn verify_k_reduction_sumcheck<Tr>(
    product_claim: Fr,
    r_h: &[Fr],
    params: &QkScoreParams,
    proof: &QkScoreDotOutput,
    transcript: &mut Tr,
) -> Option<Vec<Fr>>
where
    Tr: Transcript,
{
    let rounds = verify_sumcheck_rounds(
        product_claim,
        &proof.k_reduction.rounds,
        head_vars(*params) + log2(params.head_dim),
        transcript,
    )?;
    let (r_head, _) = rounds.point.split_at(head_vars(*params));
    let (head_scalar, _) = combine_eq_points(r_h, r_head)?;
    (rounds.final_claim == head_scalar * proof.q * proof.k).then_some(rounds.point)
}

fn verify_output_rounding_sumcheck<Tr>(
    point: Vec<Fr>,
    rem: Fr,
    msb: Fr,
    proof: &QkScoreDotRoundingBits,
    transcript: &mut Tr,
) -> Option<Vec<Fr>>
where
    Tr: Transcript,
{
    let ([rem_gamma, msb_gamma], booleanity_challenges) =
        draw_qk_score_dot_bit_booleanity_challenges(transcript)?;
    let claim = rem_gamma * rem + msb_gamma * msb;
    let rounds = verify_sumcheck_rounds(claim, &proof.rounds, point.len(), transcript)?;
    let input_evals = RoundingBitsInputEvals {
        bits: proof.dot_remainder_bits,
    };
    let public_evals = RoundingBitsPublicEvals {
        eq: eq_point_eval(&point, &rounds.challenges)?,
        rem_gamma,
        msb_gamma,
        booleanity: booleanity_challenges,
    };
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

fn bit_opening_claims(point: &[Fr], values: [Fr; FRAC_BITS]) -> BitOpeningClaims {
    values.map(|value| EvalClaim::new(value, point.to_vec()))
}

fn validate_params(params: QkScoreParams) -> bool {
    params.seq.is_power_of_two()
        && params.q_heads.is_power_of_two()
        && params.kv_heads.is_power_of_two()
        && params.head_dim.is_power_of_two()
        && params.q_heads == params.kv_heads * QWEN3_GQA_GROUP_SIZE
}

fn score_vars(params: QkScoreParams) -> usize {
    log2(params.q_heads) + log2(params.seq) + log2(params.seq)
}

fn head_vars(params: QkScoreParams) -> usize {
    log2(params.q_heads)
}

fn split_score_point<'a>(
    point: &'a [Fr],
    params: &QkScoreParams,
) -> (&'a [Fr], &'a [Fr], &'a [Fr]) {
    let q_vars = log2(params.seq);
    let h_vars = log2(params.q_heads);
    (
        &point[..q_vars],
        &point[q_vars..q_vars + h_vars],
        &point[q_vars + h_vars..],
    )
}

fn kv_head_point<'a>(r_head: &'a [Fr], params: &QkScoreParams) -> Option<&'a [Fr]> {
    let kv_vars = log2(params.kv_heads);
    (r_head.len() == kv_vars + 1).then_some(&r_head[1..])
}

fn remainder_from_bits(bits: [Fr; FRAC_BITS]) -> Fr {
    bits.into_iter()
        .enumerate()
        .map(|(bit, value)| Fr::from(1_u64 << bit) * value)
        .sum()
}

fn bit_booleanity_final_relation(bits: [Fr; FRAC_BITS], challenges: &[Fr; FRAC_BITS]) -> Fr {
    bits.iter()
        .zip_eq(challenges)
        .map(|(bit, challenge)| *challenge * *bit * (*bit - Fr::one()))
        .sum()
}

fn qk_score_scale_final_relation(
    dot: Fr,
    bits: [Fr; FRAC_BITS],
    inv_sqrt_q8: i32,
    booleanity_challenges: &[Fr; FRAC_BITS],
) -> Fr {
    let main = (Fr::from(inv_sqrt_q8 as u64) * dot - remainder_from_bits(bits)
        + Fr::from(256_u64) * bits[FRAC_BITS - 1])
        / Fr::from(256_u64);
    main + bit_booleanity_final_relation(bits, booleanity_challenges)
}
