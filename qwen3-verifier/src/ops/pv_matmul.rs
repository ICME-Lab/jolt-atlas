use ark_bn254::Fr;
use ark_ff::One;
use itertools::Itertools;
use joltworks::transcripts::Transcript;
use qwen3_common::ops::pv_matmul::{
    PvMatmulOutput, PvMatmulParams, PvMatmulRoundingBits, PvMatmulVerifierOutput,
    draw_pv_matmul_bit_booleanity_challenges, draw_pv_matmul_challenges,
};

use qwen3_common::{
    BitOpeningClaims, EvalClaim, FRAC_BITS, append_eval_claim, verify_sumcheck_rounds,
};

use crate::utils::{bits_to_rem, combine_eq_points, eq_point_eval, log2};

const QWEN3_GQA_GROUP_SIZE: usize = 2;

pub fn verify_pv_matmul<Tr>(
    claim: EvalClaim,
    params: PvMatmulParams,
    proof: &PvMatmulOutput,
    transcript: &mut Tr,
) -> Option<PvMatmulVerifierOutput>
where
    Tr: Transcript,
{
    append_eval_claim(transcript, &claim);
    validate_params(params).then_some(())?;
    (claim.point.len() == context_vars(params)).then_some(())?;
    let (r_q, r_h, r_d) = split_context_point(&claim.point, &params);

    draw_pv_matmul_challenges(transcript)?;
    let rem = EvalClaim::new(proof.rem, claim.point.clone());
    let msb = EvalClaim::new(proof.msb, claim.point.clone());
    append_eval_claim(transcript, &rem);
    append_eval_claim(transcript, &msb);

    // Product sumcheck for attention value aggregation:
    //
    //   256 * context(q, h, d) + rem(q, h, d) - 256 * msb(q, h, d)
    //     = Σ_{h_kv,k} eq_gqa(h, h_kv) * p(q, h, k) * v(k, h_kv, d)
    //
    // The GQA selector is public and is checked at the final point by
    // combine_eq_points.
    let inner_point = verify_k_reduction_sumcheck(
        Fr::from(256_u64) * claim.value + rem.value - Fr::from(256_u64) * msb.value,
        r_h,
        &params,
        proof,
        transcript,
    )?;
    // Same rounding-bit sumcheck as matmul, but the bits belong to the
    // context tensor produced by P*V.
    let rounding_bits_point = verify_output_rounding_sumcheck(
        claim.point.clone(),
        rem.value,
        msb.value,
        &proof.rounding_bits,
        transcript,
    )?;

    let (r_head, r_kpos) = inner_point.split_at(head_vars(params));
    let (_, p_head_point) = combine_eq_points(r_h, r_head)?;
    let p = EvalClaim::new(proof.p, [r_q, p_head_point.as_slice(), r_kpos].concat());
    let v = EvalClaim::new(
        proof.v,
        [r_kpos, kv_head_point(r_head, &params)?, r_d].concat(),
    );
    Some(PvMatmulVerifierOutput {
        p,
        v,
        context_remainder_bits: bit_opening_claims(
            &rounding_bits_point,
            proof.rounding_bits.context_remainder_bits,
        ),
    })
}

fn verify_k_reduction_sumcheck<Tr>(
    product_claim: Fr,
    r_h: &[Fr],
    params: &PvMatmulParams,
    proof: &PvMatmulOutput,
    transcript: &mut Tr,
) -> Option<Vec<Fr>>
where
    Tr: Transcript,
{
    let rounds = verify_sumcheck_rounds(
        product_claim,
        &proof.k_reduction.rounds,
        head_vars(*params) + log2(params.seq),
        transcript,
    )?;
    let (r_head, _) = rounds.point.split_at(head_vars(*params));
    let (head_scalar, _) = combine_eq_points(r_h, r_head)?;
    (rounds.final_claim == head_scalar * proof.p * proof.v).then_some(rounds.point)
}

fn verify_output_rounding_sumcheck<Tr>(
    point: Vec<Fr>,
    rem: Fr,
    msb: Fr,
    proof: &PvMatmulRoundingBits,
    transcript: &mut Tr,
) -> Option<Vec<Fr>>
where
    Tr: Transcript,
{
    let ([rem_gamma, msb_gamma], booleanity_challenges) =
        draw_pv_matmul_bit_booleanity_challenges(transcript)?;
    let claim = rem_gamma * rem + msb_gamma * msb;
    let rounds = verify_sumcheck_rounds(claim, &proof.rounds, point.len(), transcript)?;
    let input_evals = RoundingBitsInputEvals {
        bits: proof.context_remainder_bits,
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

fn validate_params(params: PvMatmulParams) -> bool {
    params.seq.is_power_of_two()
        && params.q_heads.is_power_of_two()
        && params.kv_heads.is_power_of_two()
        && params.head_dim.is_power_of_two()
        && params.q_heads == params.kv_heads * QWEN3_GQA_GROUP_SIZE
}

fn context_vars(params: PvMatmulParams) -> usize {
    log2(params.seq) + log2(params.q_heads) + log2(params.head_dim)
}

fn head_vars(params: PvMatmulParams) -> usize {
    log2(params.q_heads)
}

fn kv_head_point<'a>(r_head: &'a [Fr], params: &PvMatmulParams) -> Option<&'a [Fr]> {
    let kv_vars = log2(params.kv_heads);
    (r_head.len() == kv_vars + 1).then_some(&r_head[1..])
}

fn split_context_point<'a>(
    point: &'a [Fr],
    params: &PvMatmulParams,
) -> (&'a [Fr], &'a [Fr], &'a [Fr]) {
    let q_vars = log2(params.seq);
    let h_vars = log2(params.q_heads);
    let d_vars = log2(params.head_dim);
    (
        &point[..q_vars],
        &point[q_vars..q_vars + h_vars],
        &point[q_vars + h_vars..q_vars + h_vars + d_vars],
    )
}
