use ark_bn254::Fr;
use ark_ff::{One, Zero};
use itertools::Itertools;
use joltworks::transcripts::Transcript;
use qwen3_common::ops::rope::{RopeOutput, RopeParams, RopeVerifierOutput, draw_rope_challenges};

use qwen3_common::{
    BitOpeningClaims, EvalClaim, FRAC_BITS, SCALE, append_eval_claim, verify_sumcheck_rounds,
};

use crate::utils::{eq_point_eval, eval_i32_mle_at_point, log2};

pub fn verify_rope<Tr>(
    claim: EvalClaim,
    params: RopeParams,
    input: qwen3_common::LayerRopeVerifierInput,
    proof: &RopeOutput,
    transcript: &mut Tr,
) -> Option<RopeVerifierOutput>
where
    Tr: Transcript,
{
    append_eval_claim(transcript, &claim);
    validate_params(params).then_some(())?;
    (claim.point.len() == tensor_vars(params)).then_some(())?;
    let (rotation_constraint_mix, first_half_bit_booleanity, second_half_bit_booleanity) =
        draw_rope_challenges(transcript)?;
    // RoPE is one sumcheck over (seq, head, dim/2).  The output claim contains
    // the parity bit in its point; the relation uses that public parity to
    // select the first-half or second-half rotation:
    //
    //   rotated = first * cos - second * sin   when parity = 0
    //   rotated = first * sin + second * cos   when parity = 1
    //
    // The same sumcheck also enforces output rounding and booleanity for both
    // half-bit arrays.
    let rounds = verify_sumcheck_rounds(
        claim.value,
        &proof.rounds,
        log2(params.seq) + log2(params.heads) + log2(params.head_dim / 2),
        transcript,
    )?;
    let input_evals = rope_input_evals(proof);
    let public_evals = build_public_rope_evals(
        &input,
        &claim.point,
        &rounds.point,
        &rounds.challenges,
        rotation_constraint_mix,
        first_half_bit_booleanity,
        second_half_bit_booleanity,
        &params,
    )?;
    (rounds.final_claim == rope_relation(&input_evals, &public_evals)).then_some(())?;

    Some(build_rope_output_claims(
        &input_evals,
        &rounds.point,
        &params,
    ))
}

struct RopeInputEvals {
    input_first_half: Fr,
    input_second_half: Fr,
    output_first_half: Fr,
    output_second_half: Fr,
    first_half_bits: [Fr; FRAC_BITS],
    second_half_bits: [Fr; FRAC_BITS],
}

struct RopePublicEvals {
    eq: Fr,
    parity: Fr,
    cos: Fr,
    sin: Fr,
    rotation_constraint_mix: Fr,
    first_half_bit_booleanity: [Fr; FRAC_BITS],
    second_half_bit_booleanity: [Fr; FRAC_BITS],
}

fn rope_input_evals(proof: &RopeOutput) -> RopeInputEvals {
    RopeInputEvals {
        input_first_half: proof.input_first_half,
        input_second_half: proof.input_second_half,
        output_first_half: proof.output_first_half,
        output_second_half: proof.output_second_half,
        first_half_bits: proof.first_half_bits,
        second_half_bits: proof.second_half_bits,
    }
}

fn build_public_rope_evals(
    input: &qwen3_common::LayerRopeVerifierInput,
    claim_point: &[Fr],
    point: &[Fr],
    challenges: &[<Fr as joltworks::field::JoltField>::Challenge],
    rotation_constraint_mix: Fr,
    first_half_bit_booleanity: [Fr; FRAC_BITS],
    second_half_bit_booleanity: [Fr; FRAC_BITS],
    params: &RopeParams,
) -> Option<RopePublicEvals> {
    let (r_seq, r_head, r_pair, r_parity) = split_tensor_point(claim_point, params);
    (r_parity.len() == 1).then_some(())?;
    let relation_point = [r_seq, r_head, r_pair].concat();
    let (point_seq, _, point_pair) = split_relation_point(point, params);
    let coeff_point = [point_seq, point_pair].concat();
    Some(RopePublicEvals {
        eq: eq_point_eval(&relation_point, challenges)?,
        parity: r_parity[0],
        cos: eval_i32_mle_at_point(&input.cos, &coeff_point)?,
        sin: eval_i32_mle_at_point(&input.sin, &coeff_point)?,
        rotation_constraint_mix,
        first_half_bit_booleanity,
        second_half_bit_booleanity,
    })
}

fn rope_relation(input: &RopeInputEvals, public: &RopePublicEvals) -> Fr {
    let one_minus_parity = Fr::one() - public.parity;
    let output =
        one_minus_parity * input.output_first_half + public.parity * input.output_second_half;
    let first_pairs = input.first_half_bits.map(|bit| (bit, bit));
    let second_pairs = input.second_half_bits.map(|bit| (bit, bit));
    let remainder = one_minus_parity * remainder_constant(&first_pairs)
        + public.parity * remainder_constant(&second_pairs);
    let msb = one_minus_parity * input.first_half_bits[FRAC_BITS - 1]
        + public.parity * input.second_half_bits[FRAC_BITS - 1];
    let coeff_first = one_minus_parity * public.cos + public.parity * public.sin;
    let coeff_second = -one_minus_parity * public.sin + public.parity * public.cos;
    let rotation = input.input_first_half * coeff_first + input.input_second_half * coeff_second;
    let main = output
        + public.rotation_constraint_mix
            * (Fr::from(SCALE) * output + remainder - Fr::from(SCALE) * msb - rotation);
    let first_booleanity = input
        .first_half_bits
        .iter()
        .map(|bit| *bit * (*bit - Fr::one()))
        .zip_eq(public.first_half_bit_booleanity)
        .map(|(check, challenge)| challenge * check)
        .sum::<Fr>();
    let second_booleanity = input
        .second_half_bits
        .iter()
        .map(|bit| *bit * (*bit - Fr::one()))
        .zip_eq(public.second_half_bit_booleanity)
        .map(|(check, challenge)| challenge * check)
        .sum::<Fr>();
    public.eq * (main + first_booleanity + second_booleanity)
}

fn build_rope_output_claims(
    input: &RopeInputEvals,
    point: &[Fr],
    params: &RopeParams,
) -> RopeVerifierOutput {
    let (seq_point, head_point, pair_point) = split_relation_point(point, params);
    let input_first_half_point = matrix_half_point(seq_point, head_point, pair_point, Fr::zero());
    let input_second_half_point = matrix_half_point(seq_point, head_point, pair_point, Fr::one());
    let first_half_point = [seq_point, head_point, pair_point, &[Fr::zero()]].concat();
    let second_half_point = [seq_point, head_point, pair_point, &[Fr::one()]].concat();
    let input_first_half = EvalClaim::new(input.input_first_half, input_first_half_point);
    let input_second_half = EvalClaim::new(input.input_second_half, input_second_half_point);
    RopeVerifierOutput {
        input_first_half: input_first_half.clone(),
        input_second_half: input_second_half.clone(),
        first_half_bits: bit_opening_claims(&first_half_point, input.first_half_bits),
        second_half_bits: bit_opening_claims(&second_half_point, input.second_half_bits),
        input_first_half_claim: vec![EvalClaim::new(input.input_first_half, first_half_point)],
        input_second_half_claim: vec![EvalClaim::new(input.input_second_half, second_half_point)],
    }
}

fn validate_params(params: RopeParams) -> bool {
    params.seq.is_power_of_two()
        && params.heads.is_power_of_two()
        && params.head_dim.is_power_of_two()
        && params.head_dim >= 2
}

fn tensor_vars(params: RopeParams) -> usize {
    log2(params.seq) + log2(params.heads) + log2(params.head_dim)
}

fn split_tensor_point<'a>(
    point: &'a [Fr],
    params: &RopeParams,
) -> (&'a [Fr], &'a [Fr], &'a [Fr], &'a [Fr]) {
    let seq_vars = log2(params.seq);
    let head_vars = log2(params.heads);
    let pair_vars = log2(params.head_dim / 2);
    let r_seq = &point[..seq_vars];
    let r_head = &point[seq_vars..seq_vars + head_vars];
    let r_pair = &point[seq_vars + head_vars..seq_vars + head_vars + pair_vars];
    let r_parity = &point[seq_vars + head_vars + pair_vars..];
    (r_seq, r_head, r_pair, r_parity)
}

fn split_relation_point<'a>(
    point: &'a [Fr],
    params: &RopeParams,
) -> (&'a [Fr], &'a [Fr], &'a [Fr]) {
    let seq_vars = log2(params.seq);
    let head_vars = log2(params.heads);
    let r_seq = &point[..seq_vars];
    let r_head = &point[seq_vars..seq_vars + head_vars];
    let r_pair = &point[seq_vars + head_vars..];
    (r_seq, r_head, r_pair)
}

fn matrix_half_point(seq: &[Fr], head: &[Fr], pair: &[Fr], parity: Fr) -> Vec<Fr> {
    [seq, head, pair, &[parity]].concat()
}

fn bit_opening_claims(point: &[Fr], values: [Fr; FRAC_BITS]) -> BitOpeningClaims {
    values.map(|value| EvalClaim::new(value, point.to_vec()))
}

fn remainder_constant(bits: &[(Fr, Fr); FRAC_BITS]) -> Fr {
    bits.iter()
        .enumerate()
        .map(|(bit, (value, _))| Fr::from(1_u64 << bit) * *value)
        .sum()
}
