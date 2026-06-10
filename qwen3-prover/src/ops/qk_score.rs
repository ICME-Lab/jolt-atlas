use ark_bn254::Fr;
use ark_ff::{Field, One, Zero};
use itertools::Itertools;
use joltworks::{
    field::JoltField,
    poly::{multilinear_polynomial::BindingOrder, split_eq_poly::GruenSplitEqPolynomial},
    transcripts::Transcript,
};
use qwen3_common::FRAC_BITS;
pub use qwen3_common::ops::qk_score::{
    QkScoreDotClaims, QkScoreDotOutput, QkScoreDotReduction, QkScoreDotRoundingBits, QkScoreOutput,
    QkScoreParams, QkScoreScaleOutput, QkScoreVerifierOutput,
    draw_qk_score_dot_bit_booleanity_challenges, draw_qk_score_dot_challenges,
    draw_qk_score_scale_challenges, qk_score_inv_sqrt_q8,
};

use crate::{
    layer::{BitOpeningClaims, EvalClaim, append_eval_claim},
    profile,
    round_message::{RoundPolynomial, SumCheckRounds, append_round_statement},
};

const QWEN3_GQA_GROUP_SIZE: usize = 2;

pub struct QkScoreProverOutput {
    pub proof: QkScoreOutput,
    pub dot: QkScoreDotClaims,
    pub score_remainder_bits: BitOpeningClaims,
    pub dot_remainder_bits: BitOpeningClaims,
}

struct QkScoreScaleProverOutput {
    proof: QkScoreScaleOutput,
    dot: EvalClaim,
    score_remainder_bits: BitOpeningClaims,
}

struct QkScoreDotProverOutput {
    proof: QkScoreDotOutput,
    dot: QkScoreDotClaims,
    dot_remainder_bits: BitOpeningClaims,
}

pub struct QkScoreProverInput {
    pub params: QkScoreParams,
    pub witness: QkScoreWitness,
}

pub struct QkScoreWitness {
    pub q: Vec<i32>,
    pub k: Vec<i32>,
    pub dot: Vec<i32>,
    pub score_remainder_bits: [Vec<bool>; FRAC_BITS],
    pub dot_remainder_bits: [Vec<bool>; FRAC_BITS],
}

// Sumcheck relations:
//   Scale:
//     256 * score(q, h, k) - inv_sqrt_q8 * dot(q, h, k)
//       + rem(q, h, k) - 256 * msb(q, h, k) = 0
//   Dot:
//     dot(q, h, k) = Σ_d q(q, h, d) * k(k, kv_head(h), d)
//   The dot rounding bits are checked by a separate read + booleanity reduction:
//     Σ_x eq(r_score, x) *
//       (λ0 * Σ_j 2^j b_j(x) + λ1 * b_7(x) + Σ_j γ_j * b_j(x) * (b_j(x) - 1))
//     = λ0 * rem(r_score) + λ1 * b_7(r_score)
pub fn prove_qk_score<Tr>(
    claim: EvalClaim,
    input: QkScoreProverInput,
    transcript: &mut Tr,
) -> Option<QkScoreProverOutput>
where
    Tr: Transcript,
{
    append_eval_claim(transcript, &claim);
    validate_input(&input)?;
    let params = input.params;
    let witness = input.witness;

    let scale = profile::measure("qk_score.proof.scale", || {
        prove_qk_score_scale(
            claim,
            &params,
            witness.dot.clone(),
            witness.score_remainder_bits,
            transcript,
        )
    })?;
    append_eval_claim(transcript, &scale.dot);
    let dot = prove_qk_score_dot(
        scale.dot.clone(),
        &params,
        witness.q,
        witness.k,
        witness.dot,
        witness.dot_remainder_bits,
        transcript,
    )?;

    Some(QkScoreProverOutput {
        proof: QkScoreOutput {
            scale: scale.proof,
            dot: dot.proof,
        },
        dot: dot.dot,
        score_remainder_bits: scale.score_remainder_bits,
        dot_remainder_bits: dot.dot_remainder_bits,
    })
}

fn prove_qk_score_scale<Tr>(
    claim: EvalClaim,
    params: &QkScoreParams,
    dot: Vec<i32>,
    score_remainder_bits: [Vec<bool>; FRAC_BITS],
    transcript: &mut Tr,
) -> Option<QkScoreScaleProverOutput>
where
    Tr: Transcript,
{
    params.validate().then_some(())?;
    (claim.point.len() == params.score_vars()).then_some(())?;
    let len = params.q_heads * params.seq * params.seq;
    (dot.len() == len).then_some(())?;
    score_remainder_bits
        .iter()
        .all(|bits| bits.len() == len)
        .then_some(())?;

    let booleanity_challenges = draw_qk_score_scale_challenges(transcript)?;
    let split_eq_point = claim.point.iter().rev().copied().collect::<Vec<_>>();
    let mut eq = GruenSplitEqPolynomial::new(&split_eq_point, BindingOrder::LowToHigh);
    let mut dot = collect_score_table(dot, *params)?
        .into_iter()
        .map(Fr::from_i32)
        .collect::<Vec<_>>();
    let mut bits =
        RoundingBitsState::from_bytes(collect_score_bytes(&score_remainder_bits, *params)?);
    let mut claim_i = claim.value;
    let mut round_polys = Vec::with_capacity(claim.point.len());
    let mut challenges = Vec::with_capacity(claim.point.len());

    while dot.len() > 1 {
        let round = qk_score_scale_round_poly(
            &eq,
            &dot,
            &bits,
            qk_score_inv_sqrt_q8(params.head_dim),
            &booleanity_challenges,
        )?;
        append_round_statement(transcript, claim_i, &round);
        let challenge = transcript.challenge_scalar_optimized::<Fr>();
        let r = challenge.into();
        eq.bind(challenge);
        bind(&mut dot, r);
        bits.bind(r);
        claim_i = round.eval(r);
        challenges.push(challenge);
        round_polys.push(round);
    }

    let opening_point = fr_challenges(&challenges);
    let dot_claim = EvalClaim {
        value: dot[0],
        point: opening_point.clone(),
    };
    let score_remainder_bits = bits.bits_at(0);
    Some(QkScoreScaleProverOutput {
        proof: QkScoreScaleOutput {
            rounds: SumCheckRounds {
                round_polys,
                final_claim: claim_i,
            },
            dot: dot_claim.value,
            score_remainder_bits,
        },
        dot: dot_claim,
        score_remainder_bits: bit_opening_claims(&opening_point, score_remainder_bits),
    })
}

fn prove_qk_score_dot<Tr>(
    claim: EvalClaim,
    params: &QkScoreParams,
    q: Vec<i32>,
    k: Vec<i32>,
    _dot: Vec<i32>,
    dot_remainder_bits: [Vec<bool>; FRAC_BITS],
    transcript: &mut Tr,
) -> Option<QkScoreDotProverOutput>
where
    Tr: Transcript,
{
    params.validate().then_some(())?;
    (claim.point.len() == params.score_vars()).then_some(())?;

    let (r_q, r_h, r_kpos) = split_score_point(&claim.point, params);
    let mut dot_remainder_bits_polys =
        RoundingBitsState::from_bytes(collect_score_bytes(&dot_remainder_bits, *params)?);
    let (mut lhs_by_inner, mut rhs_by_inner, remainder, msb) =
        profile::measure("qk_score.prepare.dot_partials", || {
            let head_eq = eq_table(r_h);
            let q_by_head_dim = partial_q_vec(&q, params, r_q);
            let k_by_head_dim = partial_k_vec(&k, params, r_kpos);
            let remainder = dot_remainder_at_point(&dot_remainder_bits, params, r_h, r_q, r_kpos);
            let msb =
                dot_bit_at_point(&dot_remainder_bits[FRAC_BITS - 1], params, r_h, r_q, r_kpos);
            (
                apply_head_eq(&q_by_head_dim, params.q_heads, params.head_dim, &head_eq),
                k_by_head_dim,
                remainder,
                msb,
            )
        });
    let output_rem = EvalClaim {
        value: remainder,
        point: claim.point.clone(),
    };
    let output_msb = EvalClaim {
        value: msb,
        point: claim.point.clone(),
    };
    append_eval_claim(transcript, &output_rem);
    append_eval_claim(transcript, &output_msb);
    draw_qk_score_dot_challenges(transcript)?;

    let inner_product_claim =
        Fr::from(256_u64) * claim.value + output_rem.value - Fr::from(256_u64) * output_msb.value;
    let k_reduction = profile::measure("qk_score.proof.dot_product", || {
        prove_product_reduction(
            inner_product_claim,
            &mut lhs_by_inner,
            &mut rhs_by_inner,
            transcript,
        )
    })?;

    let inner_point = fr_challenges(&k_reduction.challenges);
    let (r_head, r_d) = inner_point.split_at(params.head_vars());
    let (head_scalar, q_head_point) = combine_eq_points(r_h, r_head)?;
    let q = EvalClaim::new(
        lhs_by_inner[0] / head_scalar,
        [r_q, q_head_point.as_slice(), r_d].concat(),
    );
    let k = EvalClaim::new(
        rhs_by_inner[0],
        [r_kpos, kv_head_point(r_head, params)?, r_d].concat(),
    );
    let rounding_bits = profile::measure("qk_score.proof.bit_booleanity", || {
        prove_bit_booleanity(
            claim.point.clone(),
            &mut dot_remainder_bits_polys,
            output_rem.value,
            output_msb.value,
            draw_qk_score_dot_bit_booleanity_challenges(transcript)?,
            transcript,
        )
    })?;

    let dot_remainder_bits = dot_remainder_bits_polys.bits_at(0);
    Some(QkScoreDotProverOutput {
        proof: QkScoreDotOutput {
            rounding_bits: QkScoreDotRoundingBits {
                rounds: rounding_bits.rounds,
                dot_remainder_bits,
            },
            k_reduction: QkScoreDotReduction {
                rounds: k_reduction.rounds,
            },
            rem: output_rem.value,
            msb: output_msb.value,
            q: q.value,
            k: k.value,
        },
        dot: QkScoreDotClaims { q, k },
        dot_remainder_bits: bit_opening_claims(
            &fr_challenges(&rounding_bits.challenges),
            dot_remainder_bits,
        ),
    })
}

fn validate_input(input: &QkScoreProverInput) -> Option<()> {
    let params = input.params;
    params.validate().then_some(())?;
    (input.witness.q.len() == params.seq * params.q_heads * params.head_dim).then_some(())?;
    (input.witness.k.len() == params.seq * params.kv_heads * params.head_dim).then_some(())?;
    (input.witness.dot.len() == params.q_heads * params.seq * params.seq).then_some(())?;
    input
        .witness
        .score_remainder_bits
        .iter()
        .all(|bit| bit.len() == params.q_heads * params.seq * params.seq)
        .then_some(())?;
    input
        .witness
        .dot_remainder_bits
        .iter()
        .all(|bit| bit.len() == params.q_heads * params.seq * params.seq)
        .then_some(())
}

struct ProductReduction {
    rounds: SumCheckRounds<3>,
    challenges: Vec<<Fr as JoltField>::Challenge>,
}

struct BitBooleanityReduction {
    rounds: SumCheckRounds<4>,
    challenges: Vec<<Fr as JoltField>::Challenge>,
}

fn prove_bit_booleanity<Tr>(
    point: Vec<Fr>,
    bits: &mut RoundingBitsState,
    remainder: Fr,
    round_bit: Fr,
    bit_challenges: ([Fr; 2], [Fr; FRAC_BITS]),
    transcript: &mut Tr,
) -> Option<BitBooleanityReduction>
where
    Tr: Transcript,
{
    let len = bits.len();
    (len.is_power_of_two() && point.len() == len.ilog2() as usize).then_some(())?;

    let ([rem_gamma, msb_gamma], booleanity_challenges) = bit_challenges;
    let mut claim = rem_gamma * remainder + msb_gamma * round_bit;
    let split_eq_point = point.iter().rev().copied().collect::<Vec<_>>();
    let mut eq = GruenSplitEqPolynomial::new(&split_eq_point, BindingOrder::LowToHigh);
    let mut round_polys = Vec::with_capacity(point.len());
    let mut challenges = Vec::with_capacity(point.len());

    while bits.len() > 1 {
        let round = profile::measure_detail("qk_score.bit_booleanity.round_poly", || {
            bit_read_round_poly(&eq, bits, rem_gamma, msb_gamma, &booleanity_challenges)
        })?;
        let challenge = profile::measure_detail("qk_score.bit_booleanity.transcript", || {
            append_round_statement(transcript, claim, &round);
            transcript.challenge_scalar_optimized::<Fr>()
        });
        let r = challenge.into();
        profile::measure_detail("qk_score.bit_booleanity.eq_bind", || eq.bind(challenge));
        profile::measure_detail("qk_score.bit_booleanity.bits_bind", || bits.bind(r));
        claim = round.eval(r);
        challenges.push(challenge);
        round_polys.push(round);
    }

    let final_check = eq.get_current_scalar()
        * bit_read_final_relation(
            bits.bits_at(0),
            rem_gamma,
            msb_gamma,
            &booleanity_challenges,
        );
    (claim == final_check).then_some(BitBooleanityReduction {
        rounds: SumCheckRounds {
            round_polys,
            final_claim: claim,
        },
        challenges,
    })
}

fn qk_score_scale_round_poly(
    eq: &GruenSplitEqPolynomial<Fr>,
    dot: &[Fr],
    bits: &RoundingBitsState,
    inv_sqrt_q8: i32,
    booleanity_challenges: &[Fr; FRAC_BITS],
) -> Option<RoundPolynomial<4>> {
    let len = dot.len();
    (eq.len() == len && bits.len() == len && len % 2 == 0).then_some(())?;

    let inv_sqrt_q8 = Fr::from(inv_sqrt_q8 as u64);
    let shift = Field::inverse(&Fr::from(256_u64)).unwrap();
    let relation_evals = eq.par_fold_out_in_unreduced::<9, 3>(&|index| {
        qk_score_scale_relation_evals(index, dot, bits, inv_sqrt_q8, shift, booleanity_challenges)
    });
    Some(quadratic_relation_times_eq(
        relation_evals,
        eq.get_current_w(),
        eq.get_current_scalar(),
    ))
}

fn qk_score_scale_relation_evals(
    index: usize,
    dot: &[Fr],
    bits: &RoundingBitsState,
    inv_sqrt_q8: Fr,
    shift: Fr,
    booleanity_challenges: &[Fr; FRAC_BITS],
) -> [Fr; 3] {
    let dot_0 = dot[2 * index];
    let dot_1 = dot[2 * index + 1];
    let bit_pairs = pair_bits(bits.bits_at(2 * index), bits.bits_at(2 * index + 1));
    let remainder_0 = remainder_at_zero(&bit_pairs);
    let remainder_linear = remainder_linear(&bit_pairs);
    let msb_0 = bit_pairs[FRAC_BITS - 1].0;
    let msb_linear = bit_pairs[FRAC_BITS - 1].1 - bit_pairs[FRAC_BITS - 1].0;

    let mut constant = (inv_sqrt_q8 * dot_0 - remainder_0 + Fr::from(256_u64) * msb_0) * shift;
    let mut linear =
        (inv_sqrt_q8 * (dot_1 - dot_0) - remainder_linear + Fr::from(256_u64) * msb_linear) * shift;
    let mut leading = Fr::zero();
    add_bit_booleanity(
        &mut constant,
        &mut linear,
        &mut leading,
        booleanity_challenges,
        &bit_pairs,
    );

    [
        constant,
        constant + linear + leading,
        constant + linear + linear + leading * Fr::from(4_u64),
    ]
}

fn bit_read_round_poly(
    eq: &GruenSplitEqPolynomial<Fr>,
    bits: &RoundingBitsState,
    rem_gamma: Fr,
    msb_gamma: Fr,
    booleanity_challenges: &[Fr; FRAC_BITS],
) -> Option<RoundPolynomial<4>> {
    let len = bits.len();
    (eq.len() == len && len % 2 == 0).then_some(())?;
    let relation_evals = eq.par_fold_out_in_unreduced::<9, 3>(&|index| {
        bit_read_evals(index, bits, rem_gamma, msb_gamma, booleanity_challenges)
    });
    Some(quadratic_relation_times_eq(
        relation_evals,
        eq.get_current_w(),
        eq.get_current_scalar(),
    ))
}

fn prove_product_reduction<Tr>(
    mut claim: Fr,
    left: &mut Vec<Fr>,
    right: &mut Vec<Fr>,
    transcript: &mut Tr,
) -> Option<ProductReduction>
where
    Tr: Transcript,
{
    (left.len() == right.len() && left.len().is_power_of_two()).then_some(())?;
    let mut round_polys = Vec::with_capacity(left.len().ilog2() as usize);
    let mut challenges = Vec::with_capacity(left.len().ilog2() as usize);
    while left.len() > 1 {
        let round = product_round_poly(claim, left, right)?;
        append_round_statement(transcript, claim, &round);
        let r = transcript.challenge_scalar_optimized::<Fr>();
        bind(left, r.into());
        bind(right, r.into());
        claim = round.eval(r.into());
        challenges.push(r);
        round_polys.push(round);
    }
    (claim == left[0] * right[0]).then_some(ProductReduction {
        rounds: SumCheckRounds {
            round_polys,
            final_claim: claim,
        },
        challenges,
    })
}

fn product_round_poly(claim: Fr, left: &[Fr], right: &[Fr]) -> Option<RoundPolynomial<3>> {
    (left.len() == right.len() && left.len() % 2 == 0).then_some(())?;
    let eval_at_zero = left
        .chunks_exact(2)
        .zip_eq(right.chunks_exact(2))
        .map(|(left, right)| left[0] * right[0])
        .sum::<Fr>();
    let eval_at_one = claim - eval_at_zero;
    let eval_at_two = left
        .chunks_exact(2)
        .zip_eq(right.chunks_exact(2))
        .map(|(left, right)| {
            let left_at_two = left[0] + Fr::from(2_u64) * (left[1] - left[0]);
            let right_at_two = right[0] + Fr::from(2_u64) * (right[1] - right[0]);
            left_at_two * right_at_two
        })
        .sum::<Fr>();
    Some(quadratic_from_evals(eval_at_zero, eval_at_one, eval_at_two))
}

fn quadratic_from_evals(eval_at_zero: Fr, eval_at_one: Fr, eval_at_two: Fr) -> RoundPolynomial<3> {
    let two = Fr::from(2_u64);
    let leading = (eval_at_two - two * eval_at_one + eval_at_zero) / two;
    let linear = eval_at_one - eval_at_zero - leading;
    RoundPolynomial {
        coeffs: [eval_at_zero, linear, leading],
    }
}

fn quadratic_relation_times_eq(
    relation_evals: [Fr; 3],
    current_w: Fr,
    current_scalar: Fr,
) -> RoundPolynomial<4> {
    let half = Field::inverse(&Fr::from(2_u64)).expect("2 is nonzero");
    let q0 = relation_evals[0];
    let q2 = (relation_evals[2] - relation_evals[1] - relation_evals[1] + relation_evals[0]) * half;
    let q1 = relation_evals[1] - q0 - q2;

    let eq_constant = Fr::one() - current_w;
    let eq_linear = current_w + current_w - Fr::one();

    RoundPolynomial {
        coeffs: [
            current_scalar * eq_constant * q0,
            current_scalar * (eq_constant * q1 + eq_linear * q0),
            current_scalar * (eq_constant * q2 + eq_linear * q1),
            current_scalar * eq_linear * q2,
        ],
    }
}

fn partial_q_vec(q: &[i32], params: &QkScoreParams, r_q: &[Fr]) -> Vec<Fr> {
    let q_eq = eq_table(r_q);
    let mut out = vec![Fr::zero(); params.q_heads * params.head_dim];
    for head in 0..params.q_heads {
        for d in 0..params.head_dim {
            out[head + params.q_heads * d] = (0..params.seq)
                .map(|qpos| {
                    q_eq[qpos]
                        * Fr::from_i32(q[tensor_index(qpos, head, d, params.seq, params.q_heads)])
                })
                .sum();
        }
    }
    out
}

fn partial_k_vec(k: &[i32], params: &QkScoreParams, r_kpos: &[Fr]) -> Vec<Fr> {
    let kpos_eq = eq_table(r_kpos);
    let mut out = vec![Fr::zero(); params.q_heads * params.head_dim];
    for head in 0..params.q_heads {
        let kv_head = head / QWEN3_GQA_GROUP_SIZE;
        for d in 0..params.head_dim {
            out[head + params.q_heads * d] = (0..params.seq)
                .map(|kpos| {
                    kpos_eq[kpos]
                        * Fr::from_i32(
                            k[tensor_index(kpos, kv_head, d, params.seq, params.kv_heads)],
                        )
                })
                .sum();
        }
    }
    out
}

fn apply_head_eq(values: &[Fr], heads: usize, width: usize, head_eq: &[Fr]) -> Vec<Fr> {
    let mut out = values.to_vec();
    for d in 0..width {
        for head in 0..heads {
            out[head + heads * d] *= head_eq[head];
        }
    }
    out
}

fn dot_remainder_at_point(
    bits: &[Vec<bool>; FRAC_BITS],
    params: &QkScoreParams,
    r_h: &[Fr],
    r_q: &[Fr],
    r_kpos: &[Fr],
) -> Fr {
    bits.iter()
        .enumerate()
        .map(|(bit, values)| {
            Fr::from(1_u64 << bit) * dot_bit_at_point(values, params, r_h, r_q, r_kpos)
        })
        .sum()
}

fn dot_bit_at_point(
    bits: &[bool],
    params: &QkScoreParams,
    r_h: &[Fr],
    r_q: &[Fr],
    r_kpos: &[Fr],
) -> Fr {
    let h_eq = eq_table(r_h);
    let q_eq = eq_table(r_q);
    let k_eq = eq_table(r_kpos);
    (0..params.q_heads)
        .map(|head| {
            h_eq[head]
                * (0..params.seq)
                    .map(|qpos| {
                        q_eq[qpos]
                            * (0..params.seq)
                                .filter(|kpos| bits[score_index(qpos, head, *kpos, params)])
                                .map(|kpos| k_eq[kpos])
                                .sum::<Fr>()
                    })
                    .sum::<Fr>()
        })
        .sum()
}

fn tensor_index(seq: usize, head: usize, dim: usize, seq_len: usize, heads: usize) -> usize {
    dim * (seq_len * heads) + head * seq_len + seq
}

fn score_index(qpos: usize, head: usize, kpos: usize, params: &QkScoreParams) -> usize {
    kpos * (params.seq * params.q_heads) + head * params.seq + qpos
}

fn collect_score_bytes(bits: &[Vec<bool>; FRAC_BITS], params: QkScoreParams) -> Option<Vec<u8>> {
    let len = params.q_heads * params.seq * params.seq;
    bits.iter().all(|lane| lane.len() == len).then_some(())?;
    Some(
        (0..len)
            .map(|index| {
                bits.iter().enumerate().fold(0_u8, |byte, (bit, lane)| {
                    byte | (u8::from(lane[index]) << bit)
                })
            })
            .collect(),
    )
}

fn collect_score_table(table: Vec<i32>, params: QkScoreParams) -> Option<Vec<i32>> {
    (table.len() == params.q_heads * params.seq * params.seq).then_some(table)
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

fn bind(values: &mut Vec<Fr>, r: Fr) {
    let one_minus_r = Fr::one() - r;
    for index in 0..values.len() / 2 {
        values[index] = values[2 * index] * one_minus_r + values[2 * index + 1] * r;
    }
    values.truncate(values.len() / 2);
}

enum RoundingBitsState {
    Bytes(Vec<u8>),
    AffineTags { tags: Vec<u16>, r: Fr },
    Affine2Tags { tags: Vec<u32>, values: [Fr; 16] },
    FieldBits(Vec<[Fr; FRAC_BITS]>),
}

impl RoundingBitsState {
    fn from_bytes(bytes: Vec<u8>) -> Self {
        Self::Bytes(bytes)
    }

    fn len(&self) -> usize {
        match self {
            Self::Bytes(bytes) => bytes.len(),
            Self::AffineTags { tags, .. } => tags.len(),
            Self::Affine2Tags { tags, .. } => tags.len(),
            Self::FieldBits(bits) => bits.len(),
        }
    }

    fn bits_at(&self, index: usize) -> [Fr; FRAC_BITS] {
        match self {
            Self::Bytes(bytes) => byte_bits(bytes[index]),
            Self::AffineTags { tags, r } => affine_tag_bits(tags[index], *r),
            Self::Affine2Tags { tags, values } => affine2_tag_bits(tags[index], values),
            Self::FieldBits(bits) => bits[index],
        }
    }

    fn bind(&mut self, r: Fr) {
        match self {
            Self::Bytes(bytes) => {
                let next = (0..bytes.len() / 2)
                    .map(|index| affine_tags_from_bytes(bytes[2 * index], bytes[2 * index + 1]))
                    .collect();
                *self = Self::AffineTags { tags: next, r };
            }
            Self::AffineTags { tags, r: r0 } => {
                let next = (0..tags.len() / 2)
                    .map(|index| {
                        affine2_tags_from_affine1_pair(tags[2 * index], tags[2 * index + 1])
                    })
                    .collect();
                *self = Self::Affine2Tags {
                    tags: next,
                    values: affine2_value_table(*r0, r),
                };
            }
            Self::Affine2Tags { tags, values } => {
                let bind_values = affine2_bind_table(values, r);
                let next = (0..tags.len() / 2)
                    .map(|index| {
                        bind_affine2_tags(tags[2 * index], tags[2 * index + 1], &bind_values)
                    })
                    .collect();
                *self = Self::FieldBits(next);
            }
            Self::FieldBits(bits) => {
                let next_len = bits.len() / 2;
                for index in 0..next_len {
                    bits[index] = bind_bits(bits[2 * index], bits[2 * index + 1], r);
                }
                bits.truncate(next_len);
            }
        }
    }
}

fn pair_bits(lower: [Fr; FRAC_BITS], upper: [Fr; FRAC_BITS]) -> [(Fr, Fr); FRAC_BITS] {
    std::array::from_fn(|bit| (lower[bit], upper[bit]))
}

fn bit_read_evals(
    index: usize,
    bits: &RoundingBitsState,
    rem_gamma: Fr,
    msb_gamma: Fr,
    challenges: &[Fr; FRAC_BITS],
) -> [Fr; 3] {
    match bits {
        RoundingBitsState::Bytes(bytes) => bit_read_from_bytes(
            bytes[2 * index],
            bytes[2 * index + 1],
            rem_gamma,
            msb_gamma,
            challenges,
        ),
        RoundingBitsState::AffineTags { tags, r } => {
            let lower = affine_tag_bits(tags[2 * index], *r);
            let upper = affine_tag_bits(tags[2 * index + 1], *r);
            bit_read_summary(&lower, &upper, rem_gamma, msb_gamma, challenges)
        }
        RoundingBitsState::Affine2Tags { tags, values } => {
            let lower = affine2_tag_bits(tags[2 * index], values);
            let upper = affine2_tag_bits(tags[2 * index + 1], values);
            bit_read_summary(&lower, &upper, rem_gamma, msb_gamma, challenges)
        }
        RoundingBitsState::FieldBits(bits) => bit_read_summary(
            &bits[2 * index],
            &bits[2 * index + 1],
            rem_gamma,
            msb_gamma,
            challenges,
        ),
    }
}

fn bit_read_from_bytes(
    lower_byte: u8,
    upper_byte: u8,
    rem_gamma: Fr,
    msb_gamma: Fr,
    challenges: &[Fr; FRAC_BITS],
) -> [Fr; 3] {
    let lower_rem = Fr::from(lower_byte as u64);
    let upper_rem = Fr::from(upper_byte as u64);
    let lower_msb = bit_to_fr((lower_byte >> (FRAC_BITS - 1)) != 0);
    let upper_msb = bit_to_fr((upper_byte >> (FRAC_BITS - 1)) != 0);

    let constant = rem_gamma * lower_rem + msb_gamma * lower_msb;
    let mut linear = rem_gamma * (upper_rem - lower_rem) + msb_gamma * (upper_msb - lower_msb);
    let mut leading = Fr::zero();
    for (bit, challenge) in challenges.iter().enumerate() {
        let lower = ((lower_byte >> bit) & 1) != 0;
        let upper = ((upper_byte >> bit) & 1) != 0;
        match (lower, upper) {
            (false, false) => {}
            (false, true) => {
                linear -= challenge;
                leading += challenge;
            }
            (true, false) => {
                linear -= challenge;
                leading += challenge;
            }
            (true, true) => {}
        }
    }
    [
        constant,
        constant + linear + leading,
        constant + linear + linear + leading * Fr::from(4_u64),
    ]
}

fn bit_read_summary(
    lower_bits: &[Fr; FRAC_BITS],
    upper_bits: &[Fr; FRAC_BITS],
    rem_gamma: Fr,
    msb_gamma: Fr,
    challenges: &[Fr; FRAC_BITS],
) -> [Fr; 3] {
    let lower_rem = bits_to_rem(lower_bits);
    let upper_rem = bits_to_rem(upper_bits);
    let lower_msb = lower_bits[FRAC_BITS - 1];
    let upper_msb = upper_bits[FRAC_BITS - 1];

    let mut constant = rem_gamma * lower_rem + msb_gamma * lower_msb;
    let mut linear = rem_gamma * (upper_rem - lower_rem) + msb_gamma * (upper_msb - lower_msb);
    let mut leading = Fr::zero();
    for (bit, challenge) in challenges.iter().enumerate() {
        let lower = lower_bits[bit];
        let bit_linear = upper_bits[bit] - lower;
        constant += *challenge * lower * (lower - Fr::one());
        linear += *challenge * bit_linear * (lower + lower - Fr::one());
        leading += *challenge * bit_linear * bit_linear;
    }
    [
        constant,
        constant + linear + leading,
        constant + linear + linear + leading * Fr::from(4_u64),
    ]
}

fn bit_read_final_relation(
    bits: [Fr; FRAC_BITS],
    rem_gamma: Fr,
    msb_gamma: Fr,
    booleanity_challenges: &[Fr; FRAC_BITS],
) -> Fr {
    let read = rem_gamma * bits_to_rem(&bits) + msb_gamma * bits[FRAC_BITS - 1];
    let booleanity = bits
        .iter()
        .map(|bit| *bit * (*bit - Fr::one()))
        .zip_eq(booleanity_challenges)
        .map(|(check, challenge)| *challenge * check)
        .sum::<Fr>();
    read + booleanity
}

fn bits_to_rem(bits: &[Fr; FRAC_BITS]) -> Fr {
    bits.iter()
        .enumerate()
        .map(|(bit, value)| Fr::from(1_u64 << bit) * *value)
        .sum()
}

fn bit_to_fr(bit: bool) -> Fr {
    if bit { Fr::one() } else { Fr::zero() }
}

fn byte_bits(byte: u8) -> [Fr; FRAC_BITS] {
    std::array::from_fn(|bit| {
        if ((byte >> bit) & 1) != 0 {
            Fr::one()
        } else {
            Fr::zero()
        }
    })
}

fn affine_tags_from_bytes(lower: u8, upper: u8) -> u16 {
    let mut tags = 0_u16;
    for bit in 0..FRAC_BITS {
        let lower_bit = (lower >> bit) & 1;
        let upper_bit = (upper >> bit) & 1;
        let tag = match (lower_bit, upper_bit) {
            (0, 0) => 0b00,
            (1, 1) => 0b01,
            (0, 1) => 0b10,
            (1, 0) => 0b11,
            _ => unreachable!(),
        };
        tags |= tag << (2 * bit);
    }
    tags
}

fn affine_tag(tags: u16, bit: usize) -> u16 {
    (tags >> (2 * bit)) & 0b11
}

fn affine_tag_value(tag: u16, r: Fr) -> Fr {
    match tag {
        0b00 => Fr::zero(),
        0b01 => Fr::one(),
        0b10 => r,
        0b11 => Fr::one() - r,
        _ => unreachable!(),
    }
}

fn affine_tag_bits(tags: u16, r: Fr) -> [Fr; FRAC_BITS] {
    std::array::from_fn(|bit| affine_tag_value(affine_tag(tags, bit), r))
}

fn affine2_tags_from_affine1_pair(lower: u16, upper: u16) -> u32 {
    let mut tags = 0_u32;
    for bit in 0..FRAC_BITS {
        let lower_tag = affine_tag(lower, bit) as u32;
        let upper_tag = affine_tag(upper, bit) as u32;
        tags |= (lower_tag | (upper_tag << 2)) << (4 * bit);
    }
    tags
}

fn affine2_tag(tags: u32, bit: usize) -> u32 {
    (tags >> (4 * bit)) & 0b1111
}

fn affine2_value_table(r1: Fr, r2: Fr) -> [Fr; 16] {
    std::array::from_fn(|tag| affine2_tag_value(tag as u32, r1, r2))
}

fn affine2_tag_value(tag: u32, r1: Fr, r2: Fr) -> Fr {
    let lower = affine_tag_value((tag & 0b11) as u16, r1);
    let upper = affine_tag_value(((tag >> 2) & 0b11) as u16, r1);
    lower + r2 * (upper - lower)
}

fn affine2_tag_bits(tags: u32, values: &[Fr; 16]) -> [Fr; FRAC_BITS] {
    std::array::from_fn(|bit| values[affine2_tag(tags, bit) as usize])
}

fn affine2_pair_tag(lower: u32, upper: u32, bit: usize) -> usize {
    (affine2_tag(lower, bit) | (affine2_tag(upper, bit) << 4)) as usize
}

fn affine2_bind_table(values: &[Fr; 16], r: Fr) -> [Fr; 256] {
    std::array::from_fn(|pair_tag| {
        let lower = values[pair_tag & 0b1111];
        let upper = values[pair_tag >> 4];
        lower + r * (upper - lower)
    })
}

fn bind_affine2_tags(lower: u32, upper: u32, bind_values: &[Fr; 256]) -> [Fr; FRAC_BITS] {
    std::array::from_fn(|bit| bind_values[affine2_pair_tag(lower, upper, bit)])
}

fn bind_bits(lower: [Fr; FRAC_BITS], upper: [Fr; FRAC_BITS], r: Fr) -> [Fr; FRAC_BITS] {
    std::array::from_fn(|bit| lower[bit] + r * (upper[bit] - lower[bit]))
}

fn add_bit_booleanity(
    constant: &mut Fr,
    linear: &mut Fr,
    leading: &mut Fr,
    challenges: &[Fr; FRAC_BITS],
    bits: &[(Fr, Fr); FRAC_BITS],
) {
    for (challenge, (bit_0, bit_1)) in challenges.iter().zip_eq(bits) {
        let bit_linear = *bit_1 - *bit_0;
        *constant += *challenge * *bit_0 * (*bit_0 - Fr::one());
        *linear += *challenge * bit_linear * (*bit_0 + *bit_0 - Fr::one());
        *leading += *challenge * bit_linear * bit_linear;
    }
}

fn remainder_at_zero(bits: &[(Fr, Fr)]) -> Fr {
    bits.iter()
        .enumerate()
        .map(|(index, (bit_0, _))| Fr::from(1_u64 << index) * bit_0)
        .sum()
}

fn remainder_linear(bits: &[(Fr, Fr)]) -> Fr {
    bits.iter()
        .enumerate()
        .map(|(index, (bit_0, bit_1))| Fr::from(1_u64 << index) * (bit_1 - bit_0))
        .sum()
}

fn eq_table(point: &[Fr]) -> Vec<Fr> {
    (0..(1_usize << point.len()))
        .map(|index| eq_eval(index, point))
        .collect()
}

fn eq_eval(index: usize, point: &[Fr]) -> Fr {
    point
        .iter()
        .enumerate()
        .map(|(bit, point)| {
            if ((index >> bit) & 1) == 1 {
                *point
            } else {
                Fr::one() - point
            }
        })
        .product()
}

fn fr_challenges(challenges: &[<Fr as JoltField>::Challenge]) -> Vec<Fr> {
    challenges
        .iter()
        .map(|challenge| (*challenge).into())
        .collect()
}

fn combine_eq_points(left: &[Fr], right: &[Fr]) -> Option<(Fr, Vec<Fr>)> {
    (left.len() == right.len()).then_some(())?;
    let mut scalar = Fr::one();
    let mut point = Vec::with_capacity(left.len());
    for (left, right) in left.iter().zip_eq(right) {
        let denom = (Fr::one() - left) * (Fr::one() - right) + *left * *right;
        let inv = Field::inverse(&denom)?;
        scalar *= denom;
        point.push(*left * *right * inv);
    }
    Some((scalar, point))
}

fn log2(value: usize) -> usize {
    value.ilog2() as usize
}

fn bit_opening_claims(point: &[Fr], values: [Fr; FRAC_BITS]) -> BitOpeningClaims {
    values.map(|value| EvalClaim::new(value, point.to_vec()))
}
