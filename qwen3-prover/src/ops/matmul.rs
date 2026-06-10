use ark_bn254::Fr;
use ark_ff::{One, Zero};
use itertools::Itertools;
use joltworks::{
    field::JoltField,
    poly::{multilinear_polynomial::BindingOrder, split_eq_poly::GruenSplitEqPolynomial},
    transcripts::Transcript,
};
pub use qwen3_common::ops::matmul::{
    MatMulOutput, MatMulParams, MatMulReduction, MatMulRoundingBits, MatMulVerifierOutput,
    draw_matmul_rounding_bit_challenges,
};
use qwen3_common::{FRAC_BITS, SCALE};

use crate::{
    layer::{BitOpeningClaims, EvalClaim, append_eval_claim},
    profile,
    round_message::{RoundPolynomial, SumCheckRounds, append_round_statement},
};

pub struct MatMulProverOutput {
    pub proof: MatMulOutput,
    pub lhs: EvalClaim,
    pub rhs: EvalClaim,
    pub rhs_claim: Vec<EvalClaim>,
    pub rounding_bits: BitOpeningClaims,
}

pub struct MatMulProverInput {
    pub params: MatMulParams,
    pub witness: MatMulWitness,
}

pub struct MatMulWitness {
    pub lhs: Vec<i32>,
    pub rhs: Vec<i32>,
    pub output: Vec<i32>,
    pub output_remainder: Vec<u8>,
}

pub fn prove_matmul<Tr>(
    claim: EvalClaim,
    input: MatMulProverInput,
    transcript: &mut Tr,
) -> Option<MatMulProverOutput>
where
    Tr: Transcript,
{
    append_eval_claim(transcript, &claim);

    let params = input.params;
    let witness = input.witness;
    validate_witness(&params, &witness)?;

    (claim.point.len() == params.output_shape.point_len()).then_some(())?;
    let output_point = claim.point.clone();
    let (row_point, col_point) = output_point.split_at(params.output_shape.row_vars());

    // Rounding:
    //   256 * out(row, col) + rem(row, col) - 256 * msb(row, col)
    //     = Σ_k lhs(row, k) * rhs(k, col)
    let (output_rem, output_msb) =
        prepare_output_rounding_claims(&witness, &params, row_point, col_point);
    append_eval_claim(transcript, &output_rem);
    append_eval_claim(transcript, &output_msb);

    // Product reduction:
    //   product(row, col) = Σ_k lhs(row, k) * rhs(k, col)
    let (mut lhs_by_inner, mut rhs_by_inner) =
        prepare_inner_product_partials(&witness, &params, row_point, col_point);
    let inner_product_claim =
        Fr::from(SCALE) * claim.value + output_rem.value - Fr::from(SCALE) * output_msb.value;
    let (k_reduction, k_challenges) = prove_inner_product_reduction(
        transcript,
        inner_product_claim,
        &mut lhs_by_inner,
        &mut rhs_by_inner,
    )?;

    let inner_point = fr_challenges(&k_challenges);
    let lhs = EvalClaim::new(lhs_by_inner[0], [row_point, &inner_point].concat());
    let rhs = EvalClaim::new(rhs_by_inner[0], [&inner_point, col_point].concat());

    // Rounding bits:
    //   rem(row, col) = Σ_j 2^j b_j(row, col)
    //   msb(row, col) = b_7(row, col)
    let (rounding_bits, rounding_bit_point) = prove_output_rounding_relation(
        transcript,
        &witness,
        &params,
        &output_point,
        &output_rem,
        &output_msb,
    )?;

    let rounding_bit_claims = bit_opening_claims(&rounding_bit_point, rounding_bits.bits);
    Some(MatMulProverOutput {
        proof: MatMulOutput {
            rounding_bits,
            k_reduction,
            rem: output_rem.value,
            msb: output_msb.value,
            lhs: lhs.value,
            rhs: rhs.value,
        },
        lhs,
        rhs: rhs.clone(),
        rhs_claim: vec![rhs],
        rounding_bits: rounding_bit_claims,
    })
}

fn validate_witness(params: &MatMulParams, witness: &MatMulWitness) -> Option<()> {
    (witness.lhs.len() == params.lhs_shape().len()).then_some(())?;
    (witness.rhs.len() == params.rhs_shape().len()).then_some(())?;
    (witness.output.len() == params.output_shape.len()).then_some(())?;
    (witness.output_remainder.len() == params.output_shape.len()).then_some(())
}

fn prepare_output_rounding_claims(
    witness: &MatMulWitness,
    params: &MatMulParams,
    row_point: &[Fr],
    col_point: &[Fr],
) -> (EvalClaim, EvalClaim) {
    profile::measure("matmul.prepare.output_claims", || {
        (
            output_remainder_claim(&witness.output_remainder, params, row_point, col_point),
            output_msb_claim(&witness.output_remainder, params, row_point, col_point),
        )
    })
}

fn prove_output_rounding_relation<Tr>(
    transcript: &mut Tr,
    witness: &MatMulWitness,
    params: &MatMulParams,
    output_point: &[Fr],
    output_rem: &EvalClaim,
    output_msb: &EvalClaim,
) -> Option<(MatMulRoundingBits, Vec<Fr>)>
where
    Tr: Transcript,
{
    let output_remainder_table = output_remainder_table(&witness.output_remainder, params)?;
    profile::measure("matmul.proof.rounding_bits", || {
        prove_output_rounding_bits(
            output_point.to_vec(),
            output_rem.value,
            output_msb.value,
            &output_remainder_table,
            transcript,
        )
    })
}

fn prepare_inner_product_partials(
    witness: &MatMulWitness,
    params: &MatMulParams,
    row_point: &[Fr],
    col_point: &[Fr],
) -> (Vec<Fr>, Vec<Fr>) {
    profile::measure("matmul.prepare.k_partials", || {
        let row_eq = eq_table(row_point);
        let col_eq = eq_table(col_point);
        let lhs_by_inner = lhs_partial_by_inner(&witness.lhs, params, &row_eq);
        let rhs_by_inner = rhs_partial_by_inner(&witness.rhs, params, &col_eq);
        (lhs_by_inner, rhs_by_inner)
    })
}

fn prove_inner_product_reduction<Tr>(
    transcript: &mut Tr,
    claim: Fr,
    lhs_by_inner: &mut Vec<Fr>,
    rhs_by_inner: &mut Vec<Fr>,
) -> Option<(MatMulReduction, Vec<<Fr as JoltField>::Challenge>)>
where
    Tr: Transcript,
{
    profile::measure("matmul.proof.k_reduction", || {
        prove_product_reduction(claim, lhs_by_inner, rhs_by_inner, transcript)
    })
}

fn prove_product_reduction<Tr>(
    mut claim: Fr,
    left: &mut Vec<Fr>,
    right: &mut Vec<Fr>,
    transcript: &mut Tr,
) -> Option<(MatMulReduction, Vec<<Fr as JoltField>::Challenge>)>
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

    if claim != left[0] * right[0] {
        return None;
    }
    Some((
        MatMulReduction {
            rounds: SumCheckRounds {
                round_polys,
                final_claim: claim,
            },
        },
        challenges,
    ))
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

fn bind(values: &mut Vec<Fr>, r: Fr) {
    let one_minus_r = Fr::one() - r;
    for index in 0..values.len() / 2 {
        values[index] = values[2 * index] * one_minus_r + values[2 * index + 1] * r;
    }
    values.truncate(values.len() / 2);
}

fn lhs_partial_by_inner(lhs: &[i32], params: &MatMulParams, row_eq: &[Fr]) -> Vec<Fr> {
    let rows = params.output_shape.rows;
    (0..params.inner)
        .map(|inner| {
            (0..rows)
                .map(|row| row_eq[row] * Fr::from_i32(lhs[inner * rows + row]))
                .sum()
        })
        .collect()
}

fn rhs_partial_by_inner(rhs: &[i32], params: &MatMulParams, col_eq: &[Fr]) -> Vec<Fr> {
    let cols = params.output_shape.cols;
    (0..params.inner)
        .map(|inner| {
            (0..cols)
                .map(|col| col_eq[col] * Fr::from_i32(rhs[col * params.inner + inner]))
                .sum()
        })
        .collect()
}

fn prove_output_rounding_bits<Tr>(
    point: Vec<Fr>,
    rem: Fr,
    msb: Fr,
    output_remainder: &[u8],
    transcript: &mut Tr,
) -> Option<(MatMulRoundingBits, Vec<Fr>)>
where
    Tr: Transcript,
{
    let ([rem_gamma, msb_gamma], booleanity_challenges) =
        draw_matmul_rounding_bit_challenges(transcript)?;
    let mut claim = rem_gamma * rem + msb_gamma * msb;
    let len = output_remainder.len();
    (len.is_power_of_two()).then_some(())?;
    (point.len() == len.ilog2() as usize).then_some(())?;

    let split_eq_point = point.iter().rev().copied().collect::<Vec<_>>();
    let mut eq = GruenSplitEqPolynomial::new(&split_eq_point, BindingOrder::LowToHigh);
    let mut bits = RoundingBitsState::from_bytes(output_remainder.to_vec());
    let mut round_polys = Vec::with_capacity(point.len());
    let mut challenges = Vec::with_capacity(point.len());

    while bits.len() > 1 {
        let round = profile::measure_detail("matmul.rounding_bits.round_poly", || {
            rounding_bits_round_poly(&eq, &bits, rem_gamma, msb_gamma, &booleanity_challenges)
        })?;
        let challenge = profile::measure_detail("matmul.rounding_bits.transcript", || {
            append_round_statement(transcript, claim, &round);
            transcript.challenge_scalar_optimized::<Fr>()
        });
        let r = challenge.into();
        profile::measure_detail("matmul.rounding_bits.eq_bind", || eq.bind(challenge));
        profile::measure_detail("matmul.rounding_bits.bits_bind", || bits.bind(r));
        claim = round.eval(challenge.into());
        challenges.push(challenge);
        round_polys.push(round);
    }

    let point = fr_challenges(&challenges);
    Some((
        MatMulRoundingBits {
            rounds: SumCheckRounds {
                round_polys,
                final_claim: claim,
            },
            bits: bits.bits_at(0),
        },
        point,
    ))
}

enum RoundingBitsState {
    Bytes(Vec<u8>),
    AffineTags { tags: Vec<u16>, r: Fr },
    Affine2Tags { tags: Vec<u32>, values: [Fr; 16] },
    FieldBits(Vec<[Fr; FRAC_BITS]>),
}

impl RoundingBitsState {
    // The first rounds keep the bit lanes compressed. A byte encodes the
    // initial 8 boolean lanes; after binding, tags encode the small affine
    // value set instead of allocating full Fr lanes immediately.
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

fn byte_bits(byte: u8) -> [Fr; FRAC_BITS] {
    std::array::from_fn(|bit| bit_to_fr(((byte >> bit) & 1) != 0))
}

fn bit_to_fr(bit: bool) -> Fr {
    if bit { Fr::one() } else { Fr::zero() }
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

fn rounding_bits_round_poly(
    eq: &GruenSplitEqPolynomial<Fr>,
    bits: &RoundingBitsState,
    rem_gamma: Fr,
    msb_gamma: Fr,
    booleanity_challenges: &[Fr; FRAC_BITS],
) -> Option<RoundPolynomial<4>> {
    let len = bits.len();
    (eq.len() == len && len % 2 == 0).then_some(())?;
    let relation_evals = eq.par_fold_out_in_unreduced::<9, 3>(&|index| {
        rounding_bits_relation_evals(index, bits, rem_gamma, msb_gamma, booleanity_challenges)
    });
    Some(quadratic_relation_times_eq(
        relation_evals,
        eq.get_current_w(),
        eq.get_current_scalar(),
    ))
}

fn rounding_bits_relation_evals(
    index: usize,
    bits: &RoundingBitsState,
    rem_gamma: Fr,
    msb_gamma: Fr,
    booleanity_challenges: &[Fr; FRAC_BITS],
) -> [Fr; 3] {
    match bits {
        RoundingBitsState::Bytes(bytes) => rounding_bits_summary_from_bytes(
            bytes[2 * index],
            bytes[2 * index + 1],
            rem_gamma,
            msb_gamma,
            booleanity_challenges,
        ),
        RoundingBitsState::AffineTags { tags, r } => rounding_bits_summary_from_affine_tags(
            tags[2 * index],
            tags[2 * index + 1],
            *r,
            rem_gamma,
            msb_gamma,
            booleanity_challenges,
        ),
        RoundingBitsState::Affine2Tags { tags, values } => rounding_bits_summary_from_affine2_tags(
            tags[2 * index],
            tags[2 * index + 1],
            values,
            rem_gamma,
            msb_gamma,
            booleanity_challenges,
        ),
        RoundingBitsState::FieldBits(bits) => rounding_bits_summary(
            &bits[2 * index],
            &bits[2 * index + 1],
            rem_gamma,
            msb_gamma,
            booleanity_challenges,
        ),
    }
}

fn quadratic_relation_times_eq(
    relation_coeffs: [Fr; 3],
    current_w: Fr,
    current_scalar: Fr,
) -> RoundPolynomial<4> {
    let [q0, q1, q2] = relation_coeffs;

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

fn rounding_bits_summary_from_bytes(
    lower_byte: u8,
    upper_byte: u8,
    rem_gamma: Fr,
    msb_gamma: Fr,
    booleanity_challenges: &[Fr; FRAC_BITS],
) -> [Fr; 3] {
    let lower_rem = Fr::from(lower_byte as u64);
    let upper_rem = Fr::from(upper_byte as u64);
    let lower_msb = bit_to_fr((lower_byte >> (FRAC_BITS - 1)) != 0);
    let upper_msb = bit_to_fr((upper_byte >> (FRAC_BITS - 1)) != 0);

    let constant = rem_gamma * lower_rem + msb_gamma * lower_msb;
    let mut linear = rem_gamma * (upper_rem - lower_rem) + msb_gamma * (upper_msb - lower_msb);
    let mut leading = Fr::zero();

    for bit in 0..FRAC_BITS {
        let lower = ((lower_byte >> bit) & 1) != 0;
        let upper = ((upper_byte >> bit) & 1) != 0;
        let booleanity_challenge = booleanity_challenges[bit];

        match (lower, upper) {
            (false, false) => {}
            (false, true) => {
                linear -= booleanity_challenge;
                leading += booleanity_challenge;
            }
            (true, false) => {
                linear -= booleanity_challenge;
                leading += booleanity_challenge;
            }
            (true, true) => {}
        }
    }

    [constant, linear, leading]
}

fn rounding_bits_summary_from_affine_tags(
    lower_tags: u16,
    upper_tags: u16,
    r: Fr,
    rem_gamma: Fr,
    msb_gamma: Fr,
    booleanity_challenges: &[Fr; FRAC_BITS],
) -> [Fr; 3] {
    let lower_bits = affine_tag_bits(lower_tags, r);
    let upper_bits = affine_tag_bits(upper_tags, r);
    rounding_bits_summary(
        &lower_bits,
        &upper_bits,
        rem_gamma,
        msb_gamma,
        booleanity_challenges,
    )
}

fn rounding_bits_summary_from_affine2_tags(
    lower_tags: u32,
    upper_tags: u32,
    values: &[Fr; 16],
    rem_gamma: Fr,
    msb_gamma: Fr,
    booleanity_challenges: &[Fr; FRAC_BITS],
) -> [Fr; 3] {
    let lower_bits = affine2_tag_bits(lower_tags, values);
    let upper_bits = affine2_tag_bits(upper_tags, values);
    rounding_bits_summary(
        &lower_bits,
        &upper_bits,
        rem_gamma,
        msb_gamma,
        booleanity_challenges,
    )
}

fn rounding_bits_summary(
    lower_bits: &[Fr; FRAC_BITS],
    upper_bits: &[Fr; FRAC_BITS],
    rem_gamma: Fr,
    msb_gamma: Fr,
    booleanity_challenges: &[Fr; FRAC_BITS],
) -> [Fr; 3] {
    let lower_rem = bits_to_rem(lower_bits);
    let upper_rem = bits_to_rem(upper_bits);
    let lower_msb = lower_bits[FRAC_BITS - 1];
    let upper_msb = upper_bits[FRAC_BITS - 1];

    let mut constant = rem_gamma * lower_rem + msb_gamma * lower_msb;
    let mut linear = rem_gamma * (upper_rem - lower_rem) + msb_gamma * (upper_msb - lower_msb);
    let mut leading = Fr::zero();

    for bit in 0..FRAC_BITS {
        let lower = lower_bits[bit];
        let upper = upper_bits[bit];
        let bit_linear = upper - lower;
        let booleanity_challenge = booleanity_challenges[bit];

        constant += booleanity_challenge * lower * (lower - Fr::one());
        linear += booleanity_challenge * bit_linear * (lower + lower - Fr::one());
        leading += booleanity_challenge * bit_linear * bit_linear;
    }

    [constant, linear, leading]
}

fn bits_to_rem(bits: &[Fr; FRAC_BITS]) -> Fr {
    bits.iter()
        .enumerate()
        .map(|(bit, value)| Fr::from(1_u64 << bit) * *value)
        .sum()
}

fn output_remainder_claim(
    output_remainder: &[u8],
    params: &MatMulParams,
    row_point: &[Fr],
    col_point: &[Fr],
) -> EvalClaim {
    let row_eq = eq_table(row_point);
    let col_eq = eq_table(col_point);
    EvalClaim {
        value: output_byte_at_point(output_remainder, params, &row_eq, &col_eq),
        point: [row_point, col_point].concat(),
    }
}

fn output_msb_claim(
    output_remainder: &[u8],
    params: &MatMulParams,
    row_point: &[Fr],
    col_point: &[Fr],
) -> EvalClaim {
    let row_eq = eq_table(row_point);
    let col_eq = eq_table(col_point);
    EvalClaim {
        value: output_round_bit_at_point(output_remainder, params, &row_eq, &col_eq),
        point: [row_point, col_point].concat(),
    }
}

fn output_remainder_table(output_remainder: &[u8], params: &MatMulParams) -> Option<Vec<u8>> {
    (output_remainder.len() == params.output_shape.len()).then_some(())?;
    Some(output_remainder.to_vec())
}

fn output_byte_at_point(
    output_remainder: &[u8],
    params: &MatMulParams,
    row_eq: &[Fr],
    col_eq: &[Fr],
) -> Fr {
    output_value_at_point(output_remainder, params, row_eq, col_eq, |byte| {
        Fr::from(byte as u64)
    })
}

fn output_round_bit_at_point(
    output_remainder: &[u8],
    params: &MatMulParams,
    row_eq: &[Fr],
    col_eq: &[Fr],
) -> Fr {
    output_value_at_point(output_remainder, params, row_eq, col_eq, |byte| {
        Fr::from((byte >> 7) as u64)
    })
}

fn output_value_at_point(
    output_remainder: &[u8],
    params: &MatMulParams,
    row_eq: &[Fr],
    col_eq: &[Fr],
    value: impl Fn(u8) -> Fr,
) -> Fr {
    (0..params.output_shape.cols)
        .map(|col| {
            col_eq[col]
                * (0..params.output_shape.rows)
                    .map(|row| {
                        row_eq[row] * value(output_remainder[col * params.output_shape.rows + row])
                    })
                    .sum::<Fr>()
        })
        .sum()
}

fn eq_table(point: &[Fr]) -> Vec<Fr> {
    let mut table = vec![Fr::one()];
    for r in point {
        let one_minus_r = Fr::one() - r;
        for index in 0..table.len() {
            let value = table[index];
            table[index] = value * one_minus_r;
            table.push(value * r);
        }
    }
    table
}

fn fr_challenges(challenges: &[<Fr as JoltField>::Challenge]) -> Vec<Fr> {
    challenges
        .iter()
        .map(|challenge| (*challenge).into())
        .collect()
}

fn bit_opening_claims(point: &[Fr], values: [Fr; FRAC_BITS]) -> BitOpeningClaims {
    values.map(|value| EvalClaim::new(value, point.to_vec()))
}
