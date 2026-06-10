use ark_bn254::Fr;
use ark_ff::{Field, One, Zero};
use itertools::Itertools;
use joltworks::{
    field::JoltField,
    poly::{multilinear_polynomial::BindingOrder, split_eq_poly::GruenSplitEqPolynomial},
    transcripts::Transcript,
};
pub use qwen3_common::ops::rope::{
    RopeOutput, RopeParams, RopeVerifierOutput, draw_rope_challenges,
};
use qwen3_common::{FRAC_BITS, SCALE};

use crate::{
    layer::{BitOpeningClaims, EvalClaim, append_eval_claim},
    profile,
    round_message::{RoundPolynomial, SumCheckRounds, append_round_statement},
};

pub struct RopeProverOutput {
    pub proof: RopeOutput,
    pub input_first_half: EvalClaim,
    pub input_second_half: EvalClaim,
    pub first_half_bits: BitOpeningClaims,
    pub second_half_bits: BitOpeningClaims,
    pub input_first_half_claim: Vec<EvalClaim>,
    pub input_second_half_claim: Vec<EvalClaim>,
}

pub struct RopeProverInput {
    pub params: RopeParams,
    pub witness: RopeWitness,
}

pub struct RopeWitness {
    pub input: Vec<i32>,
    pub output: Vec<i32>,
    pub output_remainder_bits: [Vec<bool>; FRAC_BITS],
    pub cos: Vec<i32>,
    pub sin: Vec<i32>,
}

struct RopeRelation {
    parity: Fr,
    rotation_constraint_mix: Fr,
    first_half_bit_booleanity_challenges: [Fr; FRAC_BITS],
    second_half_bit_booleanity_challenges: [Fr; FRAC_BITS],
}

// Sumcheck relation over x = (seq, head, dim_without_half_bit):
//
//   first_half_out  = (first_half_in * cos - second_half_in * sin) / 256
//   second_half_out = (first_half_in * sin + second_half_in * cos) / 256
//
// The incoming claim point contains the half selector as the final dim bit.
// The relation selects the corresponding half with that scalar parity:
//
//   selected_out = (1 - parity) * first_half_out + parity * second_half_out
//
// Rounding and remainder bits are checked in the same sumcheck:
//
//   256 * selected_out + rem - 256 * msb - selected_rotation = 0
//   b_j * (b_j - 1) = 0
pub fn prove_rope<Tr>(
    claim: EvalClaim,
    input: RopeProverInput,
    transcript: &mut Tr,
) -> Option<RopeProverOutput>
where
    Tr: Transcript,
{
    append_eval_claim(transcript, &claim);
    validate_input(&input)?;
    (claim.point.len() == input.params.tensor_vars()).then_some(())?;

    let params = input.params;
    let witness = input.witness;
    let (r_seq, r_head, r_pair, r_parity) = split_tensor_point(&claim.point, &params);
    (r_parity.len() == 1).then_some(())?;

    let (
        rotation_constraint_mix,
        first_half_bit_booleanity_challenges,
        second_half_bit_booleanity_challenges,
    ) = draw_rope_challenges(transcript)?;
    let relation = RopeRelation {
        parity: r_parity[0],
        rotation_constraint_mix,
        first_half_bit_booleanity_challenges,
        second_half_bit_booleanity_challenges,
    };

    let (
        mut input_first_half,
        mut input_second_half,
        mut cos_for_heads,
        mut sin_for_heads,
        mut output_first_half,
        mut output_second_half,
        mut first_half_bits,
        mut second_half_bits,
    ) = profile::measure("rope.prepare.relation_polys", || {
        Some((
            tensor_half(&witness.input, &params, 0)?,
            tensor_half(&witness.input, &params, 1)?,
            coeff_for_heads(&witness.cos, &params)?,
            coeff_for_heads(&witness.sin, &params)?,
            tensor_half(&witness.output, &params, 0)?,
            tensor_half(&witness.output, &params, 1)?,
            RoundingBitsState::from_bytes(tensor_half_bytes(
                &witness.output_remainder_bits,
                &params,
                0,
            )?),
            RoundingBitsState::from_bytes(tensor_half_bytes(
                &witness.output_remainder_bits,
                &params,
                1,
            )?),
        ))
    })?;

    let relation_point = [r_seq, r_head, r_pair].concat();
    let split_eq_point = relation_point.iter().rev().copied().collect::<Vec<_>>();
    let mut eq = GruenSplitEqPolynomial::new(&split_eq_point, BindingOrder::LowToHigh);

    let mut claim_i = claim.value;
    let mut round_polys = Vec::with_capacity(params.relation_vars());
    let mut challenges = Vec::with_capacity(params.relation_vars());

    profile::measure("rope.proof.relation", || {
        while input_first_half.len() > 1 {
            let round = profile::measure_detail("rope.relation.round_poly", || {
                rope_round_poly(
                    &eq,
                    &relation,
                    &input_first_half,
                    &input_second_half,
                    &cos_for_heads,
                    &sin_for_heads,
                    &output_first_half,
                    &output_second_half,
                    &first_half_bits,
                    &second_half_bits,
                )
            })?;
            let challenge = profile::measure_detail("rope.relation.transcript", || {
                append_round_statement(transcript, claim_i, &round);
                transcript.challenge_scalar_optimized::<Fr>()
            });
            let r = challenge.into();
            profile::measure_detail("rope.relation.eq_bind", || eq.bind(challenge));
            profile::measure_detail("rope.relation.input_first_half_bind", || {
                bind(&mut input_first_half, r)
            });
            profile::measure_detail("rope.relation.input_second_half_bind", || {
                bind(&mut input_second_half, r)
            });
            profile::measure_detail("rope.relation.cos_bind", || bind(&mut cos_for_heads, r));
            profile::measure_detail("rope.relation.sin_bind", || bind(&mut sin_for_heads, r));
            profile::measure_detail("rope.relation.output_first_half_bind", || {
                bind(&mut output_first_half, r)
            });
            profile::measure_detail("rope.relation.output_second_half_bind", || {
                bind(&mut output_second_half, r)
            });
            profile::measure_detail("rope.relation.first_half_bits_bind", || {
                first_half_bits.bind(r)
            });
            profile::measure_detail("rope.relation.second_half_bits_bind", || {
                second_half_bits.bind(r)
            });

            claim_i = round.eval(r);
            challenges.push(challenge);
            round_polys.push(round);
        }
        Some(())
    })?;

    let point = fr_challenges(&challenges);
    let (seq_point, head_point, pair_point) = split_relation_point(&point, &params);
    let input_first_half_point = matrix_half_point(seq_point, head_point, pair_point, Fr::zero());
    let input_second_half_point = matrix_half_point(seq_point, head_point, pair_point, Fr::one());
    let first_half_point = [seq_point, head_point, pair_point, &[Fr::zero()]].concat();
    let second_half_point = [seq_point, head_point, pair_point, &[Fr::one()]].concat();

    let input_first_half_claim = vec![EvalClaim {
        value: input_first_half[0],
        point: first_half_point.clone(),
    }];
    let input_second_half_claim = vec![EvalClaim {
        value: input_second_half[0],
        point: second_half_point.clone(),
    }];
    let input_first_half = EvalClaim {
        value: input_first_half[0],
        point: input_first_half_point,
    };
    let input_second_half = EvalClaim {
        value: input_second_half[0],
        point: input_second_half_point,
    };

    let first_half_bits_at_point = first_half_bits.bits_at(0);
    let second_half_bits_at_point = second_half_bits.bits_at(0);
    Some(RopeProverOutput {
        proof: RopeOutput {
            rounds: SumCheckRounds {
                round_polys,
                final_claim: claim_i,
            },
            input_first_half: input_first_half.value,
            input_second_half: input_second_half.value,
            output_first_half: output_first_half[0],
            output_second_half: output_second_half[0],
            first_half_bits: first_half_bits_at_point,
            second_half_bits: second_half_bits_at_point,
        },
        input_first_half,
        input_second_half,
        first_half_bits: bit_opening_claims(&first_half_point, first_half_bits_at_point),
        second_half_bits: bit_opening_claims(&second_half_point, second_half_bits_at_point),
        input_first_half_claim,
        input_second_half_claim,
    })
}

fn validate_input(input: &RopeProverInput) -> Option<()> {
    let params = input.params;
    params.validate().then_some(())?;
    let tensor_len = params.seq * params.heads * params.head_dim;
    let coeff_len = params.seq * (params.head_dim / 2);
    (input.witness.input.len() == tensor_len).then_some(())?;
    (input.witness.output.len() == tensor_len).then_some(())?;
    (input.witness.cos.len() == coeff_len).then_some(())?;
    (input.witness.sin.len() == coeff_len).then_some(())?;
    input
        .witness
        .output_remainder_bits
        .iter()
        .all(|bits| bits.len() == tensor_len)
        .then_some(())
}

#[allow(clippy::too_many_arguments)]
fn rope_round_poly(
    eq: &GruenSplitEqPolynomial<Fr>,
    relation: &RopeRelation,
    input_first_half: &[Fr],
    input_second_half: &[Fr],
    cos: &[Fr],
    sin: &[Fr],
    output_first_half: &[Fr],
    output_second_half: &[Fr],
    first_half_bits: &RoundingBitsState,
    second_half_bits: &RoundingBitsState,
) -> Option<RoundPolynomial<4>> {
    let len = input_first_half.len();
    (eq.len() == len
        && input_second_half.len() == len
        && cos.len() == len
        && sin.len() == len
        && output_first_half.len() == len
        && output_second_half.len() == len
        && first_half_bits.len() == len
        && second_half_bits.len() == len
        && len % 2 == 0)
        .then_some(())?;

    let relation_evals = eq.par_fold_out_in_unreduced::<9, 3>(&|index| {
        rope_relation_evals(
            index,
            relation,
            input_first_half,
            input_second_half,
            cos,
            sin,
            output_first_half,
            output_second_half,
            first_half_bits,
            second_half_bits,
        )
    });
    Some(quadratic_relation_times_eq(
        relation_evals,
        eq.get_current_w(),
        eq.get_current_scalar(),
    ))
}

#[allow(clippy::too_many_arguments)]
fn rope_relation_evals(
    index: usize,
    relation: &RopeRelation,
    input_first_half: &[Fr],
    input_second_half: &[Fr],
    cos: &[Fr],
    sin: &[Fr],
    output_first_half: &[Fr],
    output_second_half: &[Fr],
    first_half_bits: &RoundingBitsState,
    second_half_bits: &RoundingBitsState,
) -> [Fr; 3] {
    let input_first_0 = input_first_half[2 * index];
    let input_first_linear = input_first_half[2 * index + 1] - input_first_0;
    let input_second_0 = input_second_half[2 * index];
    let input_second_linear = input_second_half[2 * index + 1] - input_second_0;
    let cos_0 = cos[2 * index];
    let cos_linear = cos[2 * index + 1] - cos_0;
    let sin_0 = sin[2 * index];
    let sin_linear = sin[2 * index + 1] - sin_0;
    let output_first_0 = output_first_half[2 * index];
    let output_first_linear = output_first_half[2 * index + 1] - output_first_0;
    let output_second_0 = output_second_half[2 * index];
    let output_second_linear = output_second_half[2 * index + 1] - output_second_0;
    let first_bits = pair_bits(
        first_half_bits.bits_at(2 * index),
        first_half_bits.bits_at(2 * index + 1),
    );
    let second_bits = pair_bits(
        second_half_bits.bits_at(2 * index),
        second_half_bits.bits_at(2 * index + 1),
    );

    let one_minus_parity = Fr::one() - relation.parity;
    let output_0 = one_minus_parity * output_first_0 + relation.parity * output_second_0;
    let output_linear =
        one_minus_parity * output_first_linear + relation.parity * output_second_linear;
    let remainder_0 = one_minus_parity * remainder_constant(&first_bits)
        + relation.parity * remainder_constant(&second_bits);
    let remainder_linear = one_minus_parity * remainder_linear(&first_bits)
        + relation.parity * remainder_linear(&second_bits);
    let msb_0 = one_minus_parity * first_bits[FRAC_BITS - 1].0
        + relation.parity * second_bits[FRAC_BITS - 1].0;
    let msb_linear = one_minus_parity * (first_bits[FRAC_BITS - 1].1 - first_bits[FRAC_BITS - 1].0)
        + relation.parity * (second_bits[FRAC_BITS - 1].1 - second_bits[FRAC_BITS - 1].0);

    let coeff_first_0 = one_minus_parity * cos_0 + relation.parity * sin_0;
    let coeff_first_linear = one_minus_parity * cos_linear + relation.parity * sin_linear;
    let coeff_second_0 = -one_minus_parity * sin_0 + relation.parity * cos_0;
    let coeff_second_linear = -one_minus_parity * sin_linear + relation.parity * cos_linear;
    let rotation_0 = input_first_0 * coeff_first_0 + input_second_0 * coeff_second_0;
    let rotation_linear = input_first_0 * coeff_first_linear
        + input_first_linear * coeff_first_0
        + input_second_0 * coeff_second_linear
        + input_second_linear * coeff_second_0;
    let rotation_leading =
        input_first_linear * coeff_first_linear + input_second_linear * coeff_second_linear;

    let mut constant = output_0
        + relation.rotation_constraint_mix
            * (Fr::from(SCALE) * output_0 + remainder_0 - Fr::from(SCALE) * msb_0 - rotation_0);
    let mut linear = output_linear
        + relation.rotation_constraint_mix
            * (Fr::from(SCALE) * output_linear + remainder_linear
                - Fr::from(SCALE) * msb_linear
                - rotation_linear);
    let mut leading = -relation.rotation_constraint_mix * rotation_leading;

    add_bit_booleanity(
        &mut constant,
        &mut linear,
        &mut leading,
        &relation.first_half_bit_booleanity_challenges,
        &first_bits,
    );
    add_bit_booleanity(
        &mut constant,
        &mut linear,
        &mut leading,
        &relation.second_half_bit_booleanity_challenges,
        &second_bits,
    );

    [
        constant,
        constant + linear + leading,
        constant + linear + linear + leading * Fr::from(4_u64),
    ]
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

fn tensor_half(values: &[i32], params: &RopeParams, half_selector: usize) -> Option<Vec<Fr>> {
    (values.len() == params.seq * params.heads * params.head_dim).then_some(())?;
    Some(
        (0..params.head_dim / 2)
            .flat_map(|pair| {
                (0..params.heads).flat_map(move |head| {
                    (0..params.seq).map(move |seq| {
                        Fr::from_i32(
                            values[source_tensor_index(seq, head, pair, half_selector, params)],
                        )
                    })
                })
            })
            .collect(),
    )
}

fn tensor_half_bytes(
    bits: &[Vec<bool>; FRAC_BITS],
    params: &RopeParams,
    half_selector: usize,
) -> Option<Vec<u8>> {
    bits.iter()
        .all(|lane| lane.len() == params.seq * params.heads * params.head_dim)
        .then_some(())?;
    Some(
        (0..params.head_dim / 2)
            .flat_map(|pair| {
                (0..params.heads).flat_map(move |head| {
                    (0..params.seq).map(move |seq| {
                        let source = source_tensor_index(seq, head, pair, half_selector, params);
                        bits.iter().enumerate().fold(0_u8, |byte, (bit, lane)| {
                            byte | (u8::from(lane[source]) << bit)
                        })
                    })
                })
            })
            .collect(),
    )
}

fn coeff_for_heads(values: &[i32], params: &RopeParams) -> Option<Vec<Fr>> {
    (values.len() == params.seq * (params.head_dim / 2)).then_some(())?;
    Some(
        (0..params.head_dim / 2)
            .flat_map(|pair| {
                (0..params.heads).flat_map(move |_| {
                    (0..params.seq).map(move |seq| Fr::from_i32(values[pair * params.seq + seq]))
                })
            })
            .collect(),
    )
}

fn source_tensor_index(
    seq: usize,
    head: usize,
    pair: usize,
    half_selector: usize,
    params: &RopeParams,
) -> usize {
    let dim = pair + (params.head_dim / 2) * half_selector;
    dim * (params.seq * params.heads) + head * params.seq + seq
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

fn remainder_constant(bits: &[(Fr, Fr); FRAC_BITS]) -> Fr {
    bits.iter()
        .enumerate()
        .map(|(bit, (value, _))| Fr::from(1_u64 << bit) * *value)
        .sum()
}

fn remainder_linear(bits: &[(Fr, Fr); FRAC_BITS]) -> Fr {
    bits.iter()
        .enumerate()
        .map(|(bit, (lower, upper))| Fr::from(1_u64 << bit) * (*upper - *lower))
        .sum()
}

fn fr_challenges(challenges: &[<Fr as JoltField>::Challenge]) -> Vec<Fr> {
    challenges.iter().copied().map(Into::into).collect()
}

fn bit_opening_claims(point: &[Fr], values: [Fr; FRAC_BITS]) -> BitOpeningClaims {
    values.map(|value| EvalClaim::new(value, point.to_vec()))
}

fn log2(value: usize) -> usize {
    value.ilog2() as usize
}
