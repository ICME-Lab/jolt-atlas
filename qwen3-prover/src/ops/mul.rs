use ark_bn254::Fr;
use ark_ff::{Field, One, Zero};
use itertools::Itertools;
use joltworks::{
    field::JoltField,
    poly::{multilinear_polynomial::BindingOrder, split_eq_poly::GruenSplitEqPolynomial},
    transcripts::Transcript,
};
use qwen3_common::FRAC_BITS;
pub use qwen3_common::ops::mul::{
    MulInputEvals, MulOutput, MulParams, MulPublicEvals, MulVerifierOutput,
    draw_mul_booleanity_challenges,
};

use crate::{
    layer::{BitOpeningClaims, EvalClaim, append_eval_claim},
    profile,
    round_message::{RoundPolynomial, SumCheckRounds, append_round_statement},
};

pub struct MulProverOutput {
    pub proof: MulOutput,
    pub lhs: EvalClaim,
    pub rhs: EvalClaim,
    pub bits: BitOpeningClaims,
}

pub struct MulProverInput {
    pub params: MulParams,
    pub witness: MulWitness,
}

pub struct MulWitness {
    pub a: Vec<i32>,
    pub b: Vec<i32>,
    pub bits: [Vec<bool>; FRAC_BITS],
}

// Sumcheck relation:
//
//   out(r) = Σ_x eq(r, x) * ((a(x) * b(x) - rem_low(x) + 128 * b_7(x)) / 256)
//
// where:
//   rem_low(x) = Σ_{j=0}^6 2^j * b_j(x)
//
// Booleanity for all remainder bits is folded into the same relation:
//   Σ_x eq(r, x) * Σ_j gamma_j * b_j(x) * (b_j(x) - 1) = 0
pub fn prove_mul<Tr>(
    claim: EvalClaim,
    input: MulProverInput,
    transcript: &mut Tr,
) -> Option<MulProverOutput>
where
    Tr: Transcript,
{
    append_eval_claim(transcript, &claim);
    validate_input(&input)?;
    (claim.point.len() == input.params.len.ilog2() as usize).then_some(())?;

    let booleanity_challenges = draw_mul_booleanity_challenges(transcript)?;
    let (mut a, mut b, mut bits) = profile::measure("mul.prepare.polys", || {
        Some((
            collect_values(input.witness.a, input.params)?,
            collect_values(input.witness.b, input.params)?,
            RoundingBitsState::from_bytes(collect_bytes(input.witness.bits, input.params)?),
        ))
    })?;
    let split_eq_point = claim.point.iter().rev().copied().collect::<Vec<_>>();
    let mut eq = GruenSplitEqPolynomial::new(&split_eq_point, BindingOrder::LowToHigh);

    let mut claim_i = claim.value;
    let mut round_polys = Vec::with_capacity(claim.point.len());
    let mut challenges = Vec::with_capacity(claim.point.len());

    profile::measure("mul.proof.rounds", || {
        while a.len() > 1 {
            let round = mul_round_poly(&eq, &a, &b, &bits, &booleanity_challenges)?;
            append_round_statement(transcript, claim_i, &round);

            let challenge = transcript.challenge_scalar_optimized::<Fr>();
            let r = challenge.into();
            eq.bind(challenge);
            bind(&mut a, r);
            bind(&mut b, r);
            bits.bind(r);

            claim_i = round.eval(r);
            challenges.push(challenge);
            round_polys.push(round);
        }
        Some(())
    })?;

    let point = fr_challenges(&challenges);
    let lhs = EvalClaim {
        value: a[0],
        point: point.clone(),
    };
    let rhs = EvalClaim {
        value: b[0],
        point: point.clone(),
    };
    let bits_at_point = bits.bits_at(0);
    let proof = MulOutput {
        rounds: SumCheckRounds {
            round_polys,
            final_claim: claim_i,
        },
        lhs: lhs.value,
        rhs: rhs.value,
        bits: bits_at_point,
    };

    Some(MulProverOutput {
        proof,
        lhs,
        rhs,
        bits: bit_opening_claims(&point, bits_at_point),
    })
}

fn validate_input(input: &MulProverInput) -> Option<()> {
    (input.witness.a.len() == input.params.len).then_some(())?;
    (input.witness.b.len() == input.params.len).then_some(())?;
    input
        .witness
        .bits
        .iter()
        .all(|bits| bits.len() == input.params.len)
        .then_some(())
}

fn collect_values(values: Vec<i32>, params: MulParams) -> Option<Vec<Fr>> {
    (values.len() == params.len).then_some(())?;
    Some(values.into_iter().map(Fr::from_i32).collect())
}

fn collect_bytes(values: [Vec<bool>; FRAC_BITS], params: MulParams) -> Option<Vec<u8>> {
    values
        .iter()
        .all(|bits| bits.len() == params.len)
        .then_some(())?;
    Some(
        (0..params.len)
            .map(|index| {
                values.iter().enumerate().fold(0_u8, |byte, (bit, lane)| {
                    byte | (u8::from(lane[index]) << bit)
                })
            })
            .collect(),
    )
}

fn mul_round_poly(
    eq: &GruenSplitEqPolynomial<Fr>,
    a: &[Fr],
    b: &[Fr],
    bits: &RoundingBitsState,
    booleanity_challenges: &[Fr; FRAC_BITS],
) -> Option<RoundPolynomial<4>> {
    (eq.len() == a.len() && a.len() == b.len() && a.len() % 2 == 0).then_some(())?;
    (bits.len() == a.len()).then_some(())?;

    let shift = Field::inverse(&Fr::from(256_u64)).unwrap();
    let relation_evals = eq.par_fold_out_in_unreduced::<9, 3>(&|index| {
        mul_relation_evals(index, a, b, bits, shift, booleanity_challenges)
    });
    Some(quadratic_relation_times_eq(
        relation_evals,
        eq.get_current_w(),
        eq.get_current_scalar(),
    ))
}

fn mul_relation_evals(
    index: usize,
    a: &[Fr],
    b: &[Fr],
    bits: &RoundingBitsState,
    shift: Fr,
    booleanity_challenges: &[Fr; FRAC_BITS],
) -> [Fr; 3] {
    let a_0 = a[2 * index];
    let a_linear = a[2 * index + 1] - a_0;
    let b_0 = b[2 * index];
    let b_linear = b[2 * index + 1] - b_0;
    let bit_pairs = pair_bits(bits.bits_at(2 * index), bits.bits_at(2 * index + 1));

    let mut constant = a_0 * b_0;
    let mut linear = a_0 * b_linear + a_linear * b_0;
    let mut leading = a_linear * b_linear;

    for (bit, (bit_0, bit_1)) in bit_pairs.iter().copied().enumerate() {
        let bit_linear = bit_1 - bit_0;
        constant -= Fr::from(1_u64 << bit) * bit_0;
        linear -= Fr::from(1_u64 << bit) * bit_linear;
    }
    let (msb_0, msb_1) = bit_pairs[FRAC_BITS - 1];
    let msb_linear = msb_1 - msb_0;
    constant += Fr::from(256_u64) * msb_0;
    linear += Fr::from(256_u64) * msb_linear;

    constant *= shift;
    linear *= shift;
    leading *= shift;

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

fn fr_challenges(challenges: &[<Fr as JoltField>::Challenge]) -> Vec<Fr> {
    challenges.iter().copied().map(Into::into).collect()
}

fn bit_opening_claims(point: &[Fr], values: [Fr; FRAC_BITS]) -> BitOpeningClaims {
    values.map(|value| EvalClaim::new(value, point.to_vec()))
}
