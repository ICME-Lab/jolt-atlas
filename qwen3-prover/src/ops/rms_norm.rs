use ark_bn254::Fr;
use ark_ff::{Field, One, Zero};
use itertools::Itertools;
use joltworks::{
    field::JoltField,
    poly::{multilinear_polynomial::BindingOrder, split_eq_poly::GruenSplitEqPolynomial},
    transcripts::Transcript,
};
pub use qwen3_common::ops::rms_norm::{
    RmsNormAdvice, RmsNormNormalizedInput, RmsNormOutput, RmsNormParams, RmsNormVerifierOutput,
    RmsNormWeightedOutput, draw_rms_norm_normalized_input_challenges,
    draw_rms_norm_weighted_output_challenges,
};
use qwen3_common::{FRAC_BITS, MatrixShape};

use crate::{
    layer::{BitOpeningClaims, EvalClaim, append_eval_claim},
    profile,
    round_message::{RoundPolynomial, SumCheckRounds, append_round_statement},
};

pub struct RmsNormProverOutput {
    pub proof: RmsNormOutput,
    pub input: EvalClaim,
    pub norm_bits: BitOpeningClaims,
    pub output_bits: BitOpeningClaims,
}

struct RoundingProverData {
    rounds: SumCheckRounds<4>,
    point: Vec<Fr>,
    values: [Fr; 3],
    bits: [Fr; FRAC_BITS],
}

struct RmsNormWeightedProverOutput {
    proof: RmsNormWeightedOutput,
    norm: EvalClaim,
    output_bits: BitOpeningClaims,
}

struct RmsNormNormalizedProverOutput {
    proof: RmsNormNormalizedInput,
    input: EvalClaim,
    norm_bits: BitOpeningClaims,
}

pub struct RmsNormProverInput {
    pub params: RmsNormParams,
    pub advice: RmsNormAdvice,
    pub witness: RmsNormWitness,
}

pub struct RmsNormWitness {
    pub input: Vec<i32>,
    pub inv_rms: Vec<i32>,
    pub norm: Vec<i32>,
    pub weight: Vec<i32>,
    pub output: Vec<i32>,
    pub norm_remainder_bits: [Vec<bool>; FRAC_BITS],
    pub output_remainder_bits: [Vec<bool>; FRAC_BITS],
}

pub fn combine_rms_norm_output_claims<T>(
    transcript: &mut T,
    claims: &[EvalClaim],
) -> Option<(Fr, CombinedEqState)>
where
    T: Transcript,
{
    append_rms_norm_output_claims(transcript, claims)?;
    combine_appended_rms_norm_output_claims(transcript, claims)
}

fn append_rms_norm_output_claims<T>(transcript: &mut T, claims: &[EvalClaim]) -> Option<()>
where
    T: Transcript,
{
    (!claims.is_empty()).then_some(())?;
    let point_len = claims[0].point.len();
    claims
        .iter()
        .all(|claim| claim.point.len() == point_len)
        .then_some(())?;

    transcript.append_message(b"q3/rms_norm/output_claims/v1");
    transcript.append_scalar(&Fr::from(claims.len() as u64));
    for claim in claims {
        append_eval_claim(transcript, claim);
    }
    Some(())
}

fn combine_appended_rms_norm_output_claims<T>(
    transcript: &mut T,
    claims: &[EvalClaim],
) -> Option<(Fr, CombinedEqState)>
where
    T: Transcript,
{
    let mut coefficients = Vec::with_capacity(claims.len());
    coefficients.push(Fr::one());
    for _ in 1..claims.len() {
        coefficients.push(transcript.challenge_scalar_optimized::<Fr>().into());
    }

    let claim = claims
        .iter()
        .zip_eq(&coefficients)
        .map(|(claim, coeff)| *coeff * claim.value)
        .sum();
    let eq = CombinedEqState::new(
        claims.iter().map(|claim| claim.point.as_slice()),
        &coefficients,
    )?;
    Some((claim, eq))
}

// Sumcheck chain:
//   C. weighted output:
//      output + α * (256*output - norm*weight + rem - 256*msb) = 0
//      plus bit booleanity, under the combined output-claim eq.
//   B. normalized input + row square:
//      norm + α * (256*norm - input*inv_rms + rem - 256*msb) = 0
//      plus bit booleanity, under the norm claim eq.
//      sum_x2(row) = Σ_col input(row,col)^2, under eq(row_claim,row).
pub fn prove_rms_norm<Tr>(
    output_claims: Vec<EvalClaim>,
    input: RmsNormProverInput,
    transcript: &mut Tr,
) -> Option<RmsNormProverOutput>
where
    Tr: Transcript,
{
    validate_input(&input)?;
    validate_output_claims(&output_claims, input.params.shape)?;
    append_rms_norm_output_claims(transcript, &output_claims)?;

    let params = input.params;
    let (combined_claim, combined_eq) =
        combine_appended_rms_norm_output_claims(transcript, &output_claims)?;
    let witness = input.witness;

    let input_table = collect_matrix_table(witness.input, params.shape)?;
    let inv_rms = collect_matrix_table(witness.inv_rms, params.shape)?;
    let norm = collect_matrix_table(witness.norm, params.shape)?;
    let weight = collect_matrix_table(witness.weight, params.shape)?;
    let output = collect_matrix_table(witness.output, params.shape)?;
    let norm_bits = collect_matrix_bytes(witness.norm_remainder_bits, params.shape)?;
    let output_bits = collect_matrix_bytes(witness.output_remainder_bits, params.shape)?;

    let weighted_output = profile::measure("rms_norm.proof.weighted_output", || {
        prove_weighted_output_sumcheck(
            combined_claim,
            combined_eq,
            norm.clone(),
            weight,
            output,
            output_bits,
            transcript,
        )
    })?;

    let normalized_input = profile::measure("rms_norm.proof.normalized_input", || {
        let row_point = weighted_output.norm.point[..params.shape.row_vars()].to_vec();
        let row_square_claim = eval_i64_at_point(&input.advice.sum_x2, &row_point)?;
        prove_normalized_input_sumcheck(
            weighted_output.norm.value,
            weighted_output.norm.point.clone(),
            row_square_claim,
            row_point,
            input_table,
            inv_rms,
            norm,
            norm_bits,
            transcript,
        )
    })?;

    Some(RmsNormProverOutput {
        input: normalized_input.input,
        norm_bits: normalized_input.norm_bits,
        output_bits: weighted_output.output_bits,
        proof: RmsNormOutput {
            weighted_output: weighted_output.proof,
            normalized_input: normalized_input.proof,
        },
    })
}

fn prove_weighted_output_sumcheck<Tr>(
    claim: Fr,
    eq: CombinedEqState,
    norm: Vec<Fr>,
    weight: Vec<Fr>,
    output: Vec<Fr>,
    output_bits: Vec<u8>,
    transcript: &mut Tr,
) -> Option<RmsNormWeightedProverOutput>
where
    Tr: Transcript,
{
    let (round_mix, bit_challenges) = draw_rms_norm_weighted_output_challenges(transcript)?;
    let mut polynomials = [norm, weight, output];
    let mut bits = RoundingBitsState::from_bytes(output_bits);
    let data = prove_rounding_relation_combined_sumcheck(
        RoundingRelation::WeightedOutput {
            round_mix,
            bit_challenges,
        },
        claim,
        eq,
        &mut polynomials,
        &mut bits,
        transcript,
    )?;
    Some(RmsNormWeightedProverOutput {
        norm: EvalClaim::new(data.values[0], data.point.clone()),
        output_bits: bit_opening_claims(&data.point, data.bits),
        proof: RmsNormWeightedOutput {
            rounds: data.rounds,
            norm: data.values[0],
            output: data.values[2],
            bits: data.bits,
        },
    })
}

fn prove_normalized_input_sumcheck<Tr>(
    claim: Fr,
    point: Vec<Fr>,
    row_square_claim: Fr,
    row_point: Vec<Fr>,
    input: Vec<Fr>,
    inv_rms: Vec<Fr>,
    norm: Vec<Fr>,
    norm_bits: Vec<u8>,
    transcript: &mut Tr,
) -> Option<RmsNormNormalizedProverOutput>
where
    Tr: Transcript,
{
    let (round_mix, row_square_mix, bit_challenges) =
        draw_rms_norm_normalized_input_challenges(transcript)?;
    let mut polynomials = [input, inv_rms, norm];
    let mut bits = RoundingBitsState::from_bytes(norm_bits);
    let split_eq_point = point.iter().rev().copied().collect::<Vec<_>>();
    let eq = GruenSplitEqPolynomial::new(&split_eq_point, BindingOrder::LowToHigh);
    let split_row_point = row_point.iter().rev().copied().collect::<Vec<_>>();
    let row_eq = GruenSplitEqPolynomial::new(&split_row_point, BindingOrder::LowToHigh);
    let data = prove_normalized_input_with_row_square_sumcheck(
        RoundingRelation::NormalizedInput {
            round_mix,
            bit_challenges,
        },
        claim + row_square_mix * row_square_claim,
        eq,
        row_square_mix,
        row_eq,
        &mut polynomials,
        &mut bits,
        transcript,
    )?;
    Some(RmsNormNormalizedProverOutput {
        input: EvalClaim::new(data.values[0], data.point.clone()),
        norm_bits: bit_opening_claims(&data.point, data.bits),
        proof: RmsNormNormalizedInput {
            rounds: data.rounds,
            input: data.values[0],
            inv_rms: data.values[1],
            norm: data.values[2],
            bits: data.bits,
        },
    })
}

fn prove_rounding_relation_combined_sumcheck<Tr>(
    relation: RoundingRelation,
    mut claim: Fr,
    mut eq: CombinedEqState,
    polynomials: &mut [Vec<Fr>; 3],
    bits: &mut RoundingBitsState,
    transcript: &mut Tr,
) -> Option<RoundingProverData>
where
    Tr: Transcript,
{
    let len = polynomials[0].len();
    (len.is_power_of_two()
        && polynomials.iter().all(|poly| poly.len() == len)
        && eq.len() == len
        && bits.len() == len)
        .then_some(())?;

    let mut round_polys = Vec::with_capacity(len.ilog2() as usize);
    let mut challenges = Vec::with_capacity(len.ilog2() as usize);

    while polynomials[0].len() > 1 {
        let round = profile::measure_detail("rms_norm.rounding.round_poly", || {
            rounding_round_poly_combined(&relation, &eq, polynomials, bits)
        })?;
        let challenge = profile::measure_detail("rms_norm.rounding.transcript", || {
            append_round_statement(transcript, claim, &round);
            transcript.challenge_scalar_optimized::<Fr>()
        });
        let r = challenge.into();
        profile::measure_detail("rms_norm.rounding.poly_bind", || {
            for poly in polynomials.iter_mut() {
                bind(poly, r);
            }
        });
        profile::measure_detail("rms_norm.rounding.eq_bind", || eq.bind(challenge));
        profile::measure_detail("rms_norm.rounding.bits_bind", || bits.bind(r));
        claim = round.eval(r);
        challenges.push(challenge);
        round_polys.push(round);
    }

    Some(RoundingProverData {
        rounds: SumCheckRounds {
            round_polys,
            final_claim: claim,
        },
        point: fr_challenges(&challenges),
        values: [polynomials[0][0], polynomials[1][0], polynomials[2][0]],
        bits: bits.bits_at(0),
    })
}

fn prove_normalized_input_with_row_square_sumcheck<Tr>(
    relation: RoundingRelation,
    mut claim: Fr,
    mut eq: GruenSplitEqPolynomial<Fr>,
    row_square_mix: Fr,
    mut row_eq: GruenSplitEqPolynomial<Fr>,
    polynomials: &mut [Vec<Fr>; 3],
    bits: &mut RoundingBitsState,
    transcript: &mut Tr,
) -> Option<RoundingProverData>
where
    Tr: Transcript,
{
    let len = polynomials[0].len();
    (len.is_power_of_two()
        && polynomials.iter().all(|poly| poly.len() == len)
        && eq.len() == len
        && len % row_eq.len() == 0
        && bits.len() == len)
        .then_some(())?;

    let mut round_polys = Vec::with_capacity(len.ilog2() as usize);
    let mut challenges = Vec::with_capacity(len.ilog2() as usize);

    while polynomials[0].len() > 1 {
        let round = profile::measure_detail("rms_norm.rounding.round_poly", || {
            let mut round = rounding_round_poly_split(&relation, &eq, polynomials, bits)?;
            let square_round = row_square_round_poly(&row_eq, &polynomials[0])?;
            for (out, square) in round.coeffs.iter_mut().zip_eq(square_round.coeffs) {
                *out += row_square_mix * square;
            }
            Some(round)
        })?;
        let challenge = profile::measure_detail("rms_norm.rounding.transcript", || {
            append_round_statement(transcript, claim, &round);
            transcript.challenge_scalar_optimized::<Fr>()
        });
        let r = challenge.into();
        if row_eq.len() > 1 {
            profile::measure_detail("rms_norm.rounding.row_eq_bind", || row_eq.bind(challenge));
        }
        profile::measure_detail("rms_norm.rounding.poly_bind", || {
            for poly in polynomials.iter_mut() {
                bind(poly, r);
            }
        });
        profile::measure_detail("rms_norm.rounding.eq_bind", || eq.bind(challenge));
        profile::measure_detail("rms_norm.rounding.bits_bind", || bits.bind(r));
        claim = round.eval(r);
        challenges.push(challenge);
        round_polys.push(round);
    }

    Some(RoundingProverData {
        rounds: SumCheckRounds {
            round_polys,
            final_claim: claim,
        },
        point: fr_challenges(&challenges),
        values: [polynomials[0][0], polynomials[1][0], polynomials[2][0]],
        bits: bits.bits_at(0),
    })
}

fn rounding_round_poly_split(
    relation: &RoundingRelation,
    eq: &GruenSplitEqPolynomial<Fr>,
    polynomials: &[Vec<Fr>; 3],
    bits: &RoundingBitsState,
) -> Option<RoundPolynomial<4>> {
    let len = polynomials[0].len();
    (polynomials.iter().all(|poly| poly.len() == len)
        && eq.len() == len
        && bits.len() == len
        && len % 2 == 0)
        .then_some(())?;

    let relation_evals = eq.par_fold_out_in_unreduced::<9, 3>(&|index| {
        rounding_relation_evals(relation, polynomials, bits, index)
    });
    Some(quadratic_relation_times_eq(
        relation_evals,
        eq.get_current_w(),
        eq.get_current_scalar(),
    ))
}

fn rounding_round_poly_combined(
    relation: &RoundingRelation,
    eq: &CombinedEqState,
    polynomials: &[Vec<Fr>; 3],
    bits: &RoundingBitsState,
) -> Option<RoundPolynomial<4>> {
    let len = polynomials[0].len();
    (polynomials.iter().all(|poly| poly.len() == len)
        && eq.len() == len
        && bits.len() == len
        && len % 2 == 0)
        .then_some(())?;

    let mut coeffs = [Fr::zero(); 4];
    for (coefficient, term) in &eq.terms {
        let relation_evals = term.par_fold_out_in_unreduced::<9, 3>(&|index| {
            rounding_relation_evals(relation, polynomials, bits, index)
        });
        let round = quadratic_relation_times_eq(
            relation_evals,
            term.get_current_w(),
            term.get_current_scalar(),
        );
        for (out, term_coeff) in coeffs.iter_mut().zip_eq(round.coeffs) {
            *out += *coefficient * term_coeff;
        }
    }
    Some(RoundPolynomial { coeffs })
}

fn rounding_relation_evals(
    relation: &RoundingRelation,
    polynomials: &[Vec<Fr>; 3],
    bits: &RoundingBitsState,
    index: usize,
) -> [Fr; 3] {
    let bit_pairs = pair_bits(bits.bits_at(2 * index), bits.bits_at(2 * index + 1));
    let evals = [
        (polynomials[0][2 * index], polynomials[0][2 * index + 1]),
        (polynomials[1][2 * index], polynomials[1][2 * index + 1]),
        (polynomials[2][2 * index], polynomials[2][2 * index + 1]),
    ];
    let [constant, linear, leading] = relation.coeffs(evals, bit_pairs);
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

fn row_square_round_poly(
    eq: &GruenSplitEqPolynomial<Fr>,
    input: &[Fr],
) -> Option<RoundPolynomial<4>> {
    (input.len() % 2 == 0 && input.len() % eq.len() == 0).then_some(())?;
    if eq.len() == 1 {
        return row_square_round_poly_scalar(eq.get_current_scalar(), input);
    }

    let rows = eq.len();
    let cols = input.len() / rows;
    let relation_evals = eq.par_fold_out_in_unreduced::<9, 3>(&|row| {
        let mut evals = [Fr::zero(); 3];
        for col in 0..cols {
            let base = col * rows + 2 * row;
            let square = square_relation_evals(input[base], input[base + 1]);
            evals[0] += square[0];
            evals[1] += square[1];
            evals[2] += square[2];
        }
        evals
    });
    Some(quadratic_relation_times_eq(
        relation_evals,
        eq.get_current_w(),
        eq.get_current_scalar(),
    ))
}

fn row_square_round_poly_scalar(scalar: Fr, input: &[Fr]) -> Option<RoundPolynomial<4>> {
    (input.len() % 2 == 0).then_some(())?;
    let mut coeffs = [Fr::zero(); 4];
    for index in 0..input.len() / 2 {
        let square = square_relation_coeffs(input[2 * index], input[2 * index + 1]);
        coeffs[0] += scalar * square[0];
        coeffs[1] += scalar * square[1];
        coeffs[2] += scalar * square[2];
    }
    Some(RoundPolynomial { coeffs })
}

fn square_relation_evals(input_0: Fr, input_1: Fr) -> [Fr; 3] {
    let coeffs = square_relation_coeffs(input_0, input_1);
    [
        coeffs[0],
        coeffs[0] + coeffs[1] + coeffs[2],
        coeffs[0] + coeffs[1] + coeffs[1] + Fr::from(4_u64) * coeffs[2],
    ]
}

fn square_relation_coeffs(input_0: Fr, input_1: Fr) -> [Fr; 3] {
    let input_linear = input_1 - input_0;
    [
        input_0 * input_0,
        input_linear * (input_0 + input_0),
        input_linear * input_linear,
    ]
}

#[derive(Clone, Copy)]
enum RoundingRelation {
    WeightedOutput {
        round_mix: Fr,
        bit_challenges: [Fr; FRAC_BITS],
    },
    NormalizedInput {
        round_mix: Fr,
        bit_challenges: [Fr; FRAC_BITS],
    },
}

impl RoundingRelation {
    fn coeffs(&self, evals: [(Fr, Fr); 3], bits: [(Fr, Fr); FRAC_BITS]) -> [Fr; 3] {
        match *self {
            Self::WeightedOutput {
                round_mix,
                bit_challenges,
            } => rounding_coeffs(evals, bits, round_mix, bit_challenges, true),
            Self::NormalizedInput {
                round_mix,
                bit_challenges,
            } => rounding_coeffs(evals, bits, round_mix, bit_challenges, false),
        }
    }
}

fn rounding_coeffs(
    evals: [(Fr, Fr); 3],
    bits: [(Fr, Fr); FRAC_BITS],
    round_mix: Fr,
    bit_challenges: [Fr; FRAC_BITS],
    weighted_output: bool,
) -> [Fr; 3] {
    let [(left_0, left_1), (right_0, right_1), (out_0, out_1)] = evals;
    let left_linear = left_1 - left_0;
    let right_linear = right_1 - right_0;
    let out_linear = out_1 - out_0;
    let scale = Fr::from(256_u64);
    let remainder_0 = remainder_at_zero(&bits);
    let remainder_linear = remainder_linear(&bits);
    let msb_0 = bits[FRAC_BITS - 1].0;
    let msb_linear = bits[FRAC_BITS - 1].1 - msb_0;

    let mut constant = out_0;
    let mut linear = out_linear;
    let mut leading = Fr::zero();

    let round_constant = scale * out_0 - left_0 * right_0 + remainder_0 - scale * msb_0;
    let round_linear = scale * out_linear - (left_0 * right_linear + left_linear * right_0)
        + remainder_linear
        - scale * msb_linear;
    let round_leading = -left_linear * right_linear;

    constant += round_mix * round_constant;
    linear += round_mix * round_linear;
    leading += round_mix * round_leading;

    if !weighted_output {
        // Same polynomial form; the flag is kept to make the relation names explicit.
    }

    add_bit_booleanity(
        &mut constant,
        &mut linear,
        &mut leading,
        &bit_challenges,
        &bits,
    );
    [constant, linear, leading]
}

fn validate_output_claims(claims: &[EvalClaim], shape: MatrixShape) -> Option<()> {
    (claims.len() >= 2).then_some(())?;
    claims
        .iter()
        .all(|claim| claim.point.len() == shape.point_len())
        .then_some(())
}

fn validate_input(input: &RmsNormProverInput) -> Option<()> {
    let shape = input.params.shape;
    (input.advice.sum_x2.len() == shape.rows).then_some(())?;
    (input.witness.input.len() == shape.len()).then_some(())?;
    (input.witness.inv_rms.len() == shape.len()).then_some(())?;
    (input.witness.norm.len() == shape.len()).then_some(())?;
    (input.witness.weight.len() == shape.len()).then_some(())?;
    (input.witness.output.len() == shape.len()).then_some(())?;
    input
        .witness
        .norm_remainder_bits
        .iter()
        .all(|bit| bit.len() == shape.len())
        .then_some(())?;
    input
        .witness
        .output_remainder_bits
        .iter()
        .all(|bit| bit.len() == shape.len())
        .then_some(())
}

pub fn rms_inv_from_square_sum(square_sum: i64, cols: usize) -> i32 {
    if square_sum <= 0 || cols == 0 {
        return 0;
    }
    let input_scale = 256.0_f64;
    let output_scale = 256.0_f64;
    let mean = square_sum as f64 / cols as f64 / (input_scale * input_scale);
    let inv = 1.0 / (mean + 1e-6).sqrt();
    (inv * output_scale).round() as i32
}

fn collect_matrix_table(table: Vec<i32>, shape: MatrixShape) -> Option<Vec<Fr>> {
    (table.len() == shape.len()).then(|| table.into_iter().map(Fr::from_i32).collect())
}

fn collect_matrix_bytes(bits: [Vec<bool>; FRAC_BITS], shape: MatrixShape) -> Option<Vec<u8>> {
    bits.iter()
        .all(|bit| bit.len() == shape.len())
        .then_some(())?;
    Some(
        (0..shape.len())
            .map(|index| {
                bits.iter().enumerate().fold(0_u8, |byte, (bit, lane)| {
                    byte | (u8::from(lane[index]) << bit)
                })
            })
            .collect(),
    )
}

pub struct CombinedEqState {
    terms: Vec<(Fr, GruenSplitEqPolynomial<Fr>)>,
}

impl CombinedEqState {
    fn new<'a>(points: impl IntoIterator<Item = &'a [Fr]>, coefficients: &[Fr]) -> Option<Self> {
        let points = points.into_iter().collect::<Vec<_>>();
        (points.len() == coefficients.len() && !points.is_empty()).then_some(())?;
        let point_len = points[0].len();
        points
            .iter()
            .all(|point| point.len() == point_len)
            .then_some(())?;
        let terms = points
            .iter()
            .zip_eq(coefficients)
            .map(|(point, coefficient)| {
                let split_eq_point = point.iter().rev().copied().collect::<Vec<_>>();
                (
                    *coefficient,
                    GruenSplitEqPolynomial::new(&split_eq_point, BindingOrder::LowToHigh),
                )
            })
            .collect();
        Some(Self { terms })
    }

    fn len(&self) -> usize {
        self.terms
            .first()
            .map(|(_, term)| term.len())
            .unwrap_or_default()
    }

    fn bind(&mut self, r: <Fr as JoltField>::Challenge) {
        for (_, term) in &mut self.terms {
            term.bind(r);
        }
    }
}

fn eval_i64_at_point(values: &[i64], point: &[Fr]) -> Option<Fr> {
    (values.len() == (1_usize << point.len())).then(|| {
        values
            .iter()
            .enumerate()
            .map(|(index, value)| eq_eval(index, point) * Fr::from(*value))
            .sum()
    })
}

fn bind(values: &mut Vec<Fr>, r: Fr) {
    let one_minus_r = Fr::one() - r;
    for index in 0..values.len() / 2 {
        values[index] = values[2 * index] * one_minus_r + values[2 * index + 1] * r;
    }
    values.truncate(values.len() / 2);
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
