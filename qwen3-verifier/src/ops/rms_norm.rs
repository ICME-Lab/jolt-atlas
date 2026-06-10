use ark_bn254::Fr;
use ark_ff::{One, Zero};
use itertools::Itertools;
use joltworks::{
    poly::{multilinear_polynomial::BindingOrder, split_eq_poly::GruenSplitEqPolynomial},
    transcripts::Transcript,
};
use qwen3_common::ops::rms_norm::{
    RmsNormNormalizedInput, RmsNormOutput, RmsNormParams, RmsNormVerifierOutput,
    RmsNormWeightedOutput, draw_rms_norm_normalized_input_challenges,
    draw_rms_norm_weighted_output_challenges,
};

use qwen3_common::{
    BitOpeningClaims, EvalClaim, FRAC_BITS, MatrixShape, append_eval_claim, verify_sumcheck_rounds,
};

use crate::utils::{eq_point_eval, eval_i32_matrix_at_point};

pub fn verify_rms_norm<Tr>(
    output_claims: Vec<EvalClaim>,
    params: RmsNormParams,
    input: qwen3_common::LayerRmsNormVerifierInput,
    proof: &RmsNormOutput,
    transcript: &mut Tr,
) -> Option<RmsNormVerifierOutput>
where
    Tr: Transcript,
{
    validate_output_claims(&output_claims, params.shape)?;
    append_rms_norm_output_claims(transcript, &output_claims)?;
    let (combined_claim, combined_eq) =
        combine_appended_rms_norm_output_claims(transcript, &output_claims)?;
    // RMSNorm may feed several consumers.  We first batch those output claims:
    //
    //   Σ_i gamma_i * out(z_i)
    //
    // and carry the matching combined eq through the weighted-output sumcheck.
    let weighted_output = verify_weighted_output_sumcheck(
        combined_claim,
        &combined_eq,
        &input.weight,
        &params,
        &proof.weighted_output,
        transcript,
    )?;
    let normalized_input = verify_normalized_input_sumcheck(
        weighted_output.norm.value,
        weighted_output.norm.point.clone(),
        &input.advice.sum_x2,
        params.shape,
        &proof.normalized_input,
        transcript,
    )?;

    Some(RmsNormVerifierOutput {
        input: normalized_input.input,
        norm_bits: normalized_input.norm_bits,
        output_bits: weighted_output.output_bits,
    })
}

struct RmsNormWeightedVerifierOutput {
    norm: EvalClaim,
    output_bits: BitOpeningClaims,
}

fn verify_weighted_output_sumcheck<Tr>(
    claim: Fr,
    eq: &CombinedEqState,
    weight: &[i32],
    params: &RmsNormParams,
    proof: &RmsNormWeightedOutput,
    transcript: &mut Tr,
) -> Option<RmsNormWeightedVerifierOutput>
where
    Tr: Transcript,
{
    let (round_mix, bit_challenges) = draw_rms_norm_weighted_output_challenges(transcript)?;
    // Weighted-output sumcheck:
    //
    //   out(x) = round(norm(x) * weight(x) / 256)
    //
    // The left side may be a batched fanout claim, so the final check uses the
    // combined eq state instead of a single eq(claim.point, x).
    let sumcheck =
        verify_sumcheck_rounds(claim, &proof.rounds, params.shape.point_len(), transcript)?;
    let public_evals = build_public_weighted_output_evals(weight, params, &sumcheck.point)?;
    let relation = rounding_final_relation(
        RoundingRelation::WeightedOutput {
            round_mix,
            bit_challenges,
        },
        proof.norm,
        public_evals.weight.value,
        proof.output,
        proof.bits,
    );
    (sumcheck.final_claim == eq.eval_at(&sumcheck.challenges)? * relation).then_some(())?;
    Some(RmsNormWeightedVerifierOutput {
        norm: EvalClaim::new(proof.norm, sumcheck.point.clone()),
        output_bits: bit_opening_claims(&sumcheck.point, proof.bits),
    })
}

struct RmsNormWeightedPublicEvals {
    weight: EvalClaim,
}

fn build_public_weighted_output_evals(
    weight: &[i32],
    params: &RmsNormParams,
    point: &[Fr],
) -> Option<RmsNormWeightedPublicEvals> {
    Some(RmsNormWeightedPublicEvals {
        weight: EvalClaim::new(
            eval_i32_matrix_at_point(weight, params.shape, point)?,
            point.to_vec(),
        ),
    })
}

struct RmsNormNormalizedVerifierOutput {
    input: EvalClaim,
    norm_bits: BitOpeningClaims,
}

fn verify_normalized_input_sumcheck<Tr>(
    claim: Fr,
    claim_point: Vec<Fr>,
    sum_x2: &[i64],
    shape: MatrixShape,
    proof: &RmsNormNormalizedInput,
    transcript: &mut Tr,
) -> Option<RmsNormNormalizedVerifierOutput>
where
    Tr: Transcript,
{
    let (round_mix, row_square_mix, bit_challenges) =
        draw_rms_norm_normalized_input_challenges(transcript)?;
    let row_point = claim_point[..shape.row_vars()].to_vec();
    let row_square_claim = eval_i64_at_point(sum_x2, &row_point)?;
    // Normalized-input + row-square sumcheck:
    //
    //   norm(x) = round(input(x) * inv_rms(row(x)) / 256)
    //   sum_x2(row(norm_claim)) = Σ_col input(row(norm_claim), col)^2
    //
    // Both constraints share the same final point, so the verifier needs only
    // one input opening claim.
    let sumcheck = verify_sumcheck_rounds(
        claim + row_square_mix * row_square_claim,
        &proof.rounds,
        shape.point_len(),
        transcript,
    )?;
    let relation = rounding_final_relation(
        RoundingRelation::NormalizedInput {
            round_mix,
            bit_challenges,
        },
        proof.input,
        proof.inv_rms,
        proof.norm,
        proof.bits,
    );
    let row_bound = &sumcheck.challenges[..row_point.len()];
    let final_check = eq_point_eval(&claim_point, &sumcheck.challenges)? * relation
        + row_square_mix * eq_point_eval(&row_point, row_bound)? * proof.input * proof.input;
    (sumcheck.final_claim == final_check).then_some(())?;
    Some(RmsNormNormalizedVerifierOutput {
        input: EvalClaim::new(proof.input, sumcheck.point.clone()),
        norm_bits: bit_opening_claims(&sumcheck.point, proof.bits),
    })
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

struct CombinedEqState {
    terms: Vec<(Fr, GruenSplitEqPolynomial<Fr>)>,
    points: Vec<Vec<Fr>>,
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
        let owned_points = points
            .iter()
            .map(|point| point.to_vec())
            .collect::<Vec<_>>();
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
        Some(Self {
            terms,
            points: owned_points,
        })
    }

    fn eval_at(&self, point: &[<Fr as joltworks::field::JoltField>::Challenge]) -> Option<Fr> {
        self.terms
            .iter()
            .zip_eq(&self.points)
            .map(|((coefficient, _), eq_point)| {
                Some(*coefficient * eq_point_eval(eq_point, point)?)
            })
            .sum()
    }
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
            } => rounding_coeffs(evals, bits, round_mix, bit_challenges),
            Self::NormalizedInput {
                round_mix,
                bit_challenges,
            } => rounding_coeffs(evals, bits, round_mix, bit_challenges),
        }
    }
}

fn rounding_final_relation(
    relation: RoundingRelation,
    left: Fr,
    right: Fr,
    out: Fr,
    bits: [Fr; FRAC_BITS],
) -> Fr {
    let bit_pairs = bits.map(|bit| (bit, bit));
    relation.coeffs([(left, left), (right, right), (out, out)], bit_pairs)[0]
}

fn bit_opening_claims(point: &[Fr], values: [Fr; FRAC_BITS]) -> BitOpeningClaims {
    values.map(|value| EvalClaim::new(value, point.to_vec()))
}

fn rounding_coeffs(
    evals: [(Fr, Fr); 3],
    bits: [(Fr, Fr); FRAC_BITS],
    round_mix: Fr,
    bit_challenges: [Fr; FRAC_BITS],
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

    add_bit_booleanity(
        &mut constant,
        &mut linear,
        &mut leading,
        &bit_challenges,
        &bits,
    );
    [constant, linear, leading]
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

fn validate_output_claims(claims: &[EvalClaim], shape: MatrixShape) -> Option<()> {
    (claims.len() >= 2).then_some(())?;
    claims
        .iter()
        .all(|claim| claim.point.len() == shape.point_len())
        .then_some(())
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
