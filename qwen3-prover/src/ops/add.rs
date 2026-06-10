use ark_bn254::Fr;
use ark_ff::{One, Zero};
use itertools::Itertools;
use joltworks::{
    field::JoltField,
    poly::{multilinear_polynomial::BindingOrder, split_eq_poly::GruenSplitEqPolynomial},
    transcripts::Transcript,
};
pub use qwen3_common::ops::add::{AddOutput, AddParams, AddVerifierOutput};

use crate::{
    layer::{EvalClaim, append_eval_claim},
    round_message::{RoundPolynomial, SumCheckRounds, append_round_statement},
    shape::MatrixShape,
};

pub struct AddProverOutput {
    pub proof: AddOutput,
    pub lhs_claim: EvalClaim,
    pub rhs_claim: EvalClaim,
}

pub struct AddProverInput {
    pub params: AddParams,
    pub witness: AddWitness,
}

pub struct AddWitness {
    pub lhs: Vec<i32>,
    pub rhs: Vec<i32>,
}

pub fn prove_add<T>(
    output_claim: EvalClaim,
    input: AddProverInput,
    transcript: &mut T,
) -> Option<AddProverOutput>
where
    T: Transcript,
{
    append_eval_claim(transcript, &output_claim);
    validate_input(&input)?;
    (output_claim.point.len() == input.params.shape.point_len()).then_some(())?;

    let lhs_value = eval_matrix_i32(&input.witness.lhs, input.params.shape, &output_claim.point)?;
    let rhs_value = eval_matrix_i32(&input.witness.rhs, input.params.shape, &output_claim.point)?;
    (output_claim.value == lhs_value + rhs_value).then_some(())?;

    let lhs_claim = EvalClaim {
        value: lhs_value,
        point: output_claim.point.clone(),
    };
    let rhs_claim = EvalClaim {
        value: rhs_value,
        point: output_claim.point,
    };

    Some(AddProverOutput {
        proof: AddOutput {
            lhs: lhs_claim.value,
            rhs: rhs_claim.value,
            rounds: None,
        },
        lhs_claim,
        rhs_claim,
    })
}

pub fn prove_add_claims<T>(
    output_claims: Vec<EvalClaim>,
    input: AddProverInput,
    transcript: &mut T,
) -> Option<AddProverOutput>
where
    T: Transcript,
{
    if output_claims.len() == 1 {
        return prove_add(output_claims.into_iter().next()?, input, transcript);
    }
    validate_input(&input)?;
    output_claims
        .iter()
        .all(|claim| claim.point.len() == input.params.shape.point_len())
        .then_some(())?;
    append_output_claims(transcript, &output_claims)?;

    let (mut claim_i, mut eq) = combine_appended_output_claims(transcript, &output_claims)?;
    let mut lhs = collect_matrix_table(input.witness.lhs, input.params.shape)?;
    let mut rhs = collect_matrix_table(input.witness.rhs, input.params.shape)?;
    let mut round_polys = Vec::with_capacity(input.params.shape.point_len());
    let mut challenges = Vec::with_capacity(input.params.shape.point_len());

    while lhs.len() > 1 {
        let round = add_round_poly(&eq, &lhs, &rhs)?;
        append_round_statement(transcript, claim_i, &round);
        let challenge = transcript.challenge_scalar_optimized::<Fr>();
        let r = challenge.into();
        eq.bind(challenge);
        bind(&mut lhs, r);
        bind(&mut rhs, r);
        claim_i = round.eval(r);
        challenges.push(challenge);
        round_polys.push(round);
    }

    let point = fr_challenges(&challenges);
    let lhs_claim = EvalClaim {
        value: lhs[0],
        point: point.clone(),
    };
    let rhs_claim = EvalClaim {
        value: rhs[0],
        point,
    };

    Some(AddProverOutput {
        proof: AddOutput {
            lhs: lhs_claim.value,
            rhs: rhs_claim.value,
            rounds: Some(SumCheckRounds {
                round_polys,
                final_claim: claim_i,
            }),
        },
        lhs_claim,
        rhs_claim,
    })
}

fn validate_input(input: &AddProverInput) -> Option<()> {
    let shape = input.params.shape;
    (input.witness.lhs.len() == shape.len()).then_some(())?;
    (input.witness.rhs.len() == shape.len()).then_some(())
}

fn eval_matrix_i32(values: &[i32], shape: MatrixShape, point: &[Fr]) -> Option<Fr> {
    (values.len() == shape.len() && point.len() == shape.point_len()).then(|| {
        let (row_point, col_point) = point.split_at(shape.row_vars());
        (0..shape.rows)
            .map(|row| {
                eq_eval(row, row_point)
                    * (0..shape.cols)
                        .map(|col| {
                            eq_eval(col, col_point) * Fr::from(values[col * shape.rows + row])
                        })
                        .sum::<Fr>()
            })
            .sum()
    })
}

struct CombinedEqState {
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

fn append_output_claims<T>(transcript: &mut T, claims: &[EvalClaim]) -> Option<()>
where
    T: Transcript,
{
    (!claims.is_empty()).then_some(())?;
    let point_len = claims[0].point.len();
    claims
        .iter()
        .all(|claim| claim.point.len() == point_len)
        .then_some(())?;

    transcript.append_message(b"q3/add/output_claims/v1");
    transcript.append_scalar(&Fr::from(claims.len() as u64));
    for claim in claims {
        append_eval_claim(transcript, claim);
    }
    Some(())
}

fn combine_appended_output_claims<T>(
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
        .map(|(claim, coefficient)| *coefficient * claim.value)
        .sum();
    let eq = CombinedEqState::new(
        claims.iter().map(|claim| claim.point.as_slice()),
        &coefficients,
    )?;
    Some((claim, eq))
}

fn add_round_poly(eq: &CombinedEqState, lhs: &[Fr], rhs: &[Fr]) -> Option<RoundPolynomial<3>> {
    (eq.len() == lhs.len() && lhs.len() == rhs.len() && lhs.len() % 2 == 0).then_some(())?;
    let mut coeffs = [Fr::zero(); 3];
    for (coefficient, term) in &eq.terms {
        let relation_evals =
            term.par_fold_out_in_unreduced::<9, 3>(&|row| add_relation_evals(row, lhs, rhs));
        let round = linear_relation_times_eq(
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

fn add_relation_evals(row: usize, lhs: &[Fr], rhs: &[Fr]) -> [Fr; 3] {
    let lhs_0 = lhs[2 * row];
    let lhs_linear = lhs[2 * row + 1] - lhs_0;
    let rhs_0 = rhs[2 * row];
    let rhs_linear = rhs[2 * row + 1] - rhs_0;
    [
        lhs_0 + rhs_0,
        lhs_0 + lhs_linear + rhs_0 + rhs_linear,
        lhs_0 + Fr::from(2_u64) * lhs_linear + rhs_0 + Fr::from(2_u64) * rhs_linear,
    ]
}

fn linear_relation_times_eq(
    relation_evals: [Fr; 3],
    current_w: Fr,
    current_scalar: Fr,
) -> RoundPolynomial<3> {
    let relation_constant = relation_evals[0];
    let relation_linear = relation_evals[1] - relation_evals[0];
    let eq_constant = Fr::one() - current_w;
    let eq_linear = current_w + current_w - Fr::one();
    RoundPolynomial {
        coeffs: [
            current_scalar * eq_constant * relation_constant,
            current_scalar * (eq_constant * relation_linear + eq_linear * relation_constant),
            current_scalar * eq_linear * relation_linear,
        ],
    }
}

fn collect_matrix_table(table: Vec<i32>, shape: MatrixShape) -> Option<Vec<Fr>> {
    (table.len() == shape.len()).then(|| table.into_iter().map(Fr::from).collect())
}

fn bind(values: &mut Vec<Fr>, r: Fr) {
    let one_minus_r = Fr::one() - r;
    for index in 0..values.len() / 2 {
        values[index] = values[2 * index] * one_minus_r + values[2 * index + 1] * r;
    }
    values.truncate(values.len() / 2);
}

fn fr_challenges(challenges: &[<Fr as JoltField>::Challenge]) -> Vec<Fr> {
    challenges.iter().copied().map(Into::into).collect()
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
