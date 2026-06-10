use ark_bn254::Fr;
use ark_ff::One;
use itertools::Itertools;
use joltworks::{
    poly::{multilinear_polynomial::BindingOrder, split_eq_poly::GruenSplitEqPolynomial},
    transcripts::Transcript,
};
use qwen3_common::ops::add::{AddOutput, AddVerifierOutput};

use qwen3_common::{EvalClaim, append_eval_claim, verify_sumcheck_rounds};

use crate::utils::eq_point_eval;

pub fn verify_add<T>(
    output_claim: EvalClaim,
    proof: &AddOutput,
    transcript: &mut T,
) -> Option<AddVerifierOutput>
where
    T: Transcript,
{
    append_eval_claim(transcript, &output_claim);

    match &proof.rounds {
        None => {
            // No sumcheck is needed when the add result is used at a single
            // point: the same point is passed to both operands.
            (proof.lhs + proof.rhs == output_claim.value).then_some(())?;
            Some(AddVerifierOutput {
                lhs_claim: EvalClaim::new(proof.lhs, output_claim.point.clone()),
                rhs_claim: EvalClaim::new(proof.rhs, output_claim.point),
            })
        }
        Some(rounds) => {
            // Fanout add sumcheck for a single already-combined claim:
            //
            //   claim = Σ_x eq(z, x) * (lhs(x) + rhs(x))
            //
            // It absorbs the fanout and returns one shared point for lhs/rhs.
            let claim = output_claim.value;
            let sumcheck =
                verify_sumcheck_rounds(claim, rounds, output_claim.point.len(), transcript)?;
            (proof.lhs + proof.rhs == sumcheck.final_claim).then_some(())?;
            Some(AddVerifierOutput {
                lhs_claim: EvalClaim::new(proof.lhs, sumcheck.point.clone()),
                rhs_claim: EvalClaim::new(proof.rhs, sumcheck.point),
            })
        }
    }
}

pub fn verify_add_claims<T>(
    output_claims: Vec<EvalClaim>,
    proof: &AddOutput,
    transcript: &mut T,
) -> Option<AddVerifierOutput>
where
    T: Transcript,
{
    if output_claims.len() == 1 {
        return verify_add(output_claims.into_iter().next()?, proof, transcript);
    }
    let rounds = proof.rounds.as_ref()?;
    append_output_claims(transcript, &output_claims)?;
    // Multiple output claims are batched before the add sumcheck:
    //
    //   Σ_i gamma_i * out_i(z_i)
    //     = Σ_x (Σ_i gamma_i * eq(z_i, x)) * (lhs(x) + rhs(x))
    //
    // The combined eq is evaluated at the sumcheck final point.
    let (claim, eq) = combine_appended_output_claims(transcript, &output_claims)?;
    let sumcheck = verify_sumcheck_rounds(claim, rounds, output_claims[0].point.len(), transcript)?;
    (eq.eval_at(&sumcheck.challenges)? * (proof.lhs + proof.rhs) == sumcheck.final_claim)
        .then_some(())?;
    Some(AddVerifierOutput {
        lhs_claim: EvalClaim::new(proof.lhs, sumcheck.point.clone()),
        rhs_claim: EvalClaim::new(proof.rhs, sumcheck.point),
    })
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
            points: points.iter().map(|point| point.to_vec()).collect(),
        })
    }

    fn eval_at(&self, point: &[<Fr as joltworks::field::JoltField>::Challenge]) -> Option<Fr> {
        self.terms
            .iter()
            .zip_eq(&self.points)
            .map(|((coefficient, _), term_point)| {
                Some(*coefficient * eq_point_eval(term_point, point)?)
            })
            .sum()
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
