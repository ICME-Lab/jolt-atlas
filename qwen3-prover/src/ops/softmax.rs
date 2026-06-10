//! Softmax proof.
//!
//! The runtime computes causal softmax in Q8:
//!
//! ```text
//! row_max(row) = max_col input(row, col), col <= row % seq
//! diff         = valid(row, col) * (input - row_max)
//! index        = floor(diff / 256) - min_diff
//! frac         = diff - 256 * (min_diff + index)
//! exp_acc      = exp_table[index] * (256 + frac)
//! exp          = round(exp_acc / 256)
//! sum(row)     = Σ_col valid(row, col) * exp(row, col)
//! acc          = exp * inv_sum(sum(row))
//! floor        = floor(acc / 256)
//! output       = round(floor / 256)
//! ```
//!
//! The proof is a claim-reduction chain:
//!
//! ```text
//! target claim: output(r)
//!
//! D. output-to-acc sumcheck
//!    proves output(r) comes from floor(r), acc(r), and both rounding bit sets
//!    outputs claim acc(r)
//!
//! C. exp tensor sumcheck
//!    absorbs acc(r), the row-sum exp claim, and the row-max input claim,
//!    then proves these values come from input, index, exp_base, frac bits,
//!    exp bits, row_max, and mask
//!    outputs claims input(r), index(r), exp_base(r)
//!
//! B. row sum sumcheck
//!    proves sum(row) and row_max(row) through a public max-index selector:
//!      sum(row) + γ*row_max(row)
//!        = Σ_col valid(row,col)*exp(row,col)
//!          + γ*selector_max(row,col)*input(row,col)
//!    outputs claims for exp(row, ·) and input(row, max_index(row))
//!
//! A. lookup/RA sumcheck
//!    proves index(r), exp_base(r) come from ra(r, i) and the exp LUT
//!    outputs opening claims for ra(r, ·) / tables
//! ```

use ark_bn254::Fr;
use ark_ff::{Field, One, Zero};
use itertools::Itertools;
use joltworks::{
    field::JoltField,
    poly::{multilinear_polynomial::BindingOrder, split_eq_poly::GruenSplitEqPolynomial},
    transcripts::Transcript,
};
use qwen3_common::MatrixShape;
pub use qwen3_common::ops::softmax::{
    SoftmaxAdvice, SoftmaxBooleanityOutput, SoftmaxExpOutput, SoftmaxHammingWeightOutput,
    SoftmaxLookupOutput, SoftmaxOutput, SoftmaxOutputToAccOutput, SoftmaxParams,
    SoftmaxRaVirtualOutput, SoftmaxReadOutput, SoftmaxRowSumOutput, SoftmaxVerifierOutput,
    build_softmax_tables, draw_softmax_exp_challenges, draw_softmax_lookup_challenges,
    draw_softmax_output_to_acc_challenges, draw_softmax_row_sum_challenge, validate_softmax_range,
};
use qwen3_common::{FRAC_BITS, SCALE};

use crate::{
    layer::{BitOpeningClaims, EvalClaim, RaOpeningClaims, append_eval_claim},
    profile,
    round_message::{RoundPolynomial, SumCheckRounds, append_round_statement},
};

struct SoftmaxOutputToAccProverOutput {
    proof: SoftmaxOutputToAccOutput,
    acc: EvalClaim,
    output_bits: BitOpeningClaims,
    floor_bits: BitOpeningClaims,
}

struct SoftmaxExpProverOutput {
    proof: SoftmaxExpOutput,
    pub input: EvalClaim,
    pub index: EvalClaim,
    pub exp_base: EvalClaim,
    pub frac_bits: BitOpeningClaims,
    pub exp_bits: BitOpeningClaims,
}

struct SoftmaxRowSumProverOutput {
    proof: SoftmaxRowSumOutput,
    exp: EvalClaim,
    input: EvalClaim,
}

struct SoftmaxReadProverOutput {
    proof: SoftmaxReadOutput,
    ra: EvalClaim,
}

struct SoftmaxRaVirtualProverOutput {
    proof: SoftmaxRaVirtualOutput,
    ra: EvalClaim,
}

struct SoftmaxHammingWeightProverOutput {
    proof: SoftmaxHammingWeightOutput,
    ra: EvalClaim,
}

struct SoftmaxBooleanityProverOutput {
    proof: SoftmaxBooleanityOutput,
    ra: EvalClaim,
}

struct SoftmaxLookupProverOutput {
    proof: SoftmaxLookupOutput,
    ra: RaOpeningClaims,
}

pub struct SoftmaxProverOutput {
    pub proof: SoftmaxOutput,
    pub input: EvalClaim,
    pub output_bits: BitOpeningClaims,
    pub floor_bits: BitOpeningClaims,
    pub frac_bits: BitOpeningClaims,
    pub exp_bits: BitOpeningClaims,
    pub ra: RaOpeningClaims,
}

pub struct SoftmaxProverInput {
    pub params: SoftmaxParams,
    pub advice: SoftmaxAdvice,
    pub witness: SoftmaxWitness,
}

pub struct SoftmaxWitness {
    pub input: Vec<i32>,
    pub output: Vec<i32>,
    pub ra: Vec<u8>,
    pub exp_acc: Vec<i64>,
    pub exp: Vec<i32>,
    pub frac_bits: [Vec<bool>; FRAC_BITS],
    pub exp_remainder_bits: [Vec<bool>; FRAC_BITS],
    pub acc: Vec<i64>,
    pub floor: Vec<i32>,
    pub floor_remainder_bits: [Vec<bool>; FRAC_BITS],
    pub output_remainder_bits: [Vec<bool>; FRAC_BITS],
}

struct RoundingRelation {
    output_round_mix: Fr,
    floor_round_mix: Fr,
    output_bit_challenges: [Fr; FRAC_BITS],
    floor_bit_challenges: [Fr; FRAC_BITS],
}

struct ExpRelation {
    row_sum_exp_mix: Fr,
    row_max_input_mix: Fr,
    min_diff: Fr,
    acc_mix: Fr,
    exp_acc_mix: Fr,
    exp_round_mix: Fr,
    diff_mix: Fr,
    frac_bit_challenges: [Fr; FRAC_BITS],
    exp_bit_challenges: [Fr; FRAC_BITS],
}

struct LookupRelation {
    read_challenges: [Fr; 3],
}

// D. output-to-acc:
//   output
//   + α * (256 * output - floor + rem_out - 256 * msb_out)
//   + β * (256 * floor - acc + rem_floor)
//   + bit booleanity = output
//
// C. exp tensor:
//   Acc point:
//     acc + α * (acc - valid * inv_sum(sum(row)) * exp) + exp derivation
//   Row-sum point:
//     exp + exp derivation
//
// B. row sum:
//   sum(row) = Σ_col valid(row,col) * exp(row,col)
//
// A. lookup/RA:
//   Σ_i ra_i = 1
//   Σ_i id_i * ra_i = index
//   Σ_i exp_i * ra_i = exp_base
//   ra_i * (ra_i - 1) = 0
pub fn prove_softmax<Tr>(
    claim: EvalClaim,
    input: SoftmaxProverInput,
    transcript: &mut Tr,
) -> Option<SoftmaxProverOutput>
where
    Tr: Transcript,
{
    append_eval_claim(transcript, &claim);
    validate_input(&claim, &input)?;
    let params = input.params;
    let witness = input.witness;
    let advice = input.advice;
    let (index, exp_base, ra, id_table, exp_table) =
        profile::measure("softmax.prepare.lookup_tables", || {
            prepare_lookup_witness(&witness.ra, params.shape, advice.min_diff, advice.max_diff)
        })?;

    let output_to_acc = profile::measure("softmax.proof.output_to_acc", || {
        prove_output_to_acc(
            claim,
            params.shape,
            witness.output.clone(),
            witness.floor.clone(),
            witness.acc.clone(),
            collect_matrix_bytes(witness.output_remainder_bits, params.shape)?,
            collect_matrix_bytes(witness.floor_remainder_bits, params.shape)?,
            transcript,
        )
    })?;

    let row_sum = profile::measure("softmax.proof.row_sum", || {
        let row_point = output_to_acc.acc.point[..params.shape.row_vars()].to_vec();
        prove_row_sum(
            params.shape,
            eval_i32_at_point(&advice.sum, &row_point)?,
            eval_i32_at_point(&advice.row_max, &row_point)?,
            row_point,
            &advice.sum,
            &advice.max_index,
            witness.input.clone(),
            witness.exp.clone(),
            transcript,
        )
    })?;

    let exp = profile::measure("softmax.proof.exp_tensor", || {
        prove_exp_tensor(
            output_to_acc.acc,
            row_sum.exp,
            row_sum.input,
            params.shape,
            advice.min_diff,
            witness.input,
            advice.row_max,
            advice.sum.clone(),
            index,
            exp_base,
            witness.exp.clone(),
            witness.exp_acc,
            witness.acc,
            collect_matrix_bytes(witness.frac_bits, params.shape)?,
            collect_matrix_bytes(witness.exp_remainder_bits, params.shape)?,
            transcript,
        )
    })?;

    let lookup = profile::measure("softmax.proof.lookup", || {
        prove_softmax_lookup_with_claims(
            exp.index.point.clone(),
            exp.index.value,
            exp.exp_base.value,
            ra,
            id_table,
            exp_table,
            params.shape,
            transcript,
        )
    })?;

    Some(SoftmaxProverOutput {
        input: exp.input,
        output_bits: output_to_acc.output_bits,
        floor_bits: output_to_acc.floor_bits,
        frac_bits: exp.frac_bits,
        exp_bits: exp.exp_bits,
        ra: lookup.ra,
        proof: SoftmaxOutput {
            output_to_acc: output_to_acc.proof,
            row_sum: row_sum.proof,
            exp: exp.proof,
            lookup: lookup.proof,
        },
    })
}

fn prove_output_to_acc<Tr>(
    claim: EvalClaim,
    shape: MatrixShape,
    output: Vec<i32>,
    floor: Vec<i32>,
    acc: Vec<i64>,
    output_bits: Vec<u8>,
    floor_bits: Vec<u8>,
    transcript: &mut Tr,
) -> Option<SoftmaxOutputToAccProverOutput>
where
    Tr: Transcript,
{
    (claim.point.len() == shape.point_len()).then_some(())?;
    let (output_round_mix, floor_round_mix, output_bit_challenges, floor_bit_challenges) =
        draw_softmax_output_to_acc_challenges(transcript)?;
    let relation = RoundingRelation {
        output_round_mix,
        floor_round_mix,
        output_bit_challenges,
        floor_bit_challenges,
    };
    let mut output = collect_matrix_table(output, shape)?;
    let mut floor = collect_matrix_table(floor, shape)?;
    let mut acc = collect_matrix_i64(acc, shape)?;
    let mut output_bits = RoundingBitsState::from_bytes(output_bits);
    let mut floor_bits = RoundingBitsState::from_bytes(floor_bits);
    let split_eq_point = claim.point.iter().rev().copied().collect::<Vec<_>>();
    let mut eq = GruenSplitEqPolynomial::new(&split_eq_point, BindingOrder::LowToHigh);

    let mut claim_i = claim.value;
    let mut round_polys = Vec::with_capacity(claim.point.len());
    let mut challenges = Vec::with_capacity(claim.point.len());

    while output.len() > 1 {
        let round = output_to_acc_round_poly(
            &relation,
            &eq,
            &output,
            &floor,
            &acc,
            &output_bits,
            &floor_bits,
        )?;
        append_round_statement(transcript, claim_i, &round);
        let challenge = transcript.challenge_scalar_optimized::<Fr>();
        let r = challenge.into();
        eq.bind(challenge);
        bind(&mut output, r);
        bind(&mut floor, r);
        bind(&mut acc, r);
        output_bits.bind(r);
        floor_bits.bind(r);
        claim_i = round.eval(r);
        challenges.push(challenge);
        round_polys.push(round);
    }

    let point = fr_challenges(&challenges);
    let output_bits_at_point = output_bits.bits_at(0);
    let floor_bits_at_point = floor_bits.bits_at(0);
    let proof = SoftmaxOutputToAccOutput {
        rounds: SumCheckRounds {
            round_polys,
            final_claim: claim_i,
        },
        output: output[0],
        floor: floor[0],
        acc: acc[0],
        output_bits: output_bits_at_point,
        floor_bits: floor_bits_at_point,
    };
    Some(SoftmaxOutputToAccProverOutput {
        acc: EvalClaim::new(proof.acc, point.clone()),
        output_bits: bit_opening_claims(&point, output_bits_at_point),
        floor_bits: bit_opening_claims(&point, floor_bits_at_point),
        proof,
    })
}

#[allow(clippy::too_many_arguments)]
fn prove_exp_tensor<Tr>(
    acc_claim: EvalClaim,
    row_sum_exp_claim: EvalClaim,
    row_max_input_claim: EvalClaim,
    shape: MatrixShape,
    min_diff: i64,
    input: Vec<i32>,
    row_max: Vec<i32>,
    sum: Vec<i32>,
    index: Vec<i32>,
    exp_base: Vec<i32>,
    exp: Vec<i32>,
    exp_acc: Vec<i64>,
    acc: Vec<i64>,
    frac_bits: Vec<u8>,
    exp_bits: Vec<u8>,
    transcript: &mut Tr,
) -> Option<SoftmaxExpProverOutput>
where
    Tr: Transcript,
{
    (acc_claim.point.len() == shape.point_len()).then_some(())?;
    (row_sum_exp_claim.point.len() == shape.point_len()).then_some(())?;
    (row_max_input_claim.point.len() == shape.point_len()).then_some(())?;
    let (
        row_sum_exp_mix,
        row_max_input_mix,
        acc_mix,
        exp_acc_mix,
        exp_round_mix,
        diff_mix,
        frac_bit_challenges,
        exp_bit_challenges,
    ) = draw_softmax_exp_challenges(transcript)?;
    let relation = ExpRelation {
        row_sum_exp_mix,
        row_max_input_mix,
        min_diff: field_from_i64(min_diff),
        acc_mix,
        exp_acc_mix,
        exp_round_mix,
        diff_mix,
        frac_bit_challenges,
        exp_bit_challenges,
    };

    let mut input = collect_matrix_table(input, shape)?;
    let mut row_max = RowState::new(row_max, shape)?;
    let mut valid = collect_valid_table(&sum, shape)?;
    let mut coefficient = collect_coefficient_table(&sum, shape)?;
    let mut sum = RowState::new(sum, shape)?;
    let mut index = collect_mle_table(index, shape)?;
    let mut exp_base = collect_mle_table(exp_base, shape)?;
    let mut exp = collect_matrix_table(exp, shape)?;
    let mut exp_acc = collect_matrix_i64(exp_acc, shape)?;
    let mut acc = collect_matrix_i64(acc, shape)?;
    let mut frac_bits = RoundingBitsState::from_bytes(frac_bits);
    let mut exp_bits = RoundingBitsState::from_bytes(exp_bits);
    let acc_eq_point = acc_claim.point.iter().rev().copied().collect::<Vec<_>>();
    let row_sum_eq_point = row_sum_exp_claim
        .point
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    let row_max_input_eq_point = row_max_input_claim
        .point
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    let mut acc_eq = GruenSplitEqPolynomial::new(&acc_eq_point, BindingOrder::LowToHigh);
    let mut row_sum_eq = GruenSplitEqPolynomial::new(&row_sum_eq_point, BindingOrder::LowToHigh);
    let mut row_max_input_eq =
        GruenSplitEqPolynomial::new(&row_max_input_eq_point, BindingOrder::LowToHigh);

    let mut claim_i = acc_claim.value
        + row_sum_exp_mix * row_sum_exp_claim.value
        + row_max_input_mix * row_max_input_claim.value;
    let mut round_polys = Vec::with_capacity(acc_claim.point.len());
    let mut challenges = Vec::with_capacity(acc_claim.point.len());

    while acc.len() > 1 {
        let round = profile::measure_detail("softmax.exp_tensor.round_poly", || {
            exp_tensor_round_poly(
                &relation,
                &acc_eq,
                &row_sum_eq,
                &row_max_input_eq,
                &input,
                &row_max,
                &valid,
                &coefficient,
                &index,
                &exp_base,
                &exp,
                &exp_acc,
                &acc,
                &frac_bits,
                &exp_bits,
            )
        })?;
        append_round_statement(transcript, claim_i, &round);
        let challenge = transcript.challenge_scalar_optimized::<Fr>();
        let r = challenge.into();
        profile::measure_detail("softmax.exp_tensor.acc_eq_bind", || acc_eq.bind(challenge));
        profile::measure_detail("softmax.exp_tensor.row_sum_eq_bind", || {
            row_sum_eq.bind(challenge)
        });
        profile::measure_detail("softmax.exp_tensor.row_max_input_eq_bind", || {
            row_max_input_eq.bind(challenge)
        });
        profile::measure_detail("softmax.exp_tensor.input_bind", || bind(&mut input, r));
        profile::measure_detail("softmax.exp_tensor.row_max_bind", || row_max.bind(r));
        profile::measure_detail("softmax.exp_tensor.valid_bind", || bind(&mut valid, r));
        profile::measure_detail("softmax.exp_tensor.coefficient_bind", || {
            bind(&mut coefficient, r)
        });
        profile::measure_detail("softmax.exp_tensor.sum_bind", || sum.bind(r));
        profile::measure_detail("softmax.exp_tensor.index_bind", || bind(&mut index, r));
        profile::measure_detail("softmax.exp_tensor.exp_base_bind", || {
            bind(&mut exp_base, r)
        });
        profile::measure_detail("softmax.exp_tensor.exp_bind", || bind(&mut exp, r));
        profile::measure_detail("softmax.exp_tensor.exp_acc_bind", || bind(&mut exp_acc, r));
        profile::measure_detail("softmax.exp_tensor.acc_bind", || bind(&mut acc, r));
        profile::measure_detail("softmax.exp_tensor.frac_bits_bind", || frac_bits.bind(r));
        profile::measure_detail("softmax.exp_tensor.exp_bits_bind", || exp_bits.bind(r));
        claim_i = round.eval(r);
        challenges.push(challenge);
        round_polys.push(round);
    }

    let point = fr_challenges(&challenges);
    let input_claim = EvalClaim {
        value: input[0],
        point: point.clone(),
    };
    let index_claim = EvalClaim {
        value: index[0],
        point: point.clone(),
    };
    let exp_base_claim = EvalClaim {
        value: exp_base[0],
        point: point.clone(),
    };
    let frac_bits_at_point = frac_bits.bits_at(0);
    let exp_bits_at_point = exp_bits.bits_at(0);
    let proof = SoftmaxExpOutput {
        rounds: SumCheckRounds {
            round_polys,
            final_claim: claim_i,
        },
        input: input_claim.value,
        index: index_claim.value,
        exp_base: exp_base_claim.value,
        exp: exp[0],
        exp_acc: exp_acc[0],
        acc: acc[0],
        row_max: row_max.value_at_zero(),
        sum: sum.value_at_zero(),
        frac_bits: frac_bits_at_point,
        exp_bits: exp_bits_at_point,
    };

    Some(SoftmaxExpProverOutput {
        proof,
        input: input_claim,
        index: index_claim,
        exp_base: exp_base_claim,
        frac_bits: bit_opening_claims(&point, frac_bits_at_point),
        exp_bits: bit_opening_claims(&point, exp_bits_at_point),
    })
}

fn prove_row_sum<Tr>(
    shape: MatrixShape,
    sum_claim: Fr,
    row_max_claim: Fr,
    row_point: Vec<Fr>,
    sum: &[i32],
    max_index: &[usize],
    input: Vec<i32>,
    exp: Vec<i32>,
    transcript: &mut Tr,
) -> Option<SoftmaxRowSumProverOutput>
where
    Tr: Transcript,
{
    (row_point.len() == shape.row_vars()).then_some(())?;
    validate_max_index(max_index, shape)?;
    let row_max_mix = draw_softmax_row_sum_challenge(transcript);
    let row_eq_point = row_point.iter().rev().copied().collect::<Vec<_>>();
    let mut row_eq = GruenSplitEqPolynomial::new(&row_eq_point, BindingOrder::LowToHigh);
    let mut valid = collect_valid_table(sum, shape)?;
    let mut selector = collect_max_selector_table(max_index, shape)?;
    let mut input = collect_matrix_table(input, shape)?;
    let mut exp = collect_matrix_table(exp, shape)?;
    let mut claim_i = sum_claim + row_max_mix * row_max_claim;
    let mut round_polys = Vec::with_capacity(shape.point_len());
    let mut challenges = Vec::with_capacity(shape.point_len());

    while exp.len() > 1 {
        let round = row_sum_round_poly(&row_eq, row_max_mix, &valid, &selector, &input, &exp)?;
        append_round_statement(transcript, claim_i, &round);
        let challenge = transcript.challenge_scalar_optimized::<Fr>();
        let r = challenge.into();
        if row_eq.len() > 1 {
            row_eq.bind(challenge);
        }
        bind(&mut valid, r);
        bind(&mut selector, r);
        bind(&mut input, r);
        bind(&mut exp, r);
        claim_i = round.eval(r);
        challenges.push(challenge);
        round_polys.push(round);
    }

    let proof = SoftmaxRowSumOutput {
        rounds: SumCheckRounds {
            round_polys,
            final_claim: claim_i,
        },
        exp: exp[0],
        input: input[0],
    };
    let point = fr_challenges(&challenges);
    Some(SoftmaxRowSumProverOutput {
        exp: EvalClaim::new(proof.exp, point.clone()),
        input: EvalClaim::new(proof.input, point.clone()),
        proof,
    })
}

fn validate_max_index(max_index: &[usize], shape: MatrixShape) -> Option<()> {
    (max_index.len() == shape.rows).then_some(())?;
    max_index.iter().all(|&col| col < shape.cols).then_some(())
}

fn collect_max_selector_table(max_index: &[usize], shape: MatrixShape) -> Option<Vec<Fr>> {
    validate_max_index(max_index, shape)?;
    let mut selector = vec![Fr::zero(); shape.len()];
    for (row, &col) in max_index.iter().enumerate() {
        selector[col * shape.rows + row] = Fr::one();
    }
    Some(selector)
}

fn row_sum_round_poly(
    row_eq: &GruenSplitEqPolynomial<Fr>,
    row_max_mix: Fr,
    valid: &[Fr],
    selector: &[Fr],
    input: &[Fr],
    exp: &[Fr],
) -> Option<RoundPolynomial<4>> {
    (valid.len() == exp.len()
        && selector.len() == exp.len()
        && input.len() == exp.len()
        && exp.len() % 2 == 0
        && exp.len() % row_eq.len() == 0)
        .then_some(())?;
    if row_eq.len() == 1 {
        return row_sum_round_poly_scalar(
            row_eq.get_current_scalar(),
            row_max_mix,
            valid,
            selector,
            input,
            exp,
        );
    }

    let rows = row_eq.len();
    let cols = exp.len() / rows;
    let relation_evals = row_eq.par_fold_out_in_unreduced::<9, 3>(&|row| {
        let mut evals = [Fr::zero(); 3];
        for col in 0..cols {
            let base = col * rows + 2 * row;
            let product =
                product_relation_evals(valid[base], valid[base + 1], exp[base], exp[base + 1]);
            let selected_input = product_relation_evals(
                selector[base],
                selector[base + 1],
                input[base],
                input[base + 1],
            );
            for index in 0..3 {
                evals[index] += product[index] + row_max_mix * selected_input[index];
            }
        }
        evals
    });
    Some(quadratic_relation_times_eq(
        relation_evals,
        row_eq.get_current_w(),
        row_eq.get_current_scalar(),
    ))
}

fn row_sum_round_poly_scalar(
    scalar: Fr,
    row_max_mix: Fr,
    valid: &[Fr],
    selector: &[Fr],
    input: &[Fr],
    exp: &[Fr],
) -> Option<RoundPolynomial<4>> {
    (valid.len() == exp.len()
        && selector.len() == exp.len()
        && input.len() == exp.len()
        && exp.len() % 2 == 0)
        .then_some(())?;
    let mut coeffs = [Fr::zero(); 4];
    for index in 0..exp.len() / 2 {
        let product = product_relation_coeffs(
            valid[2 * index],
            valid[2 * index + 1],
            exp[2 * index],
            exp[2 * index + 1],
        );
        let selected_input = product_relation_coeffs(
            selector[2 * index],
            selector[2 * index + 1],
            input[2 * index],
            input[2 * index + 1],
        );
        coeffs[0] += scalar * product[0];
        coeffs[1] += scalar * product[1];
        coeffs[2] += scalar * product[2];
        coeffs[0] += scalar * row_max_mix * selected_input[0];
        coeffs[1] += scalar * row_max_mix * selected_input[1];
        coeffs[2] += scalar * row_max_mix * selected_input[2];
    }
    Some(RoundPolynomial { coeffs })
}

fn product_relation_evals(valid_0: Fr, valid_1: Fr, exp_0: Fr, exp_1: Fr) -> [Fr; 3] {
    let coeffs = product_relation_coeffs(valid_0, valid_1, exp_0, exp_1);
    [
        coeffs[0],
        coeffs[0] + coeffs[1] + coeffs[2],
        coeffs[0] + Fr::from(2_u64) * coeffs[1] + Fr::from(4_u64) * coeffs[2],
    ]
}

fn product_relation_coeffs(valid_0: Fr, valid_1: Fr, exp_0: Fr, exp_1: Fr) -> [Fr; 3] {
    let valid_linear = valid_1 - valid_0;
    let exp_linear = exp_1 - exp_0;
    [
        valid_0 * exp_0,
        valid_0 * exp_linear + valid_linear * exp_0,
        valid_linear * exp_linear,
    ]
}

fn prove_softmax_lookup_with_claims<Tr>(
    tensor_point: Vec<Fr>,
    index: Fr,
    exp_base: Fr,
    ra: Vec<u8>,
    id_table: Vec<i32>,
    exp_table: Vec<i32>,
    shape: MatrixShape,
    transcript: &mut Tr,
) -> Option<SoftmaxLookupProverOutput>
where
    Tr: Transcript,
{
    let entries = id_table.len();
    let selected = selected_lookup_rows(&ra, shape.len(), entries)?;
    let lookup_point_len = entries.ilog2() as usize;
    let (read_challenges, _) = draw_softmax_lookup_challenges(transcript)?;
    let [one_challenge, id_challenge, exp_challenge] = read_challenges;
    let lookup_claim = one_challenge + id_challenge * index + exp_challenge * exp_base;
    let relation = LookupRelation { read_challenges };

    let read = profile::measure("softmax.lookup.read_raf", || {
        prove_softmax_read_raf(
            relation,
            lookup_claim,
            partial_ra_at_matrix_point(&selected, &tensor_point, shape, entries)?,
            id_table.into_iter().map(Fr::from_i32).collect(),
            exp_table.into_iter().map(Fr::from_i32).collect(),
            tensor_point.clone(),
            transcript,
        )
    })?;

    let lookup_point = read.ra.point[..lookup_point_len].to_vec();
    (lookup_point.len() == lookup_point_len).then_some(())?;
    let ra_virtual = profile::measure("softmax.lookup.ra_virtual", || {
        prove_softmax_ra_virtual(
            tensor_point.clone(),
            lookup_point.clone(),
            read.ra.value,
            &selected,
            shape,
            entries,
            transcript,
        )
    })?;
    let hamming_weight = profile::measure("softmax.lookup.hamming_weight", || {
        prove_softmax_hamming_weight(tensor_point.clone(), &selected, shape, entries, transcript)
    })?;
    let booleanity_lookup_point = transcript.challenge_vector::<Fr>(lookup_point_len);
    let booleanity = profile::measure("softmax.lookup.booleanity", || {
        prove_softmax_ra_booleanity(
            tensor_point,
            booleanity_lookup_point,
            &selected,
            shape,
            entries,
            transcript,
        )
    })?;

    Some(SoftmaxLookupProverOutput {
        ra: RaOpeningClaims {
            read: read.ra,
            virtual_claim: ra_virtual.ra,
            hamming_weight: hamming_weight.ra,
            booleanity: booleanity.ra,
        },
        proof: SoftmaxLookupOutput {
            read: read.proof,
            ra_virtual: ra_virtual.proof,
            hamming_weight: hamming_weight.proof,
            booleanity: booleanity.proof,
        },
    })
}

fn prove_softmax_read_raf<Tr>(
    relation: LookupRelation,
    mut claim_i: Fr,
    mut ra: Vec<Fr>,
    mut id_table: Vec<Fr>,
    mut exp_table: Vec<Fr>,
    tensor_point: Vec<Fr>,
    transcript: &mut Tr,
) -> Option<SoftmaxReadProverOutput>
where
    Tr: Transcript,
{
    let mut round_polys = Vec::with_capacity(ra.len().ilog2() as usize);
    let mut challenges = Vec::with_capacity(ra.len().ilog2() as usize);

    while ra.len() > 1 {
        let round = softmax_read_round_poly(&relation, &ra, &id_table, &exp_table)?;
        append_round_statement(transcript, claim_i, &round);
        let challenge = transcript.challenge_scalar_optimized::<Fr>();
        let r = challenge.into();
        bind(&mut ra, r);
        bind(&mut id_table, r);
        bind(&mut exp_table, r);
        claim_i = round.eval(r);
        challenges.push(challenge);
        round_polys.push(round);
    }

    let lookup_point = fr_challenges(&challenges);
    let proof = SoftmaxReadOutput {
        rounds: SumCheckRounds {
            round_polys,
            final_claim: claim_i,
        },
        ra: ra[0],
    };
    Some(SoftmaxReadProverOutput {
        ra: EvalClaim::new(
            proof.ra,
            [lookup_point.as_slice(), tensor_point.as_slice()].concat(),
        ),
        proof,
    })
}

fn prove_softmax_ra_virtual<Tr>(
    tensor_point: Vec<Fr>,
    lookup_point: Vec<Fr>,
    mut claim_i: Fr,
    selected: &[usize],
    shape: MatrixShape,
    entries: usize,
    transcript: &mut Tr,
) -> Option<SoftmaxRaVirtualProverOutput>
where
    Tr: Transcript,
{
    let mut ra = ra_at_lookup_point(selected, &lookup_point, shape, entries)?;
    let split_eq_point = tensor_point.iter().rev().copied().collect::<Vec<_>>();
    let mut eq = GruenSplitEqPolynomial::new(&split_eq_point, BindingOrder::LowToHigh);
    let mut round_polys = Vec::with_capacity(shape.point_len());
    let mut challenges = Vec::with_capacity(shape.point_len());

    while ra.len() > 1 {
        let round = ra_virtual_round_poly(&eq, &ra)?;
        append_round_statement(transcript, claim_i, &round);
        let challenge = transcript.challenge_scalar_optimized::<Fr>();
        let r = challenge.into();
        eq.bind(challenge);
        bind(&mut ra, r);
        claim_i = round.eval(r);
        challenges.push(challenge);
        round_polys.push(round);
    }

    let row_point = fr_challenges(&challenges);
    let proof = SoftmaxRaVirtualOutput {
        rounds: SumCheckRounds {
            round_polys,
            final_claim: claim_i,
        },
        ra: ra[0],
    };
    Some(SoftmaxRaVirtualProverOutput {
        ra: EvalClaim::new(
            proof.ra,
            [lookup_point.as_slice(), row_point.as_slice()].concat(),
        ),
        proof,
    })
}

fn prove_softmax_hamming_weight<Tr>(
    tensor_point: Vec<Fr>,
    selected: &[usize],
    shape: MatrixShape,
    entries: usize,
    transcript: &mut Tr,
) -> Option<SoftmaxHammingWeightProverOutput>
where
    Tr: Transcript,
{
    let mut claim_i = Fr::one();
    let mut ra = partial_ra_at_matrix_point(selected, &tensor_point, shape, entries)?;
    let mut round_polys = Vec::with_capacity(entries.ilog2() as usize);
    let mut challenges = Vec::with_capacity(entries.ilog2() as usize);

    while ra.len() > 1 {
        let round = linear_sum_round_poly(claim_i, &ra)?;
        append_round_statement(transcript, claim_i, &round);
        let challenge = transcript.challenge_scalar_optimized::<Fr>();
        let r = challenge.into();
        bind(&mut ra, r);
        claim_i = round.eval(r);
        challenges.push(challenge);
        round_polys.push(round);
    }

    let lookup_point = fr_challenges(&challenges);
    let proof = SoftmaxHammingWeightOutput {
        rounds: SumCheckRounds {
            round_polys,
            final_claim: claim_i,
        },
        ra: ra[0],
    };
    Some(SoftmaxHammingWeightProverOutput {
        ra: EvalClaim::new(
            proof.ra,
            [lookup_point.as_slice(), tensor_point.as_slice()].concat(),
        ),
        proof,
    })
}

fn prove_softmax_ra_booleanity<Tr>(
    tensor_point: Vec<Fr>,
    lookup_point: Vec<Fr>,
    selected: &[usize],
    shape: MatrixShape,
    entries: usize,
    transcript: &mut Tr,
) -> Option<SoftmaxBooleanityProverOutput>
where
    Tr: Transcript,
{
    let mut claim_i = Fr::zero();
    let mut ra = collect_ra_state(selected, shape, entries)?;
    let point = [lookup_point.as_slice(), tensor_point.as_slice()].concat();
    let split_eq_point = point.iter().rev().copied().collect::<Vec<_>>();
    let mut eq = GruenSplitEqPolynomial::new(&split_eq_point, BindingOrder::LowToHigh);
    let mut round_polys = Vec::with_capacity(point.len());
    let mut challenges = Vec::with_capacity(point.len());

    while ra.len() > 1 {
        let round = profile::measure_detail("softmax.lookup.booleanity.round_poly", || {
            ra_booleanity_round_poly(&eq, &ra)
        })?;
        let challenge = profile::measure_detail("softmax.lookup.booleanity.transcript", || {
            append_round_statement(transcript, claim_i, &round);
            transcript.challenge_scalar_optimized::<Fr>()
        });
        let r = challenge.into();
        profile::measure_detail("softmax.lookup.booleanity.eq_bind", || eq.bind(challenge));
        profile::measure_detail("softmax.lookup.booleanity.ra_bind", || ra.bind(r));
        claim_i = round.eval(r);
        challenges.push(challenge);
        round_polys.push(round);
    }

    let point = fr_challenges(&challenges);
    let proof = SoftmaxBooleanityOutput {
        rounds: SumCheckRounds {
            round_polys,
            final_claim: claim_i,
        },
        ra: ra.value_at(0),
    };
    Some(SoftmaxBooleanityProverOutput {
        ra: EvalClaim::new(proof.ra, point),
        proof,
    })
}

fn output_to_acc_round_poly(
    relation: &RoundingRelation,
    eq: &GruenSplitEqPolynomial<Fr>,
    output: &[Fr],
    floor: &[Fr],
    acc: &[Fr],
    output_bits: &RoundingBitsState,
    floor_bits: &RoundingBitsState,
) -> Option<RoundPolynomial<4>> {
    (eq.len() == output.len()
        && output.len() == floor.len()
        && output.len() == acc.len()
        && output.len() % 2 == 0)
        .then_some(())?;
    (output_bits.len() == output.len() && floor_bits.len() == output.len()).then_some(())?;
    let relation_evals = eq.par_fold_out_in_unreduced::<9, 3>(&|row| {
        output_to_acc_relation_evals(row, relation, output, floor, acc, output_bits, floor_bits)
    });
    Some(quadratic_relation_times_eq(
        relation_evals,
        eq.get_current_w(),
        eq.get_current_scalar(),
    ))
}

fn output_to_acc_relation_evals(
    row: usize,
    relation: &RoundingRelation,
    output: &[Fr],
    floor: &[Fr],
    acc: &[Fr],
    output_bits: &RoundingBitsState,
    floor_bits: &RoundingBitsState,
) -> [Fr; 3] {
    let output_0 = output[2 * row];
    let output_linear = output[2 * row + 1] - output_0;
    let floor_0 = floor[2 * row];
    let floor_linear = floor[2 * row + 1] - floor_0;
    let acc_0 = acc[2 * row];
    let acc_linear = acc[2 * row + 1] - acc_0;
    let output_bit_pairs = pair_bits(
        output_bits.bits_at(2 * row),
        output_bits.bits_at(2 * row + 1),
    );
    let floor_bit_pairs = pair_bits(floor_bits.bits_at(2 * row), floor_bits.bits_at(2 * row + 1));
    let output_rem_0 = remainder_constant(&output_bit_pairs);
    let output_rem_linear = remainder_linear(&output_bit_pairs);
    let output_msb_0 = output_bit_pairs[FRAC_BITS - 1].0;
    let output_msb_linear = output_bit_pairs[FRAC_BITS - 1].1 - output_msb_0;
    let floor_rem_0 = remainder_constant(&floor_bit_pairs);
    let floor_rem_linear = remainder_linear(&floor_bit_pairs);
    let scale = Fr::from(SCALE);

    let output_constraint_0 = scale * output_0 - floor_0 + output_rem_0 - scale * output_msb_0;
    let output_constraint_linear =
        scale * output_linear - floor_linear + output_rem_linear - scale * output_msb_linear;
    let floor_constraint_0 = scale * floor_0 - acc_0 + floor_rem_0;
    let floor_constraint_linear = scale * floor_linear - acc_linear + floor_rem_linear;

    let mut constant = output_0
        + relation.output_round_mix * output_constraint_0
        + relation.floor_round_mix * floor_constraint_0;
    let mut linear = output_linear
        + relation.output_round_mix * output_constraint_linear
        + relation.floor_round_mix * floor_constraint_linear;
    let mut leading = Fr::zero();
    add_bit_booleanity(
        &mut constant,
        &mut linear,
        &mut leading,
        &relation.output_bit_challenges,
        &output_bit_pairs,
    );
    add_bit_booleanity(
        &mut constant,
        &mut linear,
        &mut leading,
        &relation.floor_bit_challenges,
        &floor_bit_pairs,
    );
    [
        constant,
        constant + linear + leading,
        constant + linear + linear + Fr::from(4_u64) * leading,
    ]
}

#[allow(clippy::too_many_arguments)]
fn exp_tensor_round_poly(
    relation: &ExpRelation,
    acc_eq: &GruenSplitEqPolynomial<Fr>,
    row_sum_eq: &GruenSplitEqPolynomial<Fr>,
    row_max_input_eq: &GruenSplitEqPolynomial<Fr>,
    input: &[Fr],
    row_max: &RowState,
    valid: &[Fr],
    coefficient: &[Fr],
    index: &[Fr],
    exp_base: &[Fr],
    exp: &[Fr],
    exp_acc: &[Fr],
    acc: &[Fr],
    frac_bits: &RoundingBitsState,
    exp_bits: &RoundingBitsState,
) -> Option<RoundPolynomial<4>> {
    let len = acc.len();
    (acc_eq.len() == len
        && row_sum_eq.len() == len
        && row_max_input_eq.len() == len
        && input.len() == len
        && valid.len() == len
        && coefficient.len() == len
        && index.len() == len
        && exp_base.len() == len
        && exp.len() == len
        && exp_acc.len() == len
        && frac_bits.len() == len
        && exp_bits.len() == len
        && len % 2 == 0)
        .then_some(())?;
    let relation_evals =
        profile::measure_detail("softmax.exp_tensor.round_poly.relation_evals", || {
            acc_eq.par_fold_out_in_unreduced::<9, 3>(&|row| {
                exp_tensor_relation_evals(
                    row,
                    relation,
                    ExpTensorTerm::Acc,
                    input,
                    row_max,
                    valid,
                    coefficient,
                    index,
                    exp_base,
                    exp,
                    exp_acc,
                    acc,
                    frac_bits,
                    exp_bits,
                )
            })
        });
    let mut round = profile::measure_detail("softmax.exp_tensor.round_poly.eq_combine", || {
        quadratic_relation_times_eq(
            relation_evals,
            acc_eq.get_current_w(),
            acc_eq.get_current_scalar(),
        )
    });
    let row_sum_relation_evals = profile::measure_detail(
        "softmax.exp_tensor.round_poly.row_sum_relation_evals",
        || {
            row_sum_eq.par_fold_out_in_unreduced::<9, 3>(&|row| {
                exp_tensor_relation_evals(
                    row,
                    relation,
                    ExpTensorTerm::RowSum,
                    input,
                    row_max,
                    valid,
                    coefficient,
                    index,
                    exp_base,
                    exp,
                    exp_acc,
                    acc,
                    frac_bits,
                    exp_bits,
                )
            })
        },
    );
    let row_sum_round =
        profile::measure_detail("softmax.exp_tensor.round_poly.row_sum_eq_combine", || {
            quadratic_relation_times_eq(
                row_sum_relation_evals,
                row_sum_eq.get_current_w(),
                row_sum_eq.get_current_scalar(),
            )
        });
    for (out, add) in round.coeffs.iter_mut().zip_eq(row_sum_round.coeffs) {
        *out += relation.row_sum_exp_mix * add;
    }
    let row_max_input_relation_evals = profile::measure_detail(
        "softmax.exp_tensor.round_poly.row_max_input_relation_evals",
        || {
            row_max_input_eq.par_fold_out_in_unreduced::<9, 3>(&|row| {
                exp_tensor_relation_evals(
                    row,
                    relation,
                    ExpTensorTerm::RowMaxInput,
                    input,
                    row_max,
                    valid,
                    coefficient,
                    index,
                    exp_base,
                    exp,
                    exp_acc,
                    acc,
                    frac_bits,
                    exp_bits,
                )
            })
        },
    );
    let row_max_input_round = profile::measure_detail(
        "softmax.exp_tensor.round_poly.row_max_input_eq_combine",
        || {
            quadratic_relation_times_eq(
                row_max_input_relation_evals,
                row_max_input_eq.get_current_w(),
                row_max_input_eq.get_current_scalar(),
            )
        },
    );
    for (out, add) in round.coeffs.iter_mut().zip_eq(row_max_input_round.coeffs) {
        *out += relation.row_max_input_mix * add;
    }
    Some(round)
}

enum ExpTensorTerm {
    Acc,
    RowSum,
    RowMaxInput,
}

#[allow(clippy::too_many_arguments)]
fn exp_tensor_relation_evals(
    row: usize,
    relation: &ExpRelation,
    term: ExpTensorTerm,
    input: &[Fr],
    row_max: &RowState,
    valid: &[Fr],
    coefficient: &[Fr],
    index: &[Fr],
    exp_base: &[Fr],
    exp: &[Fr],
    exp_acc: &[Fr],
    acc: &[Fr],
    frac_bits: &RoundingBitsState,
    exp_bits: &RoundingBitsState,
) -> [Fr; 3] {
    let input_0 = input[2 * row];
    let input_linear = input[2 * row + 1] - input_0;
    let (row_max_0, row_max_1) = row_max.pair_at(row);
    let row_max_linear = row_max_1 - row_max_0;
    let valid_0 = valid[2 * row];
    let valid_linear = valid[2 * row + 1] - valid_0;
    let coefficient_0 = coefficient[2 * row];
    let coefficient_linear = coefficient[2 * row + 1] - coefficient_0;
    let index_0 = index[2 * row];
    let index_linear = index[2 * row + 1] - index_0;
    let exp_base_0 = exp_base[2 * row];
    let exp_base_linear = exp_base[2 * row + 1] - exp_base_0;
    let exp_0 = exp[2 * row];
    let exp_linear = exp[2 * row + 1] - exp_0;
    let exp_acc_0 = exp_acc[2 * row];
    let exp_acc_linear = exp_acc[2 * row + 1] - exp_acc_0;
    let acc_0 = acc[2 * row];
    let acc_linear = acc[2 * row + 1] - acc_0;
    let frac_pairs = pair_bits(frac_bits.bits_at(2 * row), frac_bits.bits_at(2 * row + 1));
    let exp_bit_pairs = pair_bits(exp_bits.bits_at(2 * row), exp_bits.bits_at(2 * row + 1));
    let frac_0 = remainder_constant(&frac_pairs);
    let frac_linear = remainder_linear(&frac_pairs);
    let exp_rem_0 = remainder_constant(&exp_bit_pairs);
    let exp_rem_linear = remainder_linear(&exp_bit_pairs);
    let exp_msb_0 = exp_bit_pairs[FRAC_BITS - 1].0;
    let exp_msb_linear = exp_bit_pairs[FRAC_BITS - 1].1 - exp_msb_0;
    let scale = Fr::from(SCALE);

    let eval = |x: Fr| {
        let input_x = input_0 + x * input_linear;
        let row_max_x = row_max_0 + x * row_max_linear;
        let index_x = index_0 + x * index_linear;
        let exp_base_x = exp_base_0 + x * exp_base_linear;
        let exp_x = exp_0 + x * exp_linear;
        let exp_acc_x = exp_acc_0 + x * exp_acc_linear;
        let acc_x = acc_0 + x * acc_linear;
        let frac_x = frac_0 + x * frac_linear;
        let exp_rem_x = exp_rem_0 + x * exp_rem_linear;
        let exp_msb_x = exp_msb_0 + x * exp_msb_linear;
        let valid_x = valid_0 + x * valid_linear;
        let coefficient_x = coefficient_0 + x * coefficient_linear;

        let acc_constraint = acc_x - coefficient_x * exp_x;
        let exp_acc_constraint = exp_acc_x - exp_base_x * (scale + frac_x);
        let exp_round_constraint = scale * exp_x - exp_acc_x + exp_rem_x - scale * exp_msb_x;
        let diff_constraint =
            valid_x * (input_x - row_max_x) - scale * (relation.min_diff + index_x) - frac_x;

        let derivation = relation.exp_acc_mix * exp_acc_constraint
            + relation.exp_round_mix * exp_round_constraint
            + relation.diff_mix * diff_constraint;
        match term {
            ExpTensorTerm::Acc => acc_x + relation.acc_mix * acc_constraint + derivation,
            ExpTensorTerm::RowSum => exp_x + derivation,
            ExpTensorTerm::RowMaxInput => input_x + derivation,
        }
    };

    let mut values = [eval(Fr::zero()), eval(Fr::one()), eval(Fr::from(2_u64))];
    add_bit_booleanity_evals(&mut values, &relation.frac_bit_challenges, &frac_pairs);
    add_bit_booleanity_evals(&mut values, &relation.exp_bit_challenges, &exp_bit_pairs);
    values
}

fn softmax_read_round_poly(
    relation: &LookupRelation,
    ra: &[Fr],
    id_table: &[Fr],
    exp_table: &[Fr],
) -> Option<RoundPolynomial<3>> {
    let len = ra.len();
    (id_table.len() == len && exp_table.len() == len && len % 2 == 0).then_some(())?;
    let mut coeffs = [Fr::zero(); 3];
    for row in 0..len / 2 {
        let [constant, at_one, at_two] =
            softmax_read_relation_evals(row, relation, ra, id_table, exp_table);
        let round = quadratic_from_evals(constant, at_one, at_two);
        coeffs[0] += round.coeffs[0];
        coeffs[1] += round.coeffs[1];
        coeffs[2] += round.coeffs[2];
    }
    Some(RoundPolynomial { coeffs })
}

fn softmax_read_relation_evals(
    row: usize,
    relation: &LookupRelation,
    ra: &[Fr],
    id_table: &[Fr],
    exp_table: &[Fr],
) -> [Fr; 3] {
    std::array::from_fn(|point| {
        let x = Fr::from(point as u64);
        let ra_x = line_at(ra[2 * row], ra[2 * row + 1], x);
        let id_x = line_at(id_table[2 * row], id_table[2 * row + 1], x);
        let exp_x = line_at(exp_table[2 * row], exp_table[2 * row + 1], x);
        let [one_challenge, id_challenge, exp_challenge] = relation.read_challenges;
        ra_x * (one_challenge + id_challenge * id_x + exp_challenge * exp_x)
    })
}

fn ra_virtual_round_poly(eq: &GruenSplitEqPolynomial<Fr>, ra: &[Fr]) -> Option<RoundPolynomial<3>> {
    (eq.len() == ra.len() && ra.len() % 2 == 0).then_some(())?;
    let relation_evals = eq.par_fold_out_in_unreduced::<9, 3>(&|row| {
        let ra_0 = ra[2 * row];
        let ra_1 = ra[2 * row + 1];
        let ra_2 = ra_0 + Fr::from(2_u64) * (ra_1 - ra_0);
        [ra_0, ra_1, ra_2]
    });
    let round =
        quadratic_relation_times_eq(relation_evals, eq.get_current_w(), eq.get_current_scalar());
    Some(RoundPolynomial {
        coeffs: [round.coeffs[0], round.coeffs[1], round.coeffs[2]],
    })
}

fn ra_booleanity_round_poly(
    eq: &GruenSplitEqPolynomial<Fr>,
    ra: &RaState,
) -> Option<RoundPolynomial<4>> {
    (eq.len() == ra.len() && ra.len() % 2 == 0).then_some(())?;
    if let RaState::Selected {
        selected,
        address_space,
        ..
    } = ra
    {
        return ra_booleanity_round_poly_selected(eq, selected, *address_space);
    }
    if let RaState::AffineTags { tags, r } = ra {
        return ra_booleanity_round_poly_affine_tags(eq, tags, *r);
    }
    if let RaState::Affine2Tags { tags, values } = ra {
        return ra_booleanity_round_poly_affine2_tags(eq, tags, values);
    }

    let relation_evals = eq.par_fold_out_in_unreduced::<9, 3>(&|row| {
        let ra_0 = ra.value_at(2 * row);
        let ra_1 = ra.value_at(2 * row + 1);
        let ra_2 = ra_0 + Fr::from(2_u64) * (ra_1 - ra_0);
        [
            ra_0 * (ra_0 - Fr::one()),
            ra_1 * (ra_1 - Fr::one()),
            ra_2 * (ra_2 - Fr::one()),
        ]
    });
    Some(quadratic_relation_times_eq(
        relation_evals,
        eq.get_current_w(),
        eq.get_current_scalar(),
    ))
}

fn ra_booleanity_round_poly_selected(
    eq: &GruenSplitEqPolynomial<Fr>,
    selected: &[usize],
    address_space: usize,
) -> Option<RoundPolynomial<4>> {
    (eq.len() == selected.len() * (eq.len() / selected.len()) && eq.len() % 2 == 0).then_some(())?;
    let two = Fr::from(2_u64);
    let relation_evals = eq.par_fold_out_in_unreduced::<9, 3>(&|row| {
        if selected_value(row * 2, selected, address_space)
            == selected_value(row * 2 + 1, selected, address_space)
        {
            [Fr::zero(), Fr::zero(), Fr::zero()]
        } else {
            [Fr::zero(), Fr::zero(), two]
        }
    });
    Some(quadratic_relation_times_eq(
        relation_evals,
        eq.get_current_w(),
        eq.get_current_scalar(),
    ))
}

fn ra_booleanity_round_poly_affine_tags(
    eq: &GruenSplitEqPolynomial<Fr>,
    tags: &[u8],
    r: Fr,
) -> Option<RoundPolynomial<4>> {
    (eq.len() == tags.len() && tags.len() % 2 == 0).then_some(())?;
    let values: [Fr; 4] = [Fr::zero(), Fr::one(), r, Fr::one() - r];
    let pair_evals =
        booleanity_pair_eval_table::<16>(|pair| (values[pair & 0b11], values[pair >> 2]));
    let relation_evals = eq.par_fold_out_in_unreduced::<9, 3>(&|row| {
        pair_evals[tags[2 * row] as usize | ((tags[2 * row + 1] as usize) << 2)]
    });
    Some(quadratic_relation_times_eq(
        relation_evals,
        eq.get_current_w(),
        eq.get_current_scalar(),
    ))
}

fn ra_booleanity_round_poly_affine2_tags(
    eq: &GruenSplitEqPolynomial<Fr>,
    tags: &[u8],
    values: &[Fr; 16],
) -> Option<RoundPolynomial<4>> {
    (eq.len() == tags.len() && tags.len() % 2 == 0).then_some(())?;
    let pair_evals =
        booleanity_pair_eval_table::<256>(|pair| (values[pair & 0b1111], values[pair >> 4]));
    let relation_evals = eq.par_fold_out_in_unreduced::<9, 3>(&|row| {
        pair_evals[tags[2 * row] as usize | ((tags[2 * row + 1] as usize) << 4)]
    });
    Some(quadratic_relation_times_eq(
        relation_evals,
        eq.get_current_w(),
        eq.get_current_scalar(),
    ))
}

fn linear_sum_round_poly(claim: Fr, values: &[Fr]) -> Option<RoundPolynomial<2>> {
    (values.len() % 2 == 0).then_some(())?;
    let eval_at_zero = values.chunks_exact(2).map(|chunk| chunk[0]).sum::<Fr>();
    let eval_at_one = claim - eval_at_zero;
    Some(RoundPolynomial {
        coeffs: [eval_at_zero, eval_at_one - eval_at_zero],
    })
}

fn quadratic_relation_times_eq(
    relation_evals: [Fr; 3],
    current_w: Fr,
    current_scalar: Fr,
) -> RoundPolynomial<4> {
    let q = quadratic_from_evals(relation_evals[0], relation_evals[1], relation_evals[2]);
    let eq_constant = Fr::one() - current_w;
    let eq_linear = current_w + current_w - Fr::one();
    RoundPolynomial {
        coeffs: [
            current_scalar * eq_constant * q.coeffs[0],
            current_scalar * (eq_constant * q.coeffs[1] + eq_linear * q.coeffs[0]),
            current_scalar * (eq_constant * q.coeffs[2] + eq_linear * q.coeffs[1]),
            current_scalar * eq_linear * q.coeffs[2],
        ],
    }
}

fn quadratic_from_evals(eval_at_zero: Fr, eval_at_one: Fr, eval_at_two: Fr) -> RoundPolynomial<3> {
    let two = Fr::from(2_u64);
    let leading = (eval_at_two - two * eval_at_one + eval_at_zero)
        * Field::inverse(&two).expect("2 is nonzero");
    let linear = eval_at_one - eval_at_zero - leading;
    RoundPolynomial {
        coeffs: [eval_at_zero, linear, leading],
    }
}

fn line_at(v0: Fr, v1: Fr, point: Fr) -> Fr {
    v0 + point * (v1 - v0)
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

enum RaState {
    Selected {
        selected: Vec<usize>,
        address_space: usize,
        len: usize,
    },
    AffineTags {
        tags: Vec<u8>,
        r: Fr,
    },
    Affine2Tags {
        tags: Vec<u8>,
        values: [Fr; 16],
    },
    FieldValues(Vec<Fr>),
}

impl RaState {
    fn from_selected(selected: Vec<usize>, entries: usize) -> Self {
        let tensor_len = selected.len();
        Self::Selected {
            selected,
            address_space: entries,
            len: tensor_len * entries,
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Selected { len, .. } => *len,
            Self::AffineTags { tags, .. } => tags.len(),
            Self::Affine2Tags { tags, .. } => tags.len(),
            Self::FieldValues(values) => values.len(),
        }
    }

    fn value_at(&self, index: usize) -> Fr {
        match self {
            Self::Selected {
                selected,
                address_space,
                ..
            } => {
                let address = index % *address_space;
                let tensor_index = index / *address_space;
                if selected[tensor_index] == address {
                    Fr::one()
                } else {
                    Fr::zero()
                }
            }
            Self::AffineTags { tags, r } => affine_tag_value(tags[index] as u16, *r),
            Self::Affine2Tags { tags, values } => values[tags[index] as usize],
            Self::FieldValues(values) => values[index],
        }
    }

    fn bind(&mut self, r: Fr) {
        match self {
            Self::Selected {
                selected,
                address_space,
                len,
            } => {
                let next = (0..*len / 2)
                    .map(|index| {
                        let lower = u8::from(selected_value(2 * index, selected, *address_space));
                        let upper =
                            u8::from(selected_value(2 * index + 1, selected, *address_space));
                        affine_tag_from_bits(lower, upper)
                    })
                    .collect();
                *self = Self::AffineTags { tags: next, r };
            }
            Self::AffineTags { tags, r: r0 } => {
                let next = (0..tags.len() / 2)
                    .map(|index| {
                        affine2_tag_from_affine1_pair(tags[2 * index], tags[2 * index + 1])
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
                        bind_affine2_tag(tags[2 * index], tags[2 * index + 1], &bind_values)
                    })
                    .collect();
                *self = Self::FieldValues(next);
            }
            Self::FieldValues(values) => bind(values, r),
        }
    }
}

fn selected_value(index: usize, selected: &[usize], address_space: usize) -> bool {
    let address = index % address_space;
    let tensor_index = index / address_space;
    selected[tensor_index] == address
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

fn add_bit_booleanity_evals(
    values: &mut [Fr; 3],
    challenges: &[Fr; FRAC_BITS],
    bits: &[(Fr, Fr); FRAC_BITS],
) {
    let mut constant = Fr::zero();
    let mut linear = Fr::zero();
    let mut leading = Fr::zero();
    add_bit_booleanity(&mut constant, &mut linear, &mut leading, challenges, bits);
    values[0] += constant;
    values[1] += constant + linear + leading;
    values[2] += constant + linear + linear + Fr::from(4_u64) * leading;
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

fn affine_tag_from_bits(lower: u8, upper: u8) -> u8 {
    match (lower, upper) {
        (0, 0) => 0b00,
        (1, 1) => 0b01,
        (0, 1) => 0b10,
        (1, 0) => 0b11,
        _ => unreachable!("RA values are boolean"),
    }
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

fn affine2_tag_from_affine1_pair(lower: u8, upper: u8) -> u8 {
    lower | (upper << 2)
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

fn bind_affine2_tag(lower: u8, upper: u8, bind_values: &[Fr; 256]) -> Fr {
    bind_values[lower as usize | ((upper as usize) << 4)]
}

fn bind_bits(lower: [Fr; FRAC_BITS], upper: [Fr; FRAC_BITS], r: Fr) -> [Fr; FRAC_BITS] {
    std::array::from_fn(|bit| lower[bit] + r * (upper[bit] - lower[bit]))
}

fn booleanity_pair_eval_table<const N: usize>(
    value_pair: impl Fn(usize) -> (Fr, Fr),
) -> [[Fr; 3]; N] {
    std::array::from_fn(|pair| {
        let (v0, v1) = value_pair(pair);
        booleanity_pair_evals(v0, v1)
    })
}

fn booleanity_pair_evals(v0: Fr, v1: Fr) -> [Fr; 3] {
    let v2 = v0 + Fr::from(2_u64) * (v1 - v0);
    [
        v0 * (v0 - Fr::one()),
        v1 * (v1 - Fr::one()),
        v2 * (v2 - Fr::one()),
    ]
}

fn collect_matrix_table(values: Vec<i32>, shape: MatrixShape) -> Option<Vec<Fr>> {
    (values.len() == shape.len()).then_some(())?;
    Some(values.into_iter().map(Fr::from_i32).collect())
}

fn eval_i32_at_point(values: &[i32], point: &[Fr]) -> Option<Fr> {
    (values.len() == (1_usize << point.len())).then_some(())?;
    Some(
        values
            .iter()
            .enumerate()
            .map(|(index, value)| eq_eval(index, point) * Fr::from_i32(*value))
            .sum(),
    )
}

fn collect_mle_table(values: Vec<i32>, shape: MatrixShape) -> Option<Vec<Fr>> {
    (values.len() == shape.len()).then_some(())?;
    Some(values.into_iter().map(Fr::from_i32).collect())
}

fn collect_matrix_i64(values: Vec<i64>, shape: MatrixShape) -> Option<Vec<Fr>> {
    (values.len() == shape.len()).then_some(())?;
    Some(values.into_iter().map(field_from_i64).collect())
}

struct RowState {
    values: Vec<Fr>,
}

impl RowState {
    fn new(values: Vec<i32>, shape: MatrixShape) -> Option<Self> {
        (values.len() == shape.rows).then_some(Self {
            values: values.into_iter().map(Fr::from_i32).collect(),
        })
    }

    fn pair_at(&self, tensor_pair: usize) -> (Fr, Fr) {
        if self.values.len() == 1 {
            return (self.values[0], self.values[0]);
        }

        let lower_row = (2 * tensor_pair) % self.values.len();
        (self.values[lower_row], self.values[lower_row + 1])
    }

    fn bind(&mut self, r: Fr) {
        if self.values.len() > 1 {
            bind(&mut self.values, r);
        }
    }

    fn value_at_zero(&self) -> Fr {
        self.values[0]
    }
}

fn collect_matrix_bytes(bits: [Vec<bool>; FRAC_BITS], shape: MatrixShape) -> Option<Vec<u8>> {
    bits.iter()
        .all(|lane| lane.len() == shape.len())
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

fn collect_valid_table(sum: &[i32], shape: MatrixShape) -> Option<Vec<Fr>> {
    (sum.len() == shape.rows).then_some(())?;
    let mut out = vec![Fr::zero(); shape.len()];
    for col in 0..shape.cols {
        for row in 0..shape.rows {
            if sum[row] != 0 && col <= row % shape.cols {
                out[row + shape.rows * col] = Fr::one();
            }
        }
    }
    Some(out)
}

fn collect_coefficient_table(sum: &[i32], shape: MatrixShape) -> Option<Vec<Fr>> {
    (sum.len() == shape.rows).then_some(())?;
    let mut out = vec![Fr::zero(); shape.len()];
    for col in 0..shape.cols {
        for row in 0..shape.rows {
            if sum[row] != 0 && col <= row % shape.cols {
                out[row + shape.rows * col] = field_from_i64(inv_sum_q16(sum[row]));
            }
        }
    }
    Some(out)
}

fn partial_ra_at_matrix_point(
    selected: &[usize],
    tensor_point: &[Fr],
    shape: MatrixShape,
    entries: usize,
) -> Option<Vec<Fr>> {
    (tensor_point.len() == shape.point_len()).then_some(())?;
    (selected.len() == shape.len()).then_some(())?;
    let tensor_eq = gruen_eq_evals(tensor_point);
    let mut values = vec![Fr::zero(); entries];
    for (tensor_index, eq) in tensor_eq.into_iter().enumerate() {
        values[selected[tensor_index]] += eq;
    }
    Some(values)
}

fn ra_at_lookup_point(
    selected: &[usize],
    lookup_point: &[Fr],
    shape: MatrixShape,
    entries: usize,
) -> Option<Vec<Fr>> {
    (lookup_point.len() == entries.ilog2() as usize).then_some(())?;
    (selected.len() == shape.len()).then_some(())?;
    let lookup_eq = gruen_eq_evals(lookup_point);
    let mut values = vec![Fr::zero(); shape.len()];
    for (tensor_index, value) in values.iter_mut().enumerate() {
        *value = lookup_eq[selected[tensor_index]];
    }
    Some(values)
}

fn collect_ra_state(selected: &[usize], shape: MatrixShape, entries: usize) -> Option<RaState> {
    (selected.len() == shape.len()).then_some(())?;
    Some(RaState::from_selected(selected.to_vec(), entries))
}

fn prepare_lookup_witness(
    ra: &[u8],
    shape: MatrixShape,
    min_diff: i64,
    max_diff: i64,
) -> Option<(Vec<i32>, Vec<i32>, Vec<u8>, Vec<i32>, Vec<i32>)> {
    validate_softmax_range(min_diff, max_diff)?;
    let entries = usize::try_from(max_diff - min_diff + 1).ok()?;
    let tables = build_softmax_tables(&SoftmaxAdvice {
        min_diff,
        max_diff,
        row_max: Vec::new(),
        max_index: Vec::new(),
        sum: Vec::new(),
    })?;
    let lut_len = tables.id.len();
    let selected = selected_lookup_rows(ra, shape.len(), entries)?;
    let index = selected
        .iter()
        .map(|row| i32::try_from(*row).ok())
        .collect::<Option<Vec<_>>>()?;
    let exp_base = selected
        .iter()
        .map(|row| tables.exp_base.get(*row).copied())
        .collect::<Option<Vec<_>>>()?;
    let ra = expand_ra(ra, shape.len(), entries, lut_len)?;
    Some((index, exp_base, ra, tables.id, tables.exp_base))
}

fn validate_input(claim: &EvalClaim, input: &SoftmaxProverInput) -> Option<()> {
    validate_softmax_range(input.advice.min_diff, input.advice.max_diff)?;
    (claim.point.len() == input.params.shape.point_len()).then_some(())?;
    (input.advice.row_max.len() == input.params.shape.rows).then_some(())?;
    validate_max_index(&input.advice.max_index, input.params.shape)?;
    (input.advice.sum.len() == input.params.shape.rows).then_some(())?;
    (input.witness.input.len() == input.params.shape.len()).then_some(())?;
    (input.witness.output.len() == input.params.shape.len()).then_some(())?;
    (input.witness.exp_acc.len() == input.params.shape.len()).then_some(())?;
    (input.witness.exp.len() == input.params.shape.len()).then_some(())?;
    (input.witness.acc.len() == input.params.shape.len()).then_some(())?;
    (input.witness.floor.len() == input.params.shape.len()).then_some(())?;
    input
        .witness
        .frac_bits
        .iter()
        .chain(input.witness.exp_remainder_bits.iter())
        .chain(input.witness.floor_remainder_bits.iter())
        .chain(input.witness.output_remainder_bits.iter())
        .all(|bits| bits.len() == input.params.shape.len())
        .then_some(())?;
    let entries = usize::try_from(input.advice.max_diff - input.advice.min_diff + 1).ok()?;
    (input.witness.ra.len() == input.params.shape.len().checked_mul(entries)?).then_some(())
}

fn selected_lookup_rows(ra: &[u8], rows: usize, entries: usize) -> Option<Vec<usize>> {
    (ra.len() == rows.checked_mul(entries)?).then_some(())?;
    (0..rows)
        .map(|row| {
            let mut selected = None;
            for index in 0..entries {
                match (ra[index * rows + row], selected) {
                    (1, None) => selected = Some(index),
                    (1, Some(_)) => return None,
                    (0, _) => {}
                    _ => return None,
                }
            }
            selected
        })
        .collect()
}

fn expand_ra(ra: &[u8], rows: usize, entries: usize, lut_len: usize) -> Option<Vec<u8>> {
    (lut_len >= entries).then_some(())?;
    (ra.len() == rows.checked_mul(entries)?).then_some(())?;
    let mut expanded = vec![0_u8; rows.checked_mul(lut_len)?];
    for entry in 0..entries {
        let source = entry * rows;
        let target = entry * rows;
        expanded[target..target + rows].copy_from_slice(&ra[source..source + rows]);
    }
    Some(expanded)
}

fn inv_sum_q16(sum: i32) -> i64 {
    ((1_i64 << 24) as f64 / f64::from(sum)).round() as i64
}

fn field_from_i64(value: i64) -> Fr {
    if value >= 0 {
        Fr::from(value as u64)
    } else {
        -Fr::from(value.unsigned_abs())
    }
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

fn gruen_eq_evals(point: &[Fr]) -> Vec<Fr> {
    let point = point.iter().rev().copied().collect::<Vec<_>>();
    GruenSplitEqPolynomial::<Fr>::new(&point, BindingOrder::LowToHigh)
        .merge()
        .Z
}

fn fr_challenges(challenges: &[<Fr as JoltField>::Challenge]) -> Vec<Fr> {
    challenges.iter().copied().map(Into::into).collect()
}

fn bit_opening_claims(point: &[Fr], values: [Fr; FRAC_BITS]) -> BitOpeningClaims {
    values.map(|value| EvalClaim::new(value, point.to_vec()))
}
