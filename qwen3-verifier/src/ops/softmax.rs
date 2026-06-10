use ark_bn254::Fr;
use ark_ff::{One, Zero};
use itertools::Itertools;
use joltworks::field::JoltField;
use joltworks::transcripts::Transcript;
use qwen3_common::ops::softmax::{
    SoftmaxAdvice, SoftmaxBooleanityOutput, SoftmaxExpOutput, SoftmaxHammingWeightOutput,
    SoftmaxOutput, SoftmaxOutputToAccOutput, SoftmaxParams, SoftmaxRaVirtualOutput,
    SoftmaxReadOutput, SoftmaxRowSumOutput, SoftmaxVerifierOutput, build_softmax_tables,
    draw_softmax_exp_challenges, draw_softmax_lookup_challenges,
    draw_softmax_output_to_acc_challenges, draw_softmax_row_sum_challenge,
};

use qwen3_common::{
    BitOpeningClaims, EvalClaim, FRAC_BITS, MatrixShape, RaOpeningClaims, SCALE, append_eval_claim,
    verify_sumcheck_rounds,
};

use crate::utils::{eq_point_eval, eval_i32_mle_at_point};

pub fn verify_softmax<Tr>(
    claim: EvalClaim,
    params: SoftmaxParams,
    advice: SoftmaxAdvice,
    proof: &SoftmaxOutput,
    transcript: &mut Tr,
) -> Option<SoftmaxVerifierOutput>
where
    Tr: Transcript,
{
    append_eval_claim(transcript, &claim);
    (claim.point.len() == params.shape.point_len()).then_some(())?;
    let output_to_acc =
        verify_output_to_acc(claim, params.shape, &proof.output_to_acc, transcript)?;
    // Softmax is verified as a claim-reduction chain:
    // output -> floor -> exp tensor -> row sum / lookup RA.
    // Rounding, lookup-read, RA virtual, hamming weight, and RA booleanity are
    // separate sumchecks so each relation stays low-degree.
    let row_sum = verify_row_sum(
        params.shape,
        eval_i32_at_point(
            &advice.sum,
            &output_to_acc.acc.point[..params.shape.row_vars()],
        )?,
        output_to_acc.acc.point[..params.shape.row_vars()].to_vec(),
        &advice.sum,
        &advice.row_max,
        &advice.max_index,
        &proof.row_sum,
        transcript,
    )?;
    let exp = verify_exp_tensor(
        output_to_acc.acc,
        row_sum.exp,
        row_sum.input,
        params.shape,
        advice.min_diff,
        &advice.row_max,
        &advice.sum,
        &proof.exp,
        transcript,
    )?;
    let (read_challenges, _) = draw_softmax_lookup_challenges(transcript)?;
    let read = verify_softmax_read(
        exp.index.point.clone(),
        exp.index.value,
        exp.exp_base.value,
        &advice,
        read_challenges,
        &proof.lookup.read,
        transcript,
    )?;
    let lookup_point = read.ra.point[..read.ra.point.len() - params.shape.point_len()].to_vec();
    let ra_virtual = verify_softmax_ra_virtual(
        exp.index.point.clone(),
        lookup_point.clone(),
        read.ra.value,
        &proof.lookup.ra_virtual,
        transcript,
    )?;
    let hamming_weight = verify_softmax_hamming_weight(
        exp.index.point.clone(),
        lookup_point.len(),
        &proof.lookup.hamming_weight,
        transcript,
    )?;
    let booleanity_lookup_point = transcript.challenge_vector::<Fr>(lookup_point.len());
    let booleanity = verify_softmax_ra_booleanity(
        exp.index.point.clone(),
        booleanity_lookup_point,
        &proof.lookup.booleanity,
        transcript,
    )?;
    Some(SoftmaxVerifierOutput {
        input: exp.input,
        output_bits: output_to_acc.output_bits,
        floor_bits: output_to_acc.floor_bits,
        frac_bits: exp.frac_bits,
        exp_bits: exp.exp_bits,
        ra: RaOpeningClaims {
            read: read.ra,
            virtual_claim: ra_virtual,
            hamming_weight,
            booleanity,
        },
    })
}

struct RoundingRelation {
    output_round_mix: Fr,
    floor_round_mix: Fr,
    output_bit_challenges: [Fr; FRAC_BITS],
    floor_bit_challenges: [Fr; FRAC_BITS],
}

struct ExpRelation {
    min_diff: Fr,
    acc_mix: Fr,
    exp_acc_mix: Fr,
    exp_round_mix: Fr,
    diff_mix: Fr,
    frac_bit_challenges: [Fr; FRAC_BITS],
    exp_bit_challenges: [Fr; FRAC_BITS],
}

struct VerifiedOutputToAcc {
    acc: EvalClaim,
    output_bits: BitOpeningClaims,
    floor_bits: BitOpeningClaims,
}

struct VerifiedExp {
    input: EvalClaim,
    index: EvalClaim,
    exp_base: EvalClaim,
    frac_bits: BitOpeningClaims,
    exp_bits: BitOpeningClaims,
}

struct VerifiedRead {
    ra: EvalClaim,
}

struct VerifiedRowSum {
    exp: EvalClaim,
    input: EvalClaim,
}

fn verify_output_to_acc<Tr>(
    claim: EvalClaim,
    shape: MatrixShape,
    proof: &SoftmaxOutputToAccOutput,
    transcript: &mut Tr,
) -> Option<VerifiedOutputToAcc>
where
    Tr: Transcript,
{
    let (output_round_mix, floor_round_mix, output_bit_challenges, floor_bit_challenges) =
        draw_softmax_output_to_acc_challenges(transcript)?;
    let relation = RoundingRelation {
        output_round_mix,
        floor_round_mix,
        output_bit_challenges,
        floor_bit_challenges,
    };
    // Output-to-acc sumcheck combines the two rounding constraints:
    //
    //   256*output - floor + rem - 256*msb = 0
    //   256*floor - acc + rem = 0
    //
    // It returns the acc claim consumed by the exp tensor sumcheck.
    let sumcheck =
        verify_sumcheck_rounds(claim.value, &proof.rounds, shape.point_len(), transcript)?;
    let point = sumcheck.point;
    let final_relation = output_to_acc_final_relation(
        &relation,
        proof.output,
        proof.floor,
        proof.acc,
        proof.output_bits,
        proof.floor_bits,
    );
    (sumcheck.final_claim == eq_point_eval(&claim.point, &sumcheck.challenges)? * final_relation)
        .then_some(())?;
    (point.len() == shape.point_len()).then_some(())?;
    Some(VerifiedOutputToAcc {
        acc: EvalClaim::new(proof.acc, point.clone()),
        output_bits: bit_opening_claims(&point, proof.output_bits),
        floor_bits: bit_opening_claims(&point, proof.floor_bits),
    })
}

fn verify_exp_tensor<Tr>(
    acc_claim: EvalClaim,
    row_sum_exp_claim: EvalClaim,
    row_max_input_claim: EvalClaim,
    shape: MatrixShape,
    min_diff: i64,
    row_max_advice: &[i32],
    sum_advice: &[i32],
    proof: &SoftmaxExpOutput,
    transcript: &mut Tr,
) -> Option<VerifiedExp>
where
    Tr: Transcript,
{
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
        min_diff: field_from_i64(min_diff),
        acc_mix,
        exp_acc_mix,
        exp_round_mix,
        diff_mix,
        frac_bit_challenges,
        exp_bit_challenges,
    };
    // Exponential tensor sumcheck ties together the shifted input, LUT index,
    // LUT value, exp accumulator, row-sum coefficient, and exp rounding:
    //
    //   valid*(input - row_max) = 256*(min_diff + index) + frac
    //   exp_acc = exp_base * (256 + frac)
    //   exp     = round(exp_acc / 256)
    //   acc     = inv_sum(row) * exp
    //
    // It returns opening claims for input, index, exp_base, exp, row sum, and
    // both bit arrays.
    let sumcheck = verify_sumcheck_rounds(
        acc_claim.value
            + row_sum_exp_mix * row_sum_exp_claim.value
            + row_max_input_mix * row_max_input_claim.value,
        &proof.rounds,
        shape.point_len(),
        transcript,
    )?;
    let point = sumcheck.point;
    let valid = eval_valid_at_point(sum_advice, shape, &point)?;
    let coefficient = eval_coefficient_at_point(sum_advice, shape, &point)?;
    let row_point = &point[..shape.row_vars()];
    let row_max_eval = eval_i32_at_point(row_max_advice, row_point)?;
    (proof.row_max == row_max_eval).then_some(())?;
    let sum_eval = eval_i32_at_point(sum_advice, row_point)?;
    (proof.sum == sum_eval).then_some(())?;
    let final_relation = exp_final_relation(
        &relation,
        proof.input,
        proof.row_max,
        valid,
        coefficient,
        proof.index,
        proof.exp_base,
        proof.exp,
        proof.exp_acc,
        proof.acc,
        proof.frac_bits,
        proof.exp_bits,
    );
    let row_sum_final_relation = exp_derivation_final_relation(
        &relation,
        proof.input,
        proof.row_max,
        valid,
        proof.index,
        proof.exp_base,
        proof.exp,
        proof.exp_acc,
        proof.frac_bits,
        proof.exp_bits,
    );
    let row_max_input_final_relation = exp_input_final_relation(
        &relation,
        proof.input,
        proof.row_max,
        valid,
        proof.index,
        proof.exp_base,
        proof.exp,
        proof.exp_acc,
        proof.frac_bits,
        proof.exp_bits,
    );
    let final_check = eq_point_eval(&acc_claim.point, &sumcheck.challenges)? * final_relation
        + row_sum_exp_mix
            * eq_point_eval(&row_sum_exp_claim.point, &sumcheck.challenges)?
            * row_sum_final_relation
        + row_max_input_mix
            * eq_point_eval(&row_max_input_claim.point, &sumcheck.challenges)?
            * row_max_input_final_relation;
    (sumcheck.final_claim == final_check).then_some(())?;
    (point.len() == shape.point_len()).then_some(())?;
    Some(VerifiedExp {
        input: EvalClaim::new(proof.input, point.clone()),
        index: EvalClaim::new(proof.index, point.clone()),
        exp_base: EvalClaim::new(proof.exp_base, point.clone()),
        frac_bits: bit_opening_claims(&point, proof.frac_bits),
        exp_bits: bit_opening_claims(&point, proof.exp_bits),
    })
}

fn verify_row_sum<Tr>(
    shape: MatrixShape,
    sum_claim: Fr,
    row_point: Vec<Fr>,
    sum_advice: &[i32],
    row_max_advice: &[i32],
    max_index: &[usize],
    proof: &SoftmaxRowSumOutput,
    transcript: &mut Tr,
) -> Option<VerifiedRowSum>
where
    Tr: Transcript,
{
    validate_max_index(max_index, shape)?;
    let row_max_mix = draw_softmax_row_sum_challenge(transcript);
    let row_max_claim = eval_i32_at_point(row_max_advice, &row_point)?;
    // Row-sum sumcheck proves both the advice sum(row) and that row_max(row)
    // is attained at the public max_index(row):
    //
    //   sum(row) + γ*row_max(row)
    //     = Σ_col valid(row, col)*exp(row, col)
    //       + γ*selector_max(row, col)*input(row, col)
    //
    // This is the check that makes the verifier-provided inv_sum coefficient
    // meaningful and proves row_max is not just arbitrary advice.
    let sumcheck = verify_sumcheck_rounds(
        sum_claim + row_max_mix * row_max_claim,
        &proof.rounds,
        shape.point_len(),
        transcript,
    )?;
    let point = sumcheck.point;
    (point.len() == shape.point_len()).then_some(())?;
    let row_bound = &sumcheck.challenges[..shape.row_vars()];
    let row_eq = eq_point_eval(&row_point, row_bound)?;
    let valid = eval_valid_at_point(sum_advice, shape, &point)?;
    let selector = eval_max_selector_at_point(max_index, shape, &point)?;
    let final_relation = valid * proof.exp + row_max_mix * selector * proof.input;
    (sumcheck.final_claim == row_eq * final_relation).then_some(())?;
    Some(VerifiedRowSum {
        exp: EvalClaim::new(proof.exp, point.clone()),
        input: EvalClaim::new(proof.input, point),
    })
}

fn validate_max_index(max_index: &[usize], shape: MatrixShape) -> Option<()> {
    (max_index.len() == shape.rows).then_some(())?;
    max_index.iter().all(|&col| col < shape.cols).then_some(())
}

fn eval_max_selector_at_point(max_index: &[usize], shape: MatrixShape, point: &[Fr]) -> Option<Fr> {
    validate_max_index(max_index, shape)?;
    (point.len() == shape.point_len()).then_some(())?;
    let (row_point, col_point) = point.split_at(shape.row_vars());
    Some(
        max_index
            .iter()
            .enumerate()
            .map(|(row, &col)| {
                eq_index_at_point(row, row_point) * eq_index_at_point(col, col_point)
            })
            .sum(),
    )
}

fn eq_index_at_point(index: usize, point: &[Fr]) -> Fr {
    point
        .iter()
        .enumerate()
        .map(|(bit, value)| {
            if ((index >> bit) & 1) == 1 {
                *value
            } else {
                Fr::one() - value
            }
        })
        .product()
}

fn verify_softmax_read<Tr>(
    tensor_point: Vec<Fr>,
    index: Fr,
    exp_base: Fr,
    advice: &SoftmaxAdvice,
    read_challenges: [Fr; 3],
    proof: &SoftmaxReadOutput,
    transcript: &mut Tr,
) -> Option<VerifiedRead>
where
    Tr: Transcript,
{
    let [one_challenge, id_challenge, exp_challenge] = read_challenges;
    let claim = one_challenge + id_challenge * index + exp_challenge * exp_base;
    // RA read sumcheck for softmax LUT:
    //
    //   gamma_1 + gamma_id*index + gamma_exp*exp_base
    //     = Σ_addr ra(addr, x)
    //         * (gamma_1 + gamma_id*id(addr) + gamma_exp*exp_table(addr))
    //
    // This ties the tensor index/exp_base claims to the public exp table.
    let lookup_vars = build_softmax_tables(advice)?.id.len().ilog2() as usize;
    let sumcheck = verify_sumcheck_rounds(claim, &proof.rounds, lookup_vars, transcript)?;
    let lookup_point = sumcheck.point;
    let ra = EvalClaim::new(
        proof.ra,
        [lookup_point.as_slice(), tensor_point.as_slice()].concat(),
    );
    let public_evals = build_public_softmax_read_evals(advice, &lookup_point)?;
    let final_relation = proof.ra
        * (one_challenge + id_challenge * public_evals.id + exp_challenge * public_evals.exp_base);
    (sumcheck.final_claim == final_relation).then_some(())?;
    Some(VerifiedRead { ra })
}

struct SoftmaxReadPublicEvals {
    id: Fr,
    exp_base: Fr,
}

fn build_public_softmax_read_evals(
    advice: &SoftmaxAdvice,
    lookup_point: &[Fr],
) -> Option<SoftmaxReadPublicEvals> {
    let tables = build_softmax_tables(advice)?;
    Some(SoftmaxReadPublicEvals {
        id: eval_i32_mle_at_point(&tables.id, lookup_point)?,
        exp_base: eval_i32_mle_at_point(&tables.exp_base, lookup_point)?,
    })
}

fn verify_softmax_ra_virtual<Tr>(
    tensor_point: Vec<Fr>,
    lookup_point: Vec<Fr>,
    claim: Fr,
    proof: &SoftmaxRaVirtualOutput,
    transcript: &mut Tr,
) -> Option<EvalClaim>
where
    Tr: Transcript,
{
    // RA virtual sumcheck fixes the tensor point and keeps the lookup address
    // from the read sumcheck:
    //
    //   ra(addr*, x_tensor)
    //     = Σ_x eq(x_tensor, x) * ra(addr*, x)
    let sumcheck = verify_sumcheck_rounds(claim, &proof.rounds, tensor_point.len(), transcript)?;
    let row_point = sumcheck.point;
    (sumcheck.final_claim == eq_point_eval(&tensor_point, &sumcheck.challenges)? * proof.ra)
        .then_some(())?;
    Some(EvalClaim::new(
        proof.ra,
        [lookup_point.as_slice(), row_point.as_slice()].concat(),
    ))
}

fn verify_softmax_hamming_weight<Tr>(
    tensor_point: Vec<Fr>,
    lookup_vars: usize,
    proof: &SoftmaxHammingWeightOutput,
    transcript: &mut Tr,
) -> Option<EvalClaim>
where
    Tr: Transcript,
{
    // Hamming-weight sumcheck:
    //
    //   1 = Σ_addr ra(addr, x_tensor)
    //
    // This proves that the selected tensor row has total RA mass one.
    let sumcheck = verify_sumcheck_rounds(Fr::one(), &proof.rounds, lookup_vars, transcript)?;
    let lookup_point = sumcheck.point;
    (sumcheck.final_claim == proof.ra).then_some(())?;
    Some(EvalClaim::new(
        proof.ra,
        [lookup_point.as_slice(), tensor_point.as_slice()].concat(),
    ))
}

fn verify_softmax_ra_booleanity<Tr>(
    tensor_point: Vec<Fr>,
    lookup_point: Vec<Fr>,
    proof: &SoftmaxBooleanityOutput,
    transcript: &mut Tr,
) -> Option<EvalClaim>
where
    Tr: Transcript,
{
    // RA booleanity sumcheck:
    //
    //   0 = Σ_{addr,x} eq((addr*, x_tensor), (addr, x))
    //       * ra(addr, x) * (ra(addr, x) - 1)
    //
    // Hamming weight plus booleanity gives one-hotness for the row used by the
    // read sumcheck.
    let sumcheck = verify_sumcheck_rounds(
        Fr::zero(),
        &proof.rounds,
        lookup_point.len() + tensor_point.len(),
        transcript,
    )?;
    let point = sumcheck.point;
    let eq_point = [lookup_point.as_slice(), tensor_point.as_slice()].concat();
    (sumcheck.final_claim
        == eq_point_eval(&eq_point, &sumcheck.challenges)? * proof.ra * (proof.ra - Fr::one()))
    .then_some(())?;
    Some(EvalClaim::new(proof.ra, point))
}

fn bit_opening_claims(point: &[Fr], values: [Fr; FRAC_BITS]) -> BitOpeningClaims {
    values.map(|value| EvalClaim::new(value, point.to_vec()))
}

fn output_to_acc_final_relation(
    relation: &RoundingRelation,
    output: Fr,
    floor: Fr,
    acc: Fr,
    output_bits: [Fr; FRAC_BITS],
    floor_bits: [Fr; FRAC_BITS],
) -> Fr {
    let scale = Fr::from(SCALE);
    let output_rem = remainder_constant(&output_bits.map(|bit| (bit, bit)));
    let output_msb = output_bits[FRAC_BITS - 1];
    let floor_rem = remainder_constant(&floor_bits.map(|bit| (bit, bit)));
    let output_constraint = scale * output - floor + output_rem - scale * output_msb;
    let floor_constraint = scale * floor - acc + floor_rem;
    output
        + relation.output_round_mix * output_constraint
        + relation.floor_round_mix * floor_constraint
        + output_bits
            .iter()
            .zip_eq(relation.output_bit_challenges)
            .map(|(bit, challenge)| challenge * *bit * (*bit - Fr::one()))
            .sum::<Fr>()
        + floor_bits
            .iter()
            .zip_eq(relation.floor_bit_challenges)
            .map(|(bit, challenge)| challenge * *bit * (*bit - Fr::one()))
            .sum::<Fr>()
}

#[allow(clippy::too_many_arguments)]
fn exp_final_relation(
    relation: &ExpRelation,
    input: Fr,
    row_max: Fr,
    valid: Fr,
    coefficient: Fr,
    index: Fr,
    exp_base: Fr,
    exp: Fr,
    exp_acc: Fr,
    acc: Fr,
    frac_bits: [Fr; FRAC_BITS],
    exp_bits: [Fr; FRAC_BITS],
) -> Fr {
    let frac = remainder_constant(&frac_bits.map(|bit| (bit, bit)));
    let exp_rem = remainder_constant(&exp_bits.map(|bit| (bit, bit)));
    let exp_msb = exp_bits[FRAC_BITS - 1];
    let scale = Fr::from(SCALE);
    let acc_constraint = acc - coefficient * exp;
    let exp_acc_constraint = exp_acc - exp_base * (scale + frac);
    let exp_round_constraint = scale * exp - exp_acc + exp_rem - scale * exp_msb;
    let diff_constraint = valid * (input - row_max) - scale * (relation.min_diff + index) - frac;
    let mut value = acc
        + relation.acc_mix * acc_constraint
        + relation.exp_acc_mix * exp_acc_constraint
        + relation.exp_round_mix * exp_round_constraint
        + relation.diff_mix * diff_constraint;
    value += frac_bits
        .iter()
        .zip_eq(relation.frac_bit_challenges)
        .map(|(bit, challenge)| challenge * *bit * (*bit - Fr::one()))
        .sum::<Fr>();
    value += exp_bits
        .iter()
        .zip_eq(relation.exp_bit_challenges)
        .map(|(bit, challenge)| challenge * *bit * (*bit - Fr::one()))
        .sum::<Fr>();
    value
}

#[allow(clippy::too_many_arguments)]
fn exp_derivation_final_relation(
    relation: &ExpRelation,
    input: Fr,
    row_max: Fr,
    valid: Fr,
    index: Fr,
    exp_base: Fr,
    exp: Fr,
    exp_acc: Fr,
    frac_bits: [Fr; FRAC_BITS],
    exp_bits: [Fr; FRAC_BITS],
) -> Fr {
    let frac = remainder_constant(&frac_bits.map(|bit| (bit, bit)));
    let exp_rem = remainder_constant(&exp_bits.map(|bit| (bit, bit)));
    let exp_msb = exp_bits[FRAC_BITS - 1];
    let scale = Fr::from(SCALE);
    let exp_acc_constraint = exp_acc - exp_base * (scale + frac);
    let exp_round_constraint = scale * exp - exp_acc + exp_rem - scale * exp_msb;
    let diff_constraint = valid * (input - row_max) - scale * (relation.min_diff + index) - frac;
    let mut value = exp
        + relation.exp_acc_mix * exp_acc_constraint
        + relation.exp_round_mix * exp_round_constraint
        + relation.diff_mix * diff_constraint;
    value += frac_bits
        .iter()
        .zip_eq(relation.frac_bit_challenges)
        .map(|(bit, challenge)| challenge * *bit * (*bit - Fr::one()))
        .sum::<Fr>();
    value += exp_bits
        .iter()
        .zip_eq(relation.exp_bit_challenges)
        .map(|(bit, challenge)| challenge * *bit * (*bit - Fr::one()))
        .sum::<Fr>();
    value
}

#[allow(clippy::too_many_arguments)]
fn exp_input_final_relation(
    relation: &ExpRelation,
    input: Fr,
    row_max: Fr,
    valid: Fr,
    index: Fr,
    exp_base: Fr,
    exp: Fr,
    exp_acc: Fr,
    frac_bits: [Fr; FRAC_BITS],
    exp_bits: [Fr; FRAC_BITS],
) -> Fr {
    let frac = remainder_constant(&frac_bits.map(|bit| (bit, bit)));
    let exp_rem = remainder_constant(&exp_bits.map(|bit| (bit, bit)));
    let exp_msb = exp_bits[FRAC_BITS - 1];
    let scale = Fr::from(SCALE);
    let exp_acc_constraint = exp_acc - exp_base * (scale + frac);
    let exp_round_constraint = scale * exp - exp_acc + exp_rem - scale * exp_msb;
    let diff_constraint = valid * (input - row_max) - scale * (relation.min_diff + index) - frac;
    let mut value = input
        + relation.exp_acc_mix * exp_acc_constraint
        + relation.exp_round_mix * exp_round_constraint
        + relation.diff_mix * diff_constraint;
    value += frac_bits
        .iter()
        .zip_eq(relation.frac_bit_challenges)
        .map(|(bit, challenge)| challenge * *bit * (*bit - Fr::one()))
        .sum::<Fr>();
    value += exp_bits
        .iter()
        .zip_eq(relation.exp_bit_challenges)
        .map(|(bit, challenge)| challenge * *bit * (*bit - Fr::one()))
        .sum::<Fr>();
    value
}

fn eval_valid_at_point(sum: &[i32], shape: MatrixShape, point: &[Fr]) -> Option<Fr> {
    (sum.len() == shape.rows && point.len() == shape.point_len()).then_some(())?;
    Some(
        (0..shape.cols)
            .flat_map(|col| {
                (0..shape.rows).map(move |row| {
                    let index = row + shape.rows * col;
                    eq_eval(index, point)
                        * if sum[row] != 0 && col <= row % shape.cols {
                            Fr::one()
                        } else {
                            Fr::zero()
                        }
                })
            })
            .sum(),
    )
}

fn eval_coefficient_at_point(sum: &[i32], shape: MatrixShape, point: &[Fr]) -> Option<Fr> {
    (sum.len() == shape.rows && point.len() == shape.point_len()).then_some(())?;
    Some(
        (0..shape.cols)
            .flat_map(|col| {
                (0..shape.rows).map(move |row| {
                    let index = row + shape.rows * col;
                    eq_eval(index, point)
                        * if sum[row] != 0 && col <= row % shape.cols {
                            field_from_i64(inv_sum_q16(sum[row]))
                        } else {
                            Fr::zero()
                        }
                })
            })
            .sum(),
    )
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
