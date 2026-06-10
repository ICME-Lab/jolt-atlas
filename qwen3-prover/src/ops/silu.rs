//! SiLU proof.
//!
//! The runtime computes:
//!
//! ```text
//! index = round(input / 256) - min_n
//! n     = min_n + index
//! base  = base_table[index]
//! slope = slope_table[index]
//!
//! acc = base + (input - 256*n)*slope
//! out = round(acc / 256)
//! ```
//!
//! The proof is a claim-reduction chain:
//!
//! ```text
//! target claim: out(r)
//!
//! B. SiLU tensor sumcheck
//!    proves out(r) comes from input(r), index(r), base(r), slope(r),
//!    input remainder bits, and output remainder bits
//!    outputs claims input(r), index(r), base(r), slope(r)
//!
//! A. lookup/RA sumcheck
//!    proves index(r), base(r), slope(r) come from ra(r, i) and LUT tables
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
pub use qwen3_common::ops::silu::{
    SiluAdvice, SiluBooleanityOutput, SiluHammingWeightOutput, SiluLookupOutput, SiluOutput,
    SiluParams, SiluRaVirtualOutput, SiluReadOutput, SiluTensorOutput, SiluVerifierOutput,
    build_silu_tables, draw_silu_lookup_challenges, draw_silu_tensor_challenges,
    validate_silu_range,
};
use qwen3_common::{FRAC_BITS, SCALE};

use crate::{
    layer::{BitOpeningClaims, EvalClaim, RaOpeningClaims, append_eval_claim},
    profile,
    round_message::{RoundPolynomial, SumCheckRounds, append_round_statement},
    shape::MatrixShape,
};

pub struct SiluTensorProverOutput {
    pub proof: SiluTensorOutput,
    pub input: EvalClaim,
    pub index: EvalClaim,
    pub base: EvalClaim,
    pub slope: EvalClaim,
    pub input_bits: BitOpeningClaims,
    pub output_bits: BitOpeningClaims,
}

struct SiluReadProverOutput {
    proof: SiluReadOutput,
    ra: EvalClaim,
}

struct SiluRaVirtualProverOutput {
    proof: SiluRaVirtualOutput,
    ra: EvalClaim,
}

struct SiluHammingWeightProverOutput {
    proof: SiluHammingWeightOutput,
    ra: EvalClaim,
}

struct SiluBooleanityProverOutput {
    proof: SiluBooleanityOutput,
    ra: EvalClaim,
}

struct SiluLookupProverOutput {
    proof: SiluLookupOutput,
    ra: RaOpeningClaims,
}

pub struct SiluProverOutput {
    pub proof: SiluOutput,
    pub input: EvalClaim,
    pub input_bits: BitOpeningClaims,
    pub output_bits: BitOpeningClaims,
    pub ra: RaOpeningClaims,
}

pub struct SiluProverInput {
    pub params: SiluParams,
    pub advice: SiluAdvice,
    pub witness: SiluWitness,
}

pub struct SiluWitness {
    pub input: Vec<i32>,
    pub output: Vec<i32>,
    pub ra: Vec<u8>,
    pub input_remainder_bits: [Vec<bool>; FRAC_BITS],
    pub output_remainder_bits: [Vec<bool>; FRAC_BITS],
}

struct SiluTensorRelation {
    min_n: Fr,
    input_round_mix: Fr,
    input_bit_booleanity_challenges: [Fr; FRAC_BITS],
    output_bit_booleanity_challenges: [Fr; FRAC_BITS],
}

struct SiluLookupRelation {
    read_challenges: [Fr; 4],
}

// B. SiLU tensor sumcheck:
//   input + 256 * input_msb - input_rem - 256 * (min_n + index) = 0
//   256 * output - base - (input - 256 * (min_n + index)) * slope
//     + output_rem - 256 * output_msb = 0
//   input/output remainder bits satisfy b_j * (b_j - 1) = 0.
//
// A. lookup/RA sumcheck:
//   Σ_i ra_i = 1
//   Σ_i id_i * ra_i = index
//   Σ_i base_i * ra_i = base
//   Σ_i slope_i * ra_i = slope
//   ra_i * (ra_i - 1) = 0
pub fn prove_silu<Tr>(
    claim: EvalClaim,
    input: SiluProverInput,
    transcript: &mut Tr,
) -> Option<SiluProverOutput>
where
    Tr: Transcript,
{
    append_eval_claim(transcript, &claim);
    validate_input(&claim, &input)?;
    let (index, base, slope, ra, id_table, base_table, slope_table) =
        profile::measure("silu.prepare.lookup_tables", || {
            prepare_lookup_witness(&input)
        })?;

    let tensor = profile::measure("silu.proof.tensor", || {
        prove_silu_tensor(
            claim,
            input.params,
            field_from_i64(input.advice.min_n),
            input.witness.input,
            index,
            base,
            slope,
            input.witness.output,
            input.witness.input_remainder_bits,
            input.witness.output_remainder_bits,
            transcript,
        )
    })?;

    append_eval_claim(transcript, &tensor.index);
    append_eval_claim(transcript, &tensor.base);
    append_eval_claim(transcript, &tensor.slope);

    let lookup = profile::measure("silu.proof.lookup", || {
        prove_silu_lookup_with_claims(
            tensor.index.point.clone(),
            tensor.index.value,
            tensor.base.value,
            tensor.slope.value,
            ra,
            id_table,
            base_table,
            slope_table,
            input.params.shape,
            transcript,
        )
    })?;

    Some(SiluProverOutput {
        input: tensor.input.clone(),
        input_bits: tensor.input_bits,
        output_bits: tensor.output_bits,
        ra: lookup.ra,
        proof: SiluOutput {
            tensor: tensor.proof,
            lookup: lookup.proof,
        },
    })
}

#[allow(clippy::too_many_arguments)]
pub fn prove_silu_tensor<Tr>(
    claim: EvalClaim,
    params: SiluParams,
    min_n: Fr,
    input: Vec<i32>,
    index: Vec<i32>,
    base: Vec<i32>,
    slope: Vec<i32>,
    output: Vec<i32>,
    input_remainder_bits: [Vec<bool>; FRAC_BITS],
    output_remainder_bits: [Vec<bool>; FRAC_BITS],
    transcript: &mut Tr,
) -> Option<SiluTensorProverOutput>
where
    Tr: Transcript,
{
    (claim.point.len() == params.shape.point_len()).then_some(())?;
    let (input_round_mix, input_bit_booleanity_challenges, output_bit_booleanity_challenges) =
        draw_silu_tensor_challenges(transcript)?;
    let relation = SiluTensorRelation {
        min_n,
        input_round_mix,
        input_bit_booleanity_challenges,
        output_bit_booleanity_challenges,
    };

    let mut input = collect_matrix_table(input, params.shape)?;
    let mut index = collect_mle_table(index, params.shape)?;
    let mut base = collect_mle_table(base, params.shape)?;
    let mut slope = collect_mle_table(slope, params.shape)?;
    let mut output = collect_matrix_table(output, params.shape)?;
    let mut input_bits =
        RoundingBitsState::from_bytes(collect_matrix_bytes(input_remainder_bits, params.shape)?);
    let mut output_bits =
        RoundingBitsState::from_bytes(collect_matrix_bytes(output_remainder_bits, params.shape)?);

    let split_eq_point = claim.point.iter().rev().copied().collect::<Vec<_>>();
    let mut eq = GruenSplitEqPolynomial::new(&split_eq_point, BindingOrder::LowToHigh);
    let mut claim_i = Fr::zero();
    let mut round_polys = Vec::with_capacity(claim.point.len());
    let mut challenges = Vec::with_capacity(claim.point.len());

    while input.len() > 1 {
        let round = silu_tensor_round_poly(
            &eq,
            &relation,
            &input,
            &index,
            &base,
            &slope,
            &output,
            &input_bits,
            &output_bits,
        )?;
        append_round_statement(transcript, claim_i, &round);
        let challenge = transcript.challenge_scalar_optimized::<Fr>();
        let r = challenge.into();
        eq.bind(challenge);
        bind(&mut input, r);
        bind(&mut index, r);
        bind(&mut base, r);
        bind(&mut slope, r);
        bind(&mut output, r);
        input_bits.bind(r);
        output_bits.bind(r);
        claim_i = round.eval(r);
        challenges.push(challenge);
        round_polys.push(round);
    }

    let point = fr_challenges(&challenges);
    let input_bits_at_point = input_bits.bits_at(0);
    let output_bits_at_point = output_bits.bits_at(0);
    let proof = SiluTensorOutput {
        rounds: SumCheckRounds {
            round_polys,
            final_claim: claim_i,
        },
        input: input[0],
        index: index[0],
        base: base[0],
        slope: slope[0],
        output: output[0],
        input_bits: input_bits_at_point,
        output_bits: output_bits_at_point,
    };
    Some(SiluTensorProverOutput {
        input: EvalClaim::new(proof.input, point.clone()),
        index: EvalClaim::new(proof.index, point.clone()),
        base: EvalClaim::new(proof.base, point.clone()),
        slope: EvalClaim::new(proof.slope, point.clone()),
        input_bits: bit_opening_claims(&point, input_bits_at_point),
        output_bits: bit_opening_claims(&point, output_bits_at_point),
        proof,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn prove_silu_lookup<Tr>(
    tensor_point: Vec<Fr>,
    index: Fr,
    base: Fr,
    slope: Fr,
    ra: Vec<u8>,
    id_table: Vec<i32>,
    base_table: Vec<i32>,
    slope_table: Vec<i32>,
    shape: MatrixShape,
    transcript: &mut Tr,
) -> Option<SiluLookupOutput>
where
    Tr: Transcript,
{
    Some(
        prove_silu_lookup_with_claims(
            tensor_point,
            index,
            base,
            slope,
            ra,
            id_table,
            base_table,
            slope_table,
            shape,
            transcript,
        )?
        .proof,
    )
}

#[allow(clippy::too_many_arguments)]
fn prove_silu_lookup_with_claims<Tr>(
    tensor_point: Vec<Fr>,
    index: Fr,
    base: Fr,
    slope: Fr,
    ra: Vec<u8>,
    id_table: Vec<i32>,
    base_table: Vec<i32>,
    slope_table: Vec<i32>,
    shape: MatrixShape,
    transcript: &mut Tr,
) -> Option<SiluLookupProverOutput>
where
    Tr: Transcript,
{
    let (read_challenges, _) = draw_silu_lookup_challenges(transcript)?;
    let entries = id_table.len();
    (entries.is_power_of_two()
        && base_table.len() == entries
        && slope_table.len() == entries
        && tensor_point.len() == shape.point_len()
        && ra.len() == shape.len().checked_mul(entries)?)
    .then_some(())?;
    let lookup_point_len = entries.ilog2() as usize;
    let selected = selected_lookup_rows(&ra, shape.len(), entries)?;

    let read = profile::measure("silu.lookup.read_raf", || {
        prove_silu_read(
            tensor_point.clone(),
            index,
            base,
            slope,
            &selected,
            id_table.clone(),
            base_table.clone(),
            slope_table.clone(),
            shape,
            read_challenges,
            transcript,
        )
    })?;
    let ra_virtual = profile::measure("silu.lookup.ra_virtual", || {
        prove_silu_ra_virtual(
            tensor_point.clone(),
            read.ra.point[..lookup_point_len].to_vec(),
            read.ra.value,
            &selected,
            shape,
            entries,
            transcript,
        )
    })?;
    let hamming_weight = profile::measure("silu.lookup.hamming_weight", || {
        prove_silu_hamming_weight(tensor_point.clone(), &selected, shape, entries, transcript)
    })?;
    let booleanity_address_point = transcript.challenge_vector::<Fr>(entries.ilog2() as usize);
    let booleanity = profile::measure("silu.lookup.booleanity", || {
        prove_silu_ra_booleanity(
            tensor_point,
            booleanity_address_point,
            &selected,
            shape,
            entries,
            transcript,
        )
    })?;

    Some(SiluLookupProverOutput {
        ra: RaOpeningClaims {
            read: read.ra,
            virtual_claim: ra_virtual.ra,
            hamming_weight: hamming_weight.ra,
            booleanity: booleanity.ra,
        },
        proof: SiluLookupOutput {
            read: read.proof,
            ra_virtual: ra_virtual.proof,
            hamming_weight: hamming_weight.proof,
            booleanity: booleanity.proof,
        },
    })
}

#[allow(clippy::too_many_arguments)]
fn prove_silu_read<Tr>(
    tensor_point: Vec<Fr>,
    index: Fr,
    base: Fr,
    slope: Fr,
    selected: &[usize],
    id_table: Vec<i32>,
    base_table: Vec<i32>,
    slope_table: Vec<i32>,
    shape: MatrixShape,
    read_challenges: [Fr; 4],
    transcript: &mut Tr,
) -> Option<SiluReadProverOutput>
where
    Tr: Transcript,
{
    let entries = id_table.len();
    let [one_challenge, id_challenge, base_challenge, slope_challenge] = read_challenges;
    let mut claim_i =
        one_challenge + id_challenge * index + base_challenge * base + slope_challenge * slope;
    let mut ra = partial_ra_at_matrix_point(selected, &tensor_point, shape, entries)?;
    let mut id_table = id_table.into_iter().map(Fr::from_i32).collect::<Vec<_>>();
    let mut base_table = base_table.into_iter().map(Fr::from_i32).collect::<Vec<_>>();
    let mut slope_table = slope_table
        .into_iter()
        .map(Fr::from_i32)
        .collect::<Vec<_>>();
    let relation = SiluLookupRelation { read_challenges };
    let mut round_polys = Vec::with_capacity(entries.ilog2() as usize);
    let mut challenges = Vec::with_capacity(entries.ilog2() as usize);

    while ra.len() > 1 {
        let round = silu_read_round_poly(&relation, &ra, &id_table, &base_table, &slope_table)?;
        append_round_statement(transcript, claim_i, &round);
        let challenge = transcript.challenge_scalar_optimized::<Fr>();
        let r = challenge.into();
        bind(&mut ra, r);
        bind(&mut id_table, r);
        bind(&mut base_table, r);
        bind(&mut slope_table, r);
        claim_i = round.eval(r);
        challenges.push(challenge);
        round_polys.push(round);
    }

    let lookup_point = fr_challenges(&challenges);
    let proof = SiluReadOutput {
        rounds: SumCheckRounds {
            round_polys,
            final_claim: claim_i,
        },
        ra: ra[0],
    };
    Some(SiluReadProverOutput {
        ra: EvalClaim::new(
            proof.ra,
            [lookup_point.as_slice(), tensor_point.as_slice()].concat(),
        ),
        proof,
    })
}

#[allow(clippy::too_many_arguments)]
fn silu_tensor_round_poly(
    eq: &GruenSplitEqPolynomial<Fr>,
    relation: &SiluTensorRelation,
    input: &[Fr],
    index: &[Fr],
    base: &[Fr],
    slope: &[Fr],
    output: &[Fr],
    input_bits: &RoundingBitsState,
    output_bits: &RoundingBitsState,
) -> Option<RoundPolynomial<4>> {
    let len = input.len();
    (eq.len() == len
        && index.len() == len
        && base.len() == len
        && slope.len() == len
        && output.len() == len
        && input_bits.len() == len
        && output_bits.len() == len
        && len % 2 == 0)
        .then_some(())?;

    let relation_evals = eq.par_fold_out_in_unreduced::<9, 3>(&|row| {
        silu_tensor_relation_evals(
            row,
            relation,
            input,
            index,
            base,
            slope,
            output,
            input_bits,
            output_bits,
        )
    });
    Some(quadratic_relation_times_eq(
        relation_evals,
        eq.get_current_w(),
        eq.get_current_scalar(),
    ))
}

#[allow(clippy::too_many_arguments)]
fn silu_tensor_relation_evals(
    row: usize,
    relation: &SiluTensorRelation,
    input: &[Fr],
    index: &[Fr],
    base: &[Fr],
    slope: &[Fr],
    output: &[Fr],
    input_bits: &RoundingBitsState,
    output_bits: &RoundingBitsState,
) -> [Fr; 3] {
    let input_0 = input[2 * row];
    let input_linear = input[2 * row + 1] - input_0;
    let index_0 = index[2 * row];
    let index_linear = index[2 * row + 1] - index_0;
    let base_0 = base[2 * row];
    let base_linear = base[2 * row + 1] - base_0;
    let slope_0 = slope[2 * row];
    let slope_linear = slope[2 * row + 1] - slope_0;
    let output_0 = output[2 * row];
    let output_linear = output[2 * row + 1] - output_0;
    let input_bit_pairs = pair_bits(input_bits.bits_at(2 * row), input_bits.bits_at(2 * row + 1));
    let output_bit_pairs = pair_bits(
        output_bits.bits_at(2 * row),
        output_bits.bits_at(2 * row + 1),
    );

    let n_0 = relation.min_n + index_0;
    let n_linear = index_linear;
    let scale = Fr::from(SCALE);

    let input_remainder_0 = remainder_constant(&input_bit_pairs);
    let input_remainder_linear = remainder_linear(&input_bit_pairs);
    let output_remainder_0 = remainder_constant(&output_bit_pairs);
    let output_remainder_linear = remainder_linear(&output_bit_pairs);
    let input_msb_0 = input_bit_pairs[FRAC_BITS - 1].0;
    let input_msb_linear = input_bit_pairs[FRAC_BITS - 1].1 - input_msb_0;
    let output_msb_0 = output_bit_pairs[FRAC_BITS - 1].0;
    let output_msb_linear = output_bit_pairs[FRAC_BITS - 1].1 - output_msb_0;

    let output_round_constant = scale * output_0 - base_0 - (input_0 - scale * n_0) * slope_0
        + output_remainder_0
        - scale * output_msb_0;
    let output_round_linear = scale * output_linear
        - base_linear
        - ((input_0 - scale * n_0) * slope_linear + (input_linear - scale * n_linear) * slope_0)
        + output_remainder_linear
        - scale * output_msb_linear;
    let output_round_leading = -(input_linear - scale * n_linear) * slope_linear;

    let input_round_constant = input_0 + scale * input_msb_0 - input_remainder_0 - scale * n_0;
    let input_round_linear =
        input_linear + scale * input_msb_linear - input_remainder_linear - scale * n_linear;

    let mut constant = output_round_constant + relation.input_round_mix * input_round_constant;
    let mut linear = output_round_linear + relation.input_round_mix * input_round_linear;
    let mut leading = output_round_leading;

    add_bit_booleanity(
        &mut constant,
        &mut linear,
        &mut leading,
        &relation.input_bit_booleanity_challenges,
        &input_bit_pairs,
    );
    add_bit_booleanity(
        &mut constant,
        &mut linear,
        &mut leading,
        &relation.output_bit_booleanity_challenges,
        &output_bit_pairs,
    );

    [
        constant,
        constant + linear + leading,
        constant + linear + linear + leading * Fr::from(4_u64),
    ]
}

fn prove_silu_ra_virtual<Tr>(
    tensor_point: Vec<Fr>,
    lookup_point: Vec<Fr>,
    mut claim_i: Fr,
    selected: &[usize],
    shape: MatrixShape,
    entries: usize,
    transcript: &mut Tr,
) -> Option<SiluRaVirtualProverOutput>
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
    let proof = SiluRaVirtualOutput {
        rounds: SumCheckRounds {
            round_polys,
            final_claim: claim_i,
        },
        ra: ra[0],
    };
    Some(SiluRaVirtualProverOutput {
        ra: EvalClaim::new(
            proof.ra,
            [lookup_point.as_slice(), row_point.as_slice()].concat(),
        ),
        proof,
    })
}

fn prove_silu_ra_booleanity<Tr>(
    tensor_point: Vec<Fr>,
    lookup_point: Vec<Fr>,
    selected: &[usize],
    shape: MatrixShape,
    entries: usize,
    transcript: &mut Tr,
) -> Option<SiluBooleanityProverOutput>
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
        let round = profile::measure_detail("silu.lookup.booleanity.round_poly", || {
            ra_booleanity_round_poly(&eq, &ra)
        })?;
        let challenge = profile::measure_detail("silu.lookup.booleanity.transcript", || {
            append_round_statement(transcript, claim_i, &round);
            transcript.challenge_scalar_optimized::<Fr>()
        });
        let r = challenge.into();
        profile::measure_detail("silu.lookup.booleanity.eq_bind", || eq.bind(challenge));
        profile::measure_detail("silu.lookup.booleanity.ra_bind", || ra.bind(r));
        claim_i = round.eval(r);
        challenges.push(challenge);
        round_polys.push(round);
    }

    let point = fr_challenges(&challenges);
    let proof = SiluBooleanityOutput {
        rounds: SumCheckRounds {
            round_polys,
            final_claim: claim_i,
        },
        ra: ra.value_at(0),
    };
    Some(SiluBooleanityProverOutput {
        ra: EvalClaim::new(proof.ra, point),
        proof,
    })
}

fn silu_read_round_poly(
    relation: &SiluLookupRelation,
    ra: &[Fr],
    id_table: &[Fr],
    base_table: &[Fr],
    slope_table: &[Fr],
) -> Option<RoundPolynomial<3>> {
    let len = ra.len();
    (id_table.len() == len && base_table.len() == len && slope_table.len() == len && len % 2 == 0)
        .then_some(())?;

    let mut coeffs = [Fr::zero(); 3];
    for row in 0..len / 2 {
        let [constant, at_one, at_two] =
            silu_read_relation_evals(row, relation, ra, id_table, base_table, slope_table);
        let round = quadratic_from_evals(constant, at_one, at_two);
        coeffs[0] += round.coeffs[0];
        coeffs[1] += round.coeffs[1];
        coeffs[2] += round.coeffs[2];
    }
    Some(RoundPolynomial { coeffs })
}

fn silu_read_relation_evals(
    row: usize,
    relation: &SiluLookupRelation,
    ra: &[Fr],
    id_table: &[Fr],
    base_table: &[Fr],
    slope_table: &[Fr],
) -> [Fr; 3] {
    std::array::from_fn(|point| {
        let x = Fr::from(point as u64);
        let ra_x = line_at(ra[2 * row], ra[2 * row + 1], x);
        let id_x = line_at(id_table[2 * row], id_table[2 * row + 1], x);
        let base_x = line_at(base_table[2 * row], base_table[2 * row + 1], x);
        let slope_x = line_at(slope_table[2 * row], slope_table[2 * row + 1], x);
        let [one_challenge, id_challenge, base_challenge, slope_challenge] =
            relation.read_challenges;
        let read_x = one_challenge
            + id_challenge * id_x
            + base_challenge * base_x
            + slope_challenge * slope_x;
        ra_x * read_x
    })
}

fn prove_silu_hamming_weight<Tr>(
    tensor_point: Vec<Fr>,
    selected: &[usize],
    shape: MatrixShape,
    entries: usize,
    transcript: &mut Tr,
) -> Option<SiluHammingWeightProverOutput>
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
    let proof = SiluHammingWeightOutput {
        rounds: SumCheckRounds {
            round_polys,
            final_claim: claim_i,
        },
        ra: ra[0],
    };
    Some(SiluHammingWeightProverOutput {
        ra: EvalClaim::new(
            proof.ra,
            [lookup_point.as_slice(), tensor_point.as_slice()].concat(),
        ),
        proof,
    })
}

fn linear_sum_round_poly(claim: Fr, values: &[Fr]) -> Option<RoundPolynomial<2>> {
    (values.len() % 2 == 0).then_some(())?;
    let eval_at_zero = values.chunks_exact(2).map(|chunk| chunk[0]).sum::<Fr>();
    let eval_at_one = claim - eval_at_zero;
    Some(RoundPolynomial {
        coeffs: [eval_at_zero, eval_at_one - eval_at_zero],
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

fn line_at(v0: Fr, v1: Fr, point: Fr) -> Fr {
    v0 + point * (v1 - v0)
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

fn collect_matrix_table(values: Vec<i32>, shape: MatrixShape) -> Option<Vec<Fr>> {
    (values.len() == shape.len()).then_some(())?;
    Some(values.into_iter().map(Fr::from_i32).collect())
}

fn collect_mle_table(values: Vec<i32>, shape: MatrixShape) -> Option<Vec<Fr>> {
    (values.len() == shape.len()).then_some(())?;
    Some(values.into_iter().map(Fr::from_i32).collect())
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
    input: &SiluProverInput,
) -> Option<(
    Vec<i32>,
    Vec<i32>,
    Vec<i32>,
    Vec<u8>,
    Vec<i32>,
    Vec<i32>,
    Vec<i32>,
)> {
    validate_silu_range(input.advice.min_n, input.advice.max_n)?;
    let entries = usize::try_from(input.advice.max_n - input.advice.min_n + 1).ok()?;
    let tables = build_silu_tables(&input.advice)?;
    let lut_len = tables.id.len();
    let selected = selected_lookup_rows(&input.witness.ra, input.params.shape.len(), entries)?;
    let index = selected
        .iter()
        .map(|row| i32::try_from(*row).ok())
        .collect::<Option<Vec<_>>>()?;
    let base = selected
        .iter()
        .map(|row| tables.base.get(*row).copied())
        .collect::<Option<Vec<_>>>()?;
    let slope = selected
        .iter()
        .map(|row| tables.slope.get(*row).copied())
        .collect::<Option<Vec<_>>>()?;
    let ra = expand_ra(
        &input.witness.ra,
        input.params.shape.len(),
        entries,
        lut_len,
    )?;
    Some((index, base, slope, ra, tables.id, tables.base, tables.slope))
}

fn validate_input(claim: &EvalClaim, input: &SiluProverInput) -> Option<()> {
    validate_silu_range(input.advice.min_n, input.advice.max_n)?;
    (claim.point.len() == input.params.shape.point_len()).then_some(())?;
    (input.witness.input.len() == input.params.shape.len()).then_some(())?;
    (input.witness.output.len() == input.params.shape.len()).then_some(())?;
    input
        .witness
        .input_remainder_bits
        .iter()
        .chain(input.witness.output_remainder_bits.iter())
        .all(|bits| bits.len() == input.params.shape.len())
        .then_some(())?;
    let entries = usize::try_from(input.advice.max_n - input.advice.min_n + 1).ok()?;
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
