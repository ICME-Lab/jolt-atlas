use common::VirtualPoly;
use std::time::Instant;

use joltworks::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            BIG_ENDIAN, LITTLE_ENDIAN, OpeningAccumulator, OpeningId, OpeningPoint,
            ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck::{Sumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

use crate::{
    claim::{Claim, Shape, TensorId},
    error::{ProverError, Result},
    ops::{
        floor::{FloorParams, FloorProof, FloorWitness, prove_floor, verify_floor},
        round::{
            ROUND_FRAC_BITS, RoundParams, RoundProof, RoundWitness, prove_round, verify_round,
        },
    },
};

const FIXED_FRAC_BITS: usize = ROUND_FRAC_BITS;
const FIXED_SCALE: i64 = 1_i64 << FIXED_FRAC_BITS;
const MAX_SOFTMAX_LUT_ENTRIES: usize = 1 << 12;

// Attention softmax proof, specialized to the Qwen attention shape where the
// softmax axis is the last axis.
//
// Reverse flow:
//   output -> round(floor)
//   floor  -> floor(acc / 2^8)
//   acc    -> exp * inv_sum(sum_exp), where inv_sum is Q0.16
//   exp    -> round(EXP_LUT_Q8[n] * (256 + f) / 256)
//   n, f   -> floor((input - max_value) / 256), frac(input - max_value)
//
// The `floor -> round` split is intentional.  `acc` is Q0.24 and the output is
// Q0.8, but the proof system currently has an 8-bit round protocol.  Because
// softmax probabilities are nonnegative, `round(floor(acc / 2^8) / 2^8)` has
// the same result as a single 16-bit round while keeping both proof steps on
// 8-bit remainders.
//
// The LUT uses min/max diff advice exactly like the SiLU lookup design:
//
//   diff_t = valid_t * (input_t - max_value(row))
//   n_t = floor(diff_t / 2^8)
//   shifted_t = n_t - min_diff
//   id_table = [0, 1, ..., entries - 1]
//   <ra_t, id_table> = shifted_t
//
// The LUT is keyed by the integer part `n_t`, not by every QX.8 `diff_t`
// value.  The runtime `exp_lut_q8(n_t)` clips `n_t` internally to the small
// physical EXP_LUT_Q8 range [-16, 0], while the prover lookup domain covers raw
// unclipped n-values seen in the trace.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SoftmaxParams {
    pub round: RoundParams,
    pub floor: FloorParams,
    pub exp_round: RoundParams,
    pub shape: Shape,
    pub input_tensor: TensorId,
    pub output_tensor: TensorId,
    pub input_frac_bit_tensors: [TensorId; ROUND_FRAC_BITS],
    pub ra_tensor: TensorId,
    pub softmax_axis: usize,
    pub causal: bool,
}

impl SoftmaxParams {
    pub fn new(
        shape: impl Into<Vec<usize>>,
        acc_tensor: impl Into<String>,
        input_tensor: impl Into<String>,
        output_tensor: impl Into<String>,
        frac_bit_tensors: [String; ROUND_FRAC_BITS],
        input_frac_bit_tensors: [String; ROUND_FRAC_BITS],
        ra_tensor: impl Into<String>,
        softmax_axis: usize,
        causal: bool,
    ) -> Self {
        let shape = Shape::new(shape);
        let acc_tensor = acc_tensor.into();
        let output_tensor = output_tensor.into();
        let floor_tensor = format!("{output_tensor}_floor");
        let exp_tensor = format!("{output_tensor}_exp");
        let exp_acc_tensor = format!("{output_tensor}_exp_acc");
        Self {
            round: RoundParams::with_frac_bit_tensors(
                shape.dims().to_vec(),
                floor_tensor.clone(),
                output_tensor.clone(),
                frac_bit_tensors,
            ),
            floor: FloorParams::with_frac_bit_tensors(
                shape.dims().to_vec(),
                acc_tensor,
                floor_tensor,
                std::array::from_fn(|idx| format!("{output_tensor}_floor_frac_bit_{idx}")),
            ),
            exp_round: RoundParams::with_frac_bit_tensors(
                shape.dims().to_vec(),
                exp_acc_tensor,
                exp_tensor,
                std::array::from_fn(|idx| format!("{output_tensor}_exp_frac_bit_{idx}")),
            ),
            shape,
            input_tensor: TensorId::new(input_tensor),
            output_tensor: TensorId::new(output_tensor),
            input_frac_bit_tensors: input_frac_bit_tensors.map(TensorId::new),
            ra_tensor: TensorId::new(ra_tensor),
            softmax_axis,
            causal,
        }
    }

    fn rows(&self) -> usize {
        self.shape.dims()[..self.shape.dims().len() - 1]
            .iter()
            .product()
    }

    fn cols(&self) -> usize {
        *self.shape.dims().last().expect("softmax shape has rank")
    }

    fn row_shape(&self) -> Shape {
        Shape::new(self.shape.dims()[..self.shape.dims().len() - 1].to_vec())
    }

    fn ra_shape(&self, entries: usize) -> Shape {
        let mut dims = self.shape.dims().to_vec();
        dims.push(entries);
        Shape::new(dims)
    }
}

#[derive(Debug, Clone, Default)]
pub struct SoftmaxWitness {
    pub input: Vec<i32>,
    pub max_index: Vec<usize>,
    pub max: Vec<i32>,
    pub min_diff: i64,
    pub max_diff: i64,
    pub input_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    pub ra: Vec<u8>,
    pub exp_acc: Vec<i64>,
    pub exp: Vec<i32>,
    pub exp_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    pub sum: Vec<i32>,
    pub acc: Vec<i64>,
    pub floor: Vec<i32>,
    pub floor_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    pub output: Vec<i32>,
    pub frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
}

#[derive(Debug, Clone)]
pub struct SoftmaxProof<F: JoltField, T: Transcript> {
    pub round: RoundProof<F, T>,
    pub floor: FloorProof<F, T>,
    pub acc: SumcheckInstanceProof<F, T>,
    pub exp_round: RoundProof<F, T>,
    pub lookup: SumcheckInstanceProof<F, T>,
    pub exp_opening: F,
    pub input_opening: F,
    pub input_frac_bit_openings: [F; ROUND_FRAC_BITS],
    pub ra_opening: F,
    pub max_index: Vec<usize>,
    pub max: Vec<i32>,
    pub min_diff: i64,
    pub max_diff: i64,
    pub sum: Vec<i32>,
}

pub fn prove_softmax_round<F, T>(
    output_claims: Vec<Claim<F>>,
    witness: &SoftmaxWitness,
    params: &SoftmaxParams,
    transcript: &mut T,
) -> Result<(
    SoftmaxProof<F, T>,
    Claim<F>,
    [Claim<F>; ROUND_FRAC_BITS],
    [Claim<F>; ROUND_FRAC_BITS],
    [Claim<F>; ROUND_FRAC_BITS],
    Claim<F>,
)>
where
    F: JoltField,
    T: Transcript,
{
    let total_start = Instant::now();
    let mut step_start = Instant::now();
    validate_inputs(witness, params)?;
    eprintln!(
        "timing: prove_softmax.validate {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let round_witness = RoundWitness {
        input: witness
            .floor
            .iter()
            .map(|&value| i64::from(value))
            .collect(),
        output: witness.output.clone(),
        frac_bits: witness.frac_bits.clone(),
    };
    let (round_proof, floor_claim, frac_bits) =
        prove_round(output_claims, &round_witness, &params.round, transcript)?;
    eprintln!(
        "timing: prove_softmax.output_round {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let floor_witness = FloorWitness {
        input: witness.acc.clone(),
        output: witness.floor.clone(),
        frac_bits: witness.floor_frac_bits.clone(),
    };
    let (floor_proof, acc_claim, floor_frac_bits) =
        prove_floor(vec![floor_claim], &floor_witness, &params.floor, transcript)?;
    eprintln!(
        "timing: prove_softmax.floor {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    append_row_advice::<F, T>(
        &witness.max_index,
        &witness.max,
        witness.min_diff,
        witness.max_diff,
        &witness.sum,
        transcript,
    );

    let inv_sum = inv_sum_from_sum::<F>(&witness.sum);
    eprintln!(
        "timing: prove_softmax.advice_inv_sum {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let acc_eq = EqPolynomial::<F>::evals(&acc_claim.point);
    let inv = expand_rows_to_tensor(&inv_sum, params);
    let exp_poly = padded_i32_tensor(&witness.exp, &params.shape);
    let inv_poly = inv.clone();
    let valid_poly = valid_tensor::<F>(params);
    eprintln!(
        "timing: prove_softmax.acc_polys {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let mut acc_prover = AccSumcheckProver::new(
        BasicSumcheckParams::new(
            params.shape.padded_power_of_two().point_len(),
            acc_claim.value,
        ),
        acc_eq,
        exp_poly,
        inv_poly,
        valid_poly,
    );
    eprintln!(
        "timing: prove_softmax.acc_prover_init {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let mut acc_accumulator = ProverOpeningAccumulator::new();
    let (acc_proof, acc_challenges) =
        Sumcheck::prove(&mut acc_prover, &mut acc_accumulator, transcript);
    eprintln!(
        "timing: prove_softmax.acc_sumcheck {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let exp_opening = prover_opening(
        &acc_accumulator,
        OpeningId::new(VirtualPoly::QwenSoftmaxExp, softmax_acc_sumcheck_id()),
    )?;
    let exp_point = normalize_sumcheck_point::<F>(&acc_challenges.into_opening());
    let exp_claim = Claim {
        tensor: TensorId::new(format!("{}_exp", params.output_tensor.0)),
        logical_shape: params.shape.clone(),
        domain_shape: params.shape.padded_power_of_two(),
        point: exp_point,
        value: exp_opening,
    };
    eprintln!(
        "timing: prove_softmax.acc_opening_claim {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let exp_round_witness = RoundWitness {
        input: witness.exp_acc.clone(),
        output: witness.exp.clone(),
        frac_bits: witness.exp_frac_bits.clone(),
    };
    let (exp_round_proof, exp_acc_claim, _exp_frac_bits) = prove_round(
        vec![exp_claim],
        &exp_round_witness,
        &params.exp_round,
        transcript,
    )?;
    eprintln!(
        "timing: prove_softmax.exp_round {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let (lookup_proof, input_claim, input_frac_bits, ra_claim) =
        prove_lookup(exp_acc_claim, witness, params, transcript)?;
    eprintln!(
        "timing: prove_softmax.lookup_total {:.3}s",
        step_start.elapsed().as_secs_f64()
    );
    eprintln!(
        "timing: prove_softmax.total {:.3}s",
        total_start.elapsed().as_secs_f64()
    );

    Ok((
        SoftmaxProof {
            round: round_proof,
            floor: floor_proof,
            acc: acc_proof,
            exp_round: exp_round_proof,
            lookup: lookup_proof,
            exp_opening,
            input_opening: input_claim.value,
            input_frac_bit_openings: input_frac_bits.each_ref().map(|claim| claim.value),
            ra_opening: ra_claim.value,
            max_index: witness.max_index.clone(),
            max: witness.max.clone(),
            min_diff: witness.min_diff,
            max_diff: witness.max_diff,
            sum: witness.sum.clone(),
        },
        input_claim,
        frac_bits,
        floor_frac_bits,
        input_frac_bits,
        ra_claim,
    ))
}

pub fn verify_softmax_round<F, T>(
    output_claims: Vec<Claim<F>>,
    proof: &SoftmaxProof<F, T>,
    params: &SoftmaxParams,
    transcript: &mut T,
) -> std::result::Result<
    (
        Claim<F>,
        [Claim<F>; ROUND_FRAC_BITS],
        [Claim<F>; ROUND_FRAC_BITS],
        [Claim<F>; ROUND_FRAC_BITS],
        Claim<F>,
    ),
    ProofVerifyError,
>
where
    F: JoltField,
    T: Transcript,
{
    verify_advice(params, proof)?;
    let (floor_claim, frac_bits) =
        verify_round(output_claims, &proof.round, &params.round, transcript)?;
    let (acc_claim, floor_frac_bits) =
        verify_floor(vec![floor_claim], &proof.floor, &params.floor, transcript)?;

    append_row_advice::<F, T>(
        &proof.max_index,
        &proof.max,
        proof.min_diff,
        proof.max_diff,
        &proof.sum,
        transcript,
    );

    let inv_sum = inv_sum_from_sum::<F>(&proof.sum);
    let inv = expand_rows_to_tensor(&inv_sum, params);
    let acc_verifier = AccSumcheckVerifier {
        params: BasicSumcheckParams::new(
            params.shape.padded_power_of_two().point_len(),
            acc_claim.value,
        ),
        acc_point: acc_claim.point.clone(),
        inv,
        valid: valid_tensor(params),
        shape: params.shape.clone(),
    };
    let mut acc_accumulator = VerifierOpeningAccumulator::new();
    acc_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenSoftmaxExp, softmax_acc_sumcheck_id()),
        (OpeningPoint::<BIG_ENDIAN, F>::default(), proof.exp_opening),
    );
    let acc_challenges =
        Sumcheck::verify(&proof.acc, &acc_verifier, &mut acc_accumulator, transcript)?;
    let exp_claim = Claim {
        tensor: TensorId::new(format!("{}_exp", params.output_tensor.0)),
        logical_shape: params.shape.clone(),
        domain_shape: params.shape.padded_power_of_two(),
        point: normalize_sumcheck_point::<F>(&acc_challenges.into_opening()),
        value: proof.exp_opening,
    };

    let (exp_acc_claim, _exp_frac_bits) = verify_round(
        vec![exp_claim],
        &proof.exp_round,
        &params.exp_round,
        transcript,
    )?;

    let (input_claim, input_frac_bits, ra_claim) =
        verify_lookup(exp_acc_claim, proof, params, transcript)?;

    Ok((
        input_claim,
        frac_bits,
        floor_frac_bits,
        input_frac_bits,
        ra_claim,
    ))
}

fn prove_lookup<F, T>(
    exp_claim: Claim<F>,
    witness: &SoftmaxWitness,
    params: &SoftmaxParams,
    transcript: &mut T,
) -> Result<(
    SumcheckInstanceProof<F, T>,
    Claim<F>,
    [Claim<F>; ROUND_FRAC_BITS],
    Claim<F>,
)>
where
    F: JoltField,
    T: Transcript,
{
    let total_start = Instant::now();
    let mut step_start = Instant::now();
    let entries = entries_from_min_max(witness.min_diff, witness.max_diff)?;
    eprintln!(
        "timing: prove_softmax.lookup.setup {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let eq_exp = lift_tensor_evals(&EqPolynomial::<F>::evals(&exp_claim.point), entries);
    let row_point = row_point_from_tensor_point(&exp_claim.point, params);
    let row_eq = lift_tensor_evals(&row_eq_lifted(&row_point, params), entries);
    eprintln!(
        "timing: prove_softmax.lookup.eq_polys {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let max_selector = lift_tensor_evals(
        &max_selector_tensor::<F>(&witness.max_index, params),
        entries,
    );
    let valid = lift_tensor_evals(&valid_tensor(params), entries);
    let max = lift_tensor_evals(
        &expand_rows_to_padded_tensor_i32(&witness.max, params),
        entries,
    );
    eprintln!(
        "timing: prove_softmax.lookup.row_valid_max_polys {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let input = lift_i32_tensor(&witness.input, &params.shape, entries);
    let bits = std::array::from_fn(|idx| {
        lift_u8_tensor(&witness.input_frac_bits[idx], &params.shape, entries)
    });
    let ra = padded_ra_tensor(&witness.ra, params, entries);
    eprintln!(
        "timing: prove_softmax.lookup.witness_polys {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let id = softmax_table_tensor(params, entries, &id_table::<F>(entries));
    let exp_table =
        softmax_table_tensor(params, entries, &exp_table::<F>(witness.min_diff, entries));
    eprintln!(
        "timing: prove_softmax.lookup.table_polys {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let max_eval = eval_i32_advice(&witness.max, &params.row_shape(), &row_point);
    let max_mix = transcript.challenge_scalar();
    let sum_mix = transcript.challenge_scalar();
    let onehot_mix = transcript.challenge_scalar();
    let diff_mix = transcript.challenge_scalar();
    let ra_booleanity_mix = transcript.challenge_scalar();
    let bit_booleanity_mix = transcript.challenge_scalar();
    let bit_weights = transcript.challenge_scalar_powers(ROUND_FRAC_BITS);
    let mask_eval = eval_logical_mask(&params.shape, &exp_claim.point);
    let input_claim = exp_claim.value + max_mix * max_eval + onehot_mix * mask_eval;
    eprintln!(
        "timing: prove_softmax.lookup.challenges_claim {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let sc_params = LookupSumcheckParams::new(
        params.ra_shape(entries).padded_power_of_two().point_len(),
        input_claim,
    );
    let mut prover = LookupSumcheckProver::new(
        sc_params,
        eq_exp,
        row_eq,
        max_selector,
        valid,
        max,
        input,
        bits,
        ra,
        id,
        exp_table,
        max_mix,
        sum_mix,
        onehot_mix,
        diff_mix,
        ra_booleanity_mix,
        bit_booleanity_mix,
        bit_weights,
        field_from_i64(witness.min_diff),
    );
    eprintln!(
        "timing: prove_softmax.lookup.prover_init {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let mut accumulator = ProverOpeningAccumulator::new();
    let (proof, challenges) = Sumcheck::prove(&mut prover, &mut accumulator, transcript);
    eprintln!(
        "timing: prove_softmax.lookup.sumcheck {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let input_opening = prover_opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenSoftmaxInput, softmax_lookup_sumcheck_id()),
    )?;
    let bit_openings: [F; ROUND_FRAC_BITS] = std::array::from_fn(|idx| {
        prover_opening(
            &accumulator,
            OpeningId::new(
                VirtualPoly::QwenSoftmaxInputFracBit(idx),
                softmax_lookup_sumcheck_id(),
            ),
        )
        .expect("softmax input frac-bit opening must be produced")
    });
    let ra_opening = prover_opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenSoftmaxRa, softmax_lookup_sumcheck_id()),
    )?;
    let full_point = normalize_sumcheck_point::<F>(&challenges.into_opening());
    let tensor_point = tensor_point_from_full(&full_point, params);
    eprintln!(
        "timing: prove_softmax.lookup.openings_claims {:.3}s",
        step_start.elapsed().as_secs_f64()
    );
    eprintln!(
        "timing: prove_softmax.lookup.total {:.3}s",
        total_start.elapsed().as_secs_f64()
    );

    Ok((
        proof,
        Claim {
            tensor: params.input_tensor.clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: tensor_point.clone(),
            value: input_opening,
        },
        std::array::from_fn(|idx| Claim {
            tensor: params.input_frac_bit_tensors[idx].clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: tensor_point.clone(),
            value: bit_openings[idx],
        }),
        Claim {
            tensor: params.ra_tensor.clone(),
            logical_shape: params.ra_shape(entries),
            domain_shape: params.ra_shape(entries).padded_power_of_two(),
            point: full_point,
            value: ra_opening,
        },
    ))
}

fn verify_lookup<F, T>(
    exp_claim: Claim<F>,
    proof: &SoftmaxProof<F, T>,
    params: &SoftmaxParams,
    transcript: &mut T,
) -> std::result::Result<(Claim<F>, [Claim<F>; ROUND_FRAC_BITS], Claim<F>), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    let entries = entries_from_min_max(proof.min_diff, proof.max_diff)
        .map_err(|_| ProofVerifyError::InvalidInputLength(MAX_SOFTMAX_LUT_ENTRIES, 0))?;
    let row_point = row_point_from_tensor_point(&exp_claim.point, params);
    let max_eval = eval_i32_advice(&proof.max, &params.row_shape(), &row_point);
    let max_mix = transcript.challenge_scalar();
    let sum_mix = transcript.challenge_scalar();
    let onehot_mix = transcript.challenge_scalar();
    let diff_mix = transcript.challenge_scalar();
    let ra_booleanity_mix = transcript.challenge_scalar();
    let bit_booleanity_mix = transcript.challenge_scalar();
    let bit_weights = transcript.challenge_scalar_powers(ROUND_FRAC_BITS);
    let mask_eval = eval_logical_mask(&params.shape, &exp_claim.point);
    let input_claim = exp_claim.value + max_mix * max_eval + onehot_mix * mask_eval;

    let verifier = LookupSumcheckVerifier {
        params: LookupSumcheckParams::new(
            params.ra_shape(entries).padded_power_of_two().point_len(),
            input_claim,
        ),
        exp_point: exp_claim.point.clone(),
        row_point,
        max_index: proof.max_index.clone(),
        max: proof.max.clone(),
        entries,
        min_diff: proof.min_diff,
        causal: params.causal,
        shape: params.shape.clone(),
        row_shape: params.row_shape(),
        max_mix,
        sum_mix,
        onehot_mix,
        diff_mix,
        ra_booleanity_mix,
        bit_booleanity_mix,
        bit_weights,
    };
    let mut accumulator = VerifierOpeningAccumulator::new();
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenSoftmaxInput, softmax_lookup_sumcheck_id()),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.input_opening,
        ),
    );
    for idx in 0..ROUND_FRAC_BITS {
        accumulator.openings.insert(
            OpeningId::new(
                VirtualPoly::QwenSoftmaxInputFracBit(idx),
                softmax_lookup_sumcheck_id(),
            ),
            (
                OpeningPoint::<BIG_ENDIAN, F>::default(),
                proof.input_frac_bit_openings[idx],
            ),
        );
    }
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenSoftmaxRa, softmax_lookup_sumcheck_id()),
        (OpeningPoint::<BIG_ENDIAN, F>::default(), proof.ra_opening),
    );
    let challenges = Sumcheck::verify(&proof.lookup, &verifier, &mut accumulator, transcript)?;
    let full_point = normalize_sumcheck_point::<F>(&challenges.into_opening());
    let tensor_point = tensor_point_from_full(&full_point, params);

    Ok((
        Claim {
            tensor: params.input_tensor.clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: tensor_point.clone(),
            value: proof.input_opening,
        },
        std::array::from_fn(|idx| Claim {
            tensor: params.input_frac_bit_tensors[idx].clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: tensor_point.clone(),
            value: proof.input_frac_bit_openings[idx],
        }),
        Claim {
            tensor: params.ra_tensor.clone(),
            logical_shape: params.ra_shape(entries),
            domain_shape: params.ra_shape(entries).padded_power_of_two(),
            point: full_point,
            value: proof.ra_opening,
        },
    ))
}

struct BasicSumcheckParams<F: JoltField> {
    num_rounds: usize,
    input_claim: F,
}

impl<F: JoltField> BasicSumcheckParams<F> {
    fn new(num_rounds: usize, input_claim: F) -> Self {
        Self {
            num_rounds,
            input_claim,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for BasicSumcheckParams<F> {
    fn degree(&self) -> usize {
        4
    }
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }
    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.input_claim
    }
    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

type LookupSumcheckParams<F> = BasicSumcheckParams<F>;

struct AccSumcheckProver<F: JoltField> {
    eq: MultilinearPolynomial<F>,
    exp: MultilinearPolynomial<F>,
    inv: MultilinearPolynomial<F>,
    valid: MultilinearPolynomial<F>,
    params: BasicSumcheckParams<F>,
}

impl<F: JoltField> AccSumcheckProver<F> {
    fn new(
        params: BasicSumcheckParams<F>,
        eq: Vec<F>,
        exp: Vec<F>,
        inv: Vec<F>,
        valid: Vec<F>,
    ) -> Self {
        Self {
            eq: MultilinearPolynomial::from(eq),
            exp: MultilinearPolynomial::from(exp),
            inv: MultilinearPolynomial::from(inv),
            valid: MultilinearPolynomial::from(valid),
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for AccSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let mut evals = [F::zero(); 4];
        for g in 0..self.exp.len() / 2 {
            let eq = [
                self.eq.get_bound_coeff(2 * g),
                self.eq.get_bound_coeff(2 * g + 1),
            ];
            let exp = [
                self.exp.get_bound_coeff(2 * g),
                self.exp.get_bound_coeff(2 * g + 1),
            ];
            let inv = [
                self.inv.get_bound_coeff(2 * g),
                self.inv.get_bound_coeff(2 * g + 1),
            ];
            let valid = [
                self.valid.get_bound_coeff(2 * g),
                self.valid.get_bound_coeff(2 * g + 1),
            ];
            for (idx, t) in [F::zero(), F::from_u64(2), F::from_u64(3), F::from_u64(4)]
                .into_iter()
                .enumerate()
            {
                evals[idx] += lerp(eq[0], eq[1], t)
                    * lerp(valid[0], valid[1], t)
                    * lerp(exp[0], exp[1], t)
                    * lerp(inv[0], inv[1], t);
            }
        }
        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.exp.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.inv.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.valid.bind_parallel(r_j, BindingOrder::LowToHigh);
    }
    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenSoftmaxExp, softmax_acc_sumcheck_id()),
            point,
            self.exp.final_claim(),
        );
    }
}

struct AccSumcheckVerifier<F: JoltField> {
    params: BasicSumcheckParams<F>,
    acc_point: Vec<F>,
    inv: Vec<F>,
    valid: Vec<F>,
    shape: Shape,
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for AccSumcheckVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }
    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let point = normalize_sumcheck_point::<F>(&sumcheck_challenges.into_opening());
        let exp = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenSoftmaxExp,
                softmax_acc_sumcheck_id(),
            ))
            .1;
        let eq = EqPolynomial::mle(&self.acc_point, &point);
        let inv = eval_field_tensor(&self.inv, &self.shape.padded_power_of_two(), &point);
        let valid = eval_field_tensor(&self.valid, &self.shape.padded_power_of_two(), &point);
        eq * valid * exp * inv
    }
    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenSoftmaxExp, softmax_acc_sumcheck_id()),
            point,
        );
    }
}

struct LookupSumcheckProver<F: JoltField> {
    eq_exp: MultilinearPolynomial<F>,
    row_eq: MultilinearPolynomial<F>,
    max_selector: MultilinearPolynomial<F>,
    valid: MultilinearPolynomial<F>,
    max: MultilinearPolynomial<F>,
    input: MultilinearPolynomial<F>,
    bits: [MultilinearPolynomial<F>; ROUND_FRAC_BITS],
    ra: MultilinearPolynomial<F>,
    id: MultilinearPolynomial<F>,
    exp_table: MultilinearPolynomial<F>,
    max_mix: F,
    sum_mix: F,
    onehot_mix: F,
    diff_mix: F,
    ra_booleanity_mix: F,
    bit_booleanity_mix: F,
    bit_weights: Vec<F>,
    min_diff: F,
    params: LookupSumcheckParams<F>,
}

#[allow(clippy::too_many_arguments)]
impl<F: JoltField> LookupSumcheckProver<F> {
    fn new(
        params: LookupSumcheckParams<F>,
        eq_exp: Vec<F>,
        row_eq: Vec<F>,
        max_selector: Vec<F>,
        valid: Vec<F>,
        max: Vec<F>,
        input: Vec<F>,
        bits: [Vec<F>; ROUND_FRAC_BITS],
        ra: Vec<F>,
        id: Vec<F>,
        exp_table: Vec<F>,
        max_mix: F,
        sum_mix: F,
        onehot_mix: F,
        diff_mix: F,
        ra_booleanity_mix: F,
        bit_booleanity_mix: F,
        bit_weights: Vec<F>,
        min_diff: F,
    ) -> Self {
        Self {
            eq_exp: MultilinearPolynomial::from(eq_exp),
            row_eq: MultilinearPolynomial::from(row_eq),
            max_selector: MultilinearPolynomial::from(max_selector),
            valid: MultilinearPolynomial::from(valid),
            max: MultilinearPolynomial::from(max),
            input: MultilinearPolynomial::from(input),
            bits: bits.map(MultilinearPolynomial::from),
            ra: MultilinearPolynomial::from(ra),
            id: MultilinearPolynomial::from(id),
            exp_table: MultilinearPolynomial::from(exp_table),
            max_mix,
            sum_mix,
            onehot_mix,
            diff_mix,
            ra_booleanity_mix,
            bit_booleanity_mix,
            bit_weights,
            min_diff,
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for LookupSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let mut evals = [F::zero(); 4];
        for g in 0..self.ra.len() / 2 {
            let v = LookupPairValues::from_prover(self, g);
            for (idx, t) in [F::zero(), F::from_u64(2), F::from_u64(3), F::from_u64(4)]
                .into_iter()
                .enumerate()
            {
                evals[idx] += v.eval(t, self);
            }
        }
        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_exp.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.row_eq.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.max_selector
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.valid.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.max.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.input.bind_parallel(r_j, BindingOrder::LowToHigh);
        for bit in &mut self.bits {
            bit.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        self.ra.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.id.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.exp_table.bind_parallel(r_j, BindingOrder::LowToHigh);
    }
    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenSoftmaxInput, softmax_lookup_sumcheck_id()),
            point.clone(),
            self.input.final_claim(),
        );
        for idx in 0..ROUND_FRAC_BITS {
            accumulator.append_virtual(
                transcript,
                OpeningId::new(
                    VirtualPoly::QwenSoftmaxInputFracBit(idx),
                    softmax_lookup_sumcheck_id(),
                ),
                point.clone(),
                self.bits[idx].final_claim(),
            );
        }
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenSoftmaxRa, softmax_lookup_sumcheck_id()),
            point,
            self.ra.final_claim(),
        );
    }
}

struct LookupSumcheckVerifier<F: JoltField> {
    params: LookupSumcheckParams<F>,
    exp_point: Vec<F>,
    row_point: Vec<F>,
    max_index: Vec<usize>,
    max: Vec<i32>,
    entries: usize,
    min_diff: i64,
    causal: bool,
    shape: Shape,
    row_shape: Shape,
    max_mix: F,
    sum_mix: F,
    onehot_mix: F,
    diff_mix: F,
    ra_booleanity_mix: F,
    bit_booleanity_mix: F,
    bit_weights: Vec<F>,
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for LookupSumcheckVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }
    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let full_point = normalize_sumcheck_point::<F>(&sumcheck_challenges.into_opening());
        let tensor_point = tensor_point_from_full_shape(&full_point, &self.shape);
        let lut_point = &full_point[tensor_point.len()..];
        let input = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenSoftmaxInput,
                softmax_lookup_sumcheck_id(),
            ))
            .1;
        let bits = std::array::from_fn(|idx| {
            accumulator
                .get_virtual_polynomial_opening(OpeningId::new(
                    VirtualPoly::QwenSoftmaxInputFracBit(idx),
                    softmax_lookup_sumcheck_id(),
                ))
                .1
        });
        let ra = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenSoftmaxRa,
                softmax_lookup_sumcheck_id(),
            ))
            .1;
        let eq_exp = EqPolynomial::mle(&self.exp_point, &tensor_point);
        let row_eq = EqPolynomial::mle(
            &self.row_point,
            row_point_from_tensor_point_shape(&tensor_point, &self.shape),
        );
        let max_selector = eval_max_selector(&self.max_index, &self.shape, &tensor_point);
        let valid = eval_valid(self.causal, &self.shape, &tensor_point);
        let id = eval_table_at_point(&padded_table(&id_table::<F>(self.entries)), lut_point);
        let exp_table = eval_table_at_point(
            &padded_table(&exp_table::<F>(self.min_diff, self.entries)),
            lut_point,
        );
        let max = eval_i32_advice(
            &self.max,
            &self.row_shape,
            row_point_from_tensor_point_shape(&tensor_point, &self.shape),
        );
        eval_lookup_relation(
            eq_exp,
            row_eq,
            max_selector,
            valid,
            input,
            bits,
            ra,
            id,
            exp_table,
            max,
            self.max_mix,
            self.sum_mix,
            self.onehot_mix,
            self.diff_mix,
            self.ra_booleanity_mix,
            self.bit_booleanity_mix,
            &self.bit_weights,
            field_from_i64(self.min_diff),
        )
    }
    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenSoftmaxInput, softmax_lookup_sumcheck_id()),
            point.clone(),
        );
        for idx in 0..ROUND_FRAC_BITS {
            accumulator.append_virtual(
                transcript,
                OpeningId::new(
                    VirtualPoly::QwenSoftmaxInputFracBit(idx),
                    softmax_lookup_sumcheck_id(),
                ),
                point.clone(),
            );
        }
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenSoftmaxRa, softmax_lookup_sumcheck_id()),
            point,
        );
    }
}

struct LookupPairValues<F: JoltField> {
    eq_exp: [F; 2],
    row_eq: [F; 2],
    max_selector: [F; 2],
    valid: [F; 2],
    max: [F; 2],
    input: [F; 2],
    bits: [[F; 2]; ROUND_FRAC_BITS],
    ra: [F; 2],
    id: [F; 2],
    exp_table: [F; 2],
}

impl<F: JoltField> LookupPairValues<F> {
    fn from_prover(p: &LookupSumcheckProver<F>, g: usize) -> Self {
        Self {
            eq_exp: [
                p.eq_exp.get_bound_coeff(2 * g),
                p.eq_exp.get_bound_coeff(2 * g + 1),
            ],
            row_eq: [
                p.row_eq.get_bound_coeff(2 * g),
                p.row_eq.get_bound_coeff(2 * g + 1),
            ],
            max_selector: [
                p.max_selector.get_bound_coeff(2 * g),
                p.max_selector.get_bound_coeff(2 * g + 1),
            ],
            valid: [
                p.valid.get_bound_coeff(2 * g),
                p.valid.get_bound_coeff(2 * g + 1),
            ],
            max: [
                p.max.get_bound_coeff(2 * g),
                p.max.get_bound_coeff(2 * g + 1),
            ],
            input: [
                p.input.get_bound_coeff(2 * g),
                p.input.get_bound_coeff(2 * g + 1),
            ],
            bits: std::array::from_fn(|idx| {
                [
                    p.bits[idx].get_bound_coeff(2 * g),
                    p.bits[idx].get_bound_coeff(2 * g + 1),
                ]
            }),
            ra: [p.ra.get_bound_coeff(2 * g), p.ra.get_bound_coeff(2 * g + 1)],
            id: [p.id.get_bound_coeff(2 * g), p.id.get_bound_coeff(2 * g + 1)],
            exp_table: [
                p.exp_table.get_bound_coeff(2 * g),
                p.exp_table.get_bound_coeff(2 * g + 1),
            ],
        }
    }

    fn eval(&self, t: F, p: &LookupSumcheckProver<F>) -> F {
        eval_lookup_relation(
            lerp(self.eq_exp[0], self.eq_exp[1], t),
            lerp(self.row_eq[0], self.row_eq[1], t),
            lerp(self.max_selector[0], self.max_selector[1], t),
            lerp(self.valid[0], self.valid[1], t),
            lerp(self.input[0], self.input[1], t),
            std::array::from_fn(|idx| lerp(self.bits[idx][0], self.bits[idx][1], t)),
            lerp(self.ra[0], self.ra[1], t),
            lerp(self.id[0], self.id[1], t),
            lerp(self.exp_table[0], self.exp_table[1], t),
            lerp(self.max[0], self.max[1], t),
            p.max_mix,
            p.sum_mix,
            p.onehot_mix,
            p.diff_mix,
            p.ra_booleanity_mix,
            p.bit_booleanity_mix,
            &p.bit_weights,
            p.min_diff,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn eval_lookup_relation<F: JoltField>(
    eq_exp: F,
    row_eq: F,
    max_selector: F,
    valid: F,
    input: F,
    bits: [F; ROUND_FRAC_BITS],
    ra: F,
    id: F,
    exp_table: F,
    max: F,
    max_mix: F,
    sum_mix: F,
    onehot_mix: F,
    diff_mix: F,
    ra_booleanity_mix: F,
    bit_booleanity_mix: F,
    bit_weights: &[F],
    min_diff: F,
) -> F {
    let n = min_diff + id;
    let frac = bits.iter().enumerate().fold(F::zero(), |acc, (idx, bit)| {
        acc + F::from_u64(1_u64 << idx) * *bit
    });
    let corr = F::from_u64(FIXED_SCALE as u64) + frac;
    let diff_expr = valid * (input - max) - (n * F::from_u64(FIXED_SCALE as u64) + frac);
    let bit_booleanity = bits
        .iter()
        .zip(bit_weights)
        .fold(F::zero(), |acc, (bit, weight)| {
            acc + *weight * *bit * (*bit - F::one())
        });
    eq_exp * ra * exp_table * corr
        + max_mix * row_eq * max_selector * ra * input
        + sum_mix * F::zero()
        + onehot_mix * eq_exp * ra
        + diff_mix * eq_exp * ra * diff_expr
        + ra_booleanity_mix * ra * (ra - F::one())
        + bit_booleanity_mix * bit_booleanity
}

fn validate_inputs(w: &SoftmaxWitness, params: &SoftmaxParams) -> Result<()> {
    if params.softmax_axis + 1 != params.shape.dims().len() {
        return Err(ProverError::InvalidAxis {
            axis: params.softmax_axis,
            rank: params.shape.dims().len(),
        });
    }
    let len = params.shape.numel();
    let rows = params.rows();
    let cols = params.cols();
    ensure_len("softmax input", &params.shape, len, w.input.len())?;
    ensure_len("softmax exp acc", &params.shape, len, w.exp_acc.len())?;
    ensure_len("softmax exp", &params.shape, len, w.exp.len())?;
    ensure_len("softmax acc", &params.shape, len, w.acc.len())?;
    ensure_len("softmax floor", &params.shape, len, w.floor.len())?;
    ensure_len("softmax output", &params.shape, len, w.output.len())?;
    ensure_len("softmax max", &params.row_shape(), rows, w.max.len())?;
    ensure_len(
        "softmax max_index",
        &params.row_shape(),
        rows,
        w.max_index.len(),
    )?;
    ensure_len("softmax sum", &params.row_shape(), rows, w.sum.len())?;
    let entries = entries_from_min_max(w.min_diff, w.max_diff)?;
    ensure_len(
        "softmax ra",
        &params.ra_shape(entries),
        len * entries,
        w.ra.len(),
    )?;
    for bit in 0..ROUND_FRAC_BITS {
        ensure_len(
            "softmax input frac bit",
            &params.shape,
            len,
            w.input_frac_bits[bit].len(),
        )?;
        ensure_len(
            "softmax exp frac bit",
            &params.shape,
            len,
            w.exp_frac_bits[bit].len(),
        )?;
        ensure_len(
            "softmax floor frac bit",
            &params.shape,
            len,
            w.floor_frac_bits[bit].len(),
        )?;
        ensure_len(
            "softmax output frac bit",
            &params.shape,
            len,
            w.frac_bits[bit].len(),
        )?;
    }
    for row in 0..rows {
        if w.max_index[row] >= cols {
            return Err(ProverError::InvalidAxis {
                axis: w.max_index[row],
                rank: cols,
            });
        }
        if !is_valid_position(params.causal, &params.shape, row, w.max_index[row]) {
            return Err(ProverError::InvalidAxis {
                axis: w.max_index[row],
                rank: cols,
            });
        }
        let max_idx = row * cols + w.max_index[row];
        if w.input[max_idx] != w.max[row] {
            return Err(ProverError::MatMulMismatch {
                row,
                col: w.max_index[row],
                expected: w.max[row],
                actual: w.input[max_idx],
            });
        }
        let mut sum = 0_i64;
        for col in 0..cols {
            let idx = row * cols + col;
            let valid = is_valid_position(params.causal, &params.shape, row, col);
            let diff = if valid {
                let raw_diff = i64::from(w.input[idx]) - i64::from(w.max[row]);
                let frac = frac_value(&w.input_frac_bits, idx);
                if frac != raw_diff.rem_euclid(FIXED_SCALE) {
                    return Err(ProverError::MatMulAccumulatorMismatch {
                        row,
                        col,
                        expected: raw_diff.rem_euclid(FIXED_SCALE),
                        actual: frac,
                    });
                }
                raw_diff
            } else {
                0
            };
            let n = floor_shift_q8(diff);
            let shifted = i64::from(n) - w.min_diff;
            if shifted < 0 || shifted > w.max_diff - w.min_diff {
                return Err(ProverError::InvalidSumcheckDomain(entries));
            }
            let shifted = shifted as usize;
            let ra_row = &w.ra[idx * entries..(idx + 1) * entries];
            if ra_row.iter().filter(|&&v| v == 1).count() != 1 || ra_row[shifted] != 1 {
                return Err(ProverError::InvalidClaimCount {
                    name: "softmax ra row",
                    expected: 1,
                    actual: ra_row.iter().filter(|&&v| v == 1).count(),
                });
            }
            let expected_exp = softmax_exp_coarse_q8(diff);
            let expected_exp_acc = softmax_exp_acc_q8(diff);
            if w.exp_acc[idx] != expected_exp_acc {
                return Err(ProverError::MatMulAccumulatorMismatch {
                    row,
                    col,
                    expected: expected_exp_acc,
                    actual: w.exp_acc[idx],
                });
            }
            if w.exp[idx] != expected_exp {
                return Err(ProverError::MatMulMismatch {
                    row,
                    col,
                    expected: expected_exp,
                    actual: w.exp[idx],
                });
            }
            if valid {
                sum += i64::from(w.exp[idx]);
            }
            let inv = inv_sum_q16(w.sum[row]);
            let expected_acc = if valid {
                i64::from(w.exp[idx]) * inv
            } else {
                0
            };
            if w.acc[idx] != expected_acc {
                return Err(ProverError::MatMulAccumulatorMismatch {
                    row,
                    col,
                    expected: expected_acc,
                    actual: w.acc[idx],
                });
            }
            let expected_floor = floor_shift_q8(expected_acc);
            if w.floor[idx] != expected_floor {
                return Err(ProverError::MatMulMismatch {
                    row,
                    col,
                    expected: expected_floor,
                    actual: w.floor[idx],
                });
            }
            let expected_output = round_shift_q8(i64::from(expected_floor));
            if w.output[idx] != expected_output {
                return Err(ProverError::MatMulMismatch {
                    row,
                    col,
                    expected: expected_output,
                    actual: w.output[idx],
                });
            }
        }
        if i64::from(w.sum[row]) != sum {
            return Err(ProverError::MatMulAccumulatorMismatch {
                row,
                col: 0,
                expected: sum,
                actual: i64::from(w.sum[row]),
            });
        }
    }
    Ok(())
}

fn verify_advice<F: JoltField, T: Transcript>(
    params: &SoftmaxParams,
    proof: &SoftmaxProof<F, T>,
) -> std::result::Result<(), ProofVerifyError> {
    let rows = params.rows();
    if proof.max_index.len() != rows {
        return Err(ProofVerifyError::InvalidInputLength(
            rows,
            proof.max_index.len(),
        ));
    }
    if proof.max.len() != rows {
        return Err(ProofVerifyError::InvalidInputLength(rows, proof.max.len()));
    }
    if proof.sum.len() != rows {
        return Err(ProofVerifyError::InvalidInputLength(rows, proof.sum.len()));
    }
    let cols = params.cols();
    for (row, &max_index) in proof.max_index.iter().enumerate() {
        if max_index >= cols || !is_valid_position(params.causal, &params.shape, row, max_index) {
            return Err(ProofVerifyError::InvalidInputLength(cols, max_index));
        }
    }
    entries_from_min_max(proof.min_diff, proof.max_diff)
        .map_err(|_| ProofVerifyError::InvalidInputLength(MAX_SOFTMAX_LUT_ENTRIES, 0))?;
    Ok(())
}

fn append_row_advice<F: JoltField, T: Transcript>(
    max_index: &[usize],
    max: &[i32],
    min_diff: i64,
    max_diff: i64,
    sum: &[i32],
    transcript: &mut T,
) {
    transcript.append_scalar(&field_from_i64::<F>(min_diff));
    transcript.append_scalar(&field_from_i64::<F>(max_diff));
    for &idx in max_index {
        transcript.append_scalar(&F::from_u64(idx as u64));
    }
    for &value in max {
        transcript.append_scalar(&field_from_i64::<F>(i64::from(value)));
    }
    for &value in sum {
        transcript.append_scalar(&field_from_i64::<F>(i64::from(value)));
    }
}

fn inv_sum_from_sum<F: JoltField>(sum: &[i32]) -> Vec<F> {
    sum.iter()
        .map(|&s| field_from_i64(inv_sum_q16(s)))
        .collect()
}

fn inv_sum_q16(sum: i32) -> i64 {
    ((1_i64 << (FIXED_FRAC_BITS + 16)) as f64 / f64::from(sum)).round() as i64
}

fn softmax_exp_coarse_q8(delta_q8: i64) -> i32 {
    round_shift_q8(softmax_exp_acc_q8(delta_q8))
}

fn softmax_exp_acc_q8(delta_q8: i64) -> i64 {
    let n = delta_q8.div_euclid(FIXED_SCALE);
    let f = delta_q8 - n * FIXED_SCALE;
    let exp_n = exp_lut_q8(n);
    let corr = (FIXED_SCALE + f).max(0);
    exp_n * corr
}

fn exp_lut_q8(n: i64) -> i64 {
    let n = n.clamp(-16, 0);
    (f64::exp(n as f64) * FIXED_SCALE as f64).round() as i64
}

fn exp_lut_q8_unclipped(n: i64) -> i64 {
    (f64::exp(n as f64) * FIXED_SCALE as f64).round() as i64
}

fn round_shift_q8(value: i64) -> i32 {
    let frac = value.rem_euclid(FIXED_SCALE);
    ((value + ((frac >> (ROUND_FRAC_BITS - 1)) * FIXED_SCALE) - frac) / FIXED_SCALE) as i32
}

fn floor_shift_q8(value: i64) -> i32 {
    value.div_euclid(FIXED_SCALE) as i32
}

fn entries_from_min_max(min: i64, max: i64) -> Result<usize> {
    if max < min {
        return Err(ProverError::InvalidSumcheckDomain(0));
    }
    let entries = (max - min + 1) as usize;
    if entries == 0 || entries > MAX_SOFTMAX_LUT_ENTRIES {
        return Err(ProverError::InvalidSumcheckDomain(entries));
    }
    Ok(entries)
}

fn id_table<F: JoltField>(entries: usize) -> Vec<F> {
    (0..entries).map(|i| F::from_u64(i as u64)).collect()
}

fn exp_table<F: JoltField>(min_diff: i64, entries: usize) -> Vec<F> {
    (0..entries)
        .map(|i| field_from_i64(exp_lut_q8_unclipped(min_diff + i as i64)))
        .collect()
}

fn max_selector_tensor<F: JoltField>(max_index: &[usize], params: &SoftmaxParams) -> Vec<F> {
    let mut values = vec![F::zero(); params.shape.numel()];
    let cols = params.cols();
    for (row, &col) in max_index.iter().enumerate() {
        values[row * cols + col] = F::one();
    }
    padded_field_tensor(&values, &params.shape)
}

fn eval_max_selector<F: JoltField>(max_index: &[usize], shape: &Shape, point: &[F]) -> F {
    let cols = *shape.dims().last().unwrap();
    let rows = shape.numel() / cols;
    let mut values = vec![F::zero(); rows * cols];
    for (row, &col) in max_index.iter().enumerate() {
        values[row * cols + col] = F::one();
    }
    eval_field_tensor(&values, shape, point)
}

fn valid_tensor<F: JoltField>(params: &SoftmaxParams) -> Vec<F> {
    valid_tensor_for_shape(params.causal, &params.shape)
}

fn valid_tensor_for_shape<F: JoltField>(causal: bool, shape: &Shape) -> Vec<F> {
    let cols = *shape.dims().last().unwrap();
    let rows = shape.numel() / cols;
    let mut values = vec![F::zero(); shape.numel()];
    for row in 0..rows {
        for col in 0..cols {
            if is_valid_position(causal, shape, row, col) {
                values[row * cols + col] = F::one();
            }
        }
    }
    padded_field_tensor(&values, shape)
}

fn eval_valid<F: JoltField>(causal: bool, shape: &Shape, point: &[F]) -> F {
    eval_field_tensor(
        &valid_tensor_for_shape(causal, shape),
        &shape.padded_power_of_two(),
        point,
    )
}

fn is_valid_position(causal: bool, shape: &Shape, row: usize, col: usize) -> bool {
    if !causal {
        return true;
    }
    let dims = shape.dims();
    if dims.len() < 2 {
        return true;
    }
    let row_dims = &dims[..dims.len() - 1];
    let query_dim = *row_dims.last().unwrap();
    let query_pos = row % query_dim;
    col <= query_pos
}

fn row_point_from_tensor_point<F: Clone>(point: &[F], params: &SoftmaxParams) -> Vec<F> {
    row_point_from_tensor_point_shape(point, &params.shape).to_vec()
}

fn row_point_from_tensor_point_shape<'a, F>(point: &'a [F], shape: &Shape) -> &'a [F] {
    let col_vars = shape
        .dims()
        .last()
        .unwrap()
        .next_power_of_two()
        .trailing_zeros() as usize;
    &point[..point.len() - col_vars]
}

fn row_eq_lifted<F: JoltField>(row_point: &[F], params: &SoftmaxParams) -> Vec<F> {
    let row_eq = EqPolynomial::<F>::evals(row_point);
    let cols = params.cols().next_power_of_two();
    let mut out = Vec::with_capacity(row_eq.len() * cols);
    for value in row_eq {
        out.extend(std::iter::repeat_n(value, cols));
    }
    out
}

fn expand_rows_to_tensor<F: JoltField>(row_values: &[F], params: &SoftmaxParams) -> Vec<F> {
    let cols = params.cols();
    let mut values = vec![F::zero(); params.shape.numel()];
    for row in 0..params.rows() {
        for col in 0..cols {
            values[row * cols + col] = row_values[row];
        }
    }
    padded_field_tensor(&values, &params.shape)
}

fn expand_rows_to_padded_tensor_i32<F: JoltField>(
    row_values: &[i32],
    params: &SoftmaxParams,
) -> Vec<F> {
    let padded_dims = params.shape.padded_power_of_two().0;
    let len = padded_dims.iter().product();
    let padded_cols = *padded_dims.last().unwrap();
    let row_dims = &params.shape.dims()[..params.shape.dims().len() - 1];
    let padded_row_dims = &padded_dims[..padded_dims.len() - 1];
    let row_strides = row_major_strides(row_dims);
    let padded_row_strides = row_major_strides(padded_row_dims);
    let mut out = vec![F::zero(); len];
    for (row, &value) in row_values.iter().enumerate() {
        let mut padded_row = 0;
        for (dim, (&stride, &padded_stride)) in
            row_strides.iter().zip(&padded_row_strides).enumerate()
        {
            let coord = (row / stride) % row_dims[dim];
            padded_row += coord * padded_stride;
        }
        let base = padded_row * padded_cols;
        for col in 0..padded_cols {
            out[base + col] = field_from_i64(i64::from(value));
        }
    }
    out
}

fn lift_tensor_evals<F: JoltField>(values: &[F], entries: usize) -> Vec<F> {
    let padded_entries = entries.next_power_of_two();
    let mut out = Vec::with_capacity(values.len() * padded_entries);
    for &value in values {
        out.extend(std::iter::repeat_n(value, padded_entries));
    }
    out
}

fn lift_i32_tensor<F: JoltField>(values: &[i32], shape: &Shape, entries: usize) -> Vec<F> {
    lift_tensor_evals(&padded_i32_tensor(values, shape), entries)
}

fn lift_u8_tensor<F: JoltField>(values: &[u8], shape: &Shape, entries: usize) -> Vec<F> {
    lift_tensor_evals(&padded_u8_tensor(values, shape), entries)
}

fn padded_ra_tensor<F: JoltField>(values: &[u8], params: &SoftmaxParams, entries: usize) -> Vec<F> {
    let shape = params.ra_shape(entries);
    let padded_dims = shape.padded_power_of_two().0;
    let len = padded_dims.iter().product();
    let mut out = vec![F::zero(); len];
    let strides = row_major_strides(shape.dims());
    let padded_strides = row_major_strides(&padded_dims);
    for (flat, &value) in values.iter().enumerate() {
        let mut padded_flat = 0;
        for (dim, (&stride, &padded_stride)) in strides.iter().zip(&padded_strides).enumerate() {
            let coord = (flat / stride) % shape.dims()[dim];
            padded_flat += coord * padded_stride;
        }
        out[padded_flat] = F::from_u64(u64::from(value));
    }
    out
}

fn softmax_table_tensor<F: JoltField>(
    params: &SoftmaxParams,
    _entries: usize,
    table: &[F],
) -> Vec<F> {
    let tensor_len = params.shape.padded_power_of_two().numel();
    let table = padded_table(table);
    let mut out = Vec::with_capacity(tensor_len * table.len());
    for _ in 0..tensor_len {
        out.extend(table.iter().copied());
    }
    out
}

fn padded_table<F: JoltField>(table: &[F]) -> Vec<F> {
    let mut out = vec![F::zero(); table.len().next_power_of_two()];
    out[..table.len()].copy_from_slice(table);
    out
}

fn eval_table_at_point<F: JoltField>(table: &[F], point: &[F]) -> F {
    EqPolynomial::<F>::evals(point)
        .into_iter()
        .zip(table)
        .fold(F::zero(), |acc, (eq, value)| acc + eq * *value)
}

fn padded_i32_tensor<F: JoltField>(values: &[i32], shape: &Shape) -> Vec<F> {
    let padded_dims = shape.padded_power_of_two().0;
    let len = padded_dims.iter().product();
    let mut out = vec![F::zero(); len];
    let strides = row_major_strides(shape.dims());
    let padded_strides = row_major_strides(&padded_dims);
    for (flat, &value) in values.iter().enumerate() {
        let mut padded_flat = 0;
        for (dim, (&stride, &padded_stride)) in strides.iter().zip(&padded_strides).enumerate() {
            let coord = (flat / stride) % shape.dims()[dim];
            padded_flat += coord * padded_stride;
        }
        out[padded_flat] = field_from_i64(i64::from(value));
    }
    out
}

fn padded_u8_tensor<F: JoltField>(values: &[u8], shape: &Shape) -> Vec<F> {
    let padded_dims = shape.padded_power_of_two().0;
    let len = padded_dims.iter().product();
    let mut out = vec![F::zero(); len];
    let strides = row_major_strides(shape.dims());
    let padded_strides = row_major_strides(&padded_dims);
    for (flat, &value) in values.iter().enumerate() {
        let mut padded_flat = 0;
        for (dim, (&stride, &padded_stride)) in strides.iter().zip(&padded_strides).enumerate() {
            let coord = (flat / stride) % shape.dims()[dim];
            padded_flat += coord * padded_stride;
        }
        out[padded_flat] = F::from_u64(u64::from(value));
    }
    out
}

fn padded_field_tensor<F: JoltField>(values: &[F], shape: &Shape) -> Vec<F> {
    let padded_dims = shape.padded_power_of_two().0;
    let len = padded_dims.iter().product();
    let mut out = vec![F::zero(); len];
    let strides = row_major_strides(shape.dims());
    let padded_strides = row_major_strides(&padded_dims);
    for (flat, &value) in values.iter().enumerate() {
        let mut padded_flat = 0;
        for (dim, (&stride, &padded_stride)) in strides.iter().zip(&padded_strides).enumerate() {
            let coord = (flat / stride) % shape.dims()[dim];
            padded_flat += coord * padded_stride;
        }
        out[padded_flat] = value;
    }
    out
}

fn eval_i32_advice<F: JoltField>(values: &[i32], shape: &Shape, point: &[F]) -> F {
    values
        .iter()
        .enumerate()
        .fold(F::zero(), |acc, (idx, &value)| {
            acc + eq_for_flat(idx, shape, point) * field_from_i64::<F>(i64::from(value))
        })
}

fn eval_field_tensor<F: JoltField>(values: &[F], shape: &Shape, point: &[F]) -> F {
    values
        .iter()
        .enumerate()
        .fold(F::zero(), |acc, (idx, &value)| {
            acc + eq_for_flat(idx, shape, point) * value
        })
}

fn eval_logical_mask<F: JoltField>(shape: &Shape, point: &[F]) -> F {
    (0..shape.numel()).fold(F::zero(), |acc, idx| acc + eq_for_flat(idx, shape, point))
}

fn eq_for_flat<F: JoltField>(flat: usize, shape: &Shape, point: &[F]) -> F {
    let eq_by_dim = split_point(shape, point)
        .into_iter()
        .map(EqPolynomial::<F>::evals)
        .collect::<Vec<_>>();
    let strides = row_major_strides(shape.dims());
    let mut weight = F::one();
    for (dim, (&stride, eq)) in strides.iter().zip(&eq_by_dim).enumerate() {
        let coord = (flat / stride) % shape.dims()[dim];
        weight *= eq[coord];
    }
    weight
}

fn split_point<'a, F>(shape: &Shape, point: &'a [F]) -> Vec<&'a [F]> {
    let mut out = Vec::with_capacity(shape.dims().len());
    let mut offset = 0;
    for &dim in shape.dims() {
        let vars = dim.next_power_of_two().trailing_zeros() as usize;
        out.push(&point[offset..offset + vars]);
        offset += vars;
    }
    out
}

fn tensor_point_from_full<F: Clone>(point: &[F], params: &SoftmaxParams) -> Vec<F> {
    tensor_point_from_full_shape(point, &params.shape)
}

fn tensor_point_from_full_shape<F: Clone>(point: &[F], shape: &Shape) -> Vec<F> {
    let tensor_vars = shape.padded_power_of_two().point_len();
    point[..tensor_vars].to_vec()
}

fn frac_value(frac_bits: &[Vec<u8>; ROUND_FRAC_BITS], idx: usize) -> i64 {
    frac_bits
        .iter()
        .enumerate()
        .fold(0_i64, |acc, (bit, values)| {
            acc + (i64::from(values[idx]) << bit)
        })
}

fn ensure_len(name: &'static str, shape: &Shape, expected: usize, actual: usize) -> Result<()> {
    if actual != expected {
        return Err(ProverError::TensorLenMismatch {
            name,
            shape: shape.0.clone(),
            expected,
            actual,
        });
    }
    Ok(())
}

fn row_major_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; dims.len()];
    let mut stride = 1;
    for (idx, &dim) in dims.iter().enumerate().rev() {
        strides[idx] = stride;
        stride *= dim;
    }
    strides
}

fn prover_opening<F: JoltField>(
    accumulator: &ProverOpeningAccumulator<F>,
    id: OpeningId,
) -> Result<F> {
    accumulator
        .openings
        .get(&id)
        .map(|(_, value)| *value)
        .ok_or(ProverError::MissingOpening)
}

fn normalize_sumcheck_point<F: JoltField>(challenges: &[F]) -> Vec<F> {
    OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec())
        .match_endianness::<BIG_ENDIAN>()
        .r
}

fn field_from_i64<F: JoltField>(value: i64) -> F {
    if value >= 0 {
        F::from_u64(value as u64)
    } else {
        -F::from_u64(value.unsigned_abs())
    }
}

fn lerp<F: JoltField>(v0: F, v1: F, t: F) -> F {
    v0 + t * (v1 - v0)
}

fn softmax_acc_sumcheck_id() -> SumcheckId {
    SumcheckId::NodeExecution(0)
}

fn softmax_lookup_sumcheck_id() -> SumcheckId {
    SumcheckId::NodeExecution(1)
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use joltworks::{field::JoltField, transcripts::Blake2bTranscript};

    use super::*;

    #[test]
    fn proves_and_verifies_softmax_round() {
        let params = softmax_params(vec![1, 2]);
        let input = vec![256, 0];
        let max = vec![256];
        let max_index = vec![0];
        let min_diff = -1;
        let max_diff = 0;
        let entries = entries_from_min_max(min_diff, max_diff).unwrap();
        let exp = vec![softmax_exp_coarse_q8(0), softmax_exp_coarse_q8(-256)];
        let exp_acc = vec![softmax_exp_acc_q8(0), softmax_exp_acc_q8(-256)];
        let sum = vec![exp.iter().sum::<i32>()];
        let inv = inv_sum_q16(sum[0]);
        let acc = exp
            .iter()
            .map(|&value| i64::from(value) * inv)
            .collect::<Vec<_>>();
        let floor = acc
            .iter()
            .map(|&value| floor_shift_q8(value))
            .collect::<Vec<_>>();
        let output = floor
            .iter()
            .map(|&value| round_q8(i64::from(value)))
            .collect::<Vec<_>>();
        let point = vec![Fr::from(7_u64)];
        let output_claim = Claim {
            tensor: TensorId::new("softmax"),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: point.clone(),
            value: eval_i32(&output, &params.shape, &point),
        };
        let witness = SoftmaxWitness {
            input,
            max_index,
            max,
            min_diff,
            max_diff,
            input_frac_bits: frac_bits_for(&[0, -256]),
            ra: onehot_rows(&[1, 0], entries),
            exp_acc: exp_acc.clone(),
            exp,
            exp_frac_bits: frac_bits_for(&exp_acc),
            sum,
            acc: acc.clone(),
            floor: floor.clone(),
            floor_frac_bits: frac_bits_for(&acc),
            output,
            frac_bits: frac_bits_for_i32(&floor),
        };

        let mut prover_transcript = Blake2bTranscript::default();
        let (proof, input_claim, output_frac_bits, floor_frac_bits, input_frac_bits, ra_claim) =
            prove_softmax_round::<Fr, _>(
                vec![output_claim.clone()],
                &witness,
                &params,
                &mut prover_transcript,
            )
            .unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        let (
            verified_input_claim,
            verified_output_frac_bits,
            verified_floor_frac_bits,
            verified_input_frac_bits,
            verified_ra,
        ) = verify_softmax_round::<Fr, _>(
            vec![output_claim],
            &proof,
            &params,
            &mut verifier_transcript,
        )
        .unwrap();

        assert_eq!(verified_input_claim, input_claim);
        assert_eq!(verified_output_frac_bits, output_frac_bits);
        assert_eq!(verified_floor_frac_bits, floor_frac_bits);
        assert_eq!(verified_input_frac_bits, input_frac_bits);
        assert_eq!(verified_ra, ra_claim);
    }

    #[test]
    fn proves_softmax_with_non_power_of_two_logical_range() {
        let params = softmax_params(vec![1, 3]);
        let input = vec![512, 256, 0];
        let max = vec![512];
        let max_index = vec![0];
        let min_diff = -2;
        let max_diff = 0;
        let entries = entries_from_min_max(min_diff, max_diff).unwrap();
        assert_eq!(entries, 3);
        assert_eq!(entries.next_power_of_two(), 4);
        let exp = vec![
            softmax_exp_coarse_q8(0),
            softmax_exp_coarse_q8(-256),
            softmax_exp_coarse_q8(-512),
        ];
        let exp_acc = vec![
            softmax_exp_acc_q8(0),
            softmax_exp_acc_q8(-256),
            softmax_exp_acc_q8(-512),
        ];
        let sum = vec![exp.iter().sum::<i32>()];
        let inv = inv_sum_q16(sum[0]);
        let acc = exp
            .iter()
            .map(|&value| i64::from(value) * inv)
            .collect::<Vec<_>>();
        let floor = acc
            .iter()
            .map(|&value| floor_shift_q8(value))
            .collect::<Vec<_>>();
        let output = floor
            .iter()
            .map(|&value| round_q8(i64::from(value)))
            .collect::<Vec<_>>();
        let point = vec![Fr::from(5_u64), Fr::from(9_u64)];
        let output_claim = Claim {
            tensor: TensorId::new("softmax"),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: point.clone(),
            value: eval_i32(&output, &params.shape, &point),
        };
        let witness = SoftmaxWitness {
            input,
            max_index,
            max,
            min_diff,
            max_diff,
            input_frac_bits: frac_bits_for(&[0, -256, -512]),
            ra: onehot_rows(&[2, 1, 0], entries),
            exp_acc: exp_acc.clone(),
            exp,
            exp_frac_bits: frac_bits_for(&exp_acc),
            sum,
            acc: acc.clone(),
            floor: floor.clone(),
            floor_frac_bits: frac_bits_for(&acc),
            output,
            frac_bits: frac_bits_for_i32(&floor),
        };

        let mut prover_transcript = Blake2bTranscript::default();
        let (proof, _, _, _, _, _) = prove_softmax_round::<Fr, _>(
            vec![output_claim.clone()],
            &witness,
            &params,
            &mut prover_transcript,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        verify_softmax_round::<Fr, _>(
            vec![output_claim],
            &proof,
            &params,
            &mut verifier_transcript,
        )
        .unwrap();
    }

    #[test]
    fn proves_causal_softmax_without_large_mask_lut_entry() {
        let mut params = softmax_params(vec![1, 2, 2]);
        params.softmax_axis = 2;
        params.causal = true;
        let input = vec![256, -10_000, 0, 256];
        let max = vec![256, 256];
        let max_index = vec![0, 1];
        let min_diff = -1;
        let max_diff = 0;
        let entries = entries_from_min_max(min_diff, max_diff).unwrap();
        let exp = vec![
            softmax_exp_coarse_q8(0),
            softmax_exp_coarse_q8(0),
            softmax_exp_coarse_q8(-256),
            softmax_exp_coarse_q8(0),
        ];
        let exp_acc = vec![
            softmax_exp_acc_q8(0),
            softmax_exp_acc_q8(0),
            softmax_exp_acc_q8(-256),
            softmax_exp_acc_q8(0),
        ];
        let sum = vec![
            softmax_exp_coarse_q8(0),
            softmax_exp_coarse_q8(0) + softmax_exp_coarse_q8(-256),
        ];
        let acc = vec![
            i64::from(exp[0]) * inv_sum_q16(sum[0]),
            0,
            i64::from(exp[2]) * inv_sum_q16(sum[1]),
            i64::from(exp[3]) * inv_sum_q16(sum[1]),
        ];
        let floor = acc
            .iter()
            .map(|&value| floor_shift_q8(value))
            .collect::<Vec<_>>();
        let output = floor
            .iter()
            .map(|&value| round_q8(i64::from(value)))
            .collect::<Vec<_>>();
        let point = vec![Fr::from(3_u64), Fr::from(7_u64)];
        let output_claim = Claim {
            tensor: TensorId::new("softmax"),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: point.clone(),
            value: eval_i32(&output, &params.shape, &point),
        };
        let witness = SoftmaxWitness {
            input,
            max_index,
            max,
            min_diff,
            max_diff,
            input_frac_bits: frac_bits_for(&[0, 0, -256, 0]),
            ra: onehot_rows(&[1, 1, 0, 1], entries),
            exp_acc: exp_acc.clone(),
            exp,
            exp_frac_bits: frac_bits_for(&exp_acc),
            sum,
            acc: acc.clone(),
            floor: floor.clone(),
            floor_frac_bits: frac_bits_for(&acc),
            output,
            frac_bits: frac_bits_for_i32(&floor),
        };

        let mut prover_transcript = Blake2bTranscript::default();
        let (proof, _, _, _, _, _) = prove_softmax_round::<Fr, _>(
            vec![output_claim.clone()],
            &witness,
            &params,
            &mut prover_transcript,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        verify_softmax_round::<Fr, _>(
            vec![output_claim],
            &proof,
            &params,
            &mut verifier_transcript,
        )
        .unwrap();
    }

    fn softmax_params(shape: Vec<usize>) -> SoftmaxParams {
        SoftmaxParams::new(
            shape,
            "softmax_acc",
            "qk_score",
            "softmax",
            std::array::from_fn(|idx| format!("softmax_frac_bit_{idx}")),
            std::array::from_fn(|idx| format!("softmax_input_frac_bit_{idx}")),
            "softmax_ra",
            1,
            false,
        )
    }

    fn onehot_rows(selected: &[usize], entries: usize) -> Vec<u8> {
        let mut out = vec![0; selected.len() * entries];
        for (row, &idx) in selected.iter().enumerate() {
            out[row * entries + idx] = 1;
        }
        out
    }

    fn round_q8(value: i64) -> i32 {
        ((value + ((value.rem_euclid(FIXED_SCALE) >> (ROUND_FRAC_BITS - 1)) * FIXED_SCALE)
            - value.rem_euclid(FIXED_SCALE))
            / FIXED_SCALE) as i32
    }

    fn frac_bits_for(values: &[i64]) -> [Vec<u8>; ROUND_FRAC_BITS] {
        std::array::from_fn(|bit| {
            values
                .iter()
                .map(|value| ((value.rem_euclid(FIXED_SCALE) >> bit) & 1) as u8)
                .collect()
        })
    }

    fn frac_bits_for_i32(values: &[i32]) -> [Vec<u8>; ROUND_FRAC_BITS] {
        std::array::from_fn(|bit| {
            values
                .iter()
                .map(|value| ((i64::from(*value).rem_euclid(FIXED_SCALE) >> bit) & 1) as u8)
                .collect()
        })
    }

    fn eval_i32<F: JoltField>(values: &[i32], shape: &Shape, point: &[F]) -> F {
        values
            .iter()
            .enumerate()
            .fold(F::zero(), |acc, (idx, &value)| {
                acc + eq_for_flat(idx, shape, point) * field_from_i64::<F>(i64::from(value))
            })
    }
}
