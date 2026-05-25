use common::{CommittedPoly, VirtualPoly};

use joltworks::{
    config::{OneHotConfig, OneHotParams},
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
        shout::{self, RaOneHotEncoding, ReadRafProvider},
        sumcheck::{BatchedSumcheck, Sumcheck, SumcheckInstanceProof},
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

#[derive(Debug, Clone)]
pub struct SoftmaxWitness<'a> {
    pub input: &'a [i32],
    pub max_index: &'a [usize],
    pub max: &'a [i32],
    pub min_diff: i64,
    pub max_diff: i64,
    pub ra: &'a [u8],
    pub exp_acc: &'a [i64],
    pub exp: &'a [i32],
    pub exp_frac_bits: [&'a [u8]; ROUND_FRAC_BITS],
    pub sum: &'a [i32],
    pub acc: &'a [i64],
    pub floor: &'a [i32],
    pub floor_frac_bits: [&'a [u8]; ROUND_FRAC_BITS],
    pub output: &'a [i32],
    pub frac_bits: [&'a [u8]; ROUND_FRAC_BITS],
}

#[derive(Debug, Clone)]
pub struct SoftmaxProof<F: JoltField, T: Transcript> {
    pub round: RoundProof<F, T>,
    pub floor: FloorProof<F, T>,
    pub acc: SumcheckInstanceProof<F, T>,
    pub exp_round: RoundProof<F, T>,
    pub lookup: SumcheckInstanceProof<F, T>,
    pub exp_lookup: SumcheckInstanceProof<F, T>,
    pub exp_ra_onehot: SumcheckInstanceProof<F, T>,
    pub input_remainder_lookup: SumcheckInstanceProof<F, T>,
    pub input_remainder_ra_onehot: SumcheckInstanceProof<F, T>,
    pub exp_opening: F,
    pub input_opening: F,
    pub input_remainder_opening: F,
    pub exp_lut_opening: F,
    pub index_opening: F,
    pub ra_opening: F,
    pub input_remainder_ra_opening: F,
    pub exp_ra_committed_openings: SoftmaxRaCommittedOpenings<F>,
    pub input_remainder_ra_committed_openings: SoftmaxRaCommittedOpenings<F>,
    pub max_index: Vec<usize>,
    pub max: Vec<i32>,
    pub min_diff: i64,
    pub max_diff: i64,
    pub sum: Vec<i32>,
}

#[derive(Debug, Clone, Default)]
pub struct SoftmaxRaCommittedOpenings<F: JoltField> {
    pub ra_virtual: Vec<F>,
    pub hamming_weight: Vec<F>,
    pub booleanity: Vec<F>,
}

struct SoftmaxLookupProveResult<F: JoltField, T: Transcript> {
    relation: SumcheckInstanceProof<F, T>,
    exp_lookup: SumcheckInstanceProof<F, T>,
    exp_ra_onehot: SumcheckInstanceProof<F, T>,
    input_remainder_lookup: SumcheckInstanceProof<F, T>,
    input_remainder_ra_onehot: SumcheckInstanceProof<F, T>,
    input_claim: Claim<F>,
    input_remainder_ra_claim: Claim<F>,
    ra_claim: Claim<F>,
    input_remainder_opening: F,
    exp_lut_opening: F,
    index_opening: F,
    input_remainder_ra_opening: F,
    exp_ra_committed_openings: SoftmaxRaCommittedOpenings<F>,
    input_remainder_ra_committed_openings: SoftmaxRaCommittedOpenings<F>,
}

pub fn prove_softmax_round<F, T>(
    output_claims: Vec<Claim<F>>,
    witness: &SoftmaxWitness<'_>,
    params: &SoftmaxParams,
    transcript: &mut T,
) -> Result<(
    SoftmaxProof<F, T>,
    Claim<F>,
    Claim<F>,
    Claim<F>,
    Claim<F>,
    Claim<F>,
)>
where
    F: JoltField,
    T: Transcript,
{
    validate_inputs(witness, params)?;
    let floor_as_i64 = witness
        .floor
        .iter()
        .map(|&value| i64::from(value))
        .collect::<Vec<_>>();
    let round_witness = RoundWitness::from_input_output(&floor_as_i64, witness.output);
    let (round_proof, floor_claim, output_round_ra) =
        prove_round(output_claims, &round_witness, &params.round, transcript)?;
    let floor_witness = FloorWitness {
        input: witness.acc,
        output: witness.floor,
        frac_bits: witness.floor_frac_bits,
    };
    let (floor_proof, acc_claim, floor_round_ra) =
        prove_floor(vec![floor_claim], &floor_witness, &params.floor, transcript)?;
    append_row_advice::<F, T>(
        witness.max_index,
        witness.max,
        witness.min_diff,
        witness.max_diff,
        witness.sum,
        transcript,
    );

    let inv_sum = inv_sum_from_sum::<F>(witness.sum);
    let acc_eq = EqPolynomial::<F>::evals(&acc_claim.point);
    let exp_poly = padded_i32_tensor(witness.exp, &params.shape);
    let valid_poly = valid_tensor_u8(params);
    let mut acc_prover = AccSumcheckProver::new(
        BasicSumcheckParams::new(
            params.shape.padded_power_of_two().point_len(),
            acc_claim.value,
        ),
        acc_eq,
        exp_poly,
        inv_sum,
        params.row_shape(),
        params.cols(),
        valid_poly,
    );
    let mut acc_accumulator = ProverOpeningAccumulator::new();
    let (acc_proof, acc_challenges) =
        Sumcheck::prove(&mut acc_prover, &mut acc_accumulator, transcript);
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
    let exp_round_witness = RoundWitness::from_input_output(witness.exp_acc, witness.exp);
    let (exp_round_proof, exp_acc_claim, _exp_ra) = prove_round(
        vec![exp_claim],
        &exp_round_witness,
        &params.exp_round,
        transcript,
    )?;
    let lookup = prove_lookup(exp_acc_claim, witness, params, transcript)?;
    Ok((
        SoftmaxProof {
            round: round_proof,
            floor: floor_proof,
            acc: acc_proof,
            exp_round: exp_round_proof,
            lookup: lookup.relation,
            exp_lookup: lookup.exp_lookup,
            exp_ra_onehot: lookup.exp_ra_onehot,
            input_remainder_lookup: lookup.input_remainder_lookup,
            input_remainder_ra_onehot: lookup.input_remainder_ra_onehot,
            exp_opening,
            input_opening: lookup.input_claim.value,
            input_remainder_opening: lookup.input_remainder_opening,
            exp_lut_opening: lookup.exp_lut_opening,
            index_opening: lookup.index_opening,
            ra_opening: lookup.ra_claim.value,
            input_remainder_ra_opening: lookup.input_remainder_ra_opening,
            exp_ra_committed_openings: lookup.exp_ra_committed_openings,
            input_remainder_ra_committed_openings: lookup.input_remainder_ra_committed_openings,
            max_index: witness.max_index.to_vec(),
            max: witness.max.to_vec(),
            min_diff: witness.min_diff,
            max_diff: witness.max_diff,
            sum: witness.sum.to_vec(),
        },
        lookup.input_claim,
        output_round_ra,
        floor_round_ra,
        lookup.input_remainder_ra_claim,
        lookup.ra_claim,
    ))
}

pub fn verify_softmax_round<F, T>(
    output_claims: Vec<Claim<F>>,
    proof: &SoftmaxProof<F, T>,
    params: &SoftmaxParams,
    transcript: &mut T,
) -> std::result::Result<(Claim<F>, Claim<F>, Claim<F>, Claim<F>, Claim<F>), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    verify_advice(params, proof)?;
    let (floor_claim, output_round_ra) =
        verify_round(output_claims, &proof.round, &params.round, transcript)?;
    let (acc_claim, floor_round_ra) =
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
    let acc_verifier = AccSumcheckVerifier {
        params: BasicSumcheckParams::new(
            params.shape.padded_power_of_two().point_len(),
            acc_claim.value,
        ),
        acc_point: acc_claim.point.clone(),
        inv: inv_sum,
        valid: valid_tensor(params),
        row_shape: params.row_shape(),
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

    let (exp_acc_claim, _exp_ra) = verify_round(
        vec![exp_claim],
        &proof.exp_round,
        &params.exp_round,
        transcript,
    )?;

    let (input_claim, input_remainder_ra, ra_claim) =
        verify_lookup(exp_acc_claim, proof, params, transcript)?;

    Ok((
        input_claim,
        output_round_ra,
        floor_round_ra,
        input_remainder_ra,
        ra_claim,
    ))
}

fn prove_lookup<F, T>(
    exp_claim: Claim<F>,
    witness: &SoftmaxWitness<'_>,
    params: &SoftmaxParams,
    transcript: &mut T,
) -> Result<SoftmaxLookupProveResult<F, T>>
where
    F: JoltField,
    T: Transcript,
{
    let entries = entries_from_min_max(witness.min_diff, witness.max_diff)?;
    let lut_len = padded_softmax_lut_len(entries);
    let eq_exp = masked_eq_poly(&exp_claim, &params.shape);
    let row_point = row_point_from_tensor_point(&exp_claim.point, params);
    let row_eq = EqPolynomial::<F>::evals(&row_point);
    let max_selector = max_selector_tensor(witness.max_index, params);
    let valid = valid_tensor_u8(params);
    let max = expand_rows_to_padded_tensor_i32(witness.max, params);
    let logical_lookup_indices = softmax_logical_lookup_indices(witness, params, entries)?;
    let padded_exp_indices = padded_lookup_indices(&logical_lookup_indices, params, entries);
    let exp_lut_values = softmax_lookup_values(
        &logical_lookup_indices,
        witness.min_diff,
        exp_lut_q8_unclipped,
    );
    let logical_remainders = softmax_input_remainders(witness, params);
    let padded_remainders = padded_lookup_indices(&logical_remainders, params, 0);
    let input = padded_i32_tensor(witness.input, &params.shape);
    let remainder = padded_usize_evals(&padded_remainders);
    let exp_lut = padded_i64_tensor(&exp_lut_values, &params.shape);
    let index = padded_usize_evals(&padded_exp_indices);
    let exp_table = padded_i32_table(witness.min_diff, entries, lut_len, exp_lut_q8_unclipped)?;
    let max_eval = eval_i32_advice(witness.max, &params.row_shape(), &row_point);
    let max_mix = transcript.challenge_scalar();
    let diff_mix = transcript.challenge_scalar();
    let input_claim = exp_claim.value + max_mix * max_eval;
    let sc_params =
        LookupSumcheckParams::new(params.shape.padded_power_of_two().point_len(), input_claim);
    let mut prover = LookupSumcheckProver::new(
        sc_params,
        eq_exp,
        row_eq,
        params.cols(),
        max_selector,
        valid,
        max,
        input,
        remainder,
        exp_lut,
        index,
        max_mix,
        diff_mix,
        field_from_i64(witness.min_diff),
    );
    let mut accumulator = ProverOpeningAccumulator::new();
    let (proof, challenges) = Sumcheck::prove(&mut prover, &mut accumulator, transcript);
    let input_opening = prover_opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenSoftmaxInput, softmax_lookup_sumcheck_id()),
    )?;
    let input_remainder_opening = prover_opening(
        &accumulator,
        OpeningId::new(
            VirtualPoly::QwenSoftmaxInputRemainder,
            softmax_lookup_sumcheck_id(),
        ),
    )?;
    let exp_lut_opening = prover_opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenSoftmaxExpLut, softmax_lookup_sumcheck_id()),
    )?;
    let index_opening = prover_opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenSoftmaxIndex, softmax_lookup_sumcheck_id()),
    )?;
    let full_point = normalize_sumcheck_point::<F>(&challenges.into_opening());
    let tensor_point = full_point;
    let exp_shout_lookup = prove_softmax_shout_lookup(
        SoftmaxLookupKind::Exp,
        lut_len,
        tensor_point.clone(),
        exp_lut_opening,
        index_opening,
        padded_exp_indices,
        exp_table,
        &mut accumulator,
        transcript,
    )?;
    let input_remainder_shout_lookup = prove_softmax_shout_lookup(
        SoftmaxLookupKind::InputRemainder,
        256,
        tensor_point.clone(),
        input_remainder_opening,
        input_remainder_opening,
        padded_remainders,
        (0..256).collect(),
        &mut accumulator,
        transcript,
    )?;

    Ok(SoftmaxLookupProveResult {
        relation: proof,
        exp_lookup: exp_shout_lookup.read_raf,
        exp_ra_onehot: exp_shout_lookup.ra_onehot,
        input_claim: Claim {
            tensor: params.input_tensor.clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: tensor_point.clone(),
            value: input_opening,
        },
        ra_claim: Claim {
            tensor: params.ra_tensor.clone(),
            logical_shape: params.ra_shape(lut_len),
            domain_shape: params.ra_shape(lut_len).padded_power_of_two(),
            point: exp_shout_lookup.ra_point,
            value: exp_shout_lookup.ra_opening,
        },
        input_remainder_ra_claim: Claim {
            tensor: TensorId::new(format!("{}_input_remainder_ra", params.output_tensor.0)),
            logical_shape: params.ra_shape(256),
            domain_shape: params.ra_shape(256).padded_power_of_two(),
            point: input_remainder_shout_lookup.ra_point,
            value: input_remainder_shout_lookup.ra_opening,
        },
        input_remainder_opening,
        exp_lut_opening,
        index_opening,
        input_remainder_ra_opening: input_remainder_shout_lookup.ra_opening,
        input_remainder_lookup: input_remainder_shout_lookup.read_raf,
        input_remainder_ra_onehot: input_remainder_shout_lookup.ra_onehot,
        exp_ra_committed_openings: exp_shout_lookup.committed_openings,
        input_remainder_ra_committed_openings: input_remainder_shout_lookup.committed_openings,
    })
}

fn verify_lookup<F, T>(
    exp_claim: Claim<F>,
    proof: &SoftmaxProof<F, T>,
    params: &SoftmaxParams,
    transcript: &mut T,
) -> std::result::Result<(Claim<F>, Claim<F>, Claim<F>), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    let entries = entries_from_min_max(proof.min_diff, proof.max_diff)
        .map_err(|_| ProofVerifyError::InvalidInputLength(MAX_SOFTMAX_LUT_ENTRIES, 0))?;
    let lut_len = padded_softmax_lut_len(entries);
    let row_point = row_point_from_tensor_point(&exp_claim.point, params);
    let max_eval = eval_i32_advice(&proof.max, &params.row_shape(), &row_point);
    let max_mix = transcript.challenge_scalar();
    let diff_mix = transcript.challenge_scalar();
    let input_claim = exp_claim.value + max_mix * max_eval;

    let verifier = LookupSumcheckVerifier {
        params: LookupSumcheckParams::new(
            params.shape.padded_power_of_two().point_len(),
            input_claim,
        ),
        exp_point: exp_claim.point.clone(),
        row_point,
        max_index: proof.max_index.clone(),
        max: proof.max.clone(),
        min_diff: proof.min_diff,
        causal: params.causal,
        shape: params.shape.clone(),
        row_shape: params.row_shape(),
        max_mix,
        diff_mix,
    };
    let mut accumulator = VerifierOpeningAccumulator::new();
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenSoftmaxInput, softmax_lookup_sumcheck_id()),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.input_opening,
        ),
    );
    accumulator.openings.insert(
        OpeningId::new(
            VirtualPoly::QwenSoftmaxInputRemainder,
            softmax_lookup_sumcheck_id(),
        ),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.input_remainder_opening,
        ),
    );
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenSoftmaxExpLut, softmax_lookup_sumcheck_id()),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.exp_lut_opening,
        ),
    );
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenSoftmaxIndex, softmax_lookup_sumcheck_id()),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.index_opening,
        ),
    );
    let challenges = Sumcheck::verify(&proof.lookup, &verifier, &mut accumulator, transcript)?;
    let tensor_point = normalize_sumcheck_point::<F>(&challenges.into_opening());
    let exp_shout_lookup = verify_softmax_shout_lookup(
        SoftmaxLookupKind::Exp,
        entries,
        lut_len,
        proof.min_diff,
        params.shape.numel(),
        tensor_point.clone(),
        proof.exp_lut_opening,
        proof.index_opening,
        proof.ra_opening,
        &proof.exp_ra_committed_openings,
        &proof.exp_lookup,
        &proof.exp_ra_onehot,
        &mut accumulator,
        transcript,
    )?;
    let input_remainder_shout_lookup = verify_softmax_shout_lookup(
        SoftmaxLookupKind::InputRemainder,
        256,
        256,
        0,
        params.shape.numel(),
        tensor_point.clone(),
        proof.input_remainder_opening,
        proof.input_remainder_opening,
        proof.input_remainder_ra_opening,
        &proof.input_remainder_ra_committed_openings,
        &proof.input_remainder_lookup,
        &proof.input_remainder_ra_onehot,
        &mut accumulator,
        transcript,
    )?;

    Ok((
        Claim {
            tensor: params.input_tensor.clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: tensor_point.clone(),
            value: proof.input_opening,
        },
        Claim {
            tensor: TensorId::new(format!("{}_input_remainder_ra", params.output_tensor.0)),
            logical_shape: params.ra_shape(256),
            domain_shape: params.ra_shape(256).padded_power_of_two(),
            point: input_remainder_shout_lookup.ra_point,
            value: proof.input_remainder_ra_opening,
        },
        Claim {
            tensor: params.ra_tensor.clone(),
            logical_shape: params.ra_shape(lut_len),
            domain_shape: params.ra_shape(lut_len).padded_power_of_two(),
            point: exp_shout_lookup.ra_point,
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

struct RowExpandedPoly<F: JoltField> {
    row: MultilinearPolynomial<F>,
    remaining_col_vars: usize,
}

impl<F: JoltField> RowExpandedPoly<F> {
    fn from_logical_rows(row_values: Vec<F>, row_shape: Shape, cols: usize) -> Self {
        Self::from_padded_rows(padded_field_tensor(&row_values, &row_shape), cols)
    }

    fn from_padded_rows(row_values: Vec<F>, cols: usize) -> Self {
        debug_assert!(row_values.len().is_power_of_two());
        Self {
            row: MultilinearPolynomial::from(row_values),
            remaining_col_vars: cols.next_power_of_two().trailing_zeros() as usize,
        }
    }

    fn get_bound_coeff(&self, index: usize) -> F {
        self.row.get_bound_coeff(index >> self.remaining_col_vars)
    }

    fn bind_parallel(&mut self, r_j: F::Challenge) {
        if self.remaining_col_vars > 0 {
            // The polynomial is constant along the softmax column variables:
            // binding a column variable only removes that variable from the
            // domain.  Row variables are bound after all column variables
            // because all softmax sumchecks bind LowToHigh on row-major tensors.
            let _ = r_j;
            self.remaining_col_vars -= 1;
        } else {
            self.row.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
    }
}

struct AccSumcheckProver<F: JoltField> {
    eq: MultilinearPolynomial<F>,
    exp: MultilinearPolynomial<F>,
    inv: RowExpandedPoly<F>,
    valid: MultilinearPolynomial<F>,
    params: BasicSumcheckParams<F>,
}

impl<F: JoltField> AccSumcheckProver<F> {
    fn new(
        params: BasicSumcheckParams<F>,
        eq: Vec<F>,
        exp: Vec<i32>,
        inv: Vec<F>,
        row_shape: Shape,
        cols: usize,
        valid: Vec<u8>,
    ) -> Self {
        Self {
            eq: MultilinearPolynomial::from(eq),
            exp: MultilinearPolynomial::from(exp),
            inv: RowExpandedPoly::from_logical_rows(inv, row_shape, cols),
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
        self.inv.bind_parallel(r_j);
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
    row_shape: Shape,
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
        let inv = eval_field_tensor(
            &padded_field_tensor(&self.inv, &self.row_shape),
            &self.row_shape.padded_power_of_two(),
            row_point_from_tensor_point_shape(&point, &self.shape),
        );
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
    row_eq: RowExpandedPoly<F>,
    max_selector: MultilinearPolynomial<F>,
    valid: MultilinearPolynomial<F>,
    max: MultilinearPolynomial<F>,
    input: MultilinearPolynomial<F>,
    remainder: MultilinearPolynomial<F>,
    exp_lut: MultilinearPolynomial<F>,
    index: MultilinearPolynomial<F>,
    max_mix: F,
    diff_mix: F,
    min_diff: F,
    params: LookupSumcheckParams<F>,
}

#[allow(clippy::too_many_arguments)]
impl<F: JoltField> LookupSumcheckProver<F> {
    fn new(
        params: LookupSumcheckParams<F>,
        eq_exp: Vec<F>,
        row_eq: Vec<F>,
        cols: usize,
        max_selector: Vec<u8>,
        valid: Vec<u8>,
        max: Vec<i32>,
        input: Vec<i32>,
        remainder: Vec<u32>,
        exp_lut: Vec<i64>,
        index: Vec<u32>,
        max_mix: F,
        diff_mix: F,
        min_diff: F,
    ) -> Self {
        Self {
            eq_exp: MultilinearPolynomial::from(eq_exp),
            row_eq: RowExpandedPoly::from_padded_rows(row_eq, cols),
            max_selector: MultilinearPolynomial::from(max_selector),
            valid: MultilinearPolynomial::from(valid),
            max: MultilinearPolynomial::from(max),
            input: MultilinearPolynomial::from(input),
            remainder: MultilinearPolynomial::from(remainder),
            exp_lut: MultilinearPolynomial::from(exp_lut),
            index: MultilinearPolynomial::from(index),
            max_mix,
            diff_mix,
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
        for g in 0..self.eq_exp.len() / 2 {
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
        self.row_eq.bind_parallel(r_j);
        self.max_selector
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.valid.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.max.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.input.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.remainder.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.exp_lut.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.index.bind_parallel(r_j, BindingOrder::LowToHigh);
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
        accumulator.append_virtual(
            transcript,
            OpeningId::new(
                VirtualPoly::QwenSoftmaxInputRemainder,
                softmax_lookup_sumcheck_id(),
            ),
            point.clone(),
            self.remainder.final_claim(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenSoftmaxExpLut, softmax_lookup_sumcheck_id()),
            point.clone(),
            self.exp_lut.final_claim(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenSoftmaxIndex, softmax_lookup_sumcheck_id()),
            point,
            self.index.final_claim(),
        );
    }
}

struct LookupSumcheckVerifier<F: JoltField> {
    params: LookupSumcheckParams<F>,
    exp_point: Vec<F>,
    row_point: Vec<F>,
    max_index: Vec<usize>,
    max: Vec<i32>,
    min_diff: i64,
    causal: bool,
    shape: Shape,
    row_shape: Shape,
    max_mix: F,
    diff_mix: F,
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
        let input = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenSoftmaxInput,
                softmax_lookup_sumcheck_id(),
            ))
            .1;
        let remainder = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenSoftmaxInputRemainder,
                softmax_lookup_sumcheck_id(),
            ))
            .1;
        let exp_lut = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenSoftmaxExpLut,
                softmax_lookup_sumcheck_id(),
            ))
            .1;
        let index = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenSoftmaxIndex,
                softmax_lookup_sumcheck_id(),
            ))
            .1;
        let eq_exp = eval_masked_eq(&self.exp_point, &self.shape, &tensor_point);
        let row_eq = EqPolynomial::mle(
            &self.row_point,
            row_point_from_tensor_point_shape(&tensor_point, &self.shape),
        );
        let max_selector = eval_max_selector(&self.max_index, &self.shape, &tensor_point);
        let valid = eval_valid(self.causal, &self.shape, &tensor_point);
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
            remainder,
            exp_lut,
            index,
            max,
            self.max_mix,
            self.diff_mix,
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
        accumulator.append_virtual(
            transcript,
            OpeningId::new(
                VirtualPoly::QwenSoftmaxInputRemainder,
                softmax_lookup_sumcheck_id(),
            ),
            point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenSoftmaxExpLut, softmax_lookup_sumcheck_id()),
            point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenSoftmaxIndex, softmax_lookup_sumcheck_id()),
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
    remainder: [F; 2],
    exp_lut: [F; 2],
    index: [F; 2],
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
            remainder: [
                p.remainder.get_bound_coeff(2 * g),
                p.remainder.get_bound_coeff(2 * g + 1),
            ],
            exp_lut: [
                p.exp_lut.get_bound_coeff(2 * g),
                p.exp_lut.get_bound_coeff(2 * g + 1),
            ],
            index: [
                p.index.get_bound_coeff(2 * g),
                p.index.get_bound_coeff(2 * g + 1),
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
            lerp(self.remainder[0], self.remainder[1], t),
            lerp(self.exp_lut[0], self.exp_lut[1], t),
            lerp(self.index[0], self.index[1], t),
            lerp(self.max[0], self.max[1], t),
            p.max_mix,
            p.diff_mix,
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
    remainder: F,
    exp_lut: F,
    index: F,
    max: F,
    max_mix: F,
    diff_mix: F,
    min_diff: F,
) -> F {
    let n = min_diff + index;
    let corr = F::from_u64(FIXED_SCALE as u64) + remainder;
    let diff_expr = valid * (input - max) - (n * F::from_u64(FIXED_SCALE as u64) + remainder);
    eq_exp * exp_lut * corr
        + max_mix * row_eq * max_selector * input
        + diff_mix * eq_exp * diff_expr
}

#[derive(Clone)]
enum SoftmaxLookupKind {
    Exp,
    InputRemainder,
}

impl SoftmaxLookupKind {
    fn value_poly(&self) -> VirtualPoly {
        match self {
            Self::Exp => VirtualPoly::QwenSoftmaxExpLut,
            Self::InputRemainder => VirtualPoly::QwenSoftmaxInputRemainder,
        }
    }

    fn index_poly(&self) -> VirtualPoly {
        match self {
            Self::Exp => VirtualPoly::QwenSoftmaxIndex,
            Self::InputRemainder => VirtualPoly::QwenSoftmaxInputRemainder,
        }
    }

    fn ra_poly(&self) -> VirtualPoly {
        match self {
            Self::Exp => VirtualPoly::QwenSoftmaxExpRa,
            Self::InputRemainder => VirtualPoly::QwenSoftmaxInputRemainderRa,
        }
    }

    fn committed_poly(&self, d: usize) -> CommittedPoly {
        match self {
            Self::Exp => CommittedPoly::QwenSoftmaxExpRaD(d),
            Self::InputRemainder => CommittedPoly::QwenSoftmaxInputRemainderRaD(d),
        }
    }

    fn sumcheck_id(&self) -> SumcheckId {
        match self {
            Self::Exp => softmax_shout_sumcheck_id(),
            Self::InputRemainder => softmax_remainder_shout_sumcheck_id(),
        }
    }
}

#[derive(Clone)]
struct SoftmaxReadRafProvider<F: JoltField> {
    kind: SoftmaxLookupKind,
    log_k: usize,
    r_cycle: OpeningPoint<BIG_ENDIAN, F>,
    rv_claim: F,
    raf_claim: F,
}

impl<F: JoltField> ReadRafProvider<F> for SoftmaxReadRafProvider<F> {
    fn rv_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.rv_claim
    }

    fn raf_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.raf_claim
    }

    fn r(&self, _accumulator: &dyn OpeningAccumulator<F>) -> OpeningPoint<BIG_ENDIAN, F> {
        self.r_cycle.clone()
    }

    fn ra_poly(&self) -> (VirtualPoly, SumcheckId) {
        (self.kind.ra_poly(), self.kind.sumcheck_id())
    }

    fn log_K(&self) -> usize {
        self.log_k
    }
}

struct SoftmaxRaEncoding {
    kind: SoftmaxLookupKind,
    log_k: usize,
}

impl RaOneHotEncoding for SoftmaxRaEncoding {
    fn committed_poly(&self, d: usize) -> CommittedPoly {
        self.kind.committed_poly(d)
    }

    fn r_cycle_source(&self) -> OpeningId {
        OpeningId::new(self.kind.value_poly(), softmax_lookup_sumcheck_id())
    }

    fn ra_source(&self) -> OpeningId {
        OpeningId::new(self.kind.ra_poly(), self.kind.sumcheck_id())
    }

    fn log_k(&self) -> usize {
        self.log_k
    }

    fn one_hot_params(&self) -> OneHotParams {
        OneHotParams::from_config_and_log_K(&OneHotConfig::default(), self.log_k)
    }
}

struct SoftmaxShoutProof<F: JoltField, T: Transcript> {
    read_raf: SumcheckInstanceProof<F, T>,
    ra_onehot: SumcheckInstanceProof<F, T>,
    ra_point: Vec<F>,
    ra_opening: F,
    committed_openings: SoftmaxRaCommittedOpenings<F>,
}

fn prove_softmax_shout_lookup<F, T>(
    kind: SoftmaxLookupKind,
    lut_len: usize,
    tensor_point: Vec<F>,
    exp_lut_opening: F,
    index_opening: F,
    lookup_indices: Vec<usize>,
    table: Vec<i32>,
    accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
) -> Result<SoftmaxShoutProof<F, T>>
where
    F: JoltField,
    T: Transcript,
{
    let log_k = lut_len.trailing_zeros() as usize;
    let r_cycle = OpeningPoint::<BIG_ENDIAN, F>::new(tensor_point);
    let provider = SoftmaxReadRafProvider {
        kind: kind.clone(),
        log_k,
        r_cycle,
        rv_claim: exp_lut_opening,
        raf_claim: index_opening,
    };
    let mut read_prover =
        shout::read_raf_prover(&provider, &lookup_indices, &table, accumulator, transcript);
    let (read_raf, _) = Sumcheck::prove(&mut *read_prover, accumulator, transcript);

    let encoding = SoftmaxRaEncoding {
        kind: kind.clone(),
        log_k,
    };
    let [ra_prover, hw_prover, bool_prover] =
        shout::ra_onehot_provers(&encoding, &lookup_indices, accumulator, transcript);
    let use_ra_virtual = lookup_indices.len().next_power_of_two() >= 8;
    let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = if !use_ra_virtual {
        // JoltWorks' RaPolynomial specialization only exposes final_claim after
        // at least three cycle bindings.  Real Qwen tensors have many more
        // variables; this branch only keeps tiny unit-test domains executable.
        vec![hw_prover]
    } else {
        vec![ra_prover, hw_prover, bool_prover]
    };
    let (ra_onehot, _) = BatchedSumcheck::prove(
        instances.iter_mut().map(|v| &mut **v as _).collect(),
        accumulator,
        transcript,
    );
    let (ra_point, ra_opening) = accumulator
        .get_virtual_polynomial_opening(OpeningId::new(kind.ra_poly(), kind.sumcheck_id()));
    let committed_openings =
        softmax_ra_committed_openings(&kind, log_k, use_ra_virtual, accumulator)?;
    Ok(SoftmaxShoutProof {
        read_raf,
        ra_onehot,
        ra_point: ra_point.r,
        ra_opening,
        committed_openings,
    })
}

fn verify_softmax_shout_lookup<F, T>(
    kind: SoftmaxLookupKind,
    entries: usize,
    lut_len: usize,
    min_diff: i64,
    logical_len: usize,
    tensor_point: Vec<F>,
    exp_lut_opening: F,
    index_opening: F,
    ra_opening: F,
    committed_openings: &SoftmaxRaCommittedOpenings<F>,
    read_raf: &SumcheckInstanceProof<F, T>,
    ra_onehot: &SumcheckInstanceProof<F, T>,
    accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut T,
) -> std::result::Result<SoftmaxShoutProof<F, T>, ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    let log_k = lut_len.trailing_zeros() as usize;
    let r_cycle = OpeningPoint::<BIG_ENDIAN, F>::new(tensor_point);
    accumulator.openings.insert(
        OpeningId::new(kind.value_poly(), softmax_lookup_sumcheck_id()),
        (r_cycle.clone(), exp_lut_opening),
    );
    accumulator.openings.insert(
        OpeningId::new(kind.index_poly(), softmax_lookup_sumcheck_id()),
        (r_cycle.clone(), index_opening),
    );
    accumulator.openings.insert(
        OpeningId::new(kind.ra_poly(), kind.sumcheck_id()),
        (OpeningPoint::<BIG_ENDIAN, F>::default(), ra_opening),
    );
    let provider = SoftmaxReadRafProvider {
        kind: kind.clone(),
        log_k,
        r_cycle,
        rv_claim: exp_lut_opening,
        raf_claim: index_opening,
    };
    let table = match kind {
        SoftmaxLookupKind::Exp => {
            padded_i32_table(min_diff, entries, lut_len, exp_lut_q8_unclipped)
                .map_err(|_| ProofVerifyError::InvalidInputLength(lut_len, 0))?
        }
        SoftmaxLookupKind::InputRemainder => (0..lut_len)
            .map(|idx| if idx < entries { idx as i32 } else { 0 })
            .collect(),
    };
    let read_verifier = shout::read_raf_verifier(&provider, table, accumulator, transcript);
    Sumcheck::verify(read_raf, &*read_verifier, accumulator, transcript)?;

    let encoding = SoftmaxRaEncoding {
        kind: kind.clone(),
        log_k,
    };
    let [ra_verifier, hw_verifier, bool_verifier] =
        shout::ra_onehot_verifiers(&encoding, accumulator, transcript);
    let use_ra_virtual = logical_len.next_power_of_two() >= 8;
    insert_softmax_ra_committed_openings(
        &kind,
        log_k,
        use_ra_virtual,
        committed_openings,
        accumulator,
    )?;
    let verifier_instances: Vec<&dyn SumcheckInstanceVerifier<F, T>> = if !use_ra_virtual {
        vec![&*hw_verifier]
    } else {
        vec![&*ra_verifier, &*hw_verifier, &*bool_verifier]
    };
    BatchedSumcheck::verify(ra_onehot, verifier_instances, accumulator, transcript)?;
    let (ra_point, ra_opening) = accumulator
        .get_virtual_polynomial_opening(OpeningId::new(kind.ra_poly(), kind.sumcheck_id()));
    Ok(SoftmaxShoutProof {
        read_raf: read_raf.clone(),
        ra_onehot: ra_onehot.clone(),
        ra_point: ra_point.r,
        ra_opening,
        committed_openings: committed_openings.clone(),
    })
}

fn softmax_ra_committed_openings<F: JoltField>(
    kind: &SoftmaxLookupKind,
    log_k: usize,
    include_full_checks: bool,
    accumulator: &ProverOpeningAccumulator<F>,
) -> Result<SoftmaxRaCommittedOpenings<F>> {
    let d = OneHotParams::from_config_and_log_K(&OneHotConfig::default(), log_k).instruction_d;
    let collect = |sumcheck| -> Vec<F> {
        (0..d)
            .map(|idx| {
                accumulator
                    .get_committed_polynomial_opening(OpeningId::new(
                        kind.committed_poly(idx),
                        sumcheck,
                    ))
                    .1
            })
            .collect()
    };
    Ok(SoftmaxRaCommittedOpenings {
        ra_virtual: if include_full_checks {
            collect(SumcheckId::RaVirtualization)
        } else {
            vec![]
        },
        hamming_weight: collect(SumcheckId::HammingWeight),
        booleanity: if include_full_checks {
            collect(SumcheckId::Booleanity)
        } else {
            vec![]
        },
    })
}

fn insert_softmax_ra_committed_openings<F: JoltField>(
    kind: &SoftmaxLookupKind,
    log_k: usize,
    include_full_checks: bool,
    openings: &SoftmaxRaCommittedOpenings<F>,
    accumulator: &mut VerifierOpeningAccumulator<F>,
) -> std::result::Result<(), ProofVerifyError> {
    let d = OneHotParams::from_config_and_log_K(&OneHotConfig::default(), log_k).instruction_d;
    if openings.hamming_weight.len() != d {
        return Err(ProofVerifyError::InvalidInputLength(
            d,
            openings.hamming_weight.len(),
        ));
    }
    if include_full_checks && (openings.ra_virtual.len() != d || openings.booleanity.len() != d) {
        return Err(ProofVerifyError::InvalidInputLength(
            d,
            openings.ra_virtual.len().min(openings.booleanity.len()),
        ));
    }
    let mut insert_group = |sumcheck, values: &[F]| {
        for (idx, &value) in values.iter().enumerate() {
            accumulator.openings.insert(
                OpeningId::new(kind.committed_poly(idx), sumcheck),
                (OpeningPoint::<BIG_ENDIAN, F>::default(), value),
            );
        }
    };
    if include_full_checks {
        insert_group(SumcheckId::RaVirtualization, &openings.ra_virtual);
    }
    insert_group(SumcheckId::HammingWeight, &openings.hamming_weight);
    if include_full_checks {
        insert_group(SumcheckId::Booleanity, &openings.booleanity);
    }
    Ok(())
}

fn validate_inputs(w: &SoftmaxWitness<'_>, params: &SoftmaxParams) -> Result<()> {
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
                i64::from(w.input[idx]) - i64::from(w.max[row])
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

fn padded_softmax_lut_len(entries: usize) -> usize {
    // Same convention as SiLU: reserve one extra zero row for padded tensor
    // slots, and keep the Shout one-hot chunks at their minimum practical
    // size.  Real attention tensors are far larger than this minimum.
    (entries + 1).next_power_of_two().max(16)
}

fn max_selector_tensor(max_index: &[usize], params: &SoftmaxParams) -> Vec<u8> {
    let mut values = vec![0; params.shape.numel()];
    let cols = params.cols();
    for (row, &col) in max_index.iter().enumerate() {
        values[row * cols + col] = 1;
    }
    padded_u8_tensor(&values, &params.shape)
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

fn valid_tensor_u8(params: &SoftmaxParams) -> Vec<u8> {
    valid_tensor_u8_for_shape(params.causal, &params.shape)
}

fn valid_tensor<F: JoltField>(params: &SoftmaxParams) -> Vec<F> {
    valid_tensor_for_shape(params.causal, &params.shape)
}

fn valid_tensor_u8_for_shape(causal: bool, shape: &Shape) -> Vec<u8> {
    let cols = *shape.dims().last().unwrap();
    let rows = shape.numel() / cols;
    let mut values = vec![0; shape.numel()];
    for row in 0..rows {
        for col in 0..cols {
            if is_valid_position(causal, shape, row, col) {
                values[row * cols + col] = 1;
            }
        }
    }
    padded_u8_tensor(&values, shape)
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

fn expand_rows_to_padded_tensor_i32(row_values: &[i32], params: &SoftmaxParams) -> Vec<i32> {
    let padded_dims = params.shape.padded_power_of_two().0;
    let len = padded_dims.iter().product();
    let padded_cols = *padded_dims.last().unwrap();
    let row_dims = &params.shape.dims()[..params.shape.dims().len() - 1];
    let padded_row_dims = &padded_dims[..padded_dims.len() - 1];
    let row_strides = row_major_strides(row_dims);
    let padded_row_strides = row_major_strides(padded_row_dims);
    let mut out = vec![0; len];
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
            out[base + col] = value;
        }
    }
    out
}

fn padded_i32_tensor(values: &[i32], shape: &Shape) -> Vec<i32> {
    let padded_dims = shape.padded_power_of_two().0;
    let len = padded_dims.iter().product();
    let mut out = vec![0; len];
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

fn padded_i64_tensor(values: &[i64], shape: &Shape) -> Vec<i64> {
    let padded_dims = shape.padded_power_of_two().0;
    let len = padded_dims.iter().product();
    let mut out = vec![0; len];
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

fn padded_u8_tensor(values: &[u8], shape: &Shape) -> Vec<u8> {
    let padded_dims = shape.padded_power_of_two().0;
    let len = padded_dims.iter().product();
    let mut out = vec![0; len];
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

fn padded_usize_evals(values: &[usize]) -> Vec<u32> {
    values.iter().map(|&value| value as u32).collect()
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

fn softmax_logical_lookup_indices(
    witness: &SoftmaxWitness<'_>,
    params: &SoftmaxParams,
    entries: usize,
) -> Result<Vec<usize>> {
    let mut indices = Vec::with_capacity(params.shape.numel());
    let cols = params.cols();
    for row in 0..params.rows() {
        for col in 0..cols {
            let idx = row * cols + col;
            let diff = if is_valid_position(params.causal, &params.shape, row, col) {
                i64::from(witness.input[idx]) - i64::from(witness.max[row])
            } else {
                0
            };
            let n = i64::from(floor_shift_q8(diff));
            let shifted = n - witness.min_diff;
            if shifted < 0 || shifted >= entries as i64 {
                return Err(ProverError::InvalidSumcheckDomain(entries));
            }
            indices.push(shifted as usize);
        }
    }
    Ok(indices)
}

fn softmax_input_remainders(witness: &SoftmaxWitness<'_>, params: &SoftmaxParams) -> Vec<usize> {
    let mut remainders = Vec::with_capacity(params.shape.numel());
    let cols = params.cols();
    for row in 0..params.rows() {
        for col in 0..cols {
            let idx = row * cols + col;
            let diff = if is_valid_position(params.causal, &params.shape, row, col) {
                i64::from(witness.input[idx]) - i64::from(witness.max[row])
            } else {
                0
            };
            remainders.push(diff.rem_euclid(FIXED_SCALE) as usize);
        }
    }
    remainders
}

fn padded_lookup_indices(
    logical_indices: &[usize],
    params: &SoftmaxParams,
    padding_index: usize,
) -> Vec<usize> {
    let padded_dims = params.shape.padded_power_of_two().0;
    let len = padded_dims.iter().product();
    let mut out = vec![padding_index; len];
    let strides = row_major_strides(params.shape.dims());
    let padded_strides = row_major_strides(&padded_dims);
    for (flat, &value) in logical_indices.iter().enumerate() {
        let mut padded_flat = 0;
        for (dim, (&stride, &padded_stride)) in strides.iter().zip(&padded_strides).enumerate() {
            let coord = (flat / stride) % params.shape.dims()[dim];
            padded_flat += coord * padded_stride;
        }
        out[padded_flat] = value;
    }
    out
}

fn softmax_lookup_values(indices: &[usize], min_diff: i64, table_fn: fn(i64) -> i64) -> Vec<i64> {
    indices
        .iter()
        .map(|&idx| table_fn(min_diff + idx as i64))
        .collect()
}

fn padded_i32_table(
    min_diff: i64,
    entries: usize,
    padded_len: usize,
    table_fn: fn(i64) -> i64,
) -> Result<Vec<i32>> {
    if padded_len < entries || !padded_len.is_power_of_two() {
        return Err(ProverError::InvalidSumcheckDomain(padded_len));
    }
    let mut table = Vec::with_capacity(padded_len);
    for idx in 0..entries {
        let value = table_fn(min_diff + idx as i64);
        let value =
            i32::try_from(value).map_err(|_| ProverError::InvalidSumcheckDomain(entries))?;
        table.push(value);
    }
    table.resize(padded_len, 0);
    Ok(table)
}

fn eval_i32_advice<F: JoltField>(values: &[i32], shape: &Shape, point: &[F]) -> F {
    let eq_by_dim = split_point(shape, point)
        .into_iter()
        .map(EqPolynomial::<F>::evals)
        .collect::<Vec<_>>();
    let strides = row_major_strides(shape.dims());
    values
        .iter()
        .enumerate()
        .fold(F::zero(), |acc, (flat, &value)| {
            let mut weight = F::one();
            for (dim, (&stride, eq)) in strides.iter().zip(&eq_by_dim).enumerate() {
                let coord = (flat / stride) % shape.dims()[dim];
                weight *= eq[coord];
            }
            acc + weight * field_from_i64::<F>(i64::from(value))
        })
}

fn eval_field_tensor<F: JoltField>(values: &[F], shape: &Shape, point: &[F]) -> F {
    let eq_by_dim = split_point(shape, point)
        .into_iter()
        .map(EqPolynomial::<F>::evals)
        .collect::<Vec<_>>();
    let strides = row_major_strides(shape.dims());
    values
        .iter()
        .enumerate()
        .fold(F::zero(), |acc, (flat, &value)| {
            let mut weight = F::one();
            for (dim, (&stride, eq)) in strides.iter().zip(&eq_by_dim).enumerate() {
                let coord = (flat / stride) % shape.dims()[dim];
                weight *= eq[coord];
            }
            acc + weight * value
        })
}

fn masked_eq_poly<F: JoltField>(claim: &Claim<F>, shape: &Shape) -> Vec<F> {
    let padded_dims = claim.domain_shape.dims();
    let mut out = vec![F::zero(); padded_dims.iter().product()];
    let strides = row_major_strides(shape.dims());
    let padded_strides = row_major_strides(padded_dims);
    let eq_evals = EqPolynomial::<F>::evals(&claim.point);
    for flat in 0..shape.numel() {
        let mut padded_flat = 0;
        for (dim, (&stride, &padded_stride)) in strides.iter().zip(&padded_strides).enumerate() {
            let coord = (flat / stride) % shape.dims()[dim];
            padded_flat += coord * padded_stride;
        }
        out[padded_flat] = eq_evals[padded_flat];
    }
    out
}

fn eval_masked_eq<F: JoltField>(claim_point: &[F], shape: &Shape, point: &[F]) -> F {
    let claim_eq_by_dim = split_point(shape, claim_point)
        .into_iter()
        .map(EqPolynomial::<F>::evals)
        .collect::<Vec<_>>();
    let point_eq_by_dim = split_point(shape, point)
        .into_iter()
        .map(EqPolynomial::<F>::evals)
        .collect::<Vec<_>>();
    let combined_eq_by_dim = claim_eq_by_dim
        .iter()
        .zip(&point_eq_by_dim)
        .map(|(lhs, rhs)| {
            lhs.iter()
                .zip(rhs)
                .map(|(lhs, rhs)| *lhs * *rhs)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let strides = row_major_strides(shape.dims());

    let mut sum = F::zero();
    for flat in 0..shape.numel() {
        let mut weight = F::one();
        for (dim, (&stride, eq)) in strides.iter().zip(&combined_eq_by_dim).enumerate() {
            let coord = (flat / stride) % shape.dims()[dim];
            weight *= eq[coord];
        }
        sum += weight;
    }
    sum
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

fn tensor_point_from_full_shape<F: Clone>(point: &[F], shape: &Shape) -> Vec<F> {
    let tensor_vars = shape.padded_power_of_two().point_len();
    point[..tensor_vars].to_vec()
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

fn softmax_shout_sumcheck_id() -> SumcheckId {
    SumcheckId::NodeExecution(2)
}

fn softmax_remainder_shout_sumcheck_id() -> SumcheckId {
    SumcheckId::NodeExecution(3)
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
        let ra = onehot_rows(&[1, 0], entries);
        let exp_frac_bits = frac_bits_for(&exp_acc);
        let floor_frac_bits = frac_bits_for(&acc);
        let frac_bits = frac_bits_for_i32(&floor);
        let witness = SoftmaxWitness {
            input: &input,
            max_index: &max_index,
            max: &max,
            min_diff,
            max_diff,
            ra: &ra,
            exp_acc: &exp_acc,
            exp: &exp,
            exp_frac_bits: exp_frac_bits.each_ref().map(Vec::as_slice),
            sum: &sum,
            acc: &acc,
            floor: &floor,
            floor_frac_bits: floor_frac_bits.each_ref().map(Vec::as_slice),
            output: &output,
            frac_bits: frac_bits.each_ref().map(Vec::as_slice),
        };

        let mut prover_transcript = Blake2bTranscript::default();
        let (proof, input_claim, output_round_ra, floor_round_ra, input_remainder_ra, ra_claim) =
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
            verified_output_round_ra,
            verified_floor_round_ra,
            verified_input_remainder_ra,
            verified_ra,
        ) = verify_softmax_round::<Fr, _>(
            vec![output_claim],
            &proof,
            &params,
            &mut verifier_transcript,
        )
        .unwrap();

        assert_eq!(verified_input_claim, input_claim);
        assert_eq!(verified_output_round_ra, output_round_ra);
        assert_eq!(verified_floor_round_ra, floor_round_ra);
        assert_eq!(verified_input_remainder_ra, input_remainder_ra);
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
        let ra = onehot_rows(&[2, 1, 0], entries);
        let exp_frac_bits = frac_bits_for(&exp_acc);
        let floor_frac_bits = frac_bits_for(&acc);
        let frac_bits = frac_bits_for_i32(&floor);
        let witness = SoftmaxWitness {
            input: &input,
            max_index: &max_index,
            max: &max,
            min_diff,
            max_diff,
            ra: &ra,
            exp_acc: &exp_acc,
            exp: &exp,
            exp_frac_bits: exp_frac_bits.each_ref().map(Vec::as_slice),
            sum: &sum,
            acc: &acc,
            floor: &floor,
            floor_frac_bits: floor_frac_bits.each_ref().map(Vec::as_slice),
            output: &output,
            frac_bits: frac_bits.each_ref().map(Vec::as_slice),
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
        let ra = onehot_rows(&[1, 1, 0, 1], entries);
        let exp_frac_bits = frac_bits_for(&exp_acc);
        let floor_frac_bits = frac_bits_for(&acc);
        let frac_bits = frac_bits_for_i32(&floor);
        let witness = SoftmaxWitness {
            input: &input,
            max_index: &max_index,
            max: &max,
            min_diff,
            max_diff,
            ra: &ra,
            exp_acc: &exp_acc,
            exp: &exp,
            exp_frac_bits: exp_frac_bits.each_ref().map(Vec::as_slice),
            sum: &sum,
            acc: &acc,
            floor: &floor,
            floor_frac_bits: floor_frac_bits.each_ref().map(Vec::as_slice),
            output: &output,
            frac_bits: frac_bits.each_ref().map(Vec::as_slice),
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
        eval_i32_advice(values, shape, point)
    }
}
