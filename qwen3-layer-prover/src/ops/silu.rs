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
    ops::round::ROUND_FRAC_BITS,
};

// Design note for future us:
//
// The current Qwen3 fixed runtime uses the linearized SiLU approximation:
//
//     n      = round(gate / 2^8)
//     f      = gate - n * 2^8
//     base   = (n * 2^8) * sigmoid_lut[n]
//     slope  = sigmoid_lut[n] + round((n * 2^8) * sigmoid'(n) / 2^8)
//     acc    = base + f * slope
//     silu   = round(acc)
//
// This file proves the `acc` relation.  The final `round(acc) -> silu` is
// handled by `silu_round.rs`, which calls the generic round protocol first.
//
// The sumcheck domain is `tensor_domain x LUT_domain`.  Gate values and
// frac-bit tensors are lifted over the LUT axis.  The RA polynomial lives on the
// full domain and selects one LUT row per tensor element.
//
// Range handling should be done through the lookup relation, not a separate
// range check.  Let n_t be the rounded integer part.  If the prover supplies
// min_advice and max_advice, the verifier can build:
//
//     shifted_t = n_t - min_advice
//     id_table  = [0, 1, ..., L - 1]
//     L >= max_advice - min_advice + 1
//     <ra_t, id_table> = shifted_t
//
// This succeeds exactly when the advised range contains every actual n_t:
//
//     min_advice <= actual_min
//     actual_max <= max_advice
//
// The min/max advice does not need to be tight for soundness.  If min_advice is
// too large, some shifted_t is negative and cannot be produced by id_table.  If
// max_advice is too small, some shifted_t is outside the table.  If the range is
// wider than necessary, the proof remains sound but uses a larger LUT domain.

const MAX_SILU_LUT_LEN: usize = 1 << 10;
const FIXED_FRAC_BITS: usize = ROUND_FRAC_BITS;
const FIXED_SCALE: i64 = 1_i64 << FIXED_FRAC_BITS;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SiluParams {
    pub shape: Shape,
    pub gate_proj_round_tensor: TensorId,
    pub output_tensor: TensorId,
    pub frac_bit_tensors: [TensorId; ROUND_FRAC_BITS],
    pub ra_tensor: TensorId,
}

impl SiluParams {
    pub fn new(
        shape: impl Into<Vec<usize>>,
        gate_proj_round_tensor: impl Into<String>,
        output_tensor: impl Into<String>,
        frac_bit_tensors: [String; ROUND_FRAC_BITS],
        ra_tensor: impl Into<String>,
    ) -> Self {
        Self {
            shape: Shape::new(shape),
            gate_proj_round_tensor: TensorId::new(gate_proj_round_tensor),
            output_tensor: TensorId::new(output_tensor),
            frac_bit_tensors: frac_bit_tensors.map(TensorId::new),
            ra_tensor: TensorId::new(ra_tensor),
        }
    }

    fn ra_shape(&self, lut_len: usize) -> Shape {
        let mut dims = self.shape.dims().to_vec();
        dims.push(lut_len);
        Shape::new(dims)
    }
}

#[derive(Debug, Clone, Default)]
pub struct SiluWitness {
    pub min_n: i64,
    pub max_n: i64,
    pub gate_proj_round: Vec<i32>,
    pub frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    pub ra: Vec<u8>,
    pub output: Vec<i64>,
}

#[derive(Debug, Clone)]
pub struct SiluProof<F: JoltField, T: Transcript> {
    pub min_n: i64,
    pub max_n: i64,
    pub sumcheck: SumcheckInstanceProof<F, T>,
    pub gate_opening: F,
    pub frac_bit_openings: [F; ROUND_FRAC_BITS],
    pub ra_opening: F,
}

pub fn prove_silu<F, T>(
    output_claims: Vec<Claim<F>>,
    witness: &SiluWitness,
    params: &SiluParams,
    transcript: &mut T,
) -> Result<(
    SiluProof<F, T>,
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
    let entries = entries_from_min_max(witness.min_n, witness.max_n)?;
    let lut_len = entries.next_power_of_two();
    validate_inputs(&output_claims, witness, params, entries)?;
    append_range_advice::<F, T>(witness.min_n, witness.max_n, transcript);
    eprintln!(
        "timing: prove_silu.setup_validate {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let alphas = transcript.challenge_scalar_powers(output_claims.len());
    let onehot_mix = transcript.challenge_scalar();
    let round_mix = transcript.challenge_scalar();
    let ra_booleanity_mix = transcript.challenge_scalar();
    let bit_booleanity_mix = transcript.challenge_scalar();
    let bit_weights = transcript.challenge_scalar_powers(ROUND_FRAC_BITS);
    eprintln!(
        "timing: prove_silu.challenges {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let mask_eval = eval_logical_mask(&params.shape, &output_claims[0].point);
    let input_claim = batched_input_claim(&output_claims, &alphas) + onehot_mix * mask_eval;
    let eq_batch = lift_tensor_evals(&batched_eq_poly(&output_claims, &alphas), lut_len);
    let eq_onehot = lift_tensor_evals(&EqPolynomial::<F>::evals(&output_claims[0].point), lut_len);
    eprintln!(
        "timing: prove_silu.eq_polys {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let gate = lift_i32_tensor(&witness.gate_proj_round, &params.shape, lut_len);
    let bits =
        std::array::from_fn(|idx| lift_u8_tensor(&witness.frac_bits[idx], &params.shape, lut_len));
    let ra = padded_ra_tensor(&witness.ra, params, entries);
    eprintln!(
        "timing: prove_silu.witness_polys {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let id = silu_table_tensor(params, entries, &silu_id_table::<F>(entries));
    let base = silu_table_tensor(
        params,
        entries,
        &silu_base_table::<F>(witness.min_n, entries),
    );
    let slope = silu_table_tensor(
        params,
        entries,
        &silu_slope_table::<F>(witness.min_n, entries),
    );
    eprintln!(
        "timing: prove_silu.table_polys {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let sc_params = SiluSumcheckParams::new(
        params.ra_shape(entries).padded_power_of_two().point_len(),
        input_claim,
    );
    let mut prover = SiluSumcheckProver::new(
        sc_params,
        eq_batch,
        eq_onehot,
        gate,
        bits,
        ra,
        id,
        base,
        slope,
        onehot_mix,
        round_mix,
        ra_booleanity_mix,
        bit_booleanity_mix,
        bit_weights,
        field_from_i64(witness.min_n),
    );
    eprintln!(
        "timing: prove_silu.prover_init {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let mut accumulator = ProverOpeningAccumulator::new();
    let (sumcheck, challenges) = Sumcheck::prove(&mut prover, &mut accumulator, transcript);
    eprintln!(
        "timing: prove_silu.sumcheck {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let gate_opening = prover_opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenSiluGate, silu_sumcheck_id()),
    )?;
    let frac_bit_openings = std::array::from_fn(|idx| {
        prover_opening(
            &accumulator,
            OpeningId::new(VirtualPoly::QwenSiluFracBit(idx), silu_sumcheck_id()),
        )
        .expect("SiLU frac-bit opening must be produced")
    });
    let ra_opening = prover_opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenSiluRa, silu_sumcheck_id()),
    )?;
    let full_point = normalize_sumcheck_point::<F>(&challenges.into_opening());
    let tensor_point = tensor_point_from_full(&full_point, params);
    eprintln!(
        "timing: prove_silu.openings_claims {:.3}s",
        step_start.elapsed().as_secs_f64()
    );
    eprintln!(
        "timing: prove_silu.total {:.3}s",
        total_start.elapsed().as_secs_f64()
    );

    Ok((
        SiluProof {
            min_n: witness.min_n,
            max_n: witness.max_n,
            sumcheck,
            gate_opening,
            frac_bit_openings,
            ra_opening,
        },
        Claim {
            tensor: params.gate_proj_round_tensor.clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: tensor_point.clone(),
            value: gate_opening,
        },
        std::array::from_fn(|idx| Claim {
            tensor: params.frac_bit_tensors[idx].clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: tensor_point.clone(),
            value: frac_bit_openings[idx],
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

pub fn verify_silu<F, T>(
    output_claims: Vec<Claim<F>>,
    proof: &SiluProof<F, T>,
    params: &SiluParams,
    transcript: &mut T,
) -> std::result::Result<(Claim<F>, [Claim<F>; ROUND_FRAC_BITS], Claim<F>), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    let entries = entries_from_min_max(proof.min_n, proof.max_n)
        .map_err(|_| ProofVerifyError::InvalidInputLength(MAX_SILU_LUT_LEN, 0))?;
    verify_inputs(&output_claims, params, entries)?;
    append_range_advice::<F, T>(proof.min_n, proof.max_n, transcript);

    let alphas = transcript.challenge_scalar_powers(output_claims.len());
    let onehot_mix = transcript.challenge_scalar();
    let round_mix = transcript.challenge_scalar();
    let ra_booleanity_mix = transcript.challenge_scalar();
    let bit_booleanity_mix = transcript.challenge_scalar();
    let bit_weights = transcript.challenge_scalar_powers(ROUND_FRAC_BITS);
    let mask_eval = eval_logical_mask(&params.shape, &output_claims[0].point);
    let input_claim = batched_input_claim(&output_claims, &alphas) + onehot_mix * mask_eval;
    let verifier = SiluSumcheckVerifier {
        params: SiluSumcheckParams::new(
            params.ra_shape(entries).padded_power_of_two().point_len(),
            input_claim,
        ),
        output_points: output_claims
            .iter()
            .map(|claim| claim.point.clone())
            .collect(),
        alphas,
        onehot_point: output_claims[0].point.clone(),
        onehot_mix,
        round_mix,
        ra_booleanity_mix,
        bit_booleanity_mix,
        bit_weights,
        shape: params.shape.clone(),
        min_n: proof.min_n,
        entries,
    };
    let mut accumulator = VerifierOpeningAccumulator::new();
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenSiluGate, silu_sumcheck_id()),
        (OpeningPoint::<BIG_ENDIAN, F>::default(), proof.gate_opening),
    );
    for idx in 0..ROUND_FRAC_BITS {
        accumulator.openings.insert(
            OpeningId::new(VirtualPoly::QwenSiluFracBit(idx), silu_sumcheck_id()),
            (
                OpeningPoint::<BIG_ENDIAN, F>::default(),
                proof.frac_bit_openings[idx],
            ),
        );
    }
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenSiluRa, silu_sumcheck_id()),
        (OpeningPoint::<BIG_ENDIAN, F>::default(), proof.ra_opening),
    );
    let challenges = Sumcheck::verify(&proof.sumcheck, &verifier, &mut accumulator, transcript)?;
    let full_point = normalize_sumcheck_point::<F>(&challenges.into_opening());
    let tensor_point = tensor_point_from_full(&full_point, params);

    Ok((
        Claim {
            tensor: params.gate_proj_round_tensor.clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: tensor_point.clone(),
            value: proof.gate_opening,
        },
        std::array::from_fn(|idx| Claim {
            tensor: params.frac_bit_tensors[idx].clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: tensor_point.clone(),
            value: proof.frac_bit_openings[idx],
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

struct SiluSumcheckParams<F: JoltField> {
    num_rounds: usize,
    input_claim: F,
}

impl<F: JoltField> SiluSumcheckParams<F> {
    fn new(num_rounds: usize, input_claim: F) -> Self {
        Self {
            num_rounds,
            input_claim,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for SiluSumcheckParams<F> {
    fn degree(&self) -> usize {
        3
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

struct SiluSumcheckProver<F: JoltField> {
    eq_batch: MultilinearPolynomial<F>,
    eq_onehot: MultilinearPolynomial<F>,
    gate: MultilinearPolynomial<F>,
    bits: [MultilinearPolynomial<F>; ROUND_FRAC_BITS],
    ra: MultilinearPolynomial<F>,
    id: MultilinearPolynomial<F>,
    base: MultilinearPolynomial<F>,
    slope: MultilinearPolynomial<F>,
    onehot_mix: F,
    round_mix: F,
    ra_booleanity_mix: F,
    bit_booleanity_mix: F,
    bit_weights: Vec<F>,
    min_n: F,
    params: SiluSumcheckParams<F>,
}

impl<F: JoltField> SiluSumcheckProver<F> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        params: SiluSumcheckParams<F>,
        eq_batch: Vec<F>,
        eq_onehot: Vec<F>,
        gate: Vec<F>,
        bits: [Vec<F>; ROUND_FRAC_BITS],
        ra: Vec<F>,
        id: Vec<F>,
        base: Vec<F>,
        slope: Vec<F>,
        onehot_mix: F,
        round_mix: F,
        ra_booleanity_mix: F,
        bit_booleanity_mix: F,
        bit_weights: Vec<F>,
        min_n: F,
    ) -> Self {
        Self {
            eq_batch: MultilinearPolynomial::from(eq_batch),
            eq_onehot: MultilinearPolynomial::from(eq_onehot),
            gate: MultilinearPolynomial::from(gate),
            bits: bits.map(MultilinearPolynomial::from),
            ra: MultilinearPolynomial::from(ra),
            id: MultilinearPolynomial::from(id),
            base: MultilinearPolynomial::from(base),
            slope: MultilinearPolynomial::from(slope),
            onehot_mix,
            round_mix,
            ra_booleanity_mix,
            bit_booleanity_mix,
            bit_weights,
            min_n,
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for SiluSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let mut evals = [F::zero(); 3];
        for g in 0..self.ra.len() / 2 {
            let values = SiluPairValues::from_prover(self, g);
            for (idx, t) in [F::zero(), F::from_u64(2), F::from_u64(3)]
                .into_iter()
                .enumerate()
            {
                evals[idx] += values.eval(t, self);
            }
        }
        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_batch.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.eq_onehot.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.gate.bind_parallel(r_j, BindingOrder::LowToHigh);
        for bit in &mut self.bits {
            bit.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        self.ra.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.id.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.base.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.slope.bind_parallel(r_j, BindingOrder::LowToHigh);
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
            OpeningId::new(VirtualPoly::QwenSiluGate, silu_sumcheck_id()),
            point.clone(),
            self.gate.final_claim(),
        );
        for idx in 0..ROUND_FRAC_BITS {
            accumulator.append_virtual(
                transcript,
                OpeningId::new(VirtualPoly::QwenSiluFracBit(idx), silu_sumcheck_id()),
                point.clone(),
                self.bits[idx].final_claim(),
            );
        }
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenSiluRa, silu_sumcheck_id()),
            point,
            self.ra.final_claim(),
        );
    }
}

struct SiluSumcheckVerifier<F: JoltField> {
    params: SiluSumcheckParams<F>,
    output_points: Vec<Vec<F>>,
    alphas: Vec<F>,
    onehot_point: Vec<F>,
    onehot_mix: F,
    round_mix: F,
    ra_booleanity_mix: F,
    bit_booleanity_mix: F,
    bit_weights: Vec<F>,
    shape: Shape,
    min_n: i64,
    entries: usize,
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for SiluSumcheckVerifier<F> {
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
        let gate = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenSiluGate,
                silu_sumcheck_id(),
            ))
            .1;
        let bits = std::array::from_fn(|idx| {
            accumulator
                .get_virtual_polynomial_opening(OpeningId::new(
                    VirtualPoly::QwenSiluFracBit(idx),
                    silu_sumcheck_id(),
                ))
                .1
        });
        let ra = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenSiluRa,
                silu_sumcheck_id(),
            ))
            .1;
        let eq_batch = self
            .output_points
            .iter()
            .zip(&self.alphas)
            .fold(F::zero(), |acc, (point, alpha)| {
                acc + *alpha * EqPolynomial::mle(point, &tensor_point)
            });
        let eq_onehot = EqPolynomial::mle(&self.onehot_point, &tensor_point);
        let id = eval_table_at_point(&padded_table(&silu_id_table::<F>(self.entries)), lut_point);
        let base = eval_table_at_point(
            &padded_table(&silu_base_table::<F>(self.min_n, self.entries)),
            lut_point,
        );
        let slope = eval_table_at_point(
            &padded_table(&silu_slope_table::<F>(self.min_n, self.entries)),
            lut_point,
        );
        eval_silu_relation(
            eq_batch,
            eq_onehot,
            gate,
            bits,
            ra,
            id,
            base,
            slope,
            self.onehot_mix,
            self.round_mix,
            self.ra_booleanity_mix,
            self.bit_booleanity_mix,
            &self.bit_weights,
            field_from_i64(self.min_n),
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
            OpeningId::new(VirtualPoly::QwenSiluGate, silu_sumcheck_id()),
            point.clone(),
        );
        for idx in 0..ROUND_FRAC_BITS {
            accumulator.append_virtual(
                transcript,
                OpeningId::new(VirtualPoly::QwenSiluFracBit(idx), silu_sumcheck_id()),
                point.clone(),
            );
        }
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenSiluRa, silu_sumcheck_id()),
            point,
        );
    }
}

struct SiluPairValues<F: JoltField> {
    eq_batch: [F; 2],
    eq_onehot: [F; 2],
    gate: [F; 2],
    bits: [[F; 2]; ROUND_FRAC_BITS],
    ra: [F; 2],
    id: [F; 2],
    base: [F; 2],
    slope: [F; 2],
}

impl<F: JoltField> SiluPairValues<F> {
    fn from_prover(prover: &SiluSumcheckProver<F>, g: usize) -> Self {
        Self {
            eq_batch: [
                prover.eq_batch.get_bound_coeff(2 * g),
                prover.eq_batch.get_bound_coeff(2 * g + 1),
            ],
            eq_onehot: [
                prover.eq_onehot.get_bound_coeff(2 * g),
                prover.eq_onehot.get_bound_coeff(2 * g + 1),
            ],
            gate: [
                prover.gate.get_bound_coeff(2 * g),
                prover.gate.get_bound_coeff(2 * g + 1),
            ],
            bits: std::array::from_fn(|idx| {
                [
                    prover.bits[idx].get_bound_coeff(2 * g),
                    prover.bits[idx].get_bound_coeff(2 * g + 1),
                ]
            }),
            ra: [
                prover.ra.get_bound_coeff(2 * g),
                prover.ra.get_bound_coeff(2 * g + 1),
            ],
            id: [
                prover.id.get_bound_coeff(2 * g),
                prover.id.get_bound_coeff(2 * g + 1),
            ],
            base: [
                prover.base.get_bound_coeff(2 * g),
                prover.base.get_bound_coeff(2 * g + 1),
            ],
            slope: [
                prover.slope.get_bound_coeff(2 * g),
                prover.slope.get_bound_coeff(2 * g + 1),
            ],
        }
    }

    fn eval(&self, t: F, prover: &SiluSumcheckProver<F>) -> F {
        eval_silu_relation(
            lerp(self.eq_batch[0], self.eq_batch[1], t),
            lerp(self.eq_onehot[0], self.eq_onehot[1], t),
            lerp(self.gate[0], self.gate[1], t),
            std::array::from_fn(|idx| lerp(self.bits[idx][0], self.bits[idx][1], t)),
            lerp(self.ra[0], self.ra[1], t),
            lerp(self.id[0], self.id[1], t),
            lerp(self.base[0], self.base[1], t),
            lerp(self.slope[0], self.slope[1], t),
            prover.onehot_mix,
            prover.round_mix,
            prover.ra_booleanity_mix,
            prover.bit_booleanity_mix,
            &prover.bit_weights,
            prover.min_n,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn eval_silu_relation<F: JoltField>(
    eq_batch: F,
    eq_onehot: F,
    gate: F,
    bits: [F; ROUND_FRAC_BITS],
    ra: F,
    id: F,
    base: F,
    slope: F,
    onehot_mix: F,
    round_mix: F,
    ra_booleanity_mix: F,
    bit_booleanity_mix: F,
    bit_weights: &[F],
    min_n: F,
) -> F {
    let frac = bits.iter().enumerate().fold(F::zero(), |acc, (idx, bit)| {
        acc + *bit * F::from_u64(1_u64 << idx)
    });
    let bit7 = bits[ROUND_FRAC_BITS - 1];
    let scale = F::from_u64(FIXED_SCALE as u64);
    let n = min_n + id;
    let acc_expr = base + (gate - n * scale) * slope;
    let round_expr = gate + bit7 * scale - frac - n * scale;
    let bit_booleanity = bits
        .iter()
        .zip(bit_weights)
        .fold(F::zero(), |acc, (bit, weight)| {
            acc + *weight * *bit * (*bit - F::one())
        });

    eq_batch * ra * acc_expr
        + onehot_mix * eq_onehot * ra
        + round_mix * eq_batch * ra * round_expr
        + ra_booleanity_mix * ra * (ra - F::one())
        + bit_booleanity_mix * bit_booleanity
}

fn validate_inputs<F: JoltField>(
    output_claims: &[Claim<F>],
    witness: &SiluWitness,
    params: &SiluParams,
    entries: usize,
) -> Result<()> {
    if output_claims.is_empty() {
        return Err(ProverError::InvalidClaimCount {
            name: "SiLU output claims",
            expected: 1,
            actual: 0,
        });
    }
    if entries == 0 || entries > MAX_SILU_LUT_LEN {
        return Err(ProverError::InvalidSumcheckDomain(entries));
    }
    let len = params.shape.dims().iter().product();
    ensure_len(
        "SiLU gate",
        &params.shape,
        len,
        witness.gate_proj_round.len(),
    )?;
    ensure_len("SiLU output", &params.shape, len, witness.output.len())?;
    ensure_len(
        "SiLU ra",
        &params.ra_shape(entries),
        len * entries,
        witness.ra.len(),
    )?;
    for claim in output_claims {
        if claim.logical_shape != params.shape {
            return Err(ProverError::ShapeMismatch {
                name: "SiLU output claim",
                expected: params.shape.0.clone(),
                actual: claim.logical_shape.0.clone(),
            });
        }
    }
    for bit in 0..ROUND_FRAC_BITS {
        ensure_len(
            "SiLU frac bit",
            &params.shape,
            len,
            witness.frac_bits[bit].len(),
        )?;
        for (idx, &value) in witness.frac_bits[bit].iter().enumerate() {
            if value > 1 {
                return Err(ProverError::BitNotBoolean {
                    bit,
                    index: idx,
                    value,
                });
            }
        }
    }
    for (idx, &value) in witness.ra.iter().enumerate() {
        if value > 1 {
            return Err(ProverError::BitNotBoolean {
                bit: 0,
                index: idx,
                value,
            });
        }
    }
    for element in 0..len {
        let row = &witness.ra[element * entries..(element + 1) * entries];
        let Some(selected) = row.iter().position(|&value| value == 1) else {
            return Err(ProverError::InvalidClaimCount {
                name: "SiLU RA row",
                expected: 1,
                actual: 0,
            });
        };
        if row.iter().filter(|&&value| value == 1).count() != 1 {
            return Err(ProverError::InvalidClaimCount {
                name: "SiLU RA row",
                expected: 1,
                actual: row.iter().filter(|&&value| value == 1).count(),
            });
        }
        let frac = frac_value(&witness.frac_bits, element);
        let n = witness.min_n + selected as i64;
        let gate = i64::from(witness.gate_proj_round[element]);
        let expected_round = n * FIXED_SCALE;
        let actual_round =
            gate + i64::from(witness.frac_bits[ROUND_FRAC_BITS - 1][element]) * FIXED_SCALE - frac;
        if expected_round != actual_round {
            return Err(ProverError::MatMulAccumulatorMismatch {
                row: element,
                col: 0,
                expected: expected_round,
                actual: actual_round,
            });
        }
        let expected = silu_base(n) + (gate - n * FIXED_SCALE) * silu_slope(n);
        if witness.output[element] != expected {
            return Err(ProverError::MatMulAccumulatorMismatch {
                row: element,
                col: 0,
                expected,
                actual: witness.output[element],
            });
        }
    }
    Ok(())
}

fn verify_inputs<F: JoltField>(
    output_claims: &[Claim<F>],
    params: &SiluParams,
    entries: usize,
) -> std::result::Result<(), ProofVerifyError> {
    if entries == 0 || entries > MAX_SILU_LUT_LEN {
        return Err(ProofVerifyError::InvalidInputLength(
            MAX_SILU_LUT_LEN,
            entries,
        ));
    }
    if output_claims.is_empty() {
        return Err(ProofVerifyError::InvalidInputLength(1, 0));
    }
    for claim in output_claims {
        if claim.logical_shape != params.shape {
            return Err(ProofVerifyError::InvalidInputLength(
                params.shape.dims().iter().product(),
                claim.logical_shape.dims().iter().product(),
            ));
        }
    }
    Ok(())
}

fn entries_from_min_max(min_n: i64, max_n: i64) -> Result<usize> {
    if max_n < min_n {
        return Err(ProverError::InvalidSumcheckDomain(0));
    }
    let needed = (max_n - min_n + 1) as usize;
    if needed == 0 || needed > MAX_SILU_LUT_LEN {
        return Err(ProverError::InvalidSumcheckDomain(needed));
    }
    Ok(needed)
}

fn append_range_advice<F: JoltField, T: Transcript>(min_n: i64, max_n: i64, transcript: &mut T) {
    transcript.append_scalar(&field_from_i64::<F>(min_n));
    transcript.append_scalar(&field_from_i64::<F>(max_n));
}

fn batched_input_claim<F: JoltField>(claims: &[Claim<F>], alphas: &[F]) -> F {
    claims
        .iter()
        .zip(alphas)
        .fold(F::zero(), |acc, (claim, alpha)| acc + *alpha * claim.value)
}

fn batched_eq_poly<F: JoltField>(claims: &[Claim<F>], alphas: &[F]) -> Vec<F> {
    let mut out = vec![F::zero(); claims[0].domain_shape.dims().iter().product()];
    for (claim, alpha) in claims.iter().zip(alphas) {
        for (dst, value) in out.iter_mut().zip(EqPolynomial::<F>::evals(&claim.point)) {
            *dst += *alpha * value;
        }
    }
    out
}

fn lift_tensor_evals<F: JoltField>(values: &[F], table_len: usize) -> Vec<F> {
    let mut out = Vec::with_capacity(values.len() * table_len.next_power_of_two());
    let padded_table_len = table_len.next_power_of_two();
    for &value in values {
        out.extend(std::iter::repeat_n(value, padded_table_len));
    }
    out
}

fn lift_i32_tensor<F: JoltField>(values: &[i32], shape: &Shape, table_len: usize) -> Vec<F> {
    lift_tensor_evals(&padded_i32_tensor(values, shape), table_len)
}

fn lift_u8_tensor<F: JoltField>(values: &[u8], shape: &Shape, table_len: usize) -> Vec<F> {
    lift_tensor_evals(&padded_u8_tensor(values, shape), table_len)
}

fn padded_ra_tensor<F: JoltField>(values: &[u8], params: &SiluParams, entries: usize) -> Vec<F> {
    let shape = params.ra_shape(entries);
    let padded_dims = shape.padded_power_of_two().0;
    let len = padded_dims.iter().product();
    let mut out = vec![F::zero(); len];
    let logical = shape.dims();
    let strides = row_major_strides(logical);
    let padded_strides = row_major_strides(&padded_dims);
    for (flat, &value) in values.iter().enumerate() {
        let mut padded_flat = 0;
        for (dim, (&stride, &padded_stride)) in strides.iter().zip(&padded_strides).enumerate() {
            let coord = (flat / stride) % logical[dim];
            padded_flat += coord * padded_stride;
        }
        out[padded_flat] = F::from_u64(u64::from(value));
    }
    out
}

fn silu_table_tensor<F: JoltField>(params: &SiluParams, entries: usize, table: &[F]) -> Vec<F> {
    let tensor_len = params.shape.padded_power_of_two().dims().iter().product();
    let padded = padded_table(table);
    let padded_table_len = entries.next_power_of_two();
    let mut out = Vec::with_capacity(tensor_len * padded_table_len);
    for _ in 0..tensor_len {
        out.extend(padded.iter().copied());
    }
    out
}

fn padded_table<F: JoltField>(table: &[F]) -> Vec<F> {
    let mut out = vec![F::zero(); table.len().next_power_of_two()];
    out[..table.len()].copy_from_slice(table);
    out
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

fn eval_table_at_point<F: JoltField>(table: &[F], point: &[F]) -> F {
    EqPolynomial::<F>::evals(point)
        .into_iter()
        .zip(table)
        .fold(F::zero(), |acc, (eq, value)| acc + eq * *value)
}

fn silu_id_table<F: JoltField>(lut_len: usize) -> Vec<F> {
    (0..lut_len).map(|idx| field_from_i64(idx as i64)).collect()
}

fn silu_base_table<F: JoltField>(min_n: i64, lut_len: usize) -> Vec<F> {
    (0..lut_len)
        .map(|idx| field_from_i64(silu_base(min_n + idx as i64)))
        .collect()
}

fn silu_slope_table<F: JoltField>(min_n: i64, lut_len: usize) -> Vec<F> {
    (0..lut_len)
        .map(|idx| field_from_i64(silu_slope(min_n + idx as i64)))
        .collect()
}

fn silu_base(n: i64) -> i64 {
    let n_q8 = n * FIXED_SCALE;
    n_q8 * sigmoid_q8(n)
}

fn silu_slope(n: i64) -> i64 {
    let n_q8 = n * FIXED_SCALE;
    let sig = sigmoid_q8(n);
    let sig_slope = round_shift_signed_i64(sig * (FIXED_SCALE - sig), FIXED_FRAC_BITS);
    sig + round_shift_signed_i64(n_q8 * sig_slope, FIXED_FRAC_BITS)
}

fn sigmoid_q8(n: i64) -> i64 {
    let sig = 1.0 / (1.0 + (-(n as f64)).exp());
    (sig * FIXED_SCALE as f64).round() as i64
}

fn frac_value(frac_bits: &[Vec<u8>; ROUND_FRAC_BITS], idx: usize) -> i64 {
    frac_bits
        .iter()
        .enumerate()
        .fold(0_i64, |acc, (bit, values)| {
            acc + (i64::from(values[idx]) << bit)
        })
}

fn round_shift_signed_i64(x: i64, shift: usize) -> i64 {
    let q = floor_shift_signed_i64(x, shift);
    let denom = 1_i64 << shift;
    let r = x - (q << shift);
    if r * 2 >= denom { q + 1 } else { q }
}

fn floor_shift_signed_i64(x: i64, shift: usize) -> i64 {
    x.div_euclid(1_i64 << shift)
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

fn tensor_point_from_full<F: JoltField>(point: &[F], params: &SiluParams) -> Vec<F> {
    tensor_point_from_full_shape(point, &params.shape)
}

fn tensor_point_from_full_shape<F: Clone>(point: &[F], shape: &Shape) -> Vec<F> {
    let tensor_vars = shape.padded_power_of_two().point_len();
    point[..tensor_vars].to_vec()
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

fn silu_sumcheck_id() -> SumcheckId {
    SumcheckId::NodeExecution(0)
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use joltworks::{
        field::JoltField, poly::eq_poly::EqPolynomial, transcripts::Blake2bTranscript,
    };

    use super::*;

    #[test]
    fn proves_and_verifies_silu() {
        let params = SiluParams::new(
            vec![2],
            "gate",
            "silu_acc",
            std::array::from_fn(|idx| format!("silu_frac_bit_{idx}")),
            "silu_ra",
        );
        let gate = vec![256, -128];
        let min_n = 0;
        let max_n = 1;
        let entries = entries_from_min_max(min_n, max_n).unwrap();
        let mut ra = vec![0; gate.len() * entries];
        let selected = [1, 0];
        for (row, &idx) in selected.iter().enumerate() {
            ra[row * entries + idx] = 1;
        }
        let frac_bits = frac_bits_for(&gate);
        let output = gate
            .iter()
            .zip(selected)
            .map(|(&gate, idx)| {
                let n = min_n + idx as i64;
                silu_base(n) + (i64::from(gate) - n * FIXED_SCALE) * silu_slope(n)
            })
            .collect::<Vec<_>>();
        let point = vec![Fr::from(7u64)];
        let output_claim = Claim {
            tensor: TensorId::new("silu_acc"),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: point.clone(),
            value: eval_i64(&output, &params.shape, &point),
        };
        let witness = SiluWitness {
            min_n,
            max_n,
            gate_proj_round: gate,
            frac_bits,
            ra,
            output,
        };

        let mut prover_transcript = Blake2bTranscript::default();
        let (proof, gate_claim, frac_bit_claims, ra_claim) = prove_silu::<Fr, _>(
            vec![output_claim.clone()],
            &witness,
            &params,
            &mut prover_transcript,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        let (verified_gate_claim, verified_frac_bit_claims, verified_ra_claim) =
            verify_silu::<Fr, _>(
                vec![output_claim],
                &proof,
                &params,
                &mut verifier_transcript,
            )
            .unwrap();

        assert_eq!(verified_gate_claim, gate_claim);
        assert_eq!(verified_frac_bit_claims, frac_bit_claims);
        assert_eq!(verified_ra_claim, ra_claim);
        assert_eq!(verified_gate_claim.tensor.0, "gate");
    }

    #[test]
    fn proves_silu_with_non_power_of_two_logical_range() {
        let params = SiluParams::new(
            vec![1],
            "gate",
            "silu_acc",
            std::array::from_fn(|idx| format!("silu_frac_bit_{idx}")),
            "silu_ra",
        );
        let gate = vec![512];
        let min_n = 0;
        let max_n = 2;
        let entries = entries_from_min_max(min_n, max_n).unwrap();
        assert_eq!(entries, 3);
        assert_eq!(entries.next_power_of_two(), 4);
        let selected = 2;
        let mut ra = vec![0; entries];
        ra[selected] = 1;
        let frac_bits = frac_bits_for(&gate);
        let n = min_n + selected as i64;
        let output = vec![silu_base(n) + (i64::from(gate[0]) - n * FIXED_SCALE) * silu_slope(n)];
        let point = vec![];
        let output_claim = Claim {
            tensor: TensorId::new("silu_acc"),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: point.clone(),
            value: eval_i64(&output, &params.shape, &point),
        };
        let witness = SiluWitness {
            min_n,
            max_n,
            gate_proj_round: gate,
            frac_bits,
            ra,
            output,
        };

        let mut prover_transcript = Blake2bTranscript::default();
        let (proof, _, _, _) = prove_silu::<Fr, _>(
            vec![output_claim.clone()],
            &witness,
            &params,
            &mut prover_transcript,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        verify_silu::<Fr, _>(
            vec![output_claim],
            &proof,
            &params,
            &mut verifier_transcript,
        )
        .unwrap();
    }

    #[test]
    fn rejects_silu_ra_padding_row_for_non_power_of_two_range() {
        let params = SiluParams::new(
            vec![1],
            "gate",
            "silu_acc",
            std::array::from_fn(|idx| format!("silu_frac_bit_{idx}")),
            "silu_ra",
        );
        let gate = vec![768];
        let min_n = 0;
        let max_n = 2;
        let entries = entries_from_min_max(min_n, max_n).unwrap();
        let mut ra = vec![0; entries.next_power_of_two()];
        ra[3] = 1;
        let frac_bits = frac_bits_for(&gate);
        let output = vec![0];
        let point = vec![];
        let output_claim = Claim {
            tensor: TensorId::new("silu_acc"),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point,
            value: Fr::from(0u64),
        };
        let witness = SiluWitness {
            min_n,
            max_n,
            gate_proj_round: gate,
            frac_bits,
            ra,
            output,
        };

        let mut prover_transcript = Blake2bTranscript::default();
        let err = match prove_silu::<Fr, _>(
            vec![output_claim],
            &witness,
            &params,
            &mut prover_transcript,
        ) {
            Ok(_) => panic!("padding RA row must be rejected"),
            Err(err) => err,
        };
        assert!(matches!(err, ProverError::TensorLenMismatch { .. }));
    }

    fn frac_bits_for(values: &[i32]) -> [Vec<u8>; ROUND_FRAC_BITS] {
        std::array::from_fn(|bit| {
            values
                .iter()
                .map(|&value| ((i64::from(value).rem_euclid(FIXED_SCALE) >> bit) & 1) as u8)
                .collect()
        })
    }

    fn eval_i64<F: JoltField>(values: &[i64], shape: &Shape, point: &[F]) -> F {
        let eq_by_dim = split_point(shape, point)
            .into_iter()
            .map(EqPolynomial::<F>::evals)
            .collect::<Vec<_>>();
        let strides = row_major_strides(shape.dims());
        let mut out = F::zero();
        for (flat, &value) in values.iter().enumerate() {
            let mut weight = F::one();
            for (dim, (&stride, eq)) in strides.iter().zip(&eq_by_dim).enumerate() {
                let coord = (flat / stride) % shape.dims()[dim];
                weight *= eq[coord];
            }
            out += weight * field_from_i64::<F>(value);
        }
        out
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
}
