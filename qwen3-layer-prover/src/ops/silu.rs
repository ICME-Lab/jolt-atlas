use std::time::Instant;

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
// The main relation sumcheck now runs only over the tensor domain.  It proves
// the linearized SiLU arithmetic and exposes two virtual lookup claims:
// `base(n)` and `slope(n)`, plus the integer table index.  Those lookups are
// proven separately with JoltWorks Shout, avoiding the old tensor x LUT
// one-hot expansion that was too slow for Qwen-sized tensors.
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
    pub round_ra_tensor: TensorId,
    pub ra_tensor: TensorId,
}

impl SiluParams {
    pub fn new(
        shape: impl Into<Vec<usize>>,
        gate_proj_round_tensor: impl Into<String>,
        output_tensor: impl Into<String>,
        ra_tensor: impl Into<String>,
    ) -> Self {
        let gate_proj_round_tensor = gate_proj_round_tensor.into();
        Self {
            shape: Shape::new(shape),
            gate_proj_round_tensor: TensorId::new(gate_proj_round_tensor.clone()),
            output_tensor: TensorId::new(output_tensor),
            round_ra_tensor: TensorId::new(format!("{gate_proj_round_tensor}_round_ra")),
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
    pub ra: Vec<u8>,
    pub output: Vec<i64>,
}

#[derive(Debug, Clone)]
pub struct SiluProof<F: JoltField, T: Transcript> {
    pub min_n: i64,
    pub max_n: i64,
    pub relation: SumcheckInstanceProof<F, T>,
    pub base_lookup: SumcheckInstanceProof<F, T>,
    pub base_ra_onehot: SumcheckInstanceProof<F, T>,
    pub slope_lookup: SumcheckInstanceProof<F, T>,
    pub slope_ra_onehot: SumcheckInstanceProof<F, T>,
    pub round_lookup: SumcheckInstanceProof<F, T>,
    pub round_ra_onehot: SumcheckInstanceProof<F, T>,
    pub gate_opening: F,
    pub remainder_opening: F,
    pub round_bit_opening: F,
    pub base_opening: F,
    pub slope_opening: F,
    pub index_opening: F,
    pub base_ra_opening: F,
    pub slope_ra_opening: F,
    pub round_ra_opening: F,
    pub base_ra_committed_openings: SiluRaCommittedOpenings<F>,
    pub slope_ra_committed_openings: SiluRaCommittedOpenings<F>,
    pub round_ra_committed_openings: SiluRaCommittedOpenings<F>,
}

#[derive(Debug, Clone, Default)]
pub struct SiluRaCommittedOpenings<F: JoltField> {
    pub ra_virtual: Vec<F>,
    pub hamming_weight: Vec<F>,
    pub booleanity: Vec<F>,
}

pub fn prove_silu<F, T>(
    output_claims: Vec<Claim<F>>,
    witness: &SiluWitness,
    params: &SiluParams,
    transcript: &mut T,
) -> Result<(
    SiluProof<F, T>,
    Claim<F>,
    Claim<F>,
    Claim<F>,
)>
where
    F: JoltField,
    T: Transcript,
{
    let total_start = Instant::now();
    let mut step_start = Instant::now();
    let entries = entries_from_min_max(witness.min_n, witness.max_n)?;
    let lut_len = padded_silu_lut_len(entries);
    validate_inputs(&output_claims, witness, params, entries)?;
    append_range_advice::<F, T>(witness.min_n, witness.max_n, transcript);
    eprintln!(
        "timing: prove_silu.setup_validate {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let alphas = transcript.challenge_scalar_powers(output_claims.len());
    let round_mix = transcript.challenge_scalar();
    eprintln!(
        "timing: prove_silu.challenges {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let input_claim = batched_input_claim(&output_claims, &alphas);
    let eq_batch = batched_masked_eq_poly(&output_claims, &alphas, &params.shape);
    eprintln!(
        "timing: prove_silu.eq_polys {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let logical_lookup_indices = silu_logical_lookup_indices(witness, params, entries)?;
    let padded_lookup_indices = padded_lookup_indices(&logical_lookup_indices, params, entries);
    let base_values = silu_lookup_values(&logical_lookup_indices, witness.min_n, silu_base);
    let slope_values = silu_lookup_values(&logical_lookup_indices, witness.min_n, silu_slope);
    let gate = padded_i32_tensor(&witness.gate_proj_round, &params.shape);
    let remainders = round_remainders(&witness.gate_proj_round);
    let round_bits = round_bits_from_remainders(&remainders);
    let remainder = padded_usize_tensor(&remainders, &params.shape);
    let round_bit = padded_i32_tensor(&round_bits, &params.shape);
    let base = padded_i64_tensor(&base_values, &params.shape);
    let slope = padded_i64_tensor(&slope_values, &params.shape);
    let index = padded_usize_evals(&padded_lookup_indices);
    eprintln!(
        "timing: prove_silu.witness_polys {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let base_table = padded_i32_table(witness.min_n, entries, lut_len, silu_base)?;
    let slope_table = padded_i32_table(witness.min_n, entries, lut_len, silu_slope)?;
    eprintln!(
        "timing: prove_silu.table_polys {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let sc_params =
        SiluSumcheckParams::new(params.shape.padded_power_of_two().point_len(), input_claim);
    let mut prover = SiluSumcheckProver::new(
        sc_params,
        eq_batch,
        gate,
        remainder,
        round_bit,
        base,
        slope,
        index,
        round_mix,
        field_from_i64(witness.min_n),
    );
    eprintln!(
        "timing: prove_silu.prover_init {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let mut accumulator = ProverOpeningAccumulator::new();
    let (relation, challenges) = Sumcheck::prove(&mut prover, &mut accumulator, transcript);
    eprintln!(
        "timing: prove_silu.relation_sumcheck {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let gate_opening = prover_opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenSiluGate, silu_sumcheck_id()),
    )?;
    let remainder_opening = prover_opening(
        &accumulator,
        OpeningId::new(
            VirtualPoly::QwenSiluRoundRemainder,
            silu_sumcheck_id(),
        ),
    )?;
    let round_bit_opening = prover_opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenSiluRoundLut, silu_sumcheck_id()),
    )?;
    let base_opening = prover_opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenSiluBase, silu_sumcheck_id()),
    )?;
    let slope_opening = prover_opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenSiluSlope, silu_sumcheck_id()),
    )?;
    let index_opening = prover_opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenSiluIndex, silu_sumcheck_id()),
    )?;
    let full_point = normalize_sumcheck_point::<F>(&challenges.into_opening());
    let tensor_point = full_point;
    eprintln!(
        "timing: prove_silu.relation_openings {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let base_lookup = prove_silu_shout_lookup(
        SiluLookupKind::Base,
        lut_len,
        tensor_point.clone(),
        base_opening,
        index_opening,
        padded_lookup_indices.clone(),
        base_table,
        &mut accumulator,
        transcript,
    )?;
    eprintln!(
        "timing: prove_silu.base_shout {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let slope_lookup = prove_silu_shout_lookup(
        SiluLookupKind::Slope,
        lut_len,
        tensor_point.clone(),
        slope_opening,
        index_opening,
        padded_lookup_indices,
        slope_table,
        &mut accumulator,
        transcript,
    )?;
    eprintln!(
        "timing: prove_silu.slope_shout {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let round_lookup = prove_silu_shout_lookup(
        SiluLookupKind::Round,
        1 << ROUND_FRAC_BITS,
        tensor_point.clone(),
        round_bit_opening,
        remainder_opening,
        padded_lookup_indices_for_shape(&remainders, &params.shape),
        round_lut_table(),
        &mut accumulator,
        transcript,
    )?;
    eprintln!(
        "timing: prove_silu.round_shout {:.3}s",
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
            relation,
            base_lookup: base_lookup.read_raf,
            base_ra_onehot: base_lookup.ra_onehot,
            slope_lookup: slope_lookup.read_raf,
            slope_ra_onehot: slope_lookup.ra_onehot,
            round_lookup: round_lookup.read_raf,
            round_ra_onehot: round_lookup.ra_onehot,
            gate_opening,
            remainder_opening,
            round_bit_opening,
            base_opening,
            slope_opening,
            index_opening,
            base_ra_opening: base_lookup.ra_opening,
            slope_ra_opening: slope_lookup.ra_opening,
            round_ra_opening: round_lookup.ra_opening,
            base_ra_committed_openings: base_lookup.committed_openings,
            slope_ra_committed_openings: slope_lookup.committed_openings,
            round_ra_committed_openings: round_lookup.committed_openings,
        },
        Claim {
            tensor: params.gate_proj_round_tensor.clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: tensor_point.clone(),
            value: gate_opening,
        },
        Claim {
            tensor: params.round_ra_tensor.clone(),
            logical_shape: Shape::new(vec![
                params.shape.padded_power_of_two().numel(),
                1 << ROUND_FRAC_BITS,
            ]),
            domain_shape: Shape::new(vec![
                params.shape.padded_power_of_two().numel(),
                1 << ROUND_FRAC_BITS,
            ]),
            point: round_lookup.ra_point,
            value: round_lookup.ra_opening,
        },
        Claim {
            tensor: params.ra_tensor.clone(),
            logical_shape: params.ra_shape(lut_len),
            domain_shape: params.ra_shape(lut_len).padded_power_of_two(),
            point: base_lookup.ra_point,
            value: base_lookup.ra_opening,
        },
    ))
}

pub fn verify_silu<F, T>(
    output_claims: Vec<Claim<F>>,
    proof: &SiluProof<F, T>,
    params: &SiluParams,
    transcript: &mut T,
) -> std::result::Result<(Claim<F>, Claim<F>, Claim<F>), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    let total_start = Instant::now();
    let mut step_start = Instant::now();
    let entries = entries_from_min_max(proof.min_n, proof.max_n)
        .map_err(|_| ProofVerifyError::InvalidInputLength(MAX_SILU_LUT_LEN, 0))?;
    verify_inputs(&output_claims, params, entries)?;
    append_range_advice::<F, T>(proof.min_n, proof.max_n, transcript);
    eprintln!(
        "timing: verify_silu.setup_validate {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let alphas = transcript.challenge_scalar_powers(output_claims.len());
    let round_mix = transcript.challenge_scalar();
    let input_claim = batched_input_claim(&output_claims, &alphas);
    let verifier = SiluSumcheckVerifier {
        params: SiluSumcheckParams::new(
            params.shape.padded_power_of_two().point_len(),
            input_claim,
        ),
        output_points: output_claims
            .iter()
            .map(|claim| claim.point.clone())
            .collect(),
        alphas,
        round_mix,
        shape: params.shape.clone(),
        min_n: proof.min_n,
    };
    eprintln!(
        "timing: verify_silu.challenges {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let mut accumulator = VerifierOpeningAccumulator::new();
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenSiluGate, silu_sumcheck_id()),
        (OpeningPoint::<BIG_ENDIAN, F>::default(), proof.gate_opening),
    );
    accumulator.openings.insert(
        OpeningId::new(
            VirtualPoly::QwenSiluRoundRemainder,
            silu_sumcheck_id(),
        ),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.remainder_opening,
        ),
    );
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenSiluRoundLut, silu_sumcheck_id()),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.round_bit_opening,
        ),
    );
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenSiluBase, silu_sumcheck_id()),
        (OpeningPoint::<BIG_ENDIAN, F>::default(), proof.base_opening),
    );
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenSiluSlope, silu_sumcheck_id()),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.slope_opening,
        ),
    );
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenSiluIndex, silu_sumcheck_id()),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.index_opening,
        ),
    );
    let challenges = Sumcheck::verify(&proof.relation, &verifier, &mut accumulator, transcript)?;
    let tensor_point = normalize_sumcheck_point::<F>(&challenges.into_opening());
    let lut_len = padded_silu_lut_len(entries);
    eprintln!(
        "timing: verify_silu.relation_sumcheck {:.3}s",
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let base_lookup = verify_silu_shout_lookup(
        SiluLookupKind::Base,
        entries,
        lut_len,
        proof.min_n,
        params.shape.numel(),
        tensor_point.clone(),
        proof.base_opening,
        proof.index_opening,
        proof.base_ra_opening,
        &proof.base_ra_committed_openings,
        &proof.base_lookup,
        &proof.base_ra_onehot,
        &mut accumulator,
        transcript,
    )?;
    eprintln!(
        "timing: verify_silu.base_shout {:.3}s",
        step_start.elapsed().as_secs_f64()
    );
    step_start = Instant::now();
    verify_silu_shout_lookup(
        SiluLookupKind::Slope,
        entries,
        lut_len,
        proof.min_n,
        params.shape.numel(),
        tensor_point.clone(),
        proof.slope_opening,
        proof.index_opening,
        proof.slope_ra_opening,
        &proof.slope_ra_committed_openings,
        &proof.slope_lookup,
        &proof.slope_ra_onehot,
        &mut accumulator,
        transcript,
    )?;
    eprintln!(
        "timing: verify_silu.slope_shout {:.3}s",
        step_start.elapsed().as_secs_f64()
    );
    step_start = Instant::now();
    let round_lookup = verify_silu_shout_lookup(
        SiluLookupKind::Round,
        1 << ROUND_FRAC_BITS,
        1 << ROUND_FRAC_BITS,
        0,
        params.shape.padded_power_of_two().numel(),
        tensor_point.clone(),
        proof.round_bit_opening,
        proof.remainder_opening,
        proof.round_ra_opening,
        &proof.round_ra_committed_openings,
        &proof.round_lookup,
        &proof.round_ra_onehot,
        &mut accumulator,
        transcript,
    )?;
    eprintln!(
        "timing: verify_silu.round_shout {:.3}s",
        step_start.elapsed().as_secs_f64()
    );
    eprintln!(
        "timing: verify_silu.total {:.3}s",
        total_start.elapsed().as_secs_f64()
    );

    Ok((
        Claim {
            tensor: params.gate_proj_round_tensor.clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: tensor_point.clone(),
            value: proof.gate_opening,
        },
        Claim {
            tensor: params.round_ra_tensor.clone(),
            logical_shape: Shape::new(vec![
                params.shape.padded_power_of_two().numel(),
                1 << ROUND_FRAC_BITS,
            ]),
            domain_shape: Shape::new(vec![
                params.shape.padded_power_of_two().numel(),
                1 << ROUND_FRAC_BITS,
            ]),
            point: round_lookup.ra_point,
            value: proof.round_ra_opening,
        },
        Claim {
            tensor: params.ra_tensor.clone(),
            logical_shape: params.ra_shape(lut_len),
            domain_shape: params.ra_shape(lut_len).padded_power_of_two(),
            point: base_lookup.ra_point,
            value: proof.base_ra_opening,
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
    gate: MultilinearPolynomial<F>,
    remainder: MultilinearPolynomial<F>,
    round_bit: MultilinearPolynomial<F>,
    base: MultilinearPolynomial<F>,
    slope: MultilinearPolynomial<F>,
    index: MultilinearPolynomial<F>,
    round_mix: F,
    min_n: F,
    params: SiluSumcheckParams<F>,
}

impl<F: JoltField> SiluSumcheckProver<F> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        params: SiluSumcheckParams<F>,
        eq_batch: Vec<F>,
        gate: Vec<F>,
        remainder: Vec<F>,
        round_bit: Vec<F>,
        base: Vec<F>,
        slope: Vec<F>,
        index: Vec<F>,
        round_mix: F,
        min_n: F,
    ) -> Self {
        Self {
            eq_batch: MultilinearPolynomial::from(eq_batch),
            gate: MultilinearPolynomial::from(gate),
            remainder: MultilinearPolynomial::from(remainder),
            round_bit: MultilinearPolynomial::from(round_bit),
            base: MultilinearPolynomial::from(base),
            slope: MultilinearPolynomial::from(slope),
            index: MultilinearPolynomial::from(index),
            round_mix,
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
        for g in 0..self.eq_batch.len() / 2 {
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
        self.gate.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.remainder.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.round_bit.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.base.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.slope.bind_parallel(r_j, BindingOrder::LowToHigh);
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
            OpeningId::new(VirtualPoly::QwenSiluGate, silu_sumcheck_id()),
            point.clone(),
            self.gate.final_claim(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(
                VirtualPoly::QwenSiluRoundRemainder,
                silu_sumcheck_id(),
            ),
            point.clone(),
            self.remainder.final_claim(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenSiluRoundLut, silu_sumcheck_id()),
            point.clone(),
            self.round_bit.final_claim(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenSiluBase, silu_sumcheck_id()),
            point.clone(),
            self.base.final_claim(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenSiluSlope, silu_sumcheck_id()),
            point.clone(),
            self.slope.final_claim(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenSiluIndex, silu_sumcheck_id()),
            point,
            self.index.final_claim(),
        );
    }
}

struct SiluSumcheckVerifier<F: JoltField> {
    params: SiluSumcheckParams<F>,
    output_points: Vec<Vec<F>>,
    alphas: Vec<F>,
    round_mix: F,
    shape: Shape,
    min_n: i64,
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
        let gate = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenSiluGate,
                silu_sumcheck_id(),
            ))
            .1;
        let remainder = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenSiluRoundRemainder,
                silu_sumcheck_id(),
            ))
            .1;
        let round_bit = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenSiluRoundLut,
                silu_sumcheck_id(),
            ))
            .1;
        let base = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenSiluBase,
                silu_sumcheck_id(),
            ))
            .1;
        let slope = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenSiluSlope,
                silu_sumcheck_id(),
            ))
            .1;
        let index = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenSiluIndex,
                silu_sumcheck_id(),
            ))
            .1;
        let eq_batch = eval_batched_masked_eq(
            &self.output_points,
            &self.alphas,
            &self.shape,
            &tensor_point,
        );
        eval_silu_relation(
            eq_batch,
            gate,
            remainder,
            round_bit,
            base,
            slope,
            index,
            self.round_mix,
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
        accumulator.append_virtual(
            transcript,
            OpeningId::new(
                VirtualPoly::QwenSiluRoundRemainder,
                silu_sumcheck_id(),
            ),
            point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenSiluRoundLut, silu_sumcheck_id()),
            point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenSiluBase, silu_sumcheck_id()),
            point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenSiluSlope, silu_sumcheck_id()),
            point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenSiluIndex, silu_sumcheck_id()),
            point,
        );
    }
}

struct SiluPairValues<F: JoltField> {
    eq_batch: [F; 2],
    gate: [F; 2],
    remainder: [F; 2],
    round_bit: [F; 2],
    base: [F; 2],
    slope: [F; 2],
    index: [F; 2],
}

impl<F: JoltField> SiluPairValues<F> {
    fn from_prover(prover: &SiluSumcheckProver<F>, g: usize) -> Self {
        Self {
            eq_batch: [
                prover.eq_batch.get_bound_coeff(2 * g),
                prover.eq_batch.get_bound_coeff(2 * g + 1),
            ],
            gate: [
                prover.gate.get_bound_coeff(2 * g),
                prover.gate.get_bound_coeff(2 * g + 1),
            ],
            remainder: [
                prover.remainder.get_bound_coeff(2 * g),
                prover.remainder.get_bound_coeff(2 * g + 1),
            ],
            round_bit: [
                prover.round_bit.get_bound_coeff(2 * g),
                prover.round_bit.get_bound_coeff(2 * g + 1),
            ],
            base: [
                prover.base.get_bound_coeff(2 * g),
                prover.base.get_bound_coeff(2 * g + 1),
            ],
            slope: [
                prover.slope.get_bound_coeff(2 * g),
                prover.slope.get_bound_coeff(2 * g + 1),
            ],
            index: [
                prover.index.get_bound_coeff(2 * g),
                prover.index.get_bound_coeff(2 * g + 1),
            ],
        }
    }

    fn eval(&self, t: F, prover: &SiluSumcheckProver<F>) -> F {
        eval_silu_relation(
            lerp(self.eq_batch[0], self.eq_batch[1], t),
            lerp(self.gate[0], self.gate[1], t),
            lerp(self.remainder[0], self.remainder[1], t),
            lerp(self.round_bit[0], self.round_bit[1], t),
            lerp(self.base[0], self.base[1], t),
            lerp(self.slope[0], self.slope[1], t),
            lerp(self.index[0], self.index[1], t),
            prover.round_mix,
            prover.min_n,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn eval_silu_relation<F: JoltField>(
    eq_batch: F,
    gate: F,
    remainder: F,
    round_bit: F,
    base: F,
    slope: F,
    index: F,
    round_mix: F,
    min_n: F,
) -> F {
    let scale = F::from_u64(FIXED_SCALE as u64);
    let n = min_n + index;
    let acc_expr = base + (gate - n * scale) * slope;
    let round_expr = gate + round_bit * scale - remainder - n * scale;

    eq_batch * acc_expr + round_mix * eq_batch * round_expr
}

#[derive(Debug, Clone, Copy)]
enum SiluLookupKind {
    Base,
    Slope,
    Round,
}

impl SiluLookupKind {
    fn value_poly(self) -> VirtualPoly {
        match self {
            Self::Base => VirtualPoly::QwenSiluBase,
            Self::Slope => VirtualPoly::QwenSiluSlope,
            Self::Round => VirtualPoly::QwenSiluRoundLut,
        }
    }

    fn ra_poly(self) -> VirtualPoly {
        match self {
            Self::Base => VirtualPoly::QwenSiluBaseRa,
            Self::Slope => VirtualPoly::QwenSiluSlopeRa,
            Self::Round => VirtualPoly::QwenSiluRoundRa,
        }
    }

    fn index_poly(self) -> VirtualPoly {
        match self {
            Self::Base | Self::Slope => VirtualPoly::QwenSiluIndex,
            Self::Round => VirtualPoly::QwenSiluRoundRemainder,
        }
    }

    fn committed_poly(self, d: usize) -> CommittedPoly {
        match self {
            Self::Base => CommittedPoly::QwenSiluBaseRaD(d),
            Self::Slope => CommittedPoly::QwenSiluSlopeRaD(d),
            Self::Round => CommittedPoly::QwenSiluRoundRaD(d),
        }
    }
}

#[derive(Clone)]
struct SiluReadRafProvider<F: JoltField> {
    kind: SiluLookupKind,
    log_k: usize,
    r_cycle: OpeningPoint<BIG_ENDIAN, F>,
    rv_claim: F,
    raf_claim: F,
}

impl<F: JoltField> ReadRafProvider<F> for SiluReadRafProvider<F> {
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
        (self.kind.ra_poly(), silu_shout_sumcheck_id(self.kind))
    }

    fn log_K(&self) -> usize {
        self.log_k
    }
}

struct SiluRaEncoding {
    kind: SiluLookupKind,
    log_k: usize,
}

impl RaOneHotEncoding for SiluRaEncoding {
    fn committed_poly(&self, d: usize) -> CommittedPoly {
        self.kind.committed_poly(d)
    }

    fn r_cycle_source(&self) -> OpeningId {
        OpeningId::new(self.kind.value_poly(), silu_sumcheck_id())
    }

    fn ra_source(&self) -> OpeningId {
        OpeningId::new(self.kind.ra_poly(), silu_shout_sumcheck_id(self.kind))
    }

    fn log_k(&self) -> usize {
        self.log_k
    }

    fn one_hot_params(&self) -> OneHotParams {
        OneHotParams::from_config_and_log_K(&OneHotConfig::default(), self.log_k)
    }
}

struct SiluShoutProof<F: JoltField, T: Transcript> {
    read_raf: SumcheckInstanceProof<F, T>,
    ra_onehot: SumcheckInstanceProof<F, T>,
    ra_point: Vec<F>,
    ra_opening: F,
    committed_openings: SiluRaCommittedOpenings<F>,
}

fn prove_silu_shout_lookup<F, T>(
    kind: SiluLookupKind,
    lut_len: usize,
    tensor_point: Vec<F>,
    value_opening: F,
    index_opening: F,
    lookup_indices: Vec<usize>,
    table: Vec<i32>,
    accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
) -> Result<SiluShoutProof<F, T>>
where
    F: JoltField,
    T: Transcript,
{
    let log_k = lut_len.trailing_zeros() as usize;
    let r_cycle = OpeningPoint::<BIG_ENDIAN, F>::new(tensor_point);
    let provider = SiluReadRafProvider {
        kind,
        log_k,
        r_cycle: r_cycle.clone(),
        rv_claim: value_opening,
        raf_claim: index_opening,
    };
    let mut read_prover =
        shout::read_raf_prover(&provider, &lookup_indices, &table, accumulator, transcript);
    let (read_raf, _) = Sumcheck::prove(&mut *read_prover, accumulator, transcript);

    let encoding = SiluRaEncoding { kind, log_k };
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
    let (ra_point, ra_opening) = accumulator.get_virtual_polynomial_opening(OpeningId::new(
        kind.ra_poly(),
        silu_shout_sumcheck_id(kind),
    ));
    let committed_openings = silu_ra_committed_openings(kind, log_k, use_ra_virtual, accumulator)?;
    Ok(SiluShoutProof {
        read_raf,
        ra_onehot,
        ra_point: ra_point.r,
        ra_opening,
        committed_openings,
    })
}

fn verify_silu_shout_lookup<F, T>(
    kind: SiluLookupKind,
    entries: usize,
    lut_len: usize,
    min_n: i64,
    logical_len: usize,
    tensor_point: Vec<F>,
    value_opening: F,
    index_opening: F,
    ra_opening: F,
    committed_openings: &SiluRaCommittedOpenings<F>,
    read_raf: &SumcheckInstanceProof<F, T>,
    ra_onehot: &SumcheckInstanceProof<F, T>,
    accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut T,
) -> std::result::Result<SiluShoutProof<F, T>, ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    let total_start = Instant::now();
    let mut step_start = Instant::now();
    let log_k = lut_len.trailing_zeros() as usize;
    let r_cycle = OpeningPoint::<BIG_ENDIAN, F>::new(tensor_point);
    accumulator.openings.insert(
        OpeningId::new(kind.value_poly(), silu_sumcheck_id()),
        (r_cycle.clone(), value_opening),
    );
    accumulator.openings.insert(
        OpeningId::new(kind.index_poly(), silu_sumcheck_id()),
        (r_cycle.clone(), index_opening),
    );
    accumulator.openings.insert(
        OpeningId::new(kind.ra_poly(), silu_shout_sumcheck_id(kind)),
        (OpeningPoint::<BIG_ENDIAN, F>::default(), ra_opening),
    );
    let provider = SiluReadRafProvider {
        kind,
        log_k,
        r_cycle: r_cycle.clone(),
        rv_claim: value_opening,
        raf_claim: index_opening,
    };
    let table = match kind {
        SiluLookupKind::Base => padded_i32_table(min_n, entries, lut_len, silu_base),
        SiluLookupKind::Slope => padded_i32_table(min_n, entries, lut_len, silu_slope),
        SiluLookupKind::Round => Ok(round_lut_table()),
    }
    .map_err(|_| ProofVerifyError::InvalidInputLength(lut_len, 0))?;
    let read_verifier = shout::read_raf_verifier(&provider, table, accumulator, transcript);
    eprintln!(
        "timing: verify_silu.{:?}.shout_setup {:.3}s",
        kind,
        step_start.elapsed().as_secs_f64()
    );
    step_start = Instant::now();
    Sumcheck::verify(read_raf, &*read_verifier, accumulator, transcript)?;
    eprintln!(
        "timing: verify_silu.{:?}.read_raf {:.3}s",
        kind,
        step_start.elapsed().as_secs_f64()
    );

    step_start = Instant::now();
    let encoding = SiluRaEncoding { kind, log_k };
    let [ra_verifier, hw_verifier, bool_verifier] =
        shout::ra_onehot_verifiers(&encoding, accumulator, transcript);
    let use_ra_virtual = logical_len.next_power_of_two() >= 8;
    insert_silu_ra_committed_openings(
        kind,
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
    eprintln!(
        "timing: verify_silu.{:?}.ra_setup {:.3}s",
        kind,
        step_start.elapsed().as_secs_f64()
    );
    step_start = Instant::now();
    BatchedSumcheck::verify(ra_onehot, verifier_instances, accumulator, transcript)?;
    eprintln!(
        "timing: verify_silu.{:?}.ra_onehot {:.3}s",
        kind,
        step_start.elapsed().as_secs_f64()
    );
    step_start = Instant::now();
    let (ra_point, ra_opening) = accumulator.get_virtual_polynomial_opening(OpeningId::new(
        kind.ra_poly(),
        silu_shout_sumcheck_id(kind),
    ));
    eprintln!(
        "timing: verify_silu.{:?}.ra_opening {:.3}s",
        kind,
        step_start.elapsed().as_secs_f64()
    );
    eprintln!(
        "timing: verify_silu.{:?}.shout_total {:.3}s",
        kind,
        total_start.elapsed().as_secs_f64()
    );
    Ok(SiluShoutProof {
        read_raf: read_raf.clone(),
        ra_onehot: ra_onehot.clone(),
        ra_point: ra_point.r,
        ra_opening,
        committed_openings: committed_openings.clone(),
    })
}

fn silu_ra_committed_openings<F: JoltField>(
    kind: SiluLookupKind,
    log_k: usize,
    include_full_checks: bool,
    accumulator: &ProverOpeningAccumulator<F>,
) -> Result<SiluRaCommittedOpenings<F>> {
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
    Ok(SiluRaCommittedOpenings {
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

fn insert_silu_ra_committed_openings<F: JoltField>(
    kind: SiluLookupKind,
    log_k: usize,
    include_full_checks: bool,
    openings: &SiluRaCommittedOpenings<F>,
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
        let n = witness.min_n + selected as i64;
        let gate = i64::from(witness.gate_proj_round[element]);
        let expected_round = n * FIXED_SCALE;
        let rem = gate.rem_euclid(FIXED_SCALE) as usize;
        let actual_round = gate + i64::from(round_lut_q8(rem)) * FIXED_SCALE - rem as i64;
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

fn padded_silu_lut_len(entries: usize) -> usize {
    // JoltWorks Shout uses one-hot chunks with a minimum chunk size of 16.
    // Keeping tiny unit tests on the same minimum domain avoids the compact
    // polynomial edge cases that occur with 1- or 2-bit lookup tables, while
    // real Qwen SiLU ranges are already small enough that this padding is
    // irrelevant to performance.  The extra entry is a dedicated zero row for
    // padded tensor elements: Shout's trace length is the padded tensor domain,
    // so invalid tensor slots must still read a table value, and that value
    // must match the zero-padded base/slope/index polynomials.
    (entries + 1).next_power_of_two().max(16)
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

fn batched_masked_eq_poly<F: JoltField>(
    claims: &[Claim<F>],
    alphas: &[F],
    shape: &Shape,
) -> Vec<F> {
    let mut out = vec![F::zero(); claims[0].domain_shape.dims().iter().product()];
    let strides = row_major_strides(shape.dims());
    let padded_strides = row_major_strides(claims[0].domain_shape.dims());
    for (claim, alpha) in claims.iter().zip(alphas) {
        let eq_evals = EqPolynomial::<F>::evals(&claim.point);
        for flat in 0..shape.numel() {
            let mut padded_flat = 0;
            for (dim, (&stride, &padded_stride)) in strides.iter().zip(&padded_strides).enumerate()
            {
                let coord = (flat / stride) % shape.dims()[dim];
                padded_flat += coord * padded_stride;
            }
            out[padded_flat] += *alpha * eq_evals[padded_flat];
        }
    }
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

fn padded_i64_tensor<F: JoltField>(values: &[i64], shape: &Shape) -> Vec<F> {
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
        out[padded_flat] = field_from_i64(value);
    }
    out
}

fn padded_usize_tensor<F: JoltField>(values: &[usize], shape: &Shape) -> Vec<F> {
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
        out[padded_flat] = F::from_u64(value as u64);
    }
    out
}

fn padded_usize_evals<F: JoltField>(values: &[usize]) -> Vec<F> {
    values
        .iter()
        .map(|&value| F::from_u64(value as u64))
        .collect()
}

fn padded_lookup_indices_for_shape(values: &[usize], shape: &Shape) -> Vec<usize> {
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

fn silu_logical_lookup_indices(
    witness: &SiluWitness,
    params: &SiluParams,
    entries: usize,
) -> Result<Vec<usize>> {
    let len = params.shape.numel();
    let mut indices = Vec::with_capacity(len);
    for idx in 0..len {
        let n = i64::from(round_shift_signed_i64(
            i64::from(witness.gate_proj_round[idx]),
            FIXED_FRAC_BITS,
        ) as i32);
        let shifted = n - witness.min_n;
        if shifted < 0 || shifted >= entries as i64 {
            return Err(ProverError::InvalidSumcheckDomain(entries));
        }
        indices.push(shifted as usize);
    }
    Ok(indices)
}

fn round_remainders(values: &[i32]) -> Vec<usize> {
    values
        .iter()
        .map(|&value| i64::from(value).rem_euclid(FIXED_SCALE) as usize)
        .collect()
}

fn round_bits_from_remainders(remainders: &[usize]) -> Vec<i32> {
    remainders.iter().map(|&rem| round_lut_q8(rem)).collect()
}

fn round_lut_table() -> Vec<i32> {
    (0..(1 << ROUND_FRAC_BITS)).map(round_lut_q8).collect()
}

fn round_lut_q8(rem: usize) -> i32 {
    if rem >= (1 << (ROUND_FRAC_BITS - 1)) {
        1
    } else {
        0
    }
}

fn padded_lookup_indices(
    logical_indices: &[usize],
    params: &SiluParams,
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

fn silu_lookup_values(indices: &[usize], min_n: i64, table_fn: fn(i64) -> i64) -> Vec<i64> {
    indices
        .iter()
        .map(|&idx| table_fn(min_n + idx as i64))
        .collect()
}

fn padded_i32_table(
    min_n: i64,
    entries: usize,
    padded_len: usize,
    table_fn: fn(i64) -> i64,
) -> Result<Vec<i32>> {
    if padded_len < entries || !padded_len.is_power_of_two() {
        return Err(ProverError::InvalidSumcheckDomain(padded_len));
    }
    let mut table = Vec::with_capacity(padded_len);
    for idx in 0..entries {
        let value = table_fn(min_n + idx as i64);
        let value =
            i32::try_from(value).map_err(|_| ProverError::InvalidSumcheckDomain(entries))?;
        table.push(value);
    }
    table.resize(padded_len, 0);
    Ok(table)
}

fn eval_batched_masked_eq<F: JoltField>(
    points: &[Vec<F>],
    alphas: &[F],
    shape: &Shape,
    point: &[F],
) -> F {
    let rhs_eq_by_dim = split_point(shape, point)
        .into_iter()
        .map(EqPolynomial::<F>::evals)
        .collect::<Vec<_>>();
    let strides = row_major_strides(shape.dims());

    points
        .iter()
        .zip(alphas)
        .map(|(claim_point, alpha)| {
            let lhs_eq_by_dim = split_point(shape, claim_point)
                .into_iter()
                .map(EqPolynomial::<F>::evals)
                .collect::<Vec<_>>();
            let combined_eq_by_dim = lhs_eq_by_dim
                .iter()
                .zip(&rhs_eq_by_dim)
                .map(|(lhs, rhs)| {
                    lhs.iter()
                        .zip(rhs)
                        .map(|(lhs, rhs)| *lhs * *rhs)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            let mut sum = F::zero();
            for flat in 0..shape.numel() {
                let mut weight = F::one();
                for (dim, (&stride, eq)) in strides.iter().zip(&combined_eq_by_dim).enumerate() {
                    let coord = (flat / stride) % shape.dims()[dim];
                    weight *= eq[coord];
                }
                sum += weight;
            }
            *alpha * sum
        })
        .sum()
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

fn tensor_point_from_full_shape<F: Clone>(point: &[F], shape: &Shape) -> Vec<F> {
    let tensor_vars = shape.padded_power_of_two().point_len();
    point[..tensor_vars].to_vec()
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

fn silu_shout_sumcheck_id(kind: SiluLookupKind) -> SumcheckId {
    match kind {
        SiluLookupKind::Base => SumcheckId::NodeExecution(1),
        SiluLookupKind::Slope => SumcheckId::NodeExecution(2),
        SiluLookupKind::Round => SumcheckId::NodeExecution(3),
    }
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
            ra,
            output,
        };

        let mut prover_transcript = Blake2bTranscript::default();
        let (proof, gate_claim, gate_round_ra, ra_claim) = prove_silu::<Fr, _>(
            vec![output_claim.clone()],
            &witness,
            &params,
            &mut prover_transcript,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        let (verified_gate_claim, verified_gate_round_ra, verified_ra_claim) =
            verify_silu::<Fr, _>(
                vec![output_claim],
                &proof,
                &params,
                &mut verifier_transcript,
            )
            .unwrap();

        assert_eq!(verified_gate_claim, gate_claim);
        assert_eq!(verified_gate_round_ra, gate_round_ra);
        assert_eq!(verified_ra_claim, ra_claim);
        assert_eq!(verified_gate_claim.tensor.0, "gate");
    }

    #[test]
    fn proves_silu_with_non_power_of_two_logical_range() {
        let params = SiluParams::new(
            vec![1],
            "gate",
            "silu_acc",
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
    fn proves_silu_with_non_power_of_two_tensor_shape() {
        let params = SiluParams::new(
            vec![2, 3],
            "gate",
            "silu_acc",
            "silu_ra",
        );
        let gate = vec![-384, -129, 0, 255, 512, 769];
        let min_n = -2;
        let max_n = 3;
        let entries = entries_from_min_max(min_n, max_n).unwrap();
        let mut ra = vec![0; gate.len() * entries];
        let mut output = vec![0; gate.len()];
        for (idx, &gate_value) in gate.iter().enumerate() {
            let n = i64::from(round_shift_signed_i64(
                i64::from(gate_value),
                FIXED_FRAC_BITS,
            ));
            let selected = (n - min_n) as usize;
            ra[idx * entries + selected] = 1;
            output[idx] = silu_base(n) + (i64::from(gate_value) - n * FIXED_SCALE) * silu_slope(n);
        }
        let point = vec![Fr::from(3u64), Fr::from(5u64), Fr::from(7u64)];
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
            "silu_ra",
        );
        let gate = vec![768];
        let min_n = 0;
        let max_n = 2;
        let entries = entries_from_min_max(min_n, max_n).unwrap();
        let mut ra = vec![0; entries.next_power_of_two()];
        ra[3] = 1;
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
