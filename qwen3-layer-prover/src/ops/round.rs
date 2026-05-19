use common::VirtualPoly;
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
};

pub const ROUND_FRAC_BITS: usize = 8;

// Design note for future us:
//
// The fixed-point runtime uses QX.8.  For an accumulator `x`, rounded output
// `y`, and low fractional bits `b_0..b_7`, the relation is:
//
//     frac = sum_j b_j * 2^j
//     y = floor(x / 2^8) + b_7
//
// Equivalently:
//
//     x + b_7 * 2^8 - frac = y * 2^8
//
// This is why the proof returns claims for the accumulator and the eight bit
// tensors.  One sumcheck batches the round relation and bit booleanity together:
//
//   sum_i eq_batch(i) * round_relation(i)
//     + gamma * sum_i sum_j beta_j * b_j(i) * (b_j(i) - 1)
//
// The booleanity term is zero for valid bit witnesses, so the public input
// claim remains `sum_t alpha_t * y_t(r_t) * 2^8`.  The low bits use Euclidean
// decomposition, so the relation also works for negative accumulators.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoundParams {
    pub shape: Shape,
    pub input_tensor: TensorId,
    pub output_tensor: TensorId,
    pub frac_bit_tensors: [TensorId; ROUND_FRAC_BITS],
    pub frac_bits: usize,
}

impl RoundParams {
    pub fn new(
        shape: impl Into<Vec<usize>>,
        input_tensor: impl Into<String>,
        output_tensor: impl Into<String>,
    ) -> Self {
        let input_tensor = input_tensor.into();
        Self {
            shape: Shape::new(shape),
            input_tensor: TensorId::new(input_tensor.clone()),
            output_tensor: TensorId::new(output_tensor),
            frac_bit_tensors: std::array::from_fn(|idx| {
                TensorId::new(format!("{input_tensor}_frac_bit_{idx}"))
            }),
            frac_bits: ROUND_FRAC_BITS,
        }
    }

    pub fn with_frac_bit_tensors(
        shape: impl Into<Vec<usize>>,
        input_tensor: impl Into<String>,
        output_tensor: impl Into<String>,
        frac_bit_tensors: [String; ROUND_FRAC_BITS],
    ) -> Self {
        Self {
            shape: Shape::new(shape),
            input_tensor: TensorId::new(input_tensor),
            output_tensor: TensorId::new(output_tensor),
            frac_bit_tensors: frac_bit_tensors.map(TensorId::new),
            frac_bits: ROUND_FRAC_BITS,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct RoundWitness {
    pub input: Vec<i64>,
    pub output: Vec<i32>,
    pub frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
}

#[derive(Debug, Clone)]
pub struct RoundProof<F: JoltField, T: Transcript> {
    pub sumcheck: SumcheckInstanceProof<F, T>,
    pub input_opening: F,
    pub frac_bit_openings: [F; ROUND_FRAC_BITS],
}

pub fn prove_round<F, T>(
    output_claims: Vec<Claim<F>>,
    witness: &RoundWitness,
    params: &RoundParams,
    transcript: &mut T,
) -> Result<(RoundProof<F, T>, Claim<F>, [Claim<F>; ROUND_FRAC_BITS])>
where
    F: JoltField,
    T: Transcript,
{
    validate_inputs(&output_claims, witness, params)?;

    let alphas = transcript.challenge_scalar_powers(output_claims.len());
    let relation_claim = round_scale::<F>() * batched_input_claim(&output_claims, &alphas);
    let eq_batch = batched_eq_poly(&output_claims, &alphas);
    let input_poly = padded_i64_tensor(&witness.input, &params.shape);
    let bit_polys = padded_bit_tensors(&witness.frac_bits, &params.shape);
    let booleanity_mix = transcript.challenge_scalar();
    let bit_weights = transcript.challenge_scalar_powers(ROUND_FRAC_BITS);

    let relation_params = RoundRelationSumcheckParams::new(
        params.shape.padded_power_of_two().point_len(),
        relation_claim,
    );
    let mut relation_prover = RoundRelationSumcheckProver::new(
        relation_params,
        eq_batch,
        input_poly,
        bit_polys,
        booleanity_mix,
        bit_weights,
    );
    let mut relation_accumulator = ProverOpeningAccumulator::new();
    let (sumcheck, relation_challenges) =
        Sumcheck::prove(&mut relation_prover, &mut relation_accumulator, transcript);
    let input_opening = prover_opening(
        &relation_accumulator,
        OpeningId::new(VirtualPoly::QwenRebaseAcc, round_relation_sumcheck_id()),
    )?;
    let frac_bit_openings = std::array::from_fn(|idx| {
        prover_opening(
            &relation_accumulator,
            OpeningId::new(VirtualPoly::QwenRoundBit(idx), round_relation_sumcheck_id()),
        )
        .expect("round relation bit opening must be produced")
    });
    let relation_point = normalize_sumcheck_point::<F>(&relation_challenges.into_opening());

    Ok((
        RoundProof {
            sumcheck,
            input_opening,
            frac_bit_openings,
        },
        Claim {
            tensor: params.input_tensor.clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: relation_point.clone(),
            value: input_opening,
        },
        std::array::from_fn(|idx| Claim {
            tensor: params.frac_bit_tensors[idx].clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: relation_point.clone(),
            value: frac_bit_openings[idx],
        }),
    ))
}

pub fn verify_round<F, T>(
    output_claims: Vec<Claim<F>>,
    proof: &RoundProof<F, T>,
    params: &RoundParams,
    transcript: &mut T,
) -> std::result::Result<(Claim<F>, [Claim<F>; ROUND_FRAC_BITS]), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    verify_inputs(&output_claims, params)?;

    let alphas = transcript.challenge_scalar_powers(output_claims.len());
    let relation_claim = round_scale::<F>() * batched_input_claim(&output_claims, &alphas);
    let booleanity_mix = transcript.challenge_scalar();
    let bit_weights = transcript.challenge_scalar_powers(ROUND_FRAC_BITS);
    let y_points = output_claims
        .iter()
        .map(|claim| claim.point.clone())
        .collect::<Vec<_>>();
    let relation_verifier = RoundRelationSumcheckVerifier {
        params: RoundRelationSumcheckParams::new(
            params.shape.padded_power_of_two().point_len(),
            relation_claim,
        ),
        y_points,
        alphas,
        booleanity_mix,
        bit_weights,
    };
    let mut relation_accumulator = VerifierOpeningAccumulator::new();
    relation_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRebaseAcc, round_relation_sumcheck_id()),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.input_opening,
        ),
    );
    for idx in 0..ROUND_FRAC_BITS {
        relation_accumulator.openings.insert(
            OpeningId::new(VirtualPoly::QwenRoundBit(idx), round_relation_sumcheck_id()),
            (
                OpeningPoint::<BIG_ENDIAN, F>::default(),
                proof.frac_bit_openings[idx],
            ),
        );
    }
    let relation_challenges = Sumcheck::verify(
        &proof.sumcheck,
        &relation_verifier,
        &mut relation_accumulator,
        transcript,
    )?;
    let relation_point = normalize_sumcheck_point::<F>(&relation_challenges.into_opening());

    Ok((
        Claim {
            tensor: params.input_tensor.clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: relation_point.clone(),
            value: proof.input_opening,
        },
        std::array::from_fn(|idx| Claim {
            tensor: params.frac_bit_tensors[idx].clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: relation_point.clone(),
            value: proof.frac_bit_openings[idx],
        }),
    ))
}

struct RoundRelationSumcheckParams<F: JoltField> {
    num_rounds: usize,
    input_claim: F,
}

impl<F: JoltField> RoundRelationSumcheckParams<F> {
    fn new(num_rounds: usize, input_claim: F) -> Self {
        Self {
            num_rounds,
            input_claim,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for RoundRelationSumcheckParams<F> {
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

struct RoundRelationSumcheckProver<F: JoltField> {
    eq_batch: MultilinearPolynomial<F>,
    input: MultilinearPolynomial<F>,
    bits: [MultilinearPolynomial<F>; ROUND_FRAC_BITS],
    booleanity_mix: F,
    bit_weights: Vec<F>,
    params: RoundRelationSumcheckParams<F>,
}

impl<F: JoltField> RoundRelationSumcheckProver<F> {
    fn new(
        params: RoundRelationSumcheckParams<F>,
        eq_batch: Vec<F>,
        input: Vec<F>,
        bits: [Vec<F>; ROUND_FRAC_BITS],
        booleanity_mix: F,
        bit_weights: Vec<F>,
    ) -> Self {
        Self {
            eq_batch: MultilinearPolynomial::from(eq_batch),
            input: MultilinearPolynomial::from(input),
            bits: bits.map(MultilinearPolynomial::from),
            booleanity_mix,
            bit_weights,
            params,
        }
    }

    fn linear_value_at(&self, idx: usize) -> F {
        let mut frac = F::zero();
        for bit in 0..ROUND_FRAC_BITS {
            frac += self.bits[bit].get_bound_coeff(idx) * F::from_u64(1u64 << bit);
        }
        self.input.get_bound_coeff(idx)
            + self.bits[ROUND_FRAC_BITS - 1].get_bound_coeff(idx) * round_scale::<F>()
            - frac
    }

    fn bool_value_at(&self, idx0: usize, idx1: usize, t: F) -> F {
        let mut out = F::zero();
        for bit in 0..ROUND_FRAC_BITS {
            let b = lerp(
                self.bits[bit].get_bound_coeff(idx0),
                self.bits[bit].get_bound_coeff(idx1),
                t,
            );
            out += self.bit_weights[bit] * b * (b - F::one());
        }
        out
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for RoundRelationSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let mut evals = [F::zero(); 3];
        for g in 0..self.input.len() / 2 {
            let e0 = self.eq_batch.get_bound_coeff(2 * g);
            let e1 = self.eq_batch.get_bound_coeff(2 * g + 1);
            let l0 = self.linear_value_at(2 * g);
            let l1 = self.linear_value_at(2 * g + 1);
            for (idx, t) in [F::zero(), F::from_u64(2), F::from_u64(3)]
                .into_iter()
                .enumerate()
            {
                evals[idx] += lerp(e0, e1, t) * lerp(l0, l1, t)
                    + self.booleanity_mix * self.bool_value_at(2 * g, 2 * g + 1, t);
            }
        }
        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_batch.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.input.bind_parallel(r_j, BindingOrder::LowToHigh);
        for bit in &mut self.bits {
            bit.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
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
            OpeningId::new(VirtualPoly::QwenRebaseAcc, round_relation_sumcheck_id()),
            point.clone(),
            self.input.final_claim(),
        );
        for idx in 0..ROUND_FRAC_BITS {
            accumulator.append_virtual(
                transcript,
                OpeningId::new(VirtualPoly::QwenRoundBit(idx), round_relation_sumcheck_id()),
                point.clone(),
                self.bits[idx].final_claim(),
            );
        }
    }
}

struct RoundRelationSumcheckVerifier<F: JoltField> {
    params: RoundRelationSumcheckParams<F>,
    y_points: Vec<Vec<F>>,
    alphas: Vec<F>,
    booleanity_mix: F,
    bit_weights: Vec<F>,
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for RoundRelationSumcheckVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let input = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenRebaseAcc,
                round_relation_sumcheck_id(),
            ))
            .1;
        let bits: [F; ROUND_FRAC_BITS] = std::array::from_fn(|idx| {
            accumulator
                .get_virtual_polynomial_opening(OpeningId::new(
                    VirtualPoly::QwenRoundBit(idx),
                    round_relation_sumcheck_id(),
                ))
                .1
        });
        let point = normalize_sumcheck_point::<F>(&sumcheck_challenges.into_opening());
        let mut eq_batch = F::zero();
        for (alpha, y_point) in self.alphas.iter().zip(&self.y_points) {
            eq_batch += *alpha * EqPolynomial::mle(y_point, &point);
        }
        let mut frac = F::zero();
        for (idx, bit) in bits.iter().enumerate() {
            frac += *bit * F::from_u64(1u64 << idx);
        }
        let mut bool_value = F::zero();
        for (idx, bit) in bits.iter().enumerate() {
            bool_value += self.bit_weights[idx] * *bit * (*bit - F::one());
        }
        eq_batch * (input + bits[ROUND_FRAC_BITS - 1] * round_scale::<F>() - frac)
            + self.booleanity_mix * bool_value
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
            OpeningId::new(VirtualPoly::QwenRebaseAcc, round_relation_sumcheck_id()),
            point.clone(),
        );
        for idx in 0..ROUND_FRAC_BITS {
            accumulator.append_virtual(
                transcript,
                OpeningId::new(VirtualPoly::QwenRoundBit(idx), round_relation_sumcheck_id()),
                point.clone(),
            );
        }
    }
}

fn validate_inputs<F: JoltField>(
    output_claims: &[Claim<F>],
    witness: &RoundWitness,
    params: &RoundParams,
) -> Result<()> {
    if params.frac_bits != ROUND_FRAC_BITS {
        return Err(ProverError::InvalidTensorShape(vec![params.frac_bits]));
    }
    if output_claims.is_empty() {
        return Err(ProverError::InvalidClaimCount {
            name: "round output claims",
            expected: 1,
            actual: 0,
        });
    }
    let expected_len = params.shape.numel();
    if witness.input.len() != expected_len {
        return Err(ProverError::TensorLenMismatch {
            name: "round input",
            shape: params.shape.0.clone(),
            expected: expected_len,
            actual: witness.input.len(),
        });
    }
    if witness.output.len() != expected_len {
        return Err(ProverError::TensorLenMismatch {
            name: "round output",
            shape: params.shape.0.clone(),
            expected: expected_len,
            actual: witness.output.len(),
        });
    }
    for (bit, values) in witness.frac_bits.iter().enumerate() {
        if values.len() != expected_len {
            return Err(ProverError::TensorLenMismatch {
                name: "round frac bit",
                shape: params.shape.0.clone(),
                expected: expected_len,
                actual: values.len(),
            });
        }
        for (index, &value) in values.iter().enumerate() {
            if value > 1 {
                return Err(ProverError::BitNotBoolean { bit, index, value });
            }
        }
    }
    for (idx, (&input, &output)) in witness.input.iter().zip(&witness.output).enumerate() {
        let frac = compose_frac_bits(&witness.frac_bits, idx);
        let expected = (input + i64::from(frac >= 128) * round_scale_i64() - i64::from(frac))
            / round_scale_i64();
        if i64::from(output) != expected {
            return Err(ProverError::MatMulAccumulatorMismatch {
                row: idx,
                col: 0,
                expected,
                actual: i64::from(output),
            });
        }
    }
    for claim in output_claims {
        validate_output_claim(claim, params)?;
    }
    Ok(())
}

fn validate_output_claim<F: JoltField>(claim: &Claim<F>, params: &RoundParams) -> Result<()> {
    if claim.logical_shape != params.shape {
        return Err(ProverError::ShapeMismatch {
            name: "round output claim",
            expected: params.shape.0.clone(),
            actual: claim.logical_shape.0.clone(),
        });
    }
    let expected_domain = params.shape.padded_power_of_two();
    if claim.domain_shape != expected_domain {
        return Err(ProverError::ShapeMismatch {
            name: "round output claim domain",
            expected: expected_domain.0,
            actual: claim.domain_shape.0.clone(),
        });
    }
    let expected_point_len = claim.domain_shape.point_len();
    if claim.point.len() != expected_point_len {
        return Err(ProverError::ShapeMismatch {
            name: "round output claim point",
            expected: vec![expected_point_len],
            actual: vec![claim.point.len()],
        });
    }
    Ok(())
}

fn verify_inputs<F: JoltField>(
    output_claims: &[Claim<F>],
    params: &RoundParams,
) -> std::result::Result<(), ProofVerifyError> {
    if params.frac_bits != ROUND_FRAC_BITS {
        return Err(ProofVerifyError::InvalidInputLength(
            ROUND_FRAC_BITS,
            params.frac_bits,
        ));
    }
    if output_claims.is_empty() {
        return Err(ProofVerifyError::InvalidInputLength(1, 0));
    }
    for claim in output_claims {
        if claim.logical_shape != params.shape {
            return Err(ProofVerifyError::InvalidInputLength(
                params.shape.numel(),
                claim.logical_shape.numel(),
            ));
        }
        let expected_domain = params.shape.padded_power_of_two();
        if claim.domain_shape != expected_domain {
            return Err(ProofVerifyError::InvalidInputLength(
                expected_domain.numel(),
                claim.domain_shape.numel(),
            ));
        }
        let expected_point_len = claim.domain_shape.point_len();
        if claim.point.len() != expected_point_len {
            return Err(ProofVerifyError::InvalidInputLength(
                expected_point_len,
                claim.point.len(),
            ));
        }
    }
    Ok(())
}

fn batched_input_claim<F: JoltField>(claims: &[Claim<F>], alphas: &[F]) -> F {
    claims
        .iter()
        .zip(alphas)
        .map(|(claim, alpha)| claim.value * *alpha)
        .sum()
}

fn batched_eq_poly<F: JoltField>(claims: &[Claim<F>], alphas: &[F]) -> Vec<F> {
    let len = claims[0].domain_shape.numel();
    let mut out = vec![F::zero(); len];
    for (claim, alpha) in claims.iter().zip(alphas) {
        let eq = EqPolynomial::<F>::evals(&claim.point);
        for (out, eq) in out.iter_mut().zip(eq) {
            *out += *alpha * eq;
        }
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

fn padded_bit_tensors<F: JoltField>(
    values: &[Vec<u8>; ROUND_FRAC_BITS],
    shape: &Shape,
) -> [Vec<F>; ROUND_FRAC_BITS] {
    values.clone().map(|bits| {
        let padded_dims = shape.padded_power_of_two().0;
        let len = padded_dims.iter().product();
        let mut out = vec![F::zero(); len];
        let strides = row_major_strides(shape.dims());
        let padded_strides = row_major_strides(&padded_dims);
        for (flat, value) in bits.into_iter().enumerate() {
            let mut padded_flat = 0;
            for (dim, (&stride, &padded_stride)) in strides.iter().zip(&padded_strides).enumerate()
            {
                let coord = (flat / stride) % shape.dims()[dim];
                padded_flat += coord * padded_stride;
            }
            out[padded_flat] = F::from_u64(u64::from(value));
        }
        out
    })
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

fn compose_frac_bits(bits: &[Vec<u8>; ROUND_FRAC_BITS], idx: usize) -> u16 {
    bits.iter()
        .enumerate()
        .map(|(bit, values)| u16::from(values[idx]) << bit)
        .sum()
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

fn round_scale<F: JoltField>() -> F {
    F::from_u64(round_scale_i64() as u64)
}

fn round_scale_i64() -> i64 {
    1_i64 << ROUND_FRAC_BITS
}

fn round_relation_sumcheck_id() -> SumcheckId {
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
    fn proves_and_verifies_round() {
        let params = RoundParams::with_frac_bit_tensors(
            vec![2, 2],
            "acc",
            "rounded",
            std::array::from_fn(|idx| format!("frac_bit_{idx}")),
        );
        let input = vec![0, 127, 128, -129];
        let output = vec![0, 0, 1, -1];
        let frac_bits = frac_bits_for(&input);
        let point = vec![Fr::from(3u64), Fr::from(5u64)];
        let output_claim = Claim {
            tensor: TensorId::new("rounded"),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: point.clone(),
            value: eval_i32(&output, &params.shape, &point),
        };
        let witness = RoundWitness {
            input,
            output,
            frac_bits,
        };

        let mut prover_transcript = Blake2bTranscript::default();
        let (proof, input_claim, bit_claims) = prove_round::<Fr, _>(
            vec![output_claim.clone()],
            &witness,
            &params,
            &mut prover_transcript,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        let (verify_input_claim, verify_bit_claims) = verify_round::<Fr, _>(
            vec![output_claim],
            &proof,
            &params,
            &mut verifier_transcript,
        )
        .unwrap();

        assert_eq!(verify_input_claim, input_claim);
        assert_eq!(verify_bit_claims, bit_claims);
        assert_eq!(verify_input_claim.tensor.0, "acc");
        assert_eq!(verify_bit_claims[7].tensor.0, "frac_bit_7");
    }

    #[test]
    fn prover_rejects_non_boolean_round_bit() {
        let params = RoundParams::new(vec![1], "acc", "rounded");
        let point = vec![Fr::from(3u64)];
        let output_claim = Claim {
            tensor: TensorId::new("rounded"),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point,
            value: Fr::from(0u64),
        };
        let mut frac_bits = std::array::from_fn(|_| vec![0]);
        frac_bits[0][0] = 2;
        let witness = RoundWitness {
            input: vec![0],
            output: vec![0],
            frac_bits,
        };
        let mut transcript = Blake2bTranscript::default();
        let err = match prove_round::<Fr, _>(vec![output_claim], &witness, &params, &mut transcript)
        {
            Ok(_) => panic!("non-boolean bit witness must be rejected"),
            Err(err) => err,
        };
        assert!(matches!(err, ProverError::BitNotBoolean { .. }));
    }

    fn frac_bits_for(values: &[i64]) -> [Vec<u8>; ROUND_FRAC_BITS] {
        std::array::from_fn(|bit| {
            values
                .iter()
                .map(|value| ((value.rem_euclid(round_scale_i64()) >> bit) & 1) as u8)
                .collect()
        })
    }

    fn eval_i32<F: JoltField>(values: &[i32], shape: &Shape, point: &[F]) -> F {
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
            out += weight * field_from_i64::<F>(i64::from(value));
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
