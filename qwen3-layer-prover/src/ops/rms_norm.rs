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
    ops::{
        mul::{MulByVectorParams, MulByVectorProof, prove_mul_by_vector, verify_mul_by_vector},
        round::{
            ROUND_FRAC_BITS, RoundParams, RoundProof, RoundWitness, prove_round, verify_round,
        },
    },
};

// RMSNorm proof shape:
//
// 1. `round` consumes all output fanout claims and returns one accumulator claim:
//      output = round(acc_out)
//
// 2. `mul_by_vector` proves:
//      acc_out[row,col] = norm[row,col] * weight[col]
//
// 3. `round` proves the runtime's first rebase:
//      norm = round(norm_acc)
//
// 4. One combined sumcheck proves both:
//      norm_acc[row,col] = x[row,col] * inv_rms(row)
//      sum_x2[row]  = sum_col x[row,col]^2
//
//    `sum_x2` is advice carried in the proof.  The verifier evaluates that
//    advice as an MLE at the row point and deterministically computes
//    `inv_rms(row)` coefficients from the advice array.  The sumcheck input is:
//
//      norm_acc_claim + gamma * sum_x2(row_point)
//
//    and the summed polynomial is:
//
//      eq_acc(row,col) * x * inv_rms(row)
//        + gamma * eq_row(row) * x^2

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RmsNormParams {
    pub round: RoundParams,
    pub norm_round: RoundParams,
    pub weight_mul: MulByVectorParams,
    pub rows: usize,
    pub cols: usize,
    pub shape: Shape,
    pub input_tensor: TensorId,
    pub weight_tensor: TensorId,
    pub output_tensor: TensorId,
}

impl RmsNormParams {
    pub fn new(
        rows: usize,
        cols: usize,
        input_tensor: impl Into<String>,
        weight_tensor: impl Into<String>,
        acc_tensor: impl Into<String>,
        output_tensor: impl Into<String>,
        frac_bit_tensors: [String; ROUND_FRAC_BITS],
    ) -> Self {
        let output_tensor = output_tensor.into();
        let weight_tensor = weight_tensor.into();
        let norm_tensor = format!("{output_tensor}_norm");
        let norm_acc_tensor = format!("{output_tensor}_norm_acc");
        Self {
            round: RoundParams::with_frac_bit_tensors(
                vec![rows, cols],
                acc_tensor,
                output_tensor.clone(),
                frac_bit_tensors,
            ),
            norm_round: RoundParams::new(vec![rows, cols], norm_acc_tensor, norm_tensor.clone()),
            weight_mul: MulByVectorParams::new(
                vec![rows, cols],
                norm_tensor,
                weight_tensor.clone(),
                1,
            ),
            rows,
            cols,
            shape: Shape::new(vec![rows, cols]),
            input_tensor: TensorId::new(input_tensor),
            weight_tensor: TensorId::new(weight_tensor),
            output_tensor: TensorId::new(output_tensor),
        }
    }

    pub fn new_nd(
        shape: impl Into<Vec<usize>>,
        input_tensor: impl Into<String>,
        weight_tensor: impl Into<String>,
        acc_tensor: impl Into<String>,
        output_tensor: impl Into<String>,
        frac_bit_tensors: [String; ROUND_FRAC_BITS],
    ) -> Self {
        let shape = Shape::new(shape);
        let dims = shape.dims().to_vec();
        let rows = dims.iter().take(dims.len().saturating_sub(1)).product();
        let cols = dims.last().copied().unwrap_or(0);
        let output_tensor = output_tensor.into();
        let weight_tensor = weight_tensor.into();
        let norm_tensor = format!("{output_tensor}_norm");
        let norm_acc_tensor = format!("{output_tensor}_norm_acc");
        Self {
            round: RoundParams::with_frac_bit_tensors(
                dims,
                acc_tensor,
                output_tensor.clone(),
                frac_bit_tensors,
            ),
            norm_round: RoundParams::new(
                shape.dims().to_vec(),
                norm_acc_tensor,
                norm_tensor.clone(),
            ),
            weight_mul: MulByVectorParams::new(
                shape.dims().to_vec(),
                norm_tensor,
                weight_tensor.clone(),
                shape.dims().len() - 1,
            ),
            rows,
            cols,
            shape,
            input_tensor: TensorId::new(input_tensor),
            weight_tensor: TensorId::new(weight_tensor),
            output_tensor: TensorId::new(output_tensor),
        }
    }

    pub fn shape(&self) -> Shape {
        self.shape.clone()
    }

    fn row_shape(&self) -> Shape {
        let dims = self.shape.dims();
        if dims.len() <= 1 {
            Shape::new(vec![self.rows])
        } else {
            Shape::new(dims[..dims.len() - 1].to_vec())
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct RmsNormWitness {
    pub input: Vec<i32>,
    pub sum_x2: Vec<i64>,
    pub norm_acc: Vec<i64>,
    pub norm: Vec<i32>,
    pub norm_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    pub acc: Vec<i64>,
    pub output: Vec<i32>,
    pub frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
}

#[derive(Debug, Clone)]
pub struct RmsNormProof<F: JoltField, T: Transcript> {
    pub round: RoundProof<F, T>,
    pub weight_mul: MulByVectorProof<F, T>,
    pub norm_round: RoundProof<F, T>,
    pub sumcheck: SumcheckInstanceProof<F, T>,
    pub input_opening: F,
    pub sum_x2: Vec<i64>,
}

pub fn prove_rmsnorm_round<F, T>(
    output_claims: Vec<Claim<F>>,
    witness: &RmsNormWitness,
    weight: &[i32],
    params: &RmsNormParams,
    transcript: &mut T,
) -> Result<(RmsNormProof<F, T>, Claim<F>, [Claim<F>; ROUND_FRAC_BITS])>
where
    F: JoltField,
    T: Transcript,
{
    validate_inputs(witness, weight, params)?;
    let round_witness = RoundWitness {
        input: witness.acc.clone(),
        output: witness.output.clone(),
        frac_bits: witness.frac_bits.clone(),
    };
    let (round_proof, acc_claim, frac_bits) =
        prove_round(output_claims, &round_witness, &params.round, transcript)?;

    let weight_mul_result = prove_mul_by_vector(
        acc_claim,
        &witness.norm,
        weight,
        &params.weight_mul,
        transcript,
    )?;
    let norm_claim = weight_mul_result.claims.x;

    let norm_round_witness = RoundWitness {
        input: witness.norm_acc.clone(),
        output: witness.norm.clone(),
        frac_bits: witness.norm_frac_bits.clone(),
    };
    let (norm_round_proof, norm_acc_claim, _norm_frac_bits) = prove_round(
        vec![norm_claim],
        &norm_round_witness,
        &params.norm_round,
        transcript,
    )?;

    append_sum_x2_advice::<F, T>(&witness.sum_x2, transcript);
    let gamma = transcript.challenge_scalar();
    let row_point = acc_row_point(&norm_acc_claim, params);
    let sum_x2_eval = eval_i64_advice(&witness.sum_x2, &params.row_shape(), &row_point);
    let input_claim = norm_acc_claim.value + gamma * sum_x2_eval;
    let inv_rms = inv_rms_from_sum_x2::<F>(&witness.sum_x2, params.cols);
    let coeff = coeff_tensor(&inv_rms, params);
    let x_poly = padded_i32_tensor(&witness.input, &params.shape);
    let eq_acc = EqPolynomial::<F>::evals(&norm_acc_claim.point);
    let row_eq = row_eq_lifted(&row_point, params);
    let coeff_poly = padded_field_tensor(&coeff, &params.shape);

    let sc_params =
        RmsNormSumcheckParams::new(params.shape.padded_power_of_two().point_len(), input_claim);
    let mut prover =
        RmsNormSumcheckProver::new(sc_params, eq_acc, row_eq, x_poly, coeff_poly, gamma);
    let mut accumulator = ProverOpeningAccumulator::new();
    let (sumcheck, challenges) = Sumcheck::prove(&mut prover, &mut accumulator, transcript);
    let input_opening = prover_opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenRmsNormX, rms_norm_sumcheck_id()),
    )?;
    let point = normalize_sumcheck_point::<F>(&challenges.into_opening());

    Ok((
        RmsNormProof {
            round: round_proof,
            weight_mul: weight_mul_result.proof,
            norm_round: norm_round_proof,
            sumcheck,
            input_opening,
            sum_x2: witness.sum_x2.clone(),
        },
        Claim {
            tensor: params.input_tensor.clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point,
            value: input_opening,
        },
        frac_bits,
    ))
}

pub fn verify_rmsnorm_round<F, T>(
    output_claims: Vec<Claim<F>>,
    proof: &RmsNormProof<F, T>,
    weight: &[i32],
    params: &RmsNormParams,
    transcript: &mut T,
) -> std::result::Result<(Claim<F>, [Claim<F>; ROUND_FRAC_BITS]), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    verify_advice(params, &proof.sum_x2, weight)?;
    let (acc_claim, frac_bits) =
        verify_round(output_claims, &proof.round, &params.round, transcript)?;
    let norm_claim = verify_mul_by_vector(
        acc_claim,
        &proof.weight_mul,
        weight,
        &params.weight_mul,
        transcript,
    )
    .map_err(|_| ProofVerifyError::InvalidInputLength(1, 0))?
    .x;
    let (norm_acc_claim, _norm_frac_bits) = verify_round(
        vec![norm_claim],
        &proof.norm_round,
        &params.norm_round,
        transcript,
    )?;

    append_sum_x2_advice::<F, T>(&proof.sum_x2, transcript);
    let gamma = transcript.challenge_scalar();
    let row_point = acc_row_point(&norm_acc_claim, params);
    let sum_x2_eval = eval_i64_advice(&proof.sum_x2, &params.row_shape(), &row_point);
    let input_claim = norm_acc_claim.value + gamma * sum_x2_eval;
    let inv_rms = inv_rms_from_sum_x2::<F>(&proof.sum_x2, params.cols);
    let row_shape = params.row_shape();
    let acc_point = norm_acc_claim.point.clone();
    let verifier = RmsNormSumcheckVerifier {
        params: RmsNormSumcheckParams::new(
            params.shape.padded_power_of_two().point_len(),
            input_claim,
        ),
        acc_point,
        acc_row_point: row_point,
        row_shape,
        inv_rms,
        gamma,
        shape: params.shape.clone(),
    };
    let mut accumulator = VerifierOpeningAccumulator::new();
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRmsNormX, rms_norm_sumcheck_id()),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.input_opening,
        ),
    );
    let challenges = Sumcheck::verify(&proof.sumcheck, &verifier, &mut accumulator, transcript)?;
    let point = normalize_sumcheck_point::<F>(&challenges.into_opening());

    Ok((
        Claim {
            tensor: params.input_tensor.clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point,
            value: proof.input_opening,
        },
        frac_bits,
    ))
}

struct RmsNormSumcheckParams<F: JoltField> {
    num_rounds: usize,
    input_claim: F,
}

impl<F: JoltField> RmsNormSumcheckParams<F> {
    fn new(num_rounds: usize, input_claim: F) -> Self {
        Self {
            num_rounds,
            input_claim,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for RmsNormSumcheckParams<F> {
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

struct RmsNormSumcheckProver<F: JoltField> {
    eq_acc: MultilinearPolynomial<F>,
    row_eq: MultilinearPolynomial<F>,
    x: MultilinearPolynomial<F>,
    coeff: MultilinearPolynomial<F>,
    gamma: F,
    params: RmsNormSumcheckParams<F>,
}

impl<F: JoltField> RmsNormSumcheckProver<F> {
    fn new(
        params: RmsNormSumcheckParams<F>,
        eq_acc: Vec<F>,
        row_eq: Vec<F>,
        x: Vec<F>,
        coeff: Vec<F>,
        gamma: F,
    ) -> Self {
        Self {
            eq_acc: MultilinearPolynomial::from(eq_acc),
            row_eq: MultilinearPolynomial::from(row_eq),
            x: MultilinearPolynomial::from(x),
            coeff: MultilinearPolynomial::from(coeff),
            gamma,
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for RmsNormSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let mut evals = [F::zero(); 3];
        let mut actual_eval_at_1 = F::zero();
        for g in 0..self.x.len() / 2 {
            let eq = [
                self.eq_acc.get_bound_coeff(2 * g),
                self.eq_acc.get_bound_coeff(2 * g + 1),
            ];
            let row_eq = [
                self.row_eq.get_bound_coeff(2 * g),
                self.row_eq.get_bound_coeff(2 * g + 1),
            ];
            let x = [
                self.x.get_bound_coeff(2 * g),
                self.x.get_bound_coeff(2 * g + 1),
            ];
            let coeff = [
                self.coeff.get_bound_coeff(2 * g),
                self.coeff.get_bound_coeff(2 * g + 1),
            ];
            for (idx, t) in [F::zero(), F::from_u64(2), F::from_u64(3)]
                .into_iter()
                .enumerate()
            {
                let x_t = lerp(x[0], x[1], t);
                evals[idx] += lerp(eq[0], eq[1], t) * x_t * lerp(coeff[0], coeff[1], t)
                    + self.gamma * lerp(row_eq[0], row_eq[1], t) * x_t * x_t;
            }
            actual_eval_at_1 += eq[1] * x[1] * coeff[1] + self.gamma * row_eq[1] * x[1] * x[1];
        }
        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_acc.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.row_eq.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.x.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.coeff.bind_parallel(r_j, BindingOrder::LowToHigh);
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
            OpeningId::new(VirtualPoly::QwenRmsNormX, rms_norm_sumcheck_id()),
            point,
            self.x.final_claim(),
        );
    }
}

struct RmsNormSumcheckVerifier<F: JoltField> {
    params: RmsNormSumcheckParams<F>,
    acc_point: Vec<F>,
    acc_row_point: Vec<F>,
    row_shape: Shape,
    inv_rms: Vec<F>,
    gamma: F,
    shape: Shape,
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for RmsNormSumcheckVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let x = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenRmsNormX,
                rms_norm_sumcheck_id(),
            ))
            .1;
        let point = normalize_sumcheck_point::<F>(&sumcheck_challenges.into_opening());
        let (row_point, _col_point) = split_row_col_point(&self.shape, &point);
        let eq_acc = EqPolynomial::mle(&self.acc_point, &point);
        let row_eq = EqPolynomial::mle(&self.acc_row_point, row_point);
        let inv = eval_field_advice(&self.inv_rms, &self.row_shape, row_point);
        eq_acc * x * inv + self.gamma * row_eq * x * x
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
            OpeningId::new(VirtualPoly::QwenRmsNormX, rms_norm_sumcheck_id()),
            point,
        );
    }
}

fn validate_inputs(witness: &RmsNormWitness, weight: &[i32], params: &RmsNormParams) -> Result<()> {
    if params.rows == 0 || params.cols == 0 {
        return Err(ProverError::InvalidTensorShape(params.shape.0.clone()));
    }
    let expected_len = params.rows * params.cols;
    ensure_len(
        "RMSNorm input",
        &params.shape,
        expected_len,
        witness.input.len(),
    )?;
    ensure_len(
        "RMSNorm norm_acc",
        &params.shape,
        expected_len,
        witness.norm_acc.len(),
    )?;
    ensure_len(
        "RMSNorm norm",
        &params.shape,
        expected_len,
        witness.norm.len(),
    )?;
    ensure_len(
        "RMSNorm acc",
        &params.shape,
        expected_len,
        witness.acc.len(),
    )?;
    ensure_len(
        "RMSNorm output",
        &params.shape,
        expected_len,
        witness.output.len(),
    )?;
    ensure_len(
        "RMSNorm sum_x2",
        &params.row_shape(),
        params.rows,
        witness.sum_x2.len(),
    )?;
    ensure_len(
        "RMSNorm weight",
        &Shape::new(vec![params.cols]),
        params.cols,
        weight.len(),
    )?;
    for (idx, &sum) in witness.sum_x2.iter().enumerate() {
        let row = &witness.input[idx * params.cols..(idx + 1) * params.cols];
        let expected = row
            .iter()
            .map(|&value| i64::from(value) * i64::from(value))
            .sum::<i64>();
        if sum != expected {
            return Err(ProverError::MatMulAccumulatorMismatch {
                row: idx,
                col: 0,
                expected,
                actual: sum,
            });
        }
    }
    let inv_i64 = witness
        .sum_x2
        .iter()
        .map(|&sum| rms_inv_from_square_sum(sum, params.cols))
        .collect::<Vec<_>>();
    for row in 0..params.rows {
        for col in 0..params.cols {
            let idx = row * params.cols + col;
            let expected_norm_acc = i64::from(witness.input[idx]) * inv_i64[row];
            if witness.norm_acc[idx] != expected_norm_acc {
                return Err(ProverError::MatMulAccumulatorMismatch {
                    row,
                    col,
                    expected: expected_norm_acc,
                    actual: witness.norm_acc[idx],
                });
            }
            let expected_acc = i64::from(witness.norm[idx]) * i64::from(weight[col]);
            if witness.acc[idx] != expected_acc {
                return Err(ProverError::MatMulAccumulatorMismatch {
                    row,
                    col,
                    expected: expected_acc,
                    actual: witness.acc[idx],
                });
            }
        }
    }
    Ok(())
}

fn verify_advice(
    params: &RmsNormParams,
    sum_x2: &[i64],
    weight: &[i32],
) -> std::result::Result<(), ProofVerifyError> {
    if sum_x2.len() != params.rows {
        return Err(ProofVerifyError::InvalidInputLength(
            params.rows,
            sum_x2.len(),
        ));
    }
    if weight.len() != params.cols {
        return Err(ProofVerifyError::InvalidInputLength(
            params.cols,
            weight.len(),
        ));
    }
    Ok(())
}

fn append_sum_x2_advice<F: JoltField, T: Transcript>(sum_x2: &[i64], transcript: &mut T) {
    for &value in sum_x2 {
        transcript.append_scalar(&field_from_i64::<F>(value));
    }
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

fn acc_row_point<F: JoltField>(acc_claim: &Claim<F>, params: &RmsNormParams) -> Vec<F> {
    let row_len = params.row_shape().padded_power_of_two().point_len();
    acc_claim.point[..row_len].to_vec()
}

fn split_row_col_point<'a, F>(shape: &Shape, point: &'a [F]) -> (&'a [F], &'a [F]) {
    let dims = shape.dims();
    let col_len = dims
        .last()
        .copied()
        .unwrap_or(1)
        .next_power_of_two()
        .trailing_zeros() as usize;
    let row_len = point.len() - col_len;
    (&point[..row_len], &point[row_len..])
}

fn row_eq_lifted<F: JoltField>(row_point: &[F], params: &RmsNormParams) -> Vec<F> {
    let row_eq = EqPolynomial::<F>::evals(row_point);
    let padded_cols = params.cols.next_power_of_two();
    let mut out = Vec::with_capacity(row_eq.len() * padded_cols);
    for value in row_eq {
        out.extend(std::iter::repeat_n(value, padded_cols));
    }
    out
}

fn coeff_tensor<F: JoltField>(inv_rms: &[F], params: &RmsNormParams) -> Vec<F> {
    let mut out = vec![F::zero(); params.rows * params.cols];
    for row in 0..params.rows {
        for col in 0..params.cols {
            out[row * params.cols + col] = inv_rms[row];
        }
    }
    out
}

fn inv_rms_from_sum_x2<F: JoltField>(sum_x2: &[i64], cols: usize) -> Vec<F> {
    sum_x2
        .iter()
        .map(|&sum| field_from_i64(rms_inv_from_square_sum(sum, cols)))
        .collect()
}

fn rms_inv_from_square_sum(square_sum: i64, hidden_size: usize) -> i64 {
    let input_scale = 256.0_f64;
    let output_scale = 256.0_f64;
    let mean = square_sum as f64 / hidden_size as f64 / (input_scale * input_scale);
    let inv = 1.0 / (mean + 1e-6).sqrt();
    (inv * output_scale).round() as i64
}

fn eval_i64_advice<F: JoltField>(values: &[i64], shape: &Shape, point: &[F]) -> F {
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

fn eval_field_advice<F: JoltField>(values: &[F], shape: &Shape, point: &[F]) -> F {
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
        out += weight * value;
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

fn rms_norm_sumcheck_id() -> SumcheckId {
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
    fn proves_and_verifies_rmsnorm_round() {
        let params = RmsNormParams::new(
            2,
            2,
            "x",
            "weight",
            "acc",
            "y",
            std::array::from_fn(|idx| format!("frac_bit_{idx}")),
        );
        let input = vec![256, 0, 0, 256];
        let weight = vec![2, 3];
        let sum_x2 = vec![65536, 65536];
        let inv = rms_inv_from_square_sum(sum_x2[0], params.cols);
        let norm_acc = vec![
            i64::from(input[0]) * inv,
            i64::from(input[1]) * inv,
            i64::from(input[2]) * inv,
            i64::from(input[3]) * inv,
        ];
        let norm = norm_acc
            .iter()
            .map(|&value| ((value + 128).div_euclid(256)) as i32)
            .collect::<Vec<_>>();
        let norm_frac_bits = std::array::from_fn(|bit| {
            norm_acc
                .iter()
                .map(|value| ((value.rem_euclid(256) >> bit) & 1) as u8)
                .collect()
        });
        let acc = vec![
            i64::from(norm[0]) * i64::from(weight[0]),
            i64::from(norm[1]) * i64::from(weight[1]),
            i64::from(norm[2]) * i64::from(weight[0]),
            i64::from(norm[3]) * i64::from(weight[1]),
        ];
        let output = acc
            .iter()
            .map(|&value| ((value + 128).div_euclid(256)) as i32)
            .collect::<Vec<_>>();
        let frac_bits = std::array::from_fn(|bit| {
            acc.iter()
                .map(|value| ((value.rem_euclid(256) >> bit) & 1) as u8)
                .collect()
        });
        let point = vec![Fr::from(3u64), Fr::from(5u64)];
        let output_claim = Claim {
            tensor: TensorId::new("y"),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: point.clone(),
            value: eval_i32(&output, &params.shape, &point),
        };
        let witness = RmsNormWitness {
            input,
            sum_x2,
            norm_acc,
            norm,
            norm_frac_bits,
            acc,
            output,
            frac_bits,
        };

        let mut prover_transcript = Blake2bTranscript::default();
        let (proof, input_claim, frac_bit_claims) = prove_rmsnorm_round::<Fr, _>(
            vec![output_claim.clone()],
            &witness,
            &weight,
            &params,
            &mut prover_transcript,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        let (verified_input_claim, verified_frac_bit_claims) = verify_rmsnorm_round::<Fr, _>(
            vec![output_claim],
            &proof,
            &weight,
            &params,
            &mut verifier_transcript,
        )
        .unwrap();

        assert_eq!(verified_input_claim, input_claim);
        assert_eq!(verified_frac_bit_claims, frac_bit_claims);
        assert_eq!(verified_input_claim.tensor.0, "x");
    }

    #[test]
    fn proves_and_verifies_rmsnorm_round_non_power_rows() {
        let params = RmsNormParams::new(
            3,
            4,
            "x",
            "weight",
            "acc",
            "y",
            std::array::from_fn(|idx| format!("frac_bit_{idx}")),
        );
        let input = (1..=12).map(|v| v * 7).collect::<Vec<_>>();
        let weight = vec![2, -3, 5, 7];
        let sum_x2 = input
            .chunks_exact(4)
            .map(|row| row.iter().map(|&v| i64::from(v) * i64::from(v)).sum())
            .collect::<Vec<_>>();
        let inv = sum_x2
            .iter()
            .map(|&sum| rms_inv_from_square_sum(sum, params.cols))
            .collect::<Vec<_>>();
        let norm_acc = input
            .chunks_exact(4)
            .zip(&inv)
            .flat_map(|(row, &inv)| row.iter().map(move |&x| i64::from(x) * inv))
            .collect::<Vec<_>>();
        let norm = norm_acc
            .iter()
            .map(|&value| ((value + 128).div_euclid(256)) as i32)
            .collect::<Vec<_>>();
        let norm_frac_bits = std::array::from_fn(|bit| {
            norm_acc
                .iter()
                .map(|value| ((value.rem_euclid(256) >> bit) & 1) as u8)
                .collect()
        });
        let acc = norm
            .chunks_exact(4)
            .flat_map(|row| {
                row.iter()
                    .zip(&weight)
                    .map(|(&n, &w)| i64::from(n) * i64::from(w))
            })
            .collect::<Vec<_>>();
        let output = acc
            .iter()
            .map(|&value| ((value + 128).div_euclid(256)) as i32)
            .collect::<Vec<_>>();
        let frac_bits = std::array::from_fn(|bit| {
            acc.iter()
                .map(|value| ((value.rem_euclid(256) >> bit) & 1) as u8)
                .collect()
        });
        let point = vec![
            Fr::from(3u64),
            Fr::from(5u64),
            Fr::from(7u64),
            Fr::from(11u64),
        ];
        let output_claim = Claim {
            tensor: TensorId::new("y"),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: point.clone(),
            value: eval_i32(&output, &params.shape, &point),
        };
        let witness = RmsNormWitness {
            input,
            sum_x2,
            norm_acc,
            norm,
            norm_frac_bits,
            acc,
            output,
            frac_bits,
        };

        let mut prover_transcript = Blake2bTranscript::default();
        let (proof, input_claim, frac_bit_claims) = prove_rmsnorm_round::<Fr, _>(
            vec![output_claim.clone()],
            &witness,
            &weight,
            &params,
            &mut prover_transcript,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        let (verified_input_claim, verified_frac_bit_claims) = verify_rmsnorm_round::<Fr, _>(
            vec![output_claim],
            &proof,
            &weight,
            &params,
            &mut verifier_transcript,
        )
        .unwrap();

        assert_eq!(verified_input_claim, input_claim);
        assert_eq!(verified_frac_bit_claims, frac_bit_claims);
    }

    #[test]
    fn proves_and_verifies_rmsnorm_round_non_power_rows_fanout() {
        let params = RmsNormParams::new(
            3,
            4,
            "x",
            "weight",
            "acc",
            "y",
            std::array::from_fn(|idx| format!("frac_bit_{idx}")),
        );
        let input = (1..=12).map(|v| v * 7).collect::<Vec<_>>();
        let weight = vec![2, -3, 5, 7];
        let sum_x2 = input
            .chunks_exact(4)
            .map(|row| row.iter().map(|&v| i64::from(v) * i64::from(v)).sum())
            .collect::<Vec<_>>();
        let inv = sum_x2
            .iter()
            .map(|&sum| rms_inv_from_square_sum(sum, params.cols))
            .collect::<Vec<_>>();
        let norm_acc = input
            .chunks_exact(4)
            .zip(&inv)
            .flat_map(|(row, &inv)| row.iter().map(move |&x| i64::from(x) * inv))
            .collect::<Vec<_>>();
        let norm = norm_acc
            .iter()
            .map(|&value| ((value + 128).div_euclid(256)) as i32)
            .collect::<Vec<_>>();
        let norm_frac_bits = std::array::from_fn(|bit| {
            norm_acc
                .iter()
                .map(|value| ((value.rem_euclid(256) >> bit) & 1) as u8)
                .collect()
        });
        let acc = norm
            .chunks_exact(4)
            .flat_map(|row| {
                row.iter()
                    .zip(&weight)
                    .map(|(&n, &w)| i64::from(n) * i64::from(w))
            })
            .collect::<Vec<_>>();
        let output = acc
            .iter()
            .map(|&value| ((value + 128).div_euclid(256)) as i32)
            .collect::<Vec<_>>();
        let frac_bits = std::array::from_fn(|bit| {
            acc.iter()
                .map(|value| ((value.rem_euclid(256) >> bit) & 1) as u8)
                .collect()
        });
        let points = [
            vec![
                Fr::from(3u64),
                Fr::from(5u64),
                Fr::from(7u64),
                Fr::from(11u64),
            ],
            vec![
                Fr::from(13u64),
                Fr::from(17u64),
                Fr::from(19u64),
                Fr::from(23u64),
            ],
        ];
        let output_claims = points
            .iter()
            .map(|point| Claim {
                tensor: TensorId::new("y"),
                logical_shape: params.shape.clone(),
                domain_shape: params.shape.padded_power_of_two(),
                point: point.clone(),
                value: eval_i32(&output, &params.shape, point),
            })
            .collect::<Vec<_>>();
        let witness = RmsNormWitness {
            input,
            sum_x2,
            norm_acc,
            norm,
            norm_frac_bits,
            acc,
            output,
            frac_bits,
        };

        let mut prover_transcript = Blake2bTranscript::default();
        let (proof, input_claim, frac_bit_claims) = prove_rmsnorm_round::<Fr, _>(
            output_claims.clone(),
            &witness,
            &weight,
            &params,
            &mut prover_transcript,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        let (verified_input_claim, verified_frac_bit_claims) = verify_rmsnorm_round::<Fr, _>(
            output_claims,
            &proof,
            &weight,
            &params,
            &mut verifier_transcript,
        )
        .unwrap();

        assert_eq!(verified_input_claim, input_claim);
        assert_eq!(verified_frac_bit_claims, frac_bit_claims);
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
}
