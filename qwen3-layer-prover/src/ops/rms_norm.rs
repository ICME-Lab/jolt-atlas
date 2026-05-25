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
    claim::{Claim, LegacyClaim, PcsOpeningRequest, Poly, Shape, TensorId},
    error::{ProverError, Result},
    ops::round::{
        ROUND_FRAC_BITS, RoundParams, RoundProof, RoundWitness, prove_round, verify_round,
    },
    proof::ProveResult,
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
//
// Future optimization note:
//
// RMSNorm still exposes the two round steps as standalone generic round
// protocols:
//
//   norm = round(x * inv_rms / 2^8)
//   output = round(norm * weight / 2^8)
//
// The matmul-like ops already fuse their final round relation into the main
// arithmetic sumcheck and keep only the SHOUT lookup for `rem -> round_bit`.
// RMSNorm can use the same idea later.
//
// A conservative first step is to fuse only the final output round:
//
//   norm * weight + round_bit_out * 2^8 - rem_out = output * 2^8
//
// The stronger version batches both elementwise relations into one sumcheck
// with a random coefficient:
//
//   alpha * (x * inv_rms + round_bit_norm * 2^8 - rem_norm - norm * 2^8)
//       + (norm * weight + round_bit_out * 2^8 - rem_out - output * 2^8)
//     = 0
//
// The `sum_x2 = sum_col x^2` relation should stay separate unless there is a
// clear reason to combine it: it is a row-wise reduction, while the two round
// relations are elementwise.  Combining them would make the protocol harder to
// read for less obvious benefit.

#[derive(Debug, Clone, PartialEq, Eq)]
struct MulByVectorParams {
    shape: Shape,
    x_tensor: TensorId,
    coeff_tensor: TensorId,
    axis: usize,
}

impl MulByVectorParams {
    fn new(
        shape: impl Into<Vec<usize>>,
        x_tensor: impl Into<String>,
        coeff_tensor: impl Into<String>,
        axis: usize,
    ) -> Self {
        Self {
            shape: Shape::new(shape),
            x_tensor: TensorId::new(x_tensor),
            coeff_tensor: TensorId::new(coeff_tensor),
            axis,
        }
    }

    fn coeff_shape(&self) -> Shape {
        Shape::new(vec![self.shape.dims()[self.axis]])
    }
}

#[derive(Debug, Clone)]
struct MulByVectorProof<F: JoltField, T: Transcript> {
    sumcheck: SumcheckInstanceProof<F, T>,
    x_opening: F,
    coeff_opening: F,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MulByVectorClaims<F: JoltField> {
    x: LegacyClaim<F>,
    coeff: LegacyClaim<F>,
}

fn prove_mul_by_vector<F, T>(
    y_claim: LegacyClaim<F>,
    x: &[i32],
    coeff: &[i32],
    params: &MulByVectorParams,
    transcript: &mut T,
) -> Result<ProveResult<MulByVectorClaims<F>, MulByVectorProof<F, T>>>
where
    F: JoltField,
    T: Transcript,
{
    validate_mul_by_vector_inputs(&y_claim, x, params)?;
    validate_mul_by_vector_coeff(coeff, params)?;

    let sc_params = MulByVectorSumcheckParams::new(
        params.shape.padded_power_of_two().point_len(),
        y_claim.value,
        y_claim.point.clone(),
        params.clone(),
        coeff.to_vec(),
    );
    let x_poly = padded_i32_tensor(x, &params.shape);
    let coeff_poly = expanded_coeff_tensor(coeff, params);
    let mut prover = MulByVectorSumcheckProver::new(sc_params, x_poly, coeff_poly);
    let mut accumulator = ProverOpeningAccumulator::new();
    let (sumcheck, challenges) = Sumcheck::prove(&mut prover, &mut accumulator, transcript);
    let x_opening = accumulator
        .openings
        .get(&OpeningId::new(VirtualPoly::QwenMulX, mul_sumcheck_id()))
        .map(|(_, value)| *value)
        .ok_or(ProverError::MissingOpening)?;
    let point = normalize_sumcheck_point::<F>(&challenges.into_opening());
    let coeff_point = axis_point(&point, &params.shape, params.axis);
    let coeff_opening = eval_vector_at_point(coeff, coeff_point);

    Ok(ProveResult::new(
        MulByVectorClaims {
            x: LegacyClaim {
                tensor: params.x_tensor.clone(),
                logical_shape: params.shape.clone(),
                domain_shape: params.shape.padded_power_of_two(),
                point: point.clone(),
                value: x_opening,
            },
            coeff: LegacyClaim {
                tensor: params.coeff_tensor.clone(),
                logical_shape: params.coeff_shape(),
                domain_shape: params.coeff_shape().padded_power_of_two(),
                point: coeff_point.to_vec(),
                value: coeff_opening,
            },
        },
        MulByVectorProof {
            sumcheck,
            x_opening,
            coeff_opening,
        },
    ))
}

fn verify_mul_by_vector<F, T>(
    y_claim: LegacyClaim<F>,
    proof: &MulByVectorProof<F, T>,
    coeff: &[i32],
    params: &MulByVectorParams,
    transcript: &mut T,
) -> Result<MulByVectorClaims<F>>
where
    F: JoltField,
    T: Transcript,
{
    validate_mul_by_vector_claim(&y_claim, &params.shape)?;
    validate_mul_by_vector_axis(params)?;
    validate_mul_by_vector_coeff(coeff, params)?;

    let sc_params = MulByVectorSumcheckParams::new(
        params.shape.padded_power_of_two().point_len(),
        y_claim.value,
        y_claim.point.clone(),
        params.clone(),
        coeff.to_vec(),
    );
    let verifier = MulByVectorSumcheckVerifier { params: sc_params };
    let mut accumulator = VerifierOpeningAccumulator::new();
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenMulX, mul_sumcheck_id()),
        (OpeningPoint::<BIG_ENDIAN, F>::default(), proof.x_opening),
    );
    let challenges = Sumcheck::verify(&proof.sumcheck, &verifier, &mut accumulator, transcript)
        .map_err(|_| ProverError::MulMismatch)?;
    let point = normalize_sumcheck_point::<F>(&challenges.into_opening());
    let coeff_point = axis_point(&point, &params.shape, params.axis);
    let expected_coeff = eval_vector_at_point(coeff, coeff_point);
    if proof.coeff_opening != expected_coeff {
        return Err(ProverError::MulMismatch);
    }

    Ok(MulByVectorClaims {
        x: LegacyClaim {
            tensor: params.x_tensor.clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: point.clone(),
            value: proof.x_opening,
        },
        coeff: LegacyClaim {
            tensor: params.coeff_tensor.clone(),
            logical_shape: params.coeff_shape(),
            domain_shape: params.coeff_shape().padded_power_of_two(),
            point: coeff_point.to_vec(),
            value: proof.coeff_opening,
        },
    })
}

struct MulByVectorSumcheckParams<F: JoltField> {
    num_rounds: usize,
    input_claim: F,
    y_point: Vec<F>,
    op: MulByVectorParams,
    coeff: Vec<i32>,
}

impl<F: JoltField> MulByVectorSumcheckParams<F> {
    fn new(
        num_rounds: usize,
        input_claim: F,
        y_point: Vec<F>,
        op: MulByVectorParams,
        coeff: Vec<i32>,
    ) -> Self {
        Self {
            num_rounds,
            input_claim,
            y_point,
            op,
            coeff,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for MulByVectorSumcheckParams<F> {
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

struct MulByVectorSumcheckProver<F: JoltField> {
    eq_y: MultilinearPolynomial<F>,
    x: MultilinearPolynomial<F>,
    coeff: MultilinearPolynomial<F>,
    params: MulByVectorSumcheckParams<F>,
}

impl<F: JoltField> MulByVectorSumcheckProver<F> {
    fn new(params: MulByVectorSumcheckParams<F>, x: Vec<F>, coeff: Vec<F>) -> Self {
        let eq_y = MultilinearPolynomial::from(EqPolynomial::<F>::evals(&params.y_point));
        Self {
            eq_y,
            x: MultilinearPolynomial::from(x),
            coeff: MultilinearPolynomial::from(coeff),
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for MulByVectorSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let mut evals = [F::zero(); 3];
        for g in 0..self.x.len() / 2 {
            let e0 = self.eq_y.get_bound_coeff(2 * g);
            let e1 = self.eq_y.get_bound_coeff(2 * g + 1);
            let x0 = self.x.get_bound_coeff(2 * g);
            let x1 = self.x.get_bound_coeff(2 * g + 1);
            let c0 = self.coeff.get_bound_coeff(2 * g);
            let c1 = self.coeff.get_bound_coeff(2 * g + 1);
            evals[0] += e0 * x0 * c0;
            evals[1] += (e1 + e1 - e0) * (x1 + x1 - x0) * (c1 + c1 - c0);
            evals[2] += (e1 * F::from_u64(3) - e0 - e0)
                * (x1 * F::from_u64(3) - x0 - x0)
                * (c1 * F::from_u64(3) - c0 - c0);
        }
        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_y.bind_parallel(r_j, BindingOrder::LowToHigh);
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
            OpeningId::new(VirtualPoly::QwenMulX, mul_sumcheck_id()),
            point,
            self.x.final_claim(),
        );
    }
}

struct MulByVectorSumcheckVerifier<F: JoltField> {
    params: MulByVectorSumcheckParams<F>,
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for MulByVectorSumcheckVerifier<F>
{
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
                VirtualPoly::QwenMulX,
                mul_sumcheck_id(),
            ))
            .1;
        let point = normalize_sumcheck_point::<F>(&sumcheck_challenges.into_opening());
        let coeff_point = axis_point(&point, &self.params.op.shape, self.params.op.axis);
        EqPolynomial::mle(&self.params.y_point, &point)
            * x
            * eval_vector_at_point(&self.params.coeff, coeff_point)
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
            OpeningId::new(VirtualPoly::QwenMulX, mul_sumcheck_id()),
            point,
        );
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RmsNormParams {
    pub round: RoundParams,
    pub norm_round: RoundParams,
    weight_mul: MulByVectorParams,
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

    pub fn with_lookup_sites(mut self, output_round_site: usize, norm_round_site: usize) -> Self {
        self.round.lookup_site = output_round_site;
        self.norm_round.lookup_site = norm_round_site;
        self
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

#[derive(Debug, Clone)]
pub struct RmsNormWitness<'a> {
    pub input: &'a [i32],
    pub sum_x2: &'a [i64],
    pub norm_acc: &'a [i64],
    pub norm: &'a [i32],
    pub norm_frac_bits: [&'a [u8]; ROUND_FRAC_BITS],
    pub acc: &'a [i64],
    pub output: &'a [i32],
    pub frac_bits: [&'a [u8]; ROUND_FRAC_BITS],
}

#[derive(Debug, Clone)]
pub struct RmsNormProof<F: JoltField, T: Transcript> {
    pub round: RoundProof<F, T>,
    weight_mul: MulByVectorProof<F, T>,
    pub norm_round: RoundProof<F, T>,
    pub sumcheck: SumcheckInstanceProof<F, T>,
    pub input_opening: F,
    pub sum_x2: Vec<i64>,
}

impl<F: JoltField, T: Transcript> RmsNormProof<F, T> {
    pub fn pcs_opening_requests(&self) -> Vec<PcsOpeningRequest<F>> {
        let mut out = self.round.pcs_opening_requests();
        out.extend(self.norm_round.pcs_opening_requests());
        out
    }
}

pub fn prove_rmsnorm_round_from_witness<F, T>(
    output_claims: Vec<LegacyClaim<F>>,
    witness: &RmsNormWitness<'_>,
    weight: &[i32],
    params: &RmsNormParams,
    transcript: &mut T,
) -> Result<(
    RmsNormProof<F, T>,
    LegacyClaim<F>,
    LegacyClaim<F>,
    Vec<PcsOpeningRequest<F>>,
    Vec<PcsOpeningRequest<F>>,
)>
where
    F: JoltField,
    T: Transcript,
{
    validate_inputs(witness, weight, params)?;
    let round_witness = RoundWitness::from_input_output(witness.acc, witness.output);
    let (round_proof, acc_claim, round_ra) =
        prove_round(output_claims, &round_witness, &params.round, transcript)?;

    let weight_mul_result = prove_mul_by_vector(
        acc_claim,
        witness.norm,
        weight,
        &params.weight_mul,
        transcript,
    )?;
    let norm_claim = weight_mul_result.claims.x;
    let weight_claim = weight_mul_result.claims.coeff;

    let norm_round_witness = RoundWitness::from_input_output(witness.norm_acc, witness.norm);
    let (norm_round_proof, norm_acc_claim, norm_ra) = prove_round(
        vec![norm_claim],
        &norm_round_witness,
        &params.norm_round,
        transcript,
    )?;

    append_sum_x2_advice::<F, T>(witness.sum_x2, transcript);
    let gamma = transcript.challenge_scalar();
    let row_point = acc_row_point(&norm_acc_claim, params);
    let sum_x2_eval = eval_i64_advice(witness.sum_x2, &params.row_shape(), &row_point);
    let input_claim = norm_acc_claim.value + gamma * sum_x2_eval;
    let inv_rms = inv_rms_from_sum_x2::<F>(witness.sum_x2, params.cols);
    let coeff = coeff_tensor(&inv_rms, params);
    let x_poly = padded_i32_tensor(witness.input, &params.shape);
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
            sum_x2: witness.sum_x2.to_vec(),
        },
        LegacyClaim {
            tensor: params.input_tensor.clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point,
            value: input_opening,
        },
        weight_claim,
        norm_ra,
        round_ra,
    ))
}

pub fn prove_rmsnorm_round<F, T, C>(
    output_claims: Vec<Claim<F, C>>,
    input_poly: Poly<F, C>,
    weight_poly: Poly<F, C>,
    norm_round_ra: Vec<Poly<F, C>>,
    output_round_ra: Vec<Poly<F, C>>,
    params: &RmsNormParams,
    transcript: &mut T,
) -> Result<(
    RmsNormProof<F, T>,
    Claim<F, C>,
    Claim<F, C>,
    Vec<Claim<F, C>>,
    Vec<Claim<F, C>>,
)>
where
    F: JoltField,
    T: Transcript,
    C: Clone,
{
    if output_claims.is_empty() {
        return Err(ProverError::InvalidInput(
            "RMSNorm requires at least one output claim".to_string(),
        ));
    }
    for claim in &output_claims {
        validate_claim_shape(claim, &params.shape, "RMSNorm output claim")?;
    }
    validate_poly_shape(&input_poly, &params.shape, "RMSNorm input")?;
    validate_poly_shape(
        &weight_poly,
        &Shape::new(vec![params.cols]),
        "RMSNorm weight",
    )?;

    let input = logical_i32_values(&input_poly, &params.shape)?;
    let weight = logical_i32_values(&weight_poly, &Shape::new(vec![params.cols]))?;
    let output = logical_i32_values(&output_claims[0].poly, &params.shape)?;
    let sum_x2 = rms_sum_x2(&input, params);
    let inv = sum_x2
        .iter()
        .map(|&sum| rms_inv_from_square_sum(sum, params.cols))
        .collect::<Vec<_>>();
    let norm_acc = input
        .chunks_exact(params.cols)
        .zip(&inv)
        .flat_map(|(row, &inv)| row.iter().map(move |&x| i64::from(x) * inv))
        .collect::<Vec<_>>();
    let norm = norm_acc
        .iter()
        .map(|&value| round_q8(value))
        .collect::<Vec<_>>();
    let acc = norm
        .chunks_exact(params.cols)
        .flat_map(|row| {
            row.iter()
                .zip(&weight)
                .map(|(&n, &w)| i64::from(n) * i64::from(w))
        })
        .collect::<Vec<_>>();

    let legacy_claims = output_claims
        .iter()
        .map(|claim| LegacyClaim {
            tensor: params.output_tensor.clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: claim.point.clone(),
            value: claim.value,
        })
        .collect::<Vec<_>>();
    let empty_bits: [&[u8]; ROUND_FRAC_BITS] = std::array::from_fn(|_| &[][..]);
    let witness = RmsNormWitness {
        input: &input,
        sum_x2: &sum_x2,
        norm_acc: &norm_acc,
        norm: &norm,
        norm_frac_bits: empty_bits,
        acc: &acc,
        output: &output,
        frac_bits: empty_bits,
    };
    let (proof, input, weight, norm_round_requests, output_round_requests) =
        prove_rmsnorm_round_from_witness(legacy_claims, &witness, &weight, params, transcript)?;

    Ok((
        proof,
        Claim::new(input_poly, input.point, input.value),
        Claim::new(weight_poly, weight.point, weight.value),
        claims_from_requests(&norm_round_ra, norm_round_requests)?,
        claims_from_requests(&output_round_ra, output_round_requests)?,
    ))
}

pub fn verify_rmsnorm_round<F, T>(
    output_claims: Vec<LegacyClaim<F>>,
    proof: &RmsNormProof<F, T>,
    weight: &[i32],
    params: &RmsNormParams,
    transcript: &mut T,
) -> std::result::Result<
    (
        LegacyClaim<F>,
        LegacyClaim<F>,
        Vec<PcsOpeningRequest<F>>,
        Vec<PcsOpeningRequest<F>>,
    ),
    ProofVerifyError,
>
where
    F: JoltField,
    T: Transcript,
{
    verify_advice(params, &proof.sum_x2, weight)?;
    let (acc_claim, round_ra) =
        verify_round(output_claims, &proof.round, &params.round, transcript)?;
    let mul_claims = verify_mul_by_vector(
        acc_claim,
        &proof.weight_mul,
        weight,
        &params.weight_mul,
        transcript,
    )
    .map_err(|_| ProofVerifyError::InvalidInputLength(1, 0))?;
    let norm_claim = mul_claims.x;
    let weight_claim = mul_claims.coeff;
    let (norm_acc_claim, norm_ra) = verify_round(
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
        LegacyClaim {
            tensor: params.input_tensor.clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point,
            value: proof.input_opening,
        },
        weight_claim,
        norm_ra,
        round_ra,
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

fn validate_mul_by_vector_inputs<F: JoltField>(
    y_claim: &LegacyClaim<F>,
    x: &[i32],
    params: &MulByVectorParams,
) -> Result<()> {
    validate_mul_by_vector_shape(&params.shape)?;
    if x.len() != params.shape.numel() {
        return Err(ProverError::TensorLenMismatch {
            name: "X",
            shape: params.shape.0.clone(),
            expected: params.shape.numel(),
            actual: x.len(),
        });
    }
    validate_mul_by_vector_claim(y_claim, &params.shape)
}

fn validate_mul_by_vector_claim<F: JoltField>(claim: &LegacyClaim<F>, shape: &Shape) -> Result<()> {
    validate_mul_by_vector_shape(shape)?;
    if claim.logical_shape != *shape {
        return Err(ProverError::ShapeMismatch {
            name: "Y claim",
            expected: shape.0.clone(),
            actual: claim.logical_shape.0.clone(),
        });
    }
    let expected_domain = shape.padded_power_of_two();
    if claim.domain_shape != expected_domain {
        return Err(ProverError::ShapeMismatch {
            name: "Y claim domain",
            expected: expected_domain.0,
            actual: claim.domain_shape.0.clone(),
        });
    }
    let expected_point_len = claim.domain_shape.point_len();
    if claim.point.len() != expected_point_len {
        return Err(ProverError::ShapeMismatch {
            name: "Y claim point",
            expected: vec![expected_point_len],
            actual: vec![claim.point.len()],
        });
    }
    Ok(())
}

fn validate_mul_by_vector_shape(shape: &Shape) -> Result<()> {
    if shape.dims().contains(&0) {
        return Err(ProverError::InvalidTensorShape(shape.0.clone()));
    }
    Ok(())
}

fn validate_mul_by_vector_axis(params: &MulByVectorParams) -> Result<()> {
    if params.axis >= params.shape.dims().len() {
        return Err(ProverError::InvalidAxis {
            axis: params.axis,
            rank: params.shape.dims().len(),
        });
    }
    Ok(())
}

fn validate_mul_by_vector_coeff(coeff: &[i32], params: &MulByVectorParams) -> Result<()> {
    validate_mul_by_vector_axis(params)?;
    if coeff.len() != params.shape.dims()[params.axis] {
        return Err(ProverError::TensorLenMismatch {
            name: "coeff",
            shape: vec![params.shape.dims()[params.axis]],
            expected: params.shape.dims()[params.axis],
            actual: coeff.len(),
        });
    }
    Ok(())
}

fn expanded_coeff_tensor<F: JoltField>(coeff: &[i32], params: &MulByVectorParams) -> Vec<F> {
    let padded_dims = padded_dims(&params.shape);
    let len = padded_dims.iter().product();
    let padded_strides = row_major_strides(&padded_dims);
    let mut out = vec![F::zero(); len];

    for (flat, out_value) in out.iter_mut().enumerate() {
        let mut axis_coord = 0;
        for (dim, &stride) in padded_strides.iter().enumerate() {
            let coord = (flat / stride) % padded_dims[dim];
            if dim == params.axis {
                axis_coord = coord;
            }
        }
        // The coefficient is a function only of `axis`. Padding in other
        // dimensions must still carry the same coefficient so that
        // y = x * coeff(axis) remains true on padded boolean points where
        // x and y are both zero. Only padding on the coefficient axis itself
        // is outside the vector domain and therefore uses zero.
        if axis_coord < params.shape.dims()[params.axis] {
            *out_value = F::from_i32(coeff[axis_coord]);
        }
    }
    out
}

fn padded_dims(shape: &Shape) -> Vec<usize> {
    shape
        .dims()
        .iter()
        .map(|dim| dim.next_power_of_two())
        .collect()
}

fn eval_vector_at_point<F: JoltField>(values: &[i32], point: &[F]) -> F {
    let eq = EqPolynomial::<F>::evals(point);
    values
        .iter()
        .enumerate()
        .map(|(idx, &value)| eq[idx] * F::from_i32(value))
        .sum()
}

fn axis_point<'a, F>(point: &'a [F], shape: &Shape, axis: usize) -> &'a [F] {
    let mut offset = 0;
    for (dim_idx, &dim) in shape.dims().iter().enumerate() {
        let vars = dim.next_power_of_two().trailing_zeros() as usize;
        if dim_idx == axis {
            return &point[offset..offset + vars];
        }
        offset += vars;
    }
    &point[0..0]
}

fn mul_sumcheck_id() -> SumcheckId {
    SumcheckId::NodeExecution(0)
}

fn validate_inputs(
    witness: &RmsNormWitness<'_>,
    weight: &[i32],
    params: &RmsNormParams,
) -> Result<()> {
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
    for (row, &inv) in inv_i64.iter().enumerate().take(params.rows) {
        for (col, &weight_col) in weight.iter().enumerate().take(params.cols) {
            let idx = row * params.cols + col;
            let expected_norm_acc = i64::from(witness.input[idx]) * inv;
            if witness.norm_acc[idx] != expected_norm_acc {
                return Err(ProverError::MatMulAccumulatorMismatch {
                    row,
                    col,
                    expected: expected_norm_acc,
                    actual: witness.norm_acc[idx],
                });
            }
            let expected_acc = i64::from(witness.norm[idx]) * i64::from(weight_col);
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

fn validate_claim_shape<F: JoltField, C>(
    claim: &Claim<F, C>,
    shape: &Shape,
    name: &'static str,
) -> Result<()> {
    if claim.point.len() != shape.padded_power_of_two().point_len() {
        return Err(ProverError::ShapeMismatch {
            name,
            expected: vec![shape.padded_power_of_two().point_len()],
            actual: vec![claim.point.len()],
        });
    }
    Ok(())
}

fn validate_poly_shape<F: JoltField, C>(
    poly: &Poly<F, C>,
    shape: &Shape,
    name: &'static str,
) -> Result<()> {
    let expected = shape.padded_power_of_two().numel();
    if poly.data.len() != expected {
        return Err(ProverError::TensorLenMismatch {
            name,
            shape: shape.padded_power_of_two().0,
            expected,
            actual: poly.data.len(),
        });
    }
    Ok(())
}

fn logical_i32_values<F: JoltField, C>(poly: &Poly<F, C>, shape: &Shape) -> Result<Vec<i32>> {
    let padded_dims = shape.padded_power_of_two().0;
    let strides = row_major_strides(shape.dims());
    let padded_strides = row_major_strides(&padded_dims);
    let mut out = Vec::with_capacity(shape.numel());
    for flat in 0..shape.numel() {
        let mut padded_flat = 0;
        for (dim, (&stride, &padded_stride)) in strides.iter().zip(&padded_strides).enumerate() {
            let coord = (flat / stride) % shape.dims()[dim];
            padded_flat += coord * padded_stride;
        }
        out.push(poly.data.get_coeff_i64(padded_flat) as i32);
    }
    Ok(out)
}

fn claims_from_requests<F: JoltField, C: Clone>(
    polys: &[Poly<F, C>],
    requests: Vec<PcsOpeningRequest<F>>,
) -> Result<Vec<Claim<F, C>>> {
    if polys.is_empty() {
        return Ok(Vec::new());
    }
    if requests.len() % polys.len() != 0 {
        return Err(ProverError::InvalidInput(format!(
            "RA opening count mismatch: openings {} is not a multiple of chunks {}",
            requests.len(),
            polys.len()
        )));
    }
    Ok(requests
        .into_iter()
        .enumerate()
        .map(|(idx, opening)| {
            Claim::new(
                polys[idx % polys.len()].clone(),
                opening.point,
                opening.value,
            )
        })
        .collect())
}

fn rms_sum_x2(input: &[i32], params: &RmsNormParams) -> Vec<i64> {
    input
        .chunks_exact(params.cols)
        .map(|row| row.iter().map(|&v| i64::from(v) * i64::from(v)).sum())
        .collect()
}

fn round_q8(value: i64) -> i32 {
    let rem = value.rem_euclid(1_i64 << ROUND_FRAC_BITS);
    let bit = i64::from(rem >= (1_i64 << (ROUND_FRAC_BITS - 1)));
    ((value + bit * (1_i64 << ROUND_FRAC_BITS) - rem) / (1_i64 << ROUND_FRAC_BITS)) as i32
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

fn acc_row_point<F: JoltField>(acc_claim: &LegacyClaim<F>, params: &RmsNormParams) -> Vec<F> {
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
        let norm_frac_bits: [Vec<u8>; ROUND_FRAC_BITS] = std::array::from_fn(|bit| {
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
        let frac_bits: [Vec<u8>; ROUND_FRAC_BITS] = std::array::from_fn(|bit| {
            acc.iter()
                .map(|value| ((value.rem_euclid(256) >> bit) & 1) as u8)
                .collect()
        });
        let point = vec![Fr::from(3u64), Fr::from(5u64)];
        let output_claim = LegacyClaim {
            tensor: TensorId::new("y"),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: point.clone(),
            value: eval_i32(&output, &params.shape, &point),
        };
        let witness = RmsNormWitness {
            input: &input,
            sum_x2: &sum_x2,
            norm_acc: &norm_acc,
            norm: &norm,
            norm_frac_bits: norm_frac_bits.each_ref().map(Vec::as_slice),
            acc: &acc,
            output: &output,
            frac_bits: frac_bits.each_ref().map(Vec::as_slice),
        };

        let mut prover_transcript = Blake2bTranscript::default();
        let (proof, input_claim, weight_claim, norm_round_ra, output_round_ra) =
            prove_rmsnorm_round_from_witness::<Fr, _>(
                vec![output_claim.clone()],
                &witness,
                &weight,
                &params,
                &mut prover_transcript,
            )
            .unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        let (
            verified_input_claim,
            verified_weight_claim,
            verified_norm_round_ra,
            verified_output_round_ra,
        ) = verify_rmsnorm_round::<Fr, _>(
            vec![output_claim],
            &proof,
            &weight,
            &params,
            &mut verifier_transcript,
        )
        .unwrap();

        assert_eq!(verified_input_claim, input_claim);
        assert_eq!(verified_weight_claim, weight_claim);
        assert_eq!(verified_norm_round_ra, norm_round_ra);
        assert_eq!(verified_output_round_ra, output_round_ra);
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
        let norm_frac_bits: [Vec<u8>; ROUND_FRAC_BITS] = std::array::from_fn(|bit| {
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
        let frac_bits: [Vec<u8>; ROUND_FRAC_BITS] = std::array::from_fn(|bit| {
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
        let output_claim = LegacyClaim {
            tensor: TensorId::new("y"),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: point.clone(),
            value: eval_i32(&output, &params.shape, &point),
        };
        let witness = RmsNormWitness {
            input: &input,
            sum_x2: &sum_x2,
            norm_acc: &norm_acc,
            norm: &norm,
            norm_frac_bits: norm_frac_bits.each_ref().map(Vec::as_slice),
            acc: &acc,
            output: &output,
            frac_bits: frac_bits.each_ref().map(Vec::as_slice),
        };

        let mut prover_transcript = Blake2bTranscript::default();
        let (proof, input_claim, weight_claim, norm_round_ra, output_round_ra) =
            prove_rmsnorm_round_from_witness::<Fr, _>(
                vec![output_claim.clone()],
                &witness,
                &weight,
                &params,
                &mut prover_transcript,
            )
            .unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        let (
            verified_input_claim,
            verified_weight_claim,
            verified_norm_round_ra,
            verified_output_round_ra,
        ) = verify_rmsnorm_round::<Fr, _>(
            vec![output_claim],
            &proof,
            &weight,
            &params,
            &mut verifier_transcript,
        )
        .unwrap();

        assert_eq!(verified_input_claim, input_claim);
        assert_eq!(verified_weight_claim, weight_claim);
        assert_eq!(verified_norm_round_ra, norm_round_ra);
        assert_eq!(verified_output_round_ra, output_round_ra);
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
        let norm_frac_bits: [Vec<u8>; ROUND_FRAC_BITS] = std::array::from_fn(|bit| {
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
        let frac_bits: [Vec<u8>; ROUND_FRAC_BITS] = std::array::from_fn(|bit| {
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
            .map(|point| LegacyClaim {
                tensor: TensorId::new("y"),
                logical_shape: params.shape.clone(),
                domain_shape: params.shape.padded_power_of_two(),
                point: point.clone(),
                value: eval_i32(&output, &params.shape, point),
            })
            .collect::<Vec<_>>();
        let witness = RmsNormWitness {
            input: &input,
            sum_x2: &sum_x2,
            norm_acc: &norm_acc,
            norm: &norm,
            norm_frac_bits: norm_frac_bits.each_ref().map(Vec::as_slice),
            acc: &acc,
            output: &output,
            frac_bits: frac_bits.each_ref().map(Vec::as_slice),
        };

        let mut prover_transcript = Blake2bTranscript::default();
        let (proof, input_claim, weight_claim, norm_round_ra, output_round_ra) =
            prove_rmsnorm_round_from_witness::<Fr, _>(
                output_claims.clone(),
                &witness,
                &weight,
                &params,
                &mut prover_transcript,
            )
            .unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        let (
            verified_input_claim,
            verified_weight_claim,
            verified_norm_round_ra,
            verified_output_round_ra,
        ) = verify_rmsnorm_round::<Fr, _>(
            output_claims,
            &proof,
            &weight,
            &params,
            &mut verifier_transcript,
        )
        .unwrap();

        assert_eq!(verified_input_claim, input_claim);
        assert_eq!(verified_weight_claim, weight_claim);
        assert_eq!(verified_norm_round_ra, norm_round_ra);
        assert_eq!(verified_output_round_ra, output_round_ra);
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
