use common::VirtualPoly;
// Design note for future us:
//
// There are two different "multiplication by known data" cases here.
//
// `mul_const` is multiplication by one scalar.  That commutes with the MLE, so
// the verifier can check `Y(r) = c * X(r)` directly and return one `X` claim.
//
// `mul_by_vector` is elementwise multiplication by an index-dependent vector,
// such as RMSNorm gamma.  Even though this is a linear map over the full tensor,
// `MLE(X * gamma)(r) != MLE(X)(r) * MLE(gamma)(r)` in general.  Therefore it
// needs one sumcheck over the full logical domain:
//     Y(r) = sum_i eq(i, r) * X(i) * gamma(axis(i)).
// The coefficient vector is public/known to the verifier, while `X` is not; the
// verifier returns a single `X` opening claim at the sumcheck point.
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
};

use crate::{
    claim::{Claim, Shape, TensorId},
    error::{ProverError, Result},
    proof::ProveResult,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MulConstParams<F: JoltField> {
    pub shape: Shape,
    pub x_tensor: TensorId,
    pub coeff: F,
}

impl<F: JoltField> MulConstParams<F> {
    pub fn new(shape: impl Into<Vec<usize>>, x_tensor: impl Into<String>, coeff: F) -> Self {
        Self {
            shape: Shape::new(shape),
            x_tensor: TensorId::new(x_tensor),
            coeff,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MulByVectorParams {
    pub shape: Shape,
    pub x_tensor: TensorId,
    pub coeff_tensor: TensorId,
    pub axis: usize,
}

impl MulByVectorParams {
    pub fn new(
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

    pub fn coeff_shape(&self) -> Shape {
        Shape::new(vec![self.shape.dims()[self.axis]])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MulProof<F: JoltField> {
    pub x_opening: F,
}

#[derive(Debug, Clone)]
pub struct MulByVectorProof<F: JoltField, T: Transcript> {
    pub sumcheck: SumcheckInstanceProof<F, T>,
    pub x_opening: F,
    pub coeff_opening: F,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MulClaim<F> {
    pub x: Claim<F>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MulByVectorClaims<F> {
    pub x: Claim<F>,
    pub coeff: Claim<F>,
}

pub fn prove_mul_const<F, T>(
    y_claim: Claim<F>,
    x: &[i32],
    params: &MulConstParams<F>,
    transcript: &mut T,
) -> Result<ProveResult<MulClaim<F>, MulProof<F>>>
where
    F: JoltField,
    T: Transcript,
{
    validate_tensor_inputs(&y_claim, x, &params.shape, "X")?;
    let x_opening = eval_tensor_at_point(x, &params.shape, &y_claim.point);
    if y_claim.value != x_opening * params.coeff {
        return Err(ProverError::MulMismatch);
    }
    transcript.append_scalar(&x_opening);

    Ok(ProveResult::new(
        MulClaim {
            x: Claim {
                tensor: params.x_tensor.clone(),
                logical_shape: params.shape.clone(),
                domain_shape: params.shape.padded_power_of_two(),
                point: y_claim.point,
                value: x_opening,
            },
        },
        MulProof { x_opening },
    ))
}

pub fn verify_mul_const<F, T>(
    y_claim: Claim<F>,
    proof: &MulProof<F>,
    params: &MulConstParams<F>,
    transcript: &mut T,
) -> Result<MulClaim<F>>
where
    F: JoltField,
    T: Transcript,
{
    validate_claim(&y_claim, &params.shape)?;
    transcript.append_scalar(&proof.x_opening);
    if y_claim.value != proof.x_opening * params.coeff {
        return Err(ProverError::MulMismatch);
    }
    Ok(MulClaim {
        x: Claim {
            tensor: params.x_tensor.clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: y_claim.point,
            value: proof.x_opening,
        },
    })
}

pub fn prove_mul_by_vector<F, T>(
    y_claim: Claim<F>,
    x: &[i32],
    coeff: &[i32],
    params: &MulByVectorParams,
    transcript: &mut T,
) -> Result<ProveResult<MulByVectorClaims<F>, MulByVectorProof<F, T>>>
where
    F: JoltField,
    T: Transcript,
{
    validate_tensor_inputs(&y_claim, x, &params.shape, "X")?;
    validate_axis(params)?;
    if coeff.len() != params.shape.dims()[params.axis] {
        return Err(ProverError::TensorLenMismatch {
            name: "coeff",
            shape: vec![params.shape.dims()[params.axis]],
            expected: params.shape.dims()[params.axis],
            actual: coeff.len(),
        });
    }

    let sc_params = MulByVectorSumcheckParams::new(
        params.shape.padded_power_of_two().point_len(),
        y_claim.value,
        y_claim.point.clone(),
        params.clone(),
        coeff.to_vec(),
    );
    let x_poly = padded_tensor(x, &params.shape);
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
            x: Claim {
                tensor: params.x_tensor.clone(),
                logical_shape: params.shape.clone(),
                domain_shape: params.shape.padded_power_of_two(),
                point: point.clone(),
                value: x_opening,
            },
            coeff: Claim {
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

pub fn verify_mul_by_vector<F, T>(
    y_claim: Claim<F>,
    proof: &MulByVectorProof<F, T>,
    coeff: &[i32],
    params: &MulByVectorParams,
    transcript: &mut T,
) -> Result<MulByVectorClaims<F>>
where
    F: JoltField,
    T: Transcript,
{
    validate_claim(&y_claim, &params.shape)?;
    validate_axis(params)?;
    if coeff.len() != params.shape.dims()[params.axis] {
        return Err(ProverError::TensorLenMismatch {
            name: "coeff",
            shape: vec![params.shape.dims()[params.axis]],
            expected: params.shape.dims()[params.axis],
            actual: coeff.len(),
        });
    }
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
        x: Claim {
            tensor: params.x_tensor.clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: point.clone(),
            value: proof.x_opening,
        },
        coeff: Claim {
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

fn validate_tensor_inputs<F: JoltField>(
    y_claim: &Claim<F>,
    x: &[i32],
    shape: &Shape,
    name: &'static str,
) -> Result<()> {
    validate_shape(shape)?;
    if x.len() != shape.numel() {
        return Err(ProverError::TensorLenMismatch {
            name,
            shape: shape.0.clone(),
            expected: shape.numel(),
            actual: x.len(),
        });
    }
    validate_claim(y_claim, shape)
}

fn validate_claim<F: JoltField>(claim: &Claim<F>, shape: &Shape) -> Result<()> {
    validate_shape(shape)?;
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

fn validate_shape(shape: &Shape) -> Result<()> {
    if shape.dims().contains(&0) {
        return Err(ProverError::InvalidTensorShape(shape.0.clone()));
    }
    Ok(())
}

fn validate_axis(params: &MulByVectorParams) -> Result<()> {
    if params.axis >= params.shape.dims().len() {
        return Err(ProverError::InvalidAxis {
            axis: params.axis,
            rank: params.shape.dims().len(),
        });
    }
    Ok(())
}

fn padded_tensor<F: JoltField>(values: &[i32], shape: &Shape) -> Vec<F> {
    let padded_dims = padded_dims(shape);
    let len = padded_dims.iter().product();
    let mut out = vec![F::zero(); len];
    let strides = row_major_strides(shape.dims());
    let padded_strides = row_major_strides(&padded_dims);

    for (flat, &value) in values.iter().enumerate() {
        let mut padded_flat = 0;
        for (dim, &stride) in strides.iter().enumerate() {
            let coord = (flat / stride) % shape.dims()[dim];
            padded_flat += coord * padded_strides[dim];
        }
        out[padded_flat] = F::from_i32(value);
    }
    out
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
        // The coefficient is a function only of `axis`.  Padding in other
        // dimensions must still carry the same coefficient so that
        // y = x * coeff(axis) remains true on padded boolean points where
        // x and y are both zero.  Only padding on the coefficient axis itself
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

fn eval_tensor_at_point<F: JoltField>(values: &[i32], shape: &Shape, point: &[F]) -> F {
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
        out += weight * F::from_i32(value);
    }
    out
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
        let vars = log2_ceil(dim);
        if dim_idx == axis {
            return &point[offset..offset + vars];
        }
        offset += vars;
    }
    &point[0..0]
}

fn split_point<'a, F>(shape: &Shape, point: &'a [F]) -> Vec<&'a [F]> {
    let mut out = Vec::with_capacity(shape.dims().len());
    let mut offset = 0;
    for &dim in shape.dims() {
        let vars = log2_ceil(dim);
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

fn log2_ceil(value: usize) -> usize {
    value.next_power_of_two().trailing_zeros() as usize
}

fn normalize_sumcheck_point<F: JoltField>(challenges: &[F]) -> Vec<F> {
    OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec())
        .match_endianness::<BIG_ENDIAN>()
        .r
}

fn mul_sumcheck_id() -> SumcheckId {
    SumcheckId::NodeExecution(0)
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use joltworks::transcripts::Blake2bTranscript;

    use super::*;

    #[test]
    fn proves_and_verifies_mul_const() {
        let coeff = Fr::from(7u64);
        let params = MulConstParams::new(vec![2, 2], "X", coeff);
        let x = vec![1, 2, 3, 4];
        let y = vec![7, 14, 21, 28];
        let point = vec![Fr::from(3u64), Fr::from(5u64)];
        let y_claim = Claim {
            tensor: TensorId::new("Y"),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            value: eval_tensor_at_point(&y, &params.shape, &point),
            point,
        };

        let mut prover_transcript = Blake2bTranscript::default();
        let result =
            prove_mul_const::<Fr, _>(y_claim.clone(), &x, &params, &mut prover_transcript).unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        let claims =
            verify_mul_const::<Fr, _>(y_claim, &result.proof, &params, &mut verifier_transcript)
                .unwrap();

        assert_eq!(claims, result.claims);
        assert_eq!(claims.x.tensor.0, "X");
    }

    #[test]
    fn proves_and_verifies_mul_by_vector() {
        let params = MulByVectorParams::new(vec![2, 3], "X", "Gamma", 1);
        let x = vec![1, 2, 3, 4, 5, 6];
        let gamma = vec![10, 20, 30];
        let y = vec![10, 40, 90, 40, 100, 180];
        let point = vec![Fr::from(3u64), Fr::from(5u64), Fr::from(7u64)];
        let y_claim = Claim {
            tensor: TensorId::new("Y"),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            value: eval_tensor_at_point(&y, &params.shape, &point),
            point,
        };

        let mut prover_transcript = Blake2bTranscript::default();
        let result = prove_mul_by_vector::<Fr, _>(
            y_claim.clone(),
            &x,
            &gamma,
            &params,
            &mut prover_transcript,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        let claims = verify_mul_by_vector::<Fr, _>(
            y_claim,
            &result.proof,
            &gamma,
            &params,
            &mut verifier_transcript,
        )
        .unwrap();

        assert_eq!(claims, result.claims);
        assert_eq!(claims.x.tensor.0, "X");
        assert_eq!(claims.coeff.tensor.0, "Gamma");
        assert_eq!(claims.coeff.point.len(), log2_ceil(3));
    }

    #[test]
    fn proves_and_verifies_mul_by_vector_non_power_rows() {
        let params = MulByVectorParams::new(vec![3, 4], "X", "Gamma", 1);
        let x = (1..=12).collect::<Vec<_>>();
        let gamma = vec![2, -3, 5, 7];
        let y = x
            .chunks_exact(4)
            .flat_map(|row| row.iter().zip(&gamma).map(|(&x, &g)| x * g))
            .collect::<Vec<_>>();
        let point = vec![
            Fr::from(3u64),
            Fr::from(5u64),
            Fr::from(7u64),
            Fr::from(11u64),
        ];
        let y_claim = Claim {
            tensor: TensorId::new("Y"),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            value: eval_tensor_at_point(&y, &params.shape, &point),
            point,
        };

        let mut prover_transcript = Blake2bTranscript::default();
        let result = prove_mul_by_vector::<Fr, _>(
            y_claim.clone(),
            &x,
            &gamma,
            &params,
            &mut prover_transcript,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        let claims = verify_mul_by_vector::<Fr, _>(
            y_claim,
            &result.proof,
            &gamma,
            &params,
            &mut verifier_transcript,
        )
        .unwrap();

        assert_eq!(claims, result.claims);
    }

    #[test]
    fn verifier_rejects_inconsistent_mul_const_opening() {
        let params = MulConstParams::new(vec![2], "X", Fr::from(3u64));
        let y_claim = Claim {
            tensor: TensorId::new("Y"),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: vec![Fr::from(5u64)],
            value: Fr::from(10u64),
        };
        let proof = MulProof {
            x_opening: Fr::from(4u64),
        };
        let mut transcript = Blake2bTranscript::default();

        let err = verify_mul_const::<Fr, _>(y_claim, &proof, &params, &mut transcript);

        assert!(err.is_err());
    }
}
