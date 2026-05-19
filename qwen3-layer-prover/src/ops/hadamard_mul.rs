use common::VirtualPoly;
// Design note for future us:
//
// Hadamard multiplication is elementwise tensor multiplication:
//     Y(i) = A(i) * B(i)
//
// Unlike addition or multiplication by a scalar, this does not commute with the
// MLE:
//     MLE(A * B)(r) != MLE(A)(r) * MLE(B)(r).
//
// Therefore this op uses one sumcheck over the whole logical tensor domain:
//     Y(r) = sum_i eq(i, r) * A(i) * B(i).
// Neither input is public, so verification returns two opening claims at the
// sumcheck point: `A(r')` and `B(r')`.
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HadamardMulParams {
    pub shape: Shape,
    pub a_tensor: TensorId,
    pub b_tensor: TensorId,
}

impl HadamardMulParams {
    pub fn new(
        shape: impl Into<Vec<usize>>,
        a_tensor: impl Into<String>,
        b_tensor: impl Into<String>,
    ) -> Self {
        Self {
            shape: Shape::new(shape),
            a_tensor: TensorId::new(a_tensor),
            b_tensor: TensorId::new(b_tensor),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HadamardMulProof<F: JoltField, T: Transcript> {
    pub sumcheck: SumcheckInstanceProof<F, T>,
    pub a_opening: F,
    pub b_opening: F,
}

pub fn prove_hadamard_mul<F, T>(
    y_claim: Claim<F>,
    a: &[i32],
    b: &[i32],
    params: &HadamardMulParams,
    transcript: &mut T,
) -> Result<(HadamardMulProof<F, T>, Claim<F>, Claim<F>)>
where
    F: JoltField,
    T: Transcript,
{
    validate_inputs(&y_claim, a, b, params)?;
    let a_poly = padded_tensor(a, &params.shape);
    let b_poly = padded_tensor(b, &params.shape);
    let sc_params = HadamardMulSumcheckParams::new(
        params.shape.padded_power_of_two().point_len(),
        y_claim.value,
        y_claim.point.clone(),
    );
    let mut prover = HadamardMulSumcheckProver::new(sc_params, a_poly, b_poly);
    let mut accumulator = ProverOpeningAccumulator::new();
    let (sumcheck, challenges) = Sumcheck::prove(&mut prover, &mut accumulator, transcript);
    let a_opening = opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenHadamardA, hadamard_sumcheck_id()),
    )?;
    let b_opening = opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenHadamardB, hadamard_sumcheck_id()),
    )?;
    let point = normalize_sumcheck_point::<F>(&challenges.into_opening());

    Ok((
        HadamardMulProof {
            sumcheck,
            a_opening,
            b_opening,
        },
        Claim {
            tensor: params.a_tensor.clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: point.clone(),
            value: a_opening,
        },
        Claim {
            tensor: params.b_tensor.clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point,
            value: b_opening,
        },
    ))
}

pub fn verify_hadamard_mul<F, T>(
    y_claim: Claim<F>,
    proof: &HadamardMulProof<F, T>,
    params: &HadamardMulParams,
    transcript: &mut T,
) -> std::result::Result<(Claim<F>, Claim<F>), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    verify_inputs(&y_claim, params)?;
    let sc_params = HadamardMulSumcheckParams::new(
        params.shape.padded_power_of_two().point_len(),
        y_claim.value,
        y_claim.point.clone(),
    );
    let verifier = HadamardMulSumcheckVerifier { params: sc_params };
    let mut accumulator = VerifierOpeningAccumulator::new();
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenHadamardA, hadamard_sumcheck_id()),
        (OpeningPoint::<BIG_ENDIAN, F>::default(), proof.a_opening),
    );
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenHadamardB, hadamard_sumcheck_id()),
        (OpeningPoint::<BIG_ENDIAN, F>::default(), proof.b_opening),
    );
    let challenges = Sumcheck::verify(&proof.sumcheck, &verifier, &mut accumulator, transcript)?;
    let point = normalize_sumcheck_point::<F>(&challenges.into_opening());

    Ok((
        Claim {
            tensor: params.a_tensor.clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: point.clone(),
            value: proof.a_opening,
        },
        Claim {
            tensor: params.b_tensor.clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point,
            value: proof.b_opening,
        },
    ))
}

struct HadamardMulSumcheckParams<F: JoltField> {
    num_rounds: usize,
    input_claim: F,
    y_point: Vec<F>,
}

impl<F: JoltField> HadamardMulSumcheckParams<F> {
    fn new(num_rounds: usize, input_claim: F, y_point: Vec<F>) -> Self {
        Self {
            num_rounds,
            input_claim,
            y_point,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for HadamardMulSumcheckParams<F> {
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

struct HadamardMulSumcheckProver<F: JoltField> {
    eq_y: MultilinearPolynomial<F>,
    a: MultilinearPolynomial<F>,
    b: MultilinearPolynomial<F>,
    params: HadamardMulSumcheckParams<F>,
}

impl<F: JoltField> HadamardMulSumcheckProver<F> {
    fn new(params: HadamardMulSumcheckParams<F>, a: Vec<F>, b: Vec<F>) -> Self {
        let eq_y = MultilinearPolynomial::from(EqPolynomial::<F>::evals(&params.y_point));
        Self {
            eq_y,
            a: MultilinearPolynomial::from(a),
            b: MultilinearPolynomial::from(b),
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for HadamardMulSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let mut evals = [F::zero(); 3];
        for g in 0..self.a.len() / 2 {
            let eq = [
                self.eq_y.get_bound_coeff(2 * g),
                self.eq_y.get_bound_coeff(2 * g + 1),
            ];
            let a = [
                self.a.get_bound_coeff(2 * g),
                self.a.get_bound_coeff(2 * g + 1),
            ];
            let b = [
                self.b.get_bound_coeff(2 * g),
                self.b.get_bound_coeff(2 * g + 1),
            ];
            for (idx, t) in [F::zero(), F::from_u64(2), F::from_u64(3)]
                .into_iter()
                .enumerate()
            {
                evals[idx] += lerp(eq[0], eq[1], t) * lerp(a[0], a[1], t) * lerp(b[0], b[1], t);
            }
        }
        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_y.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.a.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.b.bind_parallel(r_j, BindingOrder::LowToHigh);
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
            OpeningId::new(VirtualPoly::QwenHadamardA, hadamard_sumcheck_id()),
            point.clone(),
            self.a.final_claim(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenHadamardB, hadamard_sumcheck_id()),
            point,
            self.b.final_claim(),
        );
    }
}

struct HadamardMulSumcheckVerifier<F: JoltField> {
    params: HadamardMulSumcheckParams<F>,
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for HadamardMulSumcheckVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let a = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenHadamardA,
                hadamard_sumcheck_id(),
            ))
            .1;
        let b = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenHadamardB,
                hadamard_sumcheck_id(),
            ))
            .1;
        let point = normalize_sumcheck_point::<F>(&sumcheck_challenges.into_opening());
        EqPolynomial::mle(&self.params.y_point, &point) * a * b
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
            OpeningId::new(VirtualPoly::QwenHadamardA, hadamard_sumcheck_id()),
            point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenHadamardB, hadamard_sumcheck_id()),
            point,
        );
    }
}

fn validate_inputs<F: JoltField>(
    y_claim: &Claim<F>,
    a: &[i32],
    b: &[i32],
    params: &HadamardMulParams,
) -> Result<()> {
    validate_shape(&params.shape)?;
    ensure_len("A", &params.shape, a.len())?;
    ensure_len("B", &params.shape, b.len())?;
    if y_claim.logical_shape != params.shape {
        return Err(ProverError::ShapeMismatch {
            name: "Hadamard Y claim",
            expected: params.shape.0.clone(),
            actual: y_claim.logical_shape.0.clone(),
        });
    }
    let expected_domain = params.shape.padded_power_of_two();
    if y_claim.domain_shape != expected_domain {
        return Err(ProverError::ShapeMismatch {
            name: "Hadamard Y claim domain",
            expected: expected_domain.0,
            actual: y_claim.domain_shape.0.clone(),
        });
    }
    let expected = y_claim.domain_shape.point_len();
    if y_claim.point.len() != expected {
        return Err(ProverError::ShapeMismatch {
            name: "Hadamard Y claim point",
            expected: vec![expected],
            actual: vec![y_claim.point.len()],
        });
    }
    Ok(())
}

fn verify_inputs<F: JoltField>(
    y_claim: &Claim<F>,
    params: &HadamardMulParams,
) -> std::result::Result<(), ProofVerifyError> {
    if params.shape.dims().contains(&0) {
        return Err(ProofVerifyError::InvalidInputLength(1, 0));
    }
    if y_claim.logical_shape != params.shape {
        return Err(ProofVerifyError::InvalidInputLength(
            params.shape.numel(),
            y_claim.logical_shape.numel(),
        ));
    }
    let expected_domain = params.shape.padded_power_of_two();
    if y_claim.domain_shape != expected_domain {
        return Err(ProofVerifyError::InvalidInputLength(
            expected_domain.numel(),
            y_claim.domain_shape.numel(),
        ));
    }
    let expected = y_claim.domain_shape.point_len();
    if y_claim.point.len() != expected {
        return Err(ProofVerifyError::InvalidInputLength(
            expected,
            y_claim.point.len(),
        ));
    }
    Ok(())
}

fn validate_shape(shape: &Shape) -> Result<()> {
    if shape.dims().contains(&0) {
        return Err(ProverError::InvalidTensorShape(shape.0.clone()));
    }
    Ok(())
}

fn ensure_len(name: &'static str, shape: &Shape, actual: usize) -> Result<()> {
    let expected = shape.numel();
    if actual == expected {
        Ok(())
    } else {
        Err(ProverError::TensorLenMismatch {
            name,
            shape: shape.0.clone(),
            expected,
            actual,
        })
    }
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

fn padded_dims(shape: &Shape) -> Vec<usize> {
    shape
        .dims()
        .iter()
        .map(|dim| dim.next_power_of_two())
        .collect()
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

fn opening<F: JoltField>(accumulator: &ProverOpeningAccumulator<F>, id: OpeningId) -> Result<F> {
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

fn lerp<F: JoltField>(v0: F, v1: F, t: F) -> F {
    v0 + t * (v1 - v0)
}

fn hadamard_sumcheck_id() -> SumcheckId {
    SumcheckId::NodeExecution(0)
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use joltworks::transcripts::Blake2bTranscript;

    use super::*;

    #[test]
    fn proves_and_verifies_hadamard_mul() {
        let params = HadamardMulParams::new(vec![2, 3], "A", "B");
        let a = vec![1, 2, 3, 4, 5, 6];
        let b = vec![10, 20, 30, 40, 50, 60];
        let y = vec![10, 40, 90, 160, 250, 360];
        let point = vec![Fr::from(3u64), Fr::from(5u64), Fr::from(7u64)];
        let y_claim = Claim {
            tensor: TensorId::new("Y"),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            value: eval_tensor(&y, &params.shape, &point),
            point,
        };

        let mut prover_transcript = Blake2bTranscript::default();
        let (proof, lhs, rhs) =
            prove_hadamard_mul::<Fr, _>(y_claim.clone(), &a, &b, &params, &mut prover_transcript)
                .unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        let (verify_lhs, verify_rhs) =
            verify_hadamard_mul::<Fr, _>(y_claim, &proof, &params, &mut verifier_transcript)
                .unwrap();

        assert_eq!(verify_lhs, lhs);
        assert_eq!(verify_rhs, rhs);
        assert_eq!(verify_lhs.tensor.0, "A");
        assert_eq!(verify_rhs.tensor.0, "B");
    }

    fn eval_tensor<F: JoltField>(values: &[i32], shape: &Shape, point: &[F]) -> F {
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
