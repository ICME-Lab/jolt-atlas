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
// Design note for future us:
//
// Elementwise add is linear as an MLE for one point:
//     Y(r) = A(r) + B(r).
// However residual add is the main layer-boundary branching point.  When multiple
// downstream claims refer to the same add output, direct opening at every point
// would leak branching into the rest of the graph.  `prove_matadd` batches
// those claims with random coefficients and runs one sumcheck:
//     sum_t alpha_t Y(r_t)
//       = sum_i (sum_t alpha_t eq(i, r_t)) * (A(i) + B(i)).
// This produces one `A` claim and one `B` claim at the sumcheck point.  The
// batched API is also used for a single output claim, so there is only one add
// proof path.

use crate::{
    claim::{Claim, Shape, TensorId},
    error::{ProverError, Result},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatAddParams {
    pub shape: Shape,
    pub a_tensor: TensorId,
    pub b_tensor: TensorId,
}

impl MatAddParams {
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
pub struct MatAddProof<F: JoltField, T: Transcript> {
    pub sumcheck: SumcheckInstanceProof<F, T>,
    pub a_opening: F,
    pub b_opening: F,
}

pub fn prove_matadd<F, T>(
    y_claims: Vec<Claim<F>>,
    a: &[i32],
    b: &[i32],
    params: &MatAddParams,
    transcript: &mut T,
) -> Result<(MatAddProof<F, T>, Claim<F>, Claim<F>)>
where
    F: JoltField,
    T: Transcript,
{
    validate_batched_inputs(&y_claims, a, b, params)?;
    let alphas = transcript.challenge_scalar_powers(y_claims.len());
    let input_claim = batched_input_claim(&y_claims, &alphas);
    let eq_batch = batched_eq_poly(&y_claims, &alphas);
    let a_poly = padded_tensor(a, &params.shape);
    let b_poly = padded_tensor(b, &params.shape);
    let sc_params =
        MatAddSumcheckParams::new(params.shape.padded_power_of_two().point_len(), input_claim);
    let mut prover = MatAddSumcheckProver::new(sc_params, eq_batch, a_poly, b_poly);
    let mut accumulator = ProverOpeningAccumulator::new();
    let (sumcheck, challenges) = Sumcheck::prove(&mut prover, &mut accumulator, transcript);
    let a_opening = opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenMatAddA, matadd_sumcheck_id()),
    )?;
    let b_opening = opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenMatAddB, matadd_sumcheck_id()),
    )?;
    let point = normalize_sumcheck_point::<F>(&challenges.into_opening());

    Ok((
        MatAddProof {
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

pub fn verify_matadd<F, T>(
    y_claims: Vec<Claim<F>>,
    proof: &MatAddProof<F, T>,
    params: &MatAddParams,
    transcript: &mut T,
) -> std::result::Result<(Claim<F>, Claim<F>), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    verify_batched_inputs(&y_claims, params)?;
    let alphas = transcript.challenge_scalar_powers(y_claims.len());
    let input_claim = batched_input_claim(&y_claims, &alphas);
    let y_points = y_claims
        .iter()
        .map(|claim| claim.point.clone())
        .collect::<Vec<_>>();
    let verifier = MatAddSumcheckVerifier {
        params: MatAddSumcheckParams::new(
            params.shape.padded_power_of_two().point_len(),
            input_claim,
        ),
        y_points,
        alphas,
    };
    let mut accumulator = VerifierOpeningAccumulator::new();
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenMatAddA, matadd_sumcheck_id()),
        (OpeningPoint::<BIG_ENDIAN, F>::default(), proof.a_opening),
    );
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenMatAddB, matadd_sumcheck_id()),
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

fn validate_inputs<F: JoltField>(
    y_claim: &Claim<F>,
    a: &[i32],
    b: &[i32],
    params: &MatAddParams,
) -> Result<()> {
    if params.shape.dims().contains(&0) {
        return Err(ProverError::InvalidTensorShape(params.shape.0.clone()));
    }
    let expected_len = params.shape.numel();
    if a.len() != expected_len {
        return Err(ProverError::TensorLenMismatch {
            name: "A",
            shape: params.shape.0.clone(),
            expected: expected_len,
            actual: a.len(),
        });
    }
    if b.len() != expected_len {
        return Err(ProverError::TensorLenMismatch {
            name: "B",
            shape: params.shape.0.clone(),
            expected: expected_len,
            actual: b.len(),
        });
    }
    if y_claim.logical_shape != params.shape {
        return Err(ProverError::ShapeMismatch {
            name: "Y claim",
            expected: params.shape.0.clone(),
            actual: y_claim.logical_shape.0.clone(),
        });
    }
    let expected_domain = params.shape.padded_power_of_two();
    if y_claim.domain_shape != expected_domain {
        return Err(ProverError::ShapeMismatch {
            name: "Y claim domain",
            expected: expected_domain.0,
            actual: y_claim.domain_shape.0.clone(),
        });
    }
    let expected_point_len = y_claim.domain_shape.point_len();
    if y_claim.point.len() != expected_point_len {
        return Err(ProverError::ShapeMismatch {
            name: "Y claim point",
            expected: vec![expected_point_len],
            actual: vec![y_claim.point.len()],
        });
    }
    Ok(())
}

fn verify_inputs<F: JoltField>(
    y_claim: &Claim<F>,
    params: &MatAddParams,
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
    let expected_point_len = y_claim.domain_shape.point_len();
    if y_claim.point.len() != expected_point_len {
        return Err(ProofVerifyError::InvalidInputLength(
            expected_point_len,
            y_claim.point.len(),
        ));
    }
    Ok(())
}

fn validate_batched_inputs<F: JoltField>(
    y_claims: &[Claim<F>],
    a: &[i32],
    b: &[i32],
    params: &MatAddParams,
) -> Result<()> {
    if y_claims.is_empty() {
        return Err(ProverError::InvalidTensorShape(vec![]));
    }
    for claim in y_claims {
        validate_inputs(claim, a, b, params)?;
    }
    Ok(())
}

fn verify_batched_inputs<F: JoltField>(
    y_claims: &[Claim<F>],
    params: &MatAddParams,
) -> std::result::Result<(), ProofVerifyError> {
    if y_claims.is_empty() {
        return Err(ProofVerifyError::InvalidInputLength(1, 0));
    }
    for claim in y_claims {
        verify_inputs(claim, params)?;
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

struct MatAddSumcheckParams<F: JoltField> {
    num_rounds: usize,
    input_claim: F,
}

impl<F: JoltField> MatAddSumcheckParams<F> {
    fn new(num_rounds: usize, input_claim: F) -> Self {
        Self {
            num_rounds,
            input_claim,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for MatAddSumcheckParams<F> {
    fn degree(&self) -> usize {
        2
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

struct MatAddSumcheckProver<F: JoltField> {
    eq_batch: MultilinearPolynomial<F>,
    a: MultilinearPolynomial<F>,
    b: MultilinearPolynomial<F>,
    params: MatAddSumcheckParams<F>,
}

impl<F: JoltField> MatAddSumcheckProver<F> {
    fn new(params: MatAddSumcheckParams<F>, eq_batch: Vec<F>, a: Vec<F>, b: Vec<F>) -> Self {
        Self {
            eq_batch: MultilinearPolynomial::from(eq_batch),
            a: MultilinearPolynomial::from(a),
            b: MultilinearPolynomial::from(b),
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for MatAddSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let mut evals = [F::zero(); 2];
        let mut actual_eval_at_1 = F::zero();
        for g in 0..self.a.len() / 2 {
            let e0 = self.eq_batch.get_bound_coeff(2 * g);
            let e1 = self.eq_batch.get_bound_coeff(2 * g + 1);
            let a0 = self.a.get_bound_coeff(2 * g);
            let a1 = self.a.get_bound_coeff(2 * g + 1);
            let b0 = self.b.get_bound_coeff(2 * g);
            let b1 = self.b.get_bound_coeff(2 * g + 1);
            evals[0] += e0 * (a0 + b0);
            evals[1] += (e1 + e1 - e0) * ((a1 + b1) + (a1 + b1) - (a0 + b0));
            actual_eval_at_1 += e1 * (a1 + b1);
        }
        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_batch.bind_parallel(r_j, BindingOrder::LowToHigh);
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
            OpeningId::new(VirtualPoly::QwenMatAddA, matadd_sumcheck_id()),
            point.clone(),
            self.a.final_claim(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenMatAddB, matadd_sumcheck_id()),
            point,
            self.b.final_claim(),
        );
    }
}

struct MatAddSumcheckVerifier<F: JoltField> {
    params: MatAddSumcheckParams<F>,
    y_points: Vec<Vec<F>>,
    alphas: Vec<F>,
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for MatAddSumcheckVerifier<F> {
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
                VirtualPoly::QwenMatAddA,
                matadd_sumcheck_id(),
            ))
            .1;
        let b = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenMatAddB,
                matadd_sumcheck_id(),
            ))
            .1;
        let point = normalize_sumcheck_point::<F>(&sumcheck_challenges.into_opening());
        let mut eq_batch = F::zero();
        for (alpha, y_point) in self.alphas.iter().zip(&self.y_points) {
            eq_batch += *alpha * EqPolynomial::mle(y_point, &point);
        }
        eq_batch * (a + b)
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
            OpeningId::new(VirtualPoly::QwenMatAddA, matadd_sumcheck_id()),
            point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenMatAddB, matadd_sumcheck_id()),
            point,
        );
    }
}

fn padded_tensor<F: JoltField>(values: &[i32], shape: &Shape) -> Vec<F> {
    let padded_dims = shape.padded_power_of_two().0;
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

fn matadd_sumcheck_id() -> SumcheckId {
    SumcheckId::NodeExecution(0)
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use joltworks::poly::eq_poly::EqPolynomial;
    use joltworks::transcripts::Blake2bTranscript;

    use super::*;

    #[test]
    fn proves_and_verifies_matadd() {
        let params = MatAddParams::new(vec![2, 3], "A", "B");
        let a = vec![1, 2, 3, 4, 5, 6];
        let b = vec![10, 20, 30, 40, 50, 60];
        let y = vec![11, 22, 33, 44, 55, 66];
        let points = [
            vec![Fr::from(3u64), Fr::from(5u64), Fr::from(7u64)],
            vec![Fr::from(11u64), Fr::from(13u64), Fr::from(17u64)],
        ];
        let y_claims = points
            .iter()
            .map(|point| Claim {
                tensor: TensorId::new("Y"),
                logical_shape: params.shape.clone(),
                domain_shape: params.shape.padded_power_of_two(),
                value: eval_tensor_at_point(&y, &params.shape, point),
                point: point.clone(),
            })
            .collect::<Vec<_>>();

        let mut prover_transcript = Blake2bTranscript::default();
        let (proof, lhs, rhs) =
            prove_matadd::<Fr, _>(y_claims.clone(), &a, &b, &params, &mut prover_transcript)
                .unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        let (verify_lhs, verify_rhs) =
            verify_matadd::<Fr, _>(y_claims, &proof, &params, &mut verifier_transcript).unwrap();

        assert_eq!(verify_lhs, lhs);
        assert_eq!(verify_rhs, rhs);
        assert_eq!(verify_lhs.tensor.0, "A");
        assert_eq!(verify_rhs.tensor.0, "B");
    }

    #[test]
    fn proves_and_verifies_matadd_non_power_rows() {
        let params = MatAddParams::new(vec![3, 4], "A", "B");
        let a = (1..=12).collect::<Vec<_>>();
        let b = (10..=21).collect::<Vec<_>>();
        let y = a.iter().zip(&b).map(|(&a, &b)| a + b).collect::<Vec<_>>();
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
        let y_claims = points
            .iter()
            .map(|point| Claim {
                tensor: TensorId::new("Y"),
                logical_shape: params.shape.clone(),
                domain_shape: params.shape.padded_power_of_two(),
                value: eval_tensor_at_point(&y, &params.shape, point),
                point: point.clone(),
            })
            .collect::<Vec<_>>();

        let mut prover_transcript = Blake2bTranscript::default();
        let (proof, lhs, rhs) =
            prove_matadd::<Fr, _>(y_claims.clone(), &a, &b, &params, &mut prover_transcript)
                .unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        let (verify_lhs, verify_rhs) =
            verify_matadd::<Fr, _>(y_claims, &proof, &params, &mut verifier_transcript).unwrap();

        assert_eq!(verify_lhs, lhs);
        assert_eq!(verify_rhs, rhs);
    }

    #[test]
    fn matadd_handles_single_claim() {
        let params = MatAddParams::new(vec![2, 2], "A", "B");
        let a = vec![1, 2, 3, 4];
        let b = vec![10, 20, 30, 40];
        let y = vec![11, 22, 33, 44];
        let point = vec![Fr::from(3u64), Fr::from(5u64)];
        let y_claim = Claim {
            tensor: TensorId::new("Y"),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            value: eval_tensor_at_point(&y, &params.shape, &point),
            point,
        };

        let mut prover_transcript = Blake2bTranscript::default();
        let (proof, lhs, rhs) = prove_matadd::<Fr, _>(
            vec![y_claim.clone()],
            &a,
            &b,
            &params,
            &mut prover_transcript,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        let (verify_lhs, verify_rhs) =
            verify_matadd::<Fr, _>(vec![y_claim], &proof, &params, &mut verifier_transcript)
                .unwrap();

        assert_eq!(verify_lhs, lhs);
        assert_eq!(verify_rhs, rhs);
    }

    #[test]
    fn matadd_verifier_rejects_tampered_opening() {
        let params = MatAddParams::new(vec![2, 2], "A", "B");
        let a = vec![1, 2, 3, 4];
        let b = vec![10, 20, 30, 40];
        let y = vec![11, 22, 33, 44];
        let point = vec![Fr::from(3u64), Fr::from(5u64)];
        let y_claim = Claim {
            tensor: TensorId::new("Y"),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            value: eval_tensor_at_point(&y, &params.shape, &point),
            point,
        };

        let mut prover_transcript = Blake2bTranscript::default();
        let (mut proof, _, _) = prove_matadd::<Fr, _>(
            vec![y_claim.clone()],
            &a,
            &b,
            &params,
            &mut prover_transcript,
        )
        .unwrap();
        proof.a_opening += Fr::from(1u64);

        let mut verifier_transcript = Blake2bTranscript::default();
        let err = verify_matadd::<Fr, _>(vec![y_claim], &proof, &params, &mut verifier_transcript);

        assert!(err.is_err());
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
