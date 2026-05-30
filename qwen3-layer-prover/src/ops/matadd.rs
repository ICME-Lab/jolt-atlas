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
    claim::{Claim, Poly, Shape, TensorId},
    error::{ProverError, Result},
};

/// Elementwise add:
///
///   Y = A + B
///
/// Add is linear as an MLE, but residual add is a fanout point in the layer
/// graph. `prove_matadd` therefore accepts one or more claims on `Y`, batches
/// them with transcript challenges, and returns one claim for `A` and one claim
/// for `B`.

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

impl<F: JoltField, T: Transcript> MatAddProof<F, T> {
    pub(crate) fn sumcheck_round_count(&self) -> usize {
        self.sumcheck.compressed_polys.len()
    }

    pub(crate) fn sumcheck_count(&self) -> usize {
        1
    }
}

pub fn prove_matadd<F, T, C>(
    y_claims: Vec<Claim<F, C>>,
    a_poly: Poly<F, C>,
    b_poly: Poly<F, C>,
    params: &MatAddParams,
    transcript: &mut T,
) -> Result<(MatAddProof<F, T>, Claim<F, C>, Claim<F, C>)>
where
    F: JoltField,
    T: Transcript,
    C: Clone,
{
    validate_batched_inputs(&y_claims, &a_poly, &b_poly, params)?;
    let alphas = transcript.challenge_scalar_powers(y_claims.len());
    let input_claim = batched_input_claim(&y_claims, &alphas);
    let eq_batch = batched_eq_poly(&y_claims, &alphas, params);
    let sc_params =
        MatAddSumcheckParams::new(params.shape.padded_power_of_two().point_len(), input_claim);
    let mut prover = MatAddSumcheckProver::new(
        sc_params,
        eq_batch,
        coeffs_for_domain(&a_poly, params)?,
        coeffs_for_domain(&b_poly, params)?,
    );
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
        Claim::new(a_poly, point.clone(), a_opening),
        Claim::new(b_poly, point, b_opening),
    ))
}

pub fn verify_matadd<F, T, C>(
    y_claims: Vec<Claim<F, C>>,
    proof: &MatAddProof<F, T>,
    params: &MatAddParams,
    transcript: &mut T,
) -> std::result::Result<(Claim<F, C>, Claim<F, C>), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
    C: Clone,
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
        Claim::new(dummy_poly(&params.shape), point.clone(), proof.a_opening),
        Claim::new(dummy_poly(&params.shape), point, proof.b_opening),
    ))
}

fn validate_batched_inputs<F: JoltField, C>(
    y_claims: &[Claim<F, C>],
    a: &Poly<F, C>,
    b: &Poly<F, C>,
    params: &MatAddParams,
) -> Result<()> {
    if y_claims.is_empty() {
        return Err(ProverError::InvalidTensorShape(vec![]));
    }
    if params.shape.dims().contains(&0) {
        return Err(ProverError::InvalidTensorShape(params.shape.0.clone()));
    }
    validate_poly_domain("A", a, params)?;
    validate_poly_domain("B", b, params)?;
    for claim in y_claims {
        validate_claim_domain("Y claim", claim, params)?;
    }
    Ok(())
}

fn validate_poly_domain<F: JoltField, C>(
    name: &'static str,
    poly: &Poly<F, C>,
    params: &MatAddParams,
) -> Result<()> {
    let expected = params.shape.padded_power_of_two().numel();
    if poly.data.len() != expected {
        return Err(ProverError::TensorLenMismatch {
            name,
            shape: params.shape.padded_power_of_two().0,
            expected,
            actual: poly.data.len(),
        });
    }
    Ok(())
}

fn validate_claim_domain<F: JoltField, C>(
    name: &'static str,
    claim: &Claim<F, C>,
    params: &MatAddParams,
) -> Result<()> {
    validate_poly_domain(name, &claim.poly, params)?;
    let expected = params.shape.padded_power_of_two().point_len();
    if claim.point.len() != expected {
        return Err(ProverError::ShapeMismatch {
            name,
            expected: vec![expected],
            actual: vec![claim.point.len()],
        });
    }
    Ok(())
}

fn verify_batched_inputs<F: JoltField, C>(
    y_claims: &[Claim<F, C>],
    params: &MatAddParams,
) -> std::result::Result<(), ProofVerifyError> {
    if y_claims.is_empty() {
        return Err(ProofVerifyError::InvalidInputLength(1, 0));
    }
    let expected_point_len = params.shape.padded_power_of_two().point_len();
    let expected_domain_len = params.shape.padded_power_of_two().numel();
    for claim in y_claims {
        if claim.point.len() != expected_point_len {
            return Err(ProofVerifyError::InvalidInputLength(
                expected_point_len,
                claim.point.len(),
            ));
        }
        if claim.poly.data.len() != expected_domain_len {
            return Err(ProofVerifyError::InvalidInputLength(
                expected_domain_len,
                claim.poly.data.len(),
            ));
        }
    }
    Ok(())
}

fn batched_input_claim<F: JoltField, C>(claims: &[Claim<F, C>], alphas: &[F]) -> F {
    claims
        .iter()
        .zip(alphas)
        .map(|(claim, alpha)| claim.value * *alpha)
        .sum()
}

fn batched_eq_poly<F: JoltField, C>(
    claims: &[Claim<F, C>],
    alphas: &[F],
    params: &MatAddParams,
) -> Vec<F> {
    let len = params.shape.padded_power_of_two().numel();
    let mut out = vec![F::zero(); len];
    for (claim, alpha) in claims.iter().zip(alphas) {
        let eq = EqPolynomial::<F>::evals(&claim.point);
        for (out, eq) in out.iter_mut().zip(eq) {
            *out += *alpha * eq;
        }
    }
    out
}

fn coeffs_for_domain<F: JoltField, C>(poly: &Poly<F, C>, params: &MatAddParams) -> Result<Vec<F>> {
    validate_poly_domain("poly", poly, params)?;
    Ok(poly.data.coeffs())
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

fn dummy_poly<F: JoltField, C>(shape: &Shape) -> Poly<F, C> {
    Poly::new(
        MultilinearPolynomial::from(vec![F::zero(); shape.padded_power_of_two().numel()]),
        None,
    )
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
    use joltworks::{
        poly::{eq_poly::EqPolynomial, multilinear_polynomial::MultilinearPolynomial},
        transcripts::Blake2bTranscript,
    };

    use super::*;

    #[test]
    fn proves_and_verifies_matadd_from_poly_claims() {
        let params = MatAddParams::new(vec![2, 2], "A", "B");
        let a = vec![1, 2, 3, 4];
        let b = vec![10, 20, 30, 40];
        let y = vec![11, 22, 33, 44];
        let point = vec![Fr::from(3_u64), Fr::from(5_u64)];
        let y_claim = Claim::new(
            poly_from_i32(&y, &params.shape),
            point.clone(),
            eval_flat(&y, &params.shape, &point),
        );

        let mut prover_transcript = Blake2bTranscript::default();
        let (proof, lhs, rhs) = prove_matadd(
            vec![y_claim.clone()],
            poly_from_i32(&a, &params.shape),
            poly_from_i32(&b, &params.shape),
            &params,
            &mut prover_transcript,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        let (verified_lhs, verified_rhs) =
            verify_matadd(vec![y_claim], &proof, &params, &mut verifier_transcript).unwrap();

        assert_eq!(verified_lhs.point, lhs.point);
        assert_eq!(verified_lhs.value, lhs.value);
        assert_eq!(verified_rhs.point, rhs.point);
        assert_eq!(verified_rhs.value, rhs.value);
    }

    #[test]
    fn batches_multiple_output_claims() {
        let params = MatAddParams::new(vec![3, 4], "A", "B");
        let a = (1..=12).collect::<Vec<_>>();
        let b = (10..=21).collect::<Vec<_>>();
        let y = a.iter().zip(&b).map(|(&a, &b)| a + b).collect::<Vec<_>>();
        let points = [
            vec![
                Fr::from(3_u64),
                Fr::from(5_u64),
                Fr::from(7_u64),
                Fr::from(11_u64),
            ],
            vec![
                Fr::from(13_u64),
                Fr::from(17_u64),
                Fr::from(19_u64),
                Fr::from(23_u64),
            ],
        ];
        let y_claims = points
            .iter()
            .map(|point| {
                Claim::new(
                    poly_from_i32(&y, &params.shape),
                    point.clone(),
                    eval_flat(&y, &params.shape, point),
                )
            })
            .collect::<Vec<_>>();

        let mut prover_transcript = Blake2bTranscript::default();
        let (proof, lhs, rhs) = prove_matadd(
            y_claims.clone(),
            poly_from_i32(&a, &params.shape),
            poly_from_i32(&b, &params.shape),
            &params,
            &mut prover_transcript,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        let (verified_lhs, verified_rhs) =
            verify_matadd(y_claims, &proof, &params, &mut verifier_transcript).unwrap();

        assert_eq!(verified_lhs.point, lhs.point);
        assert_eq!(verified_lhs.value, lhs.value);
        assert_eq!(verified_rhs.point, rhs.point);
        assert_eq!(verified_rhs.value, rhs.value);
    }

    fn poly_from_i32(values: &[i32], shape: &Shape) -> Poly<Fr, ()> {
        let padded_dims = shape.padded_power_of_two().0;
        let mut out = vec![0; padded_dims.iter().product()];
        let strides = row_major_strides(shape.dims());
        let padded_strides = row_major_strides(&padded_dims);
        for (flat, &value) in values.iter().enumerate() {
            let mut padded_flat = 0;
            for (dim, (&stride, &padded_stride)) in strides.iter().zip(&padded_strides).enumerate()
            {
                let coord = (flat / stride) % shape.dims()[dim];
                padded_flat += coord * padded_stride;
            }
            out[padded_flat] = value;
        }
        Poly::new(MultilinearPolynomial::from(out), None)
    }

    fn eval_flat(values: &[i32], shape: &Shape, point: &[Fr]) -> Fr {
        let padded = poly_from_i32(values, shape).data.coeffs();
        let eq = EqPolynomial::<Fr>::evals(point);
        padded
            .iter()
            .zip(eq)
            .fold(Fr::from_u64(0), |acc, (value, eq)| acc + *value * eq)
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
}
