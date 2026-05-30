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
    ops::round::{
        RoundLookupProof, RoundParams, prove_round_lookup_from_ra, round_lookup_openings_from_ra,
        verify_round_lookup_from_ra,
    },
};

/// Rounded elementwise multiplication:
///
///   Y = round((A * B) / 2^8)
///
/// The round lookup witness is supplied as RA polynomials. This op does not
/// materialize `acc`, `rem`, or `round_bit` tensors. It asks the round module to
/// evaluate `rem(r)` and `round_bit(r)` from RA, then folds those scalar
/// openings into the Hadamard sumcheck.

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HadamardRoundParams {
    pub round: RoundParams,
    pub hadamard: HadamardMulParams,
}

impl HadamardRoundParams {
    pub fn new(round: RoundParams, hadamard: HadamardMulParams) -> Self {
        Self { round, hadamard }
    }
}

#[derive(Debug, Clone)]
pub struct HadamardRoundRelationProof<F: JoltField, T: Transcript> {
    pub sumcheck: SumcheckInstanceProof<F, T>,
    pub a_opening: F,
    pub b_opening: F,
    pub remainder_opening: F,
    pub round_bit_opening: F,
}

#[derive(Debug, Clone)]
pub struct HadamardRoundProof<F: JoltField, T: Transcript> {
    pub hadamard: HadamardRoundRelationProof<F, T>,
    pub(crate) round_lookup: RoundLookupProof<F, T>,
}

impl<F: JoltField, T: Transcript> HadamardRoundProof<F, T> {
    pub(crate) fn sumcheck_round_count(&self) -> usize {
        self.hadamard.sumcheck.compressed_polys.len() + self.round_lookup.sumcheck_round_count()
    }

    pub(crate) fn sumcheck_count(&self) -> usize {
        1 + self.round_lookup.sumcheck_count()
    }
}

pub fn prove_hadamard_round<F, T, C>(
    y_claim: Claim<F, C>,
    lhs_poly: Poly<F, C>,
    rhs_poly: Poly<F, C>,
    round_ra: Vec<Poly<F, C>>,
    params: &HadamardRoundParams,
    transcript: &mut T,
) -> Result<(
    HadamardRoundProof<F, T>,
    Claim<F, C>,
    Claim<F, C>,
    Vec<Claim<F, C>>,
)>
where
    F: JoltField,
    T: Transcript,
    C: Clone,
{
    validate_claim_shape(&y_claim, &params.hadamard.shape, "Hadamard Y claim")?;
    validate_poly_shape(&lhs_poly, &params.hadamard.shape, "Hadamard lhs")?;
    validate_poly_shape(&rhs_poly, &params.hadamard.shape, "Hadamard rhs")?;
    let (remainder_opening, round_bit_opening) =
        round_lookup_openings_from_ra(&round_ra, &y_claim.point, &params.round.shape)?;

    let (relation, point, a_value, b_value, round_point, remainder_opening, round_bit_opening) =
        prove_hadamard_round_relation(
            y_claim.point.clone(),
            y_claim.value,
            &lhs_poly,
            &rhs_poly,
            remainder_opening,
            round_bit_opening,
            &params.hadamard,
            transcript,
        )?;

    let mut round_accumulator = ProverOpeningAccumulator::new();
    let round_opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(round_point);
    round_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundLut, hadamard_sumcheck_id()),
        (round_opening_point.clone(), round_bit_opening),
    );
    round_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundRemainder, hadamard_sumcheck_id()),
        (round_opening_point, remainder_opening),
    );

    let (round_lookup, round_ra_claims) = prove_round_lookup_from_ra(
        params.round.lookup_site,
        y_claim.point,
        round_ra,
        &params.round.shape,
        round_bit_opening,
        remainder_opening,
        &mut round_accumulator,
        transcript,
    )?;

    Ok((
        HadamardRoundProof {
            hadamard: relation,
            round_lookup,
        },
        Claim::new(lhs_poly, point.clone(), a_value),
        Claim::new(rhs_poly, point, b_value),
        round_ra_claims,
    ))
}

pub fn verify_hadamard_round<F, T, C>(
    y_claim: Claim<F, C>,
    proof: &HadamardRoundProof<F, T>,
    round_ra: Vec<Poly<F, C>>,
    params: &HadamardRoundParams,
    transcript: &mut T,
) -> std::result::Result<(Claim<F, C>, Claim<F, C>, Vec<Claim<F, C>>), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
    C: Clone,
{
    verify_claim_shape(&y_claim, &params.hadamard.shape)?;
    let remainder_opening = proof.round_lookup.remainder_opening;
    let round_bit_opening = proof.round_lookup.round_bit_opening;
    let (point, a_value, b_value, round_point) = verify_hadamard_round_relation(
        y_claim.point.clone(),
        y_claim.value,
        proof,
        remainder_opening,
        round_bit_opening,
        &params.hadamard,
        transcript,
    )?;

    let mut round_accumulator = VerifierOpeningAccumulator::new();
    let round_opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(round_point);
    round_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundLut, hadamard_sumcheck_id()),
        (round_opening_point.clone(), round_bit_opening),
    );
    round_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundRemainder, hadamard_sumcheck_id()),
        (round_opening_point, remainder_opening),
    );
    let (_round_lookup, round_ra_claims) = verify_round_lookup_from_ra(
        params.round.lookup_site,
        y_claim.point,
        round_ra,
        &params.round.shape,
        round_bit_opening,
        remainder_opening,
        &proof.round_lookup,
        &mut round_accumulator,
        transcript,
    )?;

    let domain_len = params.hadamard.shape.padded_power_of_two().numel();
    let placeholder = || {
        Poly::new(
            MultilinearPolynomial::from(vec![F::zero(); domain_len]),
            None,
        )
    };
    Ok((
        Claim::new(placeholder(), point.clone(), a_value),
        Claim::new(placeholder(), point, b_value),
        round_ra_claims,
    ))
}

fn prove_hadamard_round_relation<F, T, C>(
    y_point: Vec<F>,
    y_value: F,
    lhs_poly: &Poly<F, C>,
    rhs_poly: &Poly<F, C>,
    remainder_opening: F,
    round_bit_opening: F,
    params: &HadamardMulParams,
    transcript: &mut T,
) -> Result<(HadamardRoundRelationProof<F, T>, Vec<F>, F, F, Vec<F>, F, F)>
where
    F: JoltField,
    T: Transcript,
{
    validate_shape(&params.shape)?;
    validate_poly_shape(lhs_poly, &params.shape, "Hadamard lhs")?;
    validate_poly_shape(rhs_poly, &params.shape, "Hadamard rhs")?;
    if y_point.len() != params.shape.padded_power_of_two().point_len() {
        return Err(ProverError::ShapeMismatch {
            name: "Hadamard Y claim point",
            expected: vec![params.shape.padded_power_of_two().point_len()],
            actual: vec![y_point.len()],
        });
    }

    let sc_params = HadamardMulSumcheckParams::new(
        params.shape.padded_power_of_two().point_len(),
        y_value * scale_q8::<F>(),
        y_point.clone(),
    );
    let mut prover = HadamardRoundRelationProver::new(
        sc_params,
        lhs_poly.data.clone(),
        rhs_poly.data.clone(),
        remainder_opening,
        round_bit_opening,
    );
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
    let remainder_opening = opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenRoundRemainder, hadamard_sumcheck_id()),
    )?;
    let round_bit_opening = opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenRoundLut, hadamard_sumcheck_id()),
    )?;
    let point = normalize_sumcheck_point::<F>(&challenges.into_opening());

    Ok((
        HadamardRoundRelationProof {
            sumcheck,
            a_opening,
            b_opening,
            remainder_opening,
            round_bit_opening,
        },
        point,
        a_opening,
        b_opening,
        y_point,
        remainder_opening,
        round_bit_opening,
    ))
}

fn verify_hadamard_round_relation<F, T>(
    y_point: Vec<F>,
    y_value: F,
    proof: &HadamardRoundProof<F, T>,
    remainder_opening: F,
    round_bit_opening: F,
    params: &HadamardMulParams,
    transcript: &mut T,
) -> std::result::Result<(Vec<F>, F, F, Vec<F>), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    verify_shape(&params.shape)?;
    if y_point.len() != params.shape.padded_power_of_two().point_len() {
        return Err(ProofVerifyError::InvalidInputLength(
            params.shape.padded_power_of_two().point_len(),
            y_point.len(),
        ));
    }
    let sc_params = HadamardMulSumcheckParams::new(
        params.shape.padded_power_of_two().point_len(),
        y_value * scale_q8::<F>(),
        y_point.clone(),
    );
    let verifier = HadamardRoundRelationVerifier { params: sc_params };
    let mut accumulator = VerifierOpeningAccumulator::new();
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenHadamardA, hadamard_sumcheck_id()),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.hadamard.a_opening,
        ),
    );
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenHadamardB, hadamard_sumcheck_id()),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.hadamard.b_opening,
        ),
    );
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundRemainder, hadamard_sumcheck_id()),
        (OpeningPoint::<BIG_ENDIAN, F>::default(), remainder_opening),
    );
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundLut, hadamard_sumcheck_id()),
        (OpeningPoint::<BIG_ENDIAN, F>::default(), round_bit_opening),
    );
    let challenges = Sumcheck::verify(
        &proof.hadamard.sumcheck,
        &verifier,
        &mut accumulator,
        transcript,
    )?;
    let point = normalize_sumcheck_point::<F>(&challenges.into_opening());

    Ok((
        point,
        proof.hadamard.a_opening,
        proof.hadamard.b_opening,
        y_point,
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

struct HadamardRoundRelationProver<F: JoltField> {
    eq_y: MultilinearPolynomial<F>,
    eq_zero: LazyEqZeroLowToHigh<F>,
    a: MultilinearPolynomial<F>,
    b: MultilinearPolynomial<F>,
    remainder: F,
    round_bit: F,
    params: HadamardMulSumcheckParams<F>,
}

impl<F: JoltField> HadamardRoundRelationProver<F> {
    fn new(
        params: HadamardMulSumcheckParams<F>,
        a: MultilinearPolynomial<F>,
        b: MultilinearPolynomial<F>,
        remainder: F,
        round_bit: F,
    ) -> Self {
        let eq_y = MultilinearPolynomial::from(EqPolynomial::<F>::evals(&params.y_point));
        Self {
            eq_y,
            eq_zero: LazyEqZeroLowToHigh::new(params.num_rounds),
            a,
            b,
            remainder,
            round_bit,
            params,
        }
    }

    fn round_term(&self) -> F {
        self.round_bit * scale_q8::<F>() - self.remainder
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for HadamardRoundRelationProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let mut evals = [F::zero(); 3];
        let round_term = self.round_term();
        for g in 0..self.a.len() / 2 {
            let eq = [
                self.eq_y.get_bound_coeff(2 * g),
                self.eq_y.get_bound_coeff(2 * g + 1),
            ];
            let z = [
                self.eq_zero.get_bound_coeff(2 * g),
                self.eq_zero.get_bound_coeff(2 * g + 1),
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
                evals[idx] += lerp(eq[0], eq[1], t) * lerp(a[0], a[1], t) * lerp(b[0], b[1], t)
                    + lerp(z[0], z[1], t) * round_term;
            }
        }
        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_y.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.eq_zero.bind(r_j);
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
            point.clone(),
            self.b.final_claim(),
        );
        let round_point = OpeningPoint::<BIG_ENDIAN, F>::new(self.params.y_point.clone());
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRoundRemainder, hadamard_sumcheck_id()),
            round_point.clone(),
            self.remainder,
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRoundLut, hadamard_sumcheck_id()),
            round_point,
            self.round_bit,
        );
    }
}

struct HadamardRoundRelationVerifier<F: JoltField> {
    params: HadamardMulSumcheckParams<F>,
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for HadamardRoundRelationVerifier<F>
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
        let remainder = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenRoundRemainder,
                hadamard_sumcheck_id(),
            ))
            .1;
        let round_bit = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenRoundLut,
                hadamard_sumcheck_id(),
            ))
            .1;
        let point = normalize_sumcheck_point::<F>(&sumcheck_challenges.into_opening());
        let eq_zero = EqPolynomial::mle(&vec![F::zero(); self.params.num_rounds], &point);
        EqPolynomial::mle(&self.params.y_point, &point) * a * b
            + eq_zero * (round_bit * scale_q8::<F>() - remainder)
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
        let round_point = OpeningPoint::<BIG_ENDIAN, F>::new(self.params.y_point.clone());
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRoundRemainder, hadamard_sumcheck_id()),
            round_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRoundLut, hadamard_sumcheck_id()),
            round_point,
        );
    }
}

struct LazyEqZeroLowToHigh<F: JoltField> {
    remaining_len: usize,
    bound_scale: F,
}

impl<F: JoltField> LazyEqZeroLowToHigh<F> {
    fn new(vars: usize) -> Self {
        Self {
            remaining_len: 1usize << vars,
            bound_scale: F::one(),
        }
    }

    fn get_bound_coeff(&self, index: usize) -> F {
        debug_assert!(index < self.remaining_len);
        if index == 0 {
            self.bound_scale
        } else {
            F::zero()
        }
    }

    fn bind(&mut self, r_j: F::Challenge) {
        debug_assert!(self.remaining_len > 1);
        let r_j: F = r_j.into();
        self.bound_scale *= F::one() - r_j;
        self.remaining_len /= 2;
    }
}

fn validate_claim_shape<F: JoltField, C>(
    claim: &Claim<F, C>,
    shape: &Shape,
    name: &'static str,
) -> Result<()> {
    let domain = shape.padded_power_of_two();
    if claim.poly.data.len() != domain.numel() {
        return Err(ProverError::ShapeMismatch {
            name,
            expected: domain.0,
            actual: vec![claim.poly.data.len()],
        });
    }
    if claim.point.len() != domain.point_len() {
        return Err(ProverError::ShapeMismatch {
            name,
            expected: vec![domain.point_len()],
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
            shape: shape.0.clone(),
            expected,
            actual: poly.data.len(),
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

fn verify_claim_shape<F: JoltField, C>(
    claim: &Claim<F, C>,
    shape: &Shape,
) -> std::result::Result<(), ProofVerifyError> {
    let domain = shape.padded_power_of_two();
    if claim.poly.data.len() != domain.numel() {
        return Err(ProofVerifyError::InvalidInputLength(
            domain.numel(),
            claim.poly.data.len(),
        ));
    }
    if claim.point.len() != domain.point_len() {
        return Err(ProofVerifyError::InvalidInputLength(
            domain.point_len(),
            claim.point.len(),
        ));
    }
    Ok(())
}

fn verify_shape(shape: &Shape) -> std::result::Result<(), ProofVerifyError> {
    if shape.dims().contains(&0) {
        return Err(ProofVerifyError::InvalidInputLength(1, 0));
    }
    Ok(())
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

fn scale_q8<F: JoltField>() -> F {
    F::from_u64(256)
}

fn hadamard_sumcheck_id() -> SumcheckId {
    SumcheckId::NodeExecution(0)
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use joltworks::{
        config::{OneHotConfig, OneHotParams},
        poly::{
            eq_poly::EqPolynomial, multilinear_polynomial::MultilinearPolynomial,
            one_hot_polynomial::OneHotPolynomial,
        },
        transcripts::Blake2bTranscript,
    };

    use super::*;
    use crate::ops::round::{ROUND_LUT_LEN, round_lut_q8};

    #[test]
    fn proves_and_verifies_hadamard_round_from_poly_and_ra() {
        let shape = Shape::new(vec![2, 2]);
        let params = HadamardRoundParams::new(
            RoundParams::new(vec![2, 2], "acc", "Y"),
            HadamardMulParams::new(vec![2, 2], "A", "B"),
        );
        let a = vec![3, 5, 7, 11];
        let b = vec![13, 17, 19, 23];
        let acc = a
            .iter()
            .zip(&b)
            .map(|(&a, &b)| i64::from(a) * i64::from(b))
            .collect::<Vec<_>>();
        let y = acc.iter().map(|&v| round_q8_to_i32(v)).collect::<Vec<_>>();
        let point = vec![Fr::from(3_u64), Fr::from(5_u64)];
        let y_claim = Claim::new(poly_from_i32(&y), point.clone(), eval_flat(&y, &point));
        let round_ra = round_ra_from_acc(&acc, shape.padded_power_of_two().numel());

        let mut prover_transcript = Blake2bTranscript::default();
        let (proof, lhs, rhs, ra) = prove_hadamard_round(
            y_claim,
            poly_from_i32(&a),
            poly_from_i32(&b),
            round_ra,
            &params,
            &mut prover_transcript,
        )
        .unwrap();
        assert!(!ra.is_empty());

        let y_claim = Claim::new(poly_from_i32(&y), point.clone(), eval_flat(&y, &point));
        let round_ra = round_ra_from_acc(&acc, shape.padded_power_of_two().numel());
        let mut verifier_transcript = Blake2bTranscript::default();
        let (verified_lhs, verified_rhs, verified_ra) =
            verify_hadamard_round(y_claim, &proof, round_ra, &params, &mut verifier_transcript)
                .unwrap();

        assert_eq!(verified_lhs.point, lhs.point);
        assert_eq!(verified_lhs.value, lhs.value);
        assert_eq!(verified_rhs.point, rhs.point);
        assert_eq!(verified_rhs.value, rhs.value);
        assert_eq!(verified_ra.len(), ra.len());
    }

    fn poly_from_i32(values: &[i32]) -> Poly<Fr, ()> {
        Poly::new(MultilinearPolynomial::from(values.to_vec()), None)
    }

    fn eval_flat(values: &[i32], point: &[Fr]) -> Fr {
        let eq = EqPolynomial::<Fr>::evals(point);
        values
            .iter()
            .zip(eq)
            .fold(Fr::from_u64(0), |acc, (&value, eq)| {
                acc + Fr::from_i32(value) * eq
            })
    }

    fn round_q8_to_i32(value: i64) -> i32 {
        let rem = value.rem_euclid(ROUND_LUT_LEN as i64);
        let rounded = (value + i64::from(round_lut_q8(rem as usize)) * 256 - rem) / 256;
        i32::try_from(rounded).expect("rounded output exceeds i32")
    }

    fn round_ra_from_acc(acc: &[i64], padded_len: usize) -> Vec<Poly<Fr, ()>> {
        let log_k = ROUND_LUT_LEN.trailing_zeros() as usize;
        let params = OneHotParams::from_config_and_log_K(&OneHotConfig::default(), log_k);
        (0..params.instruction_d)
            .map(|chunk| {
                let indices = (0..padded_len)
                    .map(|idx| {
                        let rem = acc
                            .get(idx)
                            .copied()
                            .unwrap_or_default()
                            .rem_euclid(ROUND_LUT_LEN as i64)
                            as u64;
                        Some(u16::from(params.lookup_index_chunk(rem, chunk)))
                    })
                    .collect::<Vec<_>>();
                Poly::new(
                    MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                        indices,
                        params.k_chunk,
                    )),
                    None,
                )
            })
            .collect()
    }
}
