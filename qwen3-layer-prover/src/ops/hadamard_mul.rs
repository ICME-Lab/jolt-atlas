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
//     Y(r) * 2^8 = sum_i eq(i, r)
//       * (A(i) * B(i) + round_bit(i) * 2^8 - rem(i)).
//
// The SHOUT lookup proving `rem -> round_bit` is handled by
// `prove_hadamard_round` below. Neither input is public, so verification
// returns two opening claims at the sumcheck point: `A(r')` and `B(r')`.
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
    claim::{Claim, CommittedOpeningClaim, Shape, TensorId},
    error::{ProverError, Result},
    ops::round::{
        ROUND_FRAC_BITS, RoundLookupProof, RoundParams, RoundWitness, padded_lookup_indices,
        prove_round_lookup, round_lut_table, verify_round_lookup,
    },
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
pub struct HadamardRoundRelationProof<F: JoltField, T: Transcript> {
    pub sumcheck: SumcheckInstanceProof<F, T>,
    pub a_opening: F,
    pub b_opening: F,
    pub remainder_opening: F,
    pub round_bit_opening: F,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HadamardRoundRelationClaims<F> {
    pub lhs: Claim<F>,
    pub rhs: Claim<F>,
    pub round_point: Vec<F>,
    pub remainder_opening: F,
    pub round_bit_opening: F,
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
pub struct HadamardRoundWitness<'a> {
    pub lhs: &'a [i32],
    pub rhs: &'a [i32],
    pub acc: &'a [i64],
    pub output: &'a [i32],
    pub frac_bits: [&'a [u8]; ROUND_FRAC_BITS],
}

#[derive(Debug, Clone)]
pub struct HadamardRoundProof<F: JoltField, T: Transcript> {
    pub hadamard: HadamardRoundRelationProof<F, T>,
    pub(crate) round_lookup: RoundLookupProof<F, T>,
}

impl<F: JoltField, T: Transcript> HadamardRoundProof<F, T> {
    pub fn committed_opening_claims(&self) -> Vec<CommittedOpeningClaim<F>> {
        self.round_lookup.committed_opening_claims()
    }
}

pub fn prove_hadamard_round<F, T>(
    y_claim: Claim<F>,
    witness: &HadamardRoundWitness<'_>,
    params: &HadamardRoundParams,
    transcript: &mut T,
) -> Result<(
    HadamardRoundProof<F, T>,
    Claim<F>,
    Claim<F>,
    Vec<CommittedOpeningClaim<F>>,
)>
where
    F: JoltField,
    T: Transcript,
{
    let round_witness = RoundWitness::from_input_output(witness.acc, witness.output);
    let (hadamard_proof, hadamard_claims) = prove_hadamard_round_relation(
        y_claim,
        witness.lhs,
        witness.rhs,
        &round_witness.remainder,
        &round_witness.round_bit,
        &params.hadamard,
        transcript,
    )?;
    let mut round_accumulator = ProverOpeningAccumulator::new();
    let round_point = OpeningPoint::<BIG_ENDIAN, F>::new(hadamard_claims.round_point.clone());
    round_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundLut, SumcheckId::NodeExecution(0)),
        (round_point.clone(), hadamard_claims.round_bit_opening),
    );
    round_accumulator.openings.insert(
        OpeningId::new(
            VirtualPoly::QwenRoundRemainder,
            SumcheckId::NodeExecution(0),
        ),
        (round_point, hadamard_claims.remainder_opening),
    );
    let round_lookup = prove_round_lookup(
        params.round.lookup_site,
        hadamard_claims.round_point,
        hadamard_claims.round_bit_opening,
        hadamard_claims.remainder_opening,
        padded_lookup_indices(&round_witness.remainder, &params.round.shape),
        round_lut_table(),
        &mut round_accumulator,
        transcript,
    )?;
    let round_ra = round_lookup.committed_opening_claims();

    Ok((
        HadamardRoundProof {
            hadamard: hadamard_proof,
            round_lookup,
        },
        hadamard_claims.lhs,
        hadamard_claims.rhs,
        round_ra,
    ))
}

pub fn verify_hadamard_round<F, T>(
    y_claim: Claim<F>,
    proof: &HadamardRoundProof<F, T>,
    params: &HadamardRoundParams,
    transcript: &mut T,
) -> std::result::Result<(Claim<F>, Claim<F>, Vec<CommittedOpeningClaim<F>>), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    let hadamard_claims =
        verify_hadamard_round_relation(y_claim, &proof.hadamard, &params.hadamard, transcript)?;
    let round_lookup = verify_round_lookup(
        params.round.lookup_site,
        params.round.shape.padded_power_of_two().numel(),
        hadamard_claims.round_point,
        hadamard_claims.round_bit_opening,
        hadamard_claims.remainder_opening,
        proof.round_lookup.ra_opening,
        &proof.round_lookup.committed_openings,
        &proof.round_lookup.read_raf,
        &proof.round_lookup.ra_onehot,
        &mut VerifierOpeningAccumulator::new(),
        transcript,
    )?;
    let round_ra = round_lookup.committed_opening_claims();

    Ok((hadamard_claims.lhs, hadamard_claims.rhs, round_ra))
}

pub fn prove_hadamard_round_relation<F, T>(
    y_claim: Claim<F>,
    a: &[i32],
    b: &[i32],
    remainder: &[usize],
    round_bit: &[i32],
    params: &HadamardMulParams,
    transcript: &mut T,
) -> Result<(
    HadamardRoundRelationProof<F, T>,
    HadamardRoundRelationClaims<F>,
)>
where
    F: JoltField,
    T: Transcript,
{
    validate_inputs(&y_claim, a, b, params)?;
    validate_round_witness(remainder, round_bit, params)?;
    let a_poly = padded_tensor(a, &params.shape);
    let b_poly = padded_tensor(b, &params.shape);
    let remainder_poly = padded_usize_tensor(remainder, &params.shape);
    let round_bit_poly = padded_tensor(round_bit, &params.shape);
    let sc_params = HadamardMulSumcheckParams::new(
        params.shape.padded_power_of_two().point_len(),
        y_claim.value * scale_q8::<F>(),
        y_claim.point.clone(),
    );
    let mut prover =
        HadamardRoundRelationProver::new(sc_params, a_poly, b_poly, remainder_poly, round_bit_poly);
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
        HadamardRoundRelationClaims {
            lhs: Claim {
                tensor: params.a_tensor.clone(),
                logical_shape: params.shape.clone(),
                domain_shape: params.shape.padded_power_of_two(),
                point: point.clone(),
                value: a_opening,
            },
            rhs: Claim {
                tensor: params.b_tensor.clone(),
                logical_shape: params.shape.clone(),
                domain_shape: params.shape.padded_power_of_two(),
                point: point.clone(),
                value: b_opening,
            },
            round_point: point,
            remainder_opening,
            round_bit_opening,
        },
    ))
}

pub fn verify_hadamard_round_relation<F, T>(
    y_claim: Claim<F>,
    proof: &HadamardRoundRelationProof<F, T>,
    params: &HadamardMulParams,
    transcript: &mut T,
) -> std::result::Result<HadamardRoundRelationClaims<F>, ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    verify_inputs(&y_claim, params)?;
    let sc_params = HadamardMulSumcheckParams::new(
        params.shape.padded_power_of_two().point_len(),
        y_claim.value * scale_q8::<F>(),
        y_claim.point.clone(),
    );
    let verifier = HadamardRoundRelationVerifier { params: sc_params };
    let mut accumulator = VerifierOpeningAccumulator::new();
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenHadamardA, hadamard_sumcheck_id()),
        (OpeningPoint::<BIG_ENDIAN, F>::default(), proof.a_opening),
    );
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenHadamardB, hadamard_sumcheck_id()),
        (OpeningPoint::<BIG_ENDIAN, F>::default(), proof.b_opening),
    );
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundRemainder, hadamard_sumcheck_id()),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.remainder_opening,
        ),
    );
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundLut, hadamard_sumcheck_id()),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.round_bit_opening,
        ),
    );
    let challenges = Sumcheck::verify(&proof.sumcheck, &verifier, &mut accumulator, transcript)?;
    let point = normalize_sumcheck_point::<F>(&challenges.into_opening());

    Ok(HadamardRoundRelationClaims {
        lhs: Claim {
            tensor: params.a_tensor.clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: point.clone(),
            value: proof.a_opening,
        },
        rhs: Claim {
            tensor: params.b_tensor.clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: point.clone(),
            value: proof.b_opening,
        },
        round_point: point,
        remainder_opening: proof.remainder_opening,
        round_bit_opening: proof.round_bit_opening,
    })
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
    a: MultilinearPolynomial<F>,
    b: MultilinearPolynomial<F>,
    remainder: MultilinearPolynomial<F>,
    round_bit: MultilinearPolynomial<F>,
    params: HadamardMulSumcheckParams<F>,
}

impl<F: JoltField> HadamardRoundRelationProver<F> {
    fn new(
        params: HadamardMulSumcheckParams<F>,
        a: Vec<F>,
        b: Vec<F>,
        remainder: Vec<F>,
        round_bit: Vec<F>,
    ) -> Self {
        let eq_y = MultilinearPolynomial::from(EqPolynomial::<F>::evals(&params.y_point));
        Self {
            eq_y,
            a: MultilinearPolynomial::from(a),
            b: MultilinearPolynomial::from(b),
            remainder: MultilinearPolynomial::from(remainder),
            round_bit: MultilinearPolynomial::from(round_bit),
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for HadamardRoundRelationProver<F> {
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
            let rem = [
                self.remainder.get_bound_coeff(2 * g),
                self.remainder.get_bound_coeff(2 * g + 1),
            ];
            let bit = [
                self.round_bit.get_bound_coeff(2 * g),
                self.round_bit.get_bound_coeff(2 * g + 1),
            ];
            for (idx, t) in [F::zero(), F::from_u64(2), F::from_u64(3)]
                .into_iter()
                .enumerate()
            {
                evals[idx] += lerp(eq[0], eq[1], t)
                    * (lerp(a[0], a[1], t) * lerp(b[0], b[1], t)
                        + lerp(bit[0], bit[1], t) * scale_q8::<F>()
                        - lerp(rem[0], rem[1], t));
            }
        }
        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_y.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.a.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.b.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.remainder.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.round_bit.bind_parallel(r_j, BindingOrder::LowToHigh);
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
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRoundRemainder, hadamard_sumcheck_id()),
            point.clone(),
            self.remainder.final_claim(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRoundLut, hadamard_sumcheck_id()),
            point,
            self.round_bit.final_claim(),
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
        EqPolynomial::mle(&self.params.y_point, &point)
            * (a * b + round_bit * scale_q8::<F>() - remainder)
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
            point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRoundRemainder, hadamard_sumcheck_id()),
            point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRoundLut, hadamard_sumcheck_id()),
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

fn validate_round_witness(
    remainder: &[usize],
    round_bit: &[i32],
    params: &HadamardMulParams,
) -> Result<()> {
    let expected = params.shape.numel();
    if remainder.len() != expected {
        return Err(ProverError::TensorLenMismatch {
            name: "hadamard round remainder",
            shape: params.shape.0.clone(),
            expected,
            actual: remainder.len(),
        });
    }
    if round_bit.len() != expected {
        return Err(ProverError::TensorLenMismatch {
            name: "hadamard round bit",
            shape: params.shape.0.clone(),
            expected,
            actual: round_bit.len(),
        });
    }
    for (&rem, &bit) in remainder.iter().zip(round_bit) {
        if rem >= 256 || !(bit == 0 || bit == 1) {
            return Err(ProverError::InvalidSumcheckDomain(rem));
        }
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

fn padded_usize_tensor<F: JoltField>(values: &[usize], shape: &Shape) -> Vec<F> {
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
        out[padded_flat] = F::from_u64(value as u64);
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

fn scale_q8<F: JoltField>() -> F {
    F::from_u64(256)
}

fn hadamard_sumcheck_id() -> SumcheckId {
    SumcheckId::NodeExecution(0)
}
