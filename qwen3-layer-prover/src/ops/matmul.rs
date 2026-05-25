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

/// Rounded fixed-point matmul:
///
///   Y = round((A @ W) / 2^8)
///
/// The op is written in reverse claim flow. The caller gives a claim on `Y`.
/// This op proves that claim from `A`, `W`, and the round lookup RA polys, then
/// returns exactly the claims that still need to be discharged:
///
/// - `A` claim: consumed by the previous op in the layer graph.
/// - `W` claim: directly evaluated because weights are fixed inputs.
/// - round RA claims: opened by PCS.
///
/// No witness struct is accepted here. `A`, `W`, and RA are already represented
/// as `Poly`, which is the only value container used by the layer IOP.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatMulParams {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub a_tensor: TensorId,
    pub w_tensor: TensorId,
}

impl MatMulParams {
    pub fn new(
        m: usize,
        k: usize,
        n: usize,
        a_tensor: impl Into<String>,
        w_tensor: impl Into<String>,
    ) -> Self {
        Self {
            m,
            k,
            n,
            a_tensor: TensorId::new(a_tensor),
            w_tensor: TensorId::new(w_tensor),
        }
    }

    pub fn a_shape(&self) -> Shape {
        Shape::new(vec![self.m, self.k])
    }

    pub fn w_shape(&self) -> Shape {
        Shape::new(vec![self.k, self.n])
    }

    pub fn y_shape(&self) -> Shape {
        Shape::new(vec![self.m, self.n])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatMulRoundParams {
    pub round: RoundParams,
    pub matmul: MatMulParams,
}

impl MatMulRoundParams {
    pub fn new(round: RoundParams, matmul: MatMulParams) -> Self {
        Self { round, matmul }
    }
}

#[derive(Debug, Clone)]
pub struct MatMulRoundRelationProof<F: JoltField, T: Transcript> {
    pub k_sumcheck: SumcheckInstanceProof<F, T>,
    pub m_sumcheck: SumcheckInstanceProof<F, T>,
    pub a_r_opening: F,
    pub w_r_opening: F,
    pub a_opening: F,
    pub remainder_opening: F,
    pub round_bit_opening: F,
}

#[derive(Debug, Clone)]
pub struct MatMulRoundProof<F: JoltField, T: Transcript> {
    pub matmul: MatMulRoundRelationProof<F, T>,
    pub(crate) round_lookup: RoundLookupProof<F, T>,
}

pub fn prove_matmul_round<F, T, C>(
    y_claim: Claim<F, C>,
    a_poly: Poly<F, C>,
    w_poly: Poly<F, C>,
    round_ra: Vec<Poly<F, C>>,
    params: &MatMulRoundParams,
    transcript: &mut T,
) -> Result<(
    MatMulRoundProof<F, T>,
    Claim<F, C>,
    Claim<F, C>,
    Vec<Claim<F, C>>,
)>
where
    F: JoltField,
    T: Transcript,
    C: Clone,
{
    validate_claim_shape(&y_claim, &params.matmul.y_shape(), "Y claim")?;
    validate_poly_shape(&a_poly, &params.matmul.a_shape(), "A")?;
    validate_poly_shape(&w_poly, &params.matmul.w_shape(), "W")?;
    let (remainder_opening, round_bit_opening) =
        round_lookup_openings_from_ra(&round_ra, &y_claim.point, &params.round.shape)?;

    let (
        relation,
        a_point,
        a_value,
        w_point,
        w_value,
        round_point,
        remainder_opening,
        round_bit_opening,
    ) = prove_matmul_round_relation(
        y_claim.point.clone(),
        y_claim.value,
        &a_poly,
        &w_poly,
        remainder_opening,
        round_bit_opening,
        &params.matmul,
        transcript,
    )?;

    let mut round_accumulator = ProverOpeningAccumulator::new();
    let round_opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(round_point);
    round_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundLut, k_sumcheck_id()),
        (round_opening_point.clone(), round_bit_opening),
    );
    round_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundRemainder, k_sumcheck_id()),
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
        MatMulRoundProof {
            matmul: relation,
            round_lookup,
        },
        Claim::new(a_poly, a_point, a_value),
        Claim::new(w_poly, w_point, w_value),
        round_ra_claims,
    ))
}

pub fn verify_matmul_round<F, T, C>(
    y_claim: Claim<F, C>,
    proof: &MatMulRoundProof<F, T>,
    w_poly: Poly<F, C>,
    round_ra: Vec<Poly<F, C>>,
    params: &MatMulRoundParams,
    transcript: &mut T,
) -> std::result::Result<(Claim<F, C>, Claim<F, C>, Vec<Claim<F, C>>), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
    C: Clone,
{
    verify_claim_shape(&y_claim, &params.matmul.y_shape())?;
    verify_poly_shape(&w_poly, &params.matmul.w_shape())?;
    let (remainder_opening, round_bit_opening) =
        round_lookup_openings_from_ra(&round_ra, &y_claim.point, &params.round.shape)
            .map_err(|_| ProofVerifyError::SumcheckVerificationError)?;

    let (a_point, a_value, w_point, w_value, round_point) = verify_matmul_round_relation(
        y_claim.point.clone(),
        y_claim.value,
        proof,
        &w_poly,
        remainder_opening,
        round_bit_opening,
        &params.matmul,
        transcript,
    )?;

    let mut round_accumulator = VerifierOpeningAccumulator::new();
    let round_opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(round_point);
    round_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundLut, k_sumcheck_id()),
        (round_opening_point.clone(), round_bit_opening),
    );
    round_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundRemainder, k_sumcheck_id()),
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

    let a_poly = Poly::new(
        MultilinearPolynomial::from(vec![
            F::zero();
            params.matmul.a_shape().padded_power_of_two().numel()
        ]),
        None,
    );
    Ok((
        Claim::new(a_poly, a_point, a_value),
        Claim::new(w_poly, w_point, w_value),
        round_ra_claims,
    ))
}

fn prove_matmul_round_relation<F, T, C>(
    y_point: Vec<F>,
    y_value: F,
    a_poly: &Poly<F, C>,
    w_poly: &Poly<F, C>,
    remainder_opening_at_y: F,
    round_bit_opening_at_y: F,
    params: &MatMulParams,
    transcript: &mut T,
) -> Result<(
    MatMulRoundRelationProof<F, T>,
    Vec<F>,
    F,
    Vec<F>,
    F,
    Vec<F>,
    F,
    F,
)>
where
    F: JoltField,
    T: Transcript,
{
    validate_dimensions(params)?;
    validate_poly_shape(a_poly, &params.a_shape(), "A")?;
    validate_poly_shape(w_poly, &params.w_shape(), "W")?;
    let (r_m, r_n) = split_y_point(&y_point, params)?;

    let partials = MatMulPartials::new(a_poly, w_poly, params, r_m, r_n);

    let k_params = KSumcheckParams::new(log2_ceil(params.k), y_value * scale_q8::<F>());
    let mut k_prover = KRoundSumcheckProver::new(
        k_params,
        partials.a_r,
        partials.w_r,
        remainder_opening_at_y,
        round_bit_opening_at_y,
        y_point.clone(),
    );
    let mut k_accumulator = ProverOpeningAccumulator::new();
    let (k_sumcheck, k_challenges) = Sumcheck::prove(&mut k_prover, &mut k_accumulator, transcript);
    let a_r_opening = prover_opening(
        &k_accumulator,
        OpeningId::new(VirtualPoly::QwenMatMulPartialA, k_sumcheck_id()),
    )?;
    let w_r_opening = prover_opening(
        &k_accumulator,
        OpeningId::new(VirtualPoly::QwenMatMulPartialW, k_sumcheck_id()),
    )?;
    let remainder_opening = prover_opening(
        &k_accumulator,
        OpeningId::new(VirtualPoly::QwenRoundRemainder, k_sumcheck_id()),
    )?;
    let round_bit_opening = prover_opening(
        &k_accumulator,
        OpeningId::new(VirtualPoly::QwenRoundLut, k_sumcheck_id()),
    )?;
    let k_challenges = k_challenges.into_opening();
    let r_k = normalize_sumcheck_point::<F>(&k_challenges);

    let a_by_m = fix_a_k(a_poly, params, &r_k);
    let m_params = MSumcheckParams::new(log2_ceil(params.m), a_r_opening, r_m.to_vec());
    let mut m_prover = MSumcheckProver::new(m_params, a_by_m);
    let mut m_accumulator = ProverOpeningAccumulator::new();
    let (m_sumcheck, m_challenges) = Sumcheck::prove(&mut m_prover, &mut m_accumulator, transcript);
    let a_opening = prover_opening(
        &m_accumulator,
        OpeningId::new(VirtualPoly::QwenMatMulA, m_sumcheck_id()),
    )?;

    let m_challenges = m_challenges.into_opening();
    let mut a_point = normalize_sumcheck_point::<F>(&m_challenges);
    a_point.extend(r_k.clone());
    let mut w_point = r_k;
    w_point.extend(r_n);

    Ok((
        MatMulRoundRelationProof {
            k_sumcheck,
            m_sumcheck,
            a_r_opening,
            w_r_opening,
            a_opening,
            remainder_opening,
            round_bit_opening,
        },
        a_point,
        a_opening,
        w_point,
        w_r_opening,
        y_point,
        remainder_opening,
        round_bit_opening,
    ))
}

fn verify_matmul_round_relation<F, T, C>(
    y_point: Vec<F>,
    y_value: F,
    proof: &MatMulRoundProof<F, T>,
    w_poly: &Poly<F, C>,
    remainder_opening_at_y: F,
    round_bit_opening_at_y: F,
    params: &MatMulParams,
    transcript: &mut T,
) -> std::result::Result<(Vec<F>, F, Vec<F>, F, Vec<F>), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    verify_dimensions(params)?;
    verify_poly_shape(w_poly, &params.w_shape())?;
    let (r_m, r_n) = split_y_point_verify(&y_point, params)?;

    let k_params = KSumcheckParams::new(log2_ceil(params.k), y_value * scale_q8::<F>());
    let k_verifier = KRoundSumcheckVerifier {
        params: k_params,
        y_point: y_point.clone(),
    };
    let mut k_accumulator = VerifierOpeningAccumulator::new();
    k_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenMatMulPartialA, k_sumcheck_id()),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.matmul.a_r_opening,
        ),
    );
    k_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenMatMulPartialW, k_sumcheck_id()),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.matmul.w_r_opening,
        ),
    );
    k_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundRemainder, k_sumcheck_id()),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            remainder_opening_at_y,
        ),
    );
    k_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundLut, k_sumcheck_id()),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            round_bit_opening_at_y,
        ),
    );
    let r_k_challenges = Sumcheck::verify(
        &proof.matmul.k_sumcheck,
        &k_verifier,
        &mut k_accumulator,
        transcript,
    )?
    .into_opening();
    let r_k = normalize_sumcheck_point::<F>(&r_k_challenges);

    let expected_w = eval_w_poly(w_poly, params, &r_k, r_n);
    if proof.matmul.w_r_opening != expected_w {
        return Err(ProofVerifyError::SumcheckVerificationError);
    }

    let m_params =
        MSumcheckParams::new(log2_ceil(params.m), proof.matmul.a_r_opening, r_m.to_vec());
    let m_verifier = MSumcheckVerifier { params: m_params };
    let mut m_accumulator = VerifierOpeningAccumulator::new();
    m_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenMatMulA, m_sumcheck_id()),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.matmul.a_opening,
        ),
    );
    let r_m_prime_challenges = Sumcheck::verify(
        &proof.matmul.m_sumcheck,
        &m_verifier,
        &mut m_accumulator,
        transcript,
    )?
    .into_opening();

    let mut a_point = normalize_sumcheck_point::<F>(&r_m_prime_challenges);
    a_point.extend(r_k.clone());
    let mut w_point = r_k;
    w_point.extend(r_n);
    Ok((
        a_point,
        proof.matmul.a_opening,
        w_point,
        proof.matmul.w_r_opening,
        y_point,
    ))
}

struct MatMulPartials<F: JoltField> {
    a_r: Vec<F>,
    w_r: Vec<F>,
}

impl<F: JoltField> MatMulPartials<F> {
    fn new<C>(
        a_poly: &Poly<F, C>,
        w_poly: &Poly<F, C>,
        params: &MatMulParams,
        r_m: &[F],
        r_n: &[F],
    ) -> Self {
        let k_pad = params.k.next_power_of_two();
        let row_eq = EqPolynomial::<F>::evals(r_m);
        let col_eq = EqPolynomial::<F>::evals(r_n);

        let mut a_r = vec![F::zero(); k_pad];
        for (kk, out) in a_r.iter_mut().enumerate().take(params.k) {
            *out = (0..params.m)
                .map(|row| row_eq[row] * logical_matrix_coeff(a_poly, &params.a_shape(), row, kk))
                .sum();
        }

        let mut w_r = vec![F::zero(); k_pad];
        for (kk, out) in w_r.iter_mut().enumerate().take(params.k) {
            *out = (0..params.n)
                .map(|col| col_eq[col] * logical_matrix_coeff(w_poly, &params.w_shape(), kk, col))
                .sum();
        }

        Self { a_r, w_r }
    }
}

struct KSumcheckParams<F: JoltField> {
    k_vars: usize,
    input_claim: F,
}

impl<F: JoltField> KSumcheckParams<F> {
    fn new(k_vars: usize, input_claim: F) -> Self {
        Self {
            k_vars,
            input_claim,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for KSumcheckParams<F> {
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.k_vars
    }

    fn input_claim(
        &self,
        _accumulator: &dyn joltworks::poly::opening_proof::OpeningAccumulator<F>,
    ) -> F {
        self.input_claim
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

struct KRoundSumcheckProver<F: JoltField> {
    a_r: MultilinearPolynomial<F>,
    w_r: MultilinearPolynomial<F>,
    eq_zero_k: LazyEqZeroLowToHigh<F>,
    remainder: F,
    round_bit: F,
    y_point: Vec<F>,
    params: KSumcheckParams<F>,
}

impl<F: JoltField> KRoundSumcheckProver<F> {
    fn new(
        params: KSumcheckParams<F>,
        a_r: Vec<F>,
        w_r: Vec<F>,
        remainder: F,
        round_bit: F,
        y_point: Vec<F>,
    ) -> Self {
        Self {
            a_r: MultilinearPolynomial::from(a_r),
            w_r: MultilinearPolynomial::from(w_r),
            eq_zero_k: LazyEqZeroLowToHigh::new(params.k_vars),
            remainder,
            round_bit,
            y_point,
            params,
        }
    }

    fn round_term(&self) -> F {
        self.round_bit * scale_q8::<F>() - self.remainder
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for KRoundSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let mut eval_at_0 = F::zero();
        let mut eval_at_2 = F::zero();
        let round_term = self.round_term();
        for g in 0..self.a_r.len() / 2 {
            let a0 = self.a_r.get_bound_coeff(2 * g);
            let a1 = self.a_r.get_bound_coeff(2 * g + 1);
            let w0 = self.w_r.get_bound_coeff(2 * g);
            let w1 = self.w_r.get_bound_coeff(2 * g + 1);
            let z0 = self.eq_zero_k.get_bound_coeff(2 * g);
            let z1 = self.eq_zero_k.get_bound_coeff(2 * g + 1);
            eval_at_0 += a0 * w0 + z0 * round_term;
            eval_at_2 += (a1 + a1 - a0) * (w1 + w1 - w0) + (z1 + z1 - z0) * round_term;
        }
        UniPoly::from_evals_and_hint(previous_claim, &[eval_at_0, eval_at_2])
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.a_r.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.w_r.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.eq_zero_k.bind(r_j);
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
            OpeningId::new(VirtualPoly::QwenMatMulPartialA, k_sumcheck_id()),
            point.clone(),
            self.a_r.final_claim(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenMatMulPartialW, k_sumcheck_id()),
            point,
            self.w_r.final_claim(),
        );
        let round_point = OpeningPoint::<BIG_ENDIAN, F>::new(self.y_point.clone());
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRoundRemainder, k_sumcheck_id()),
            round_point.clone(),
            self.remainder,
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRoundLut, k_sumcheck_id()),
            round_point,
            self.round_bit,
        );
    }
}

struct KRoundSumcheckVerifier<F: JoltField> {
    params: KSumcheckParams<F>,
    y_point: Vec<F>,
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for KRoundSumcheckVerifier<F> {
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
                VirtualPoly::QwenMatMulPartialA,
                k_sumcheck_id(),
            ))
            .1;
        let w = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenMatMulPartialW,
                k_sumcheck_id(),
            ))
            .1;
        let remainder = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenRoundRemainder,
                k_sumcheck_id(),
            ))
            .1;
        let round_bit = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenRoundLut,
                k_sumcheck_id(),
            ))
            .1;
        let r_k = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        let eq_zero = EqPolynomial::mle(&vec![F::zero(); self.params.k_vars], &r_k.r);
        a * w + eq_zero * (round_bit * scale_q8::<F>() - remainder)
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
            OpeningId::new(VirtualPoly::QwenMatMulPartialA, k_sumcheck_id()),
            point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenMatMulPartialW, k_sumcheck_id()),
            point,
        );
        let round_point = OpeningPoint::<BIG_ENDIAN, F>::new(self.y_point.clone());
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRoundRemainder, k_sumcheck_id()),
            round_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRoundLut, k_sumcheck_id()),
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

struct MSumcheckParams<F: JoltField> {
    m_vars: usize,
    input_claim: F,
    r_m: Vec<F>,
}

impl<F: JoltField> MSumcheckParams<F> {
    fn new(m_vars: usize, input_claim: F, r_m: Vec<F>) -> Self {
        Self {
            m_vars,
            input_claim,
            r_m,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for MSumcheckParams<F> {
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.m_vars
    }

    fn input_claim(
        &self,
        _accumulator: &dyn joltworks::poly::opening_proof::OpeningAccumulator<F>,
    ) -> F {
        self.input_claim
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

struct MSumcheckProver<F: JoltField> {
    eq_m: MultilinearPolynomial<F>,
    a: MultilinearPolynomial<F>,
    params: MSumcheckParams<F>,
}

impl<F: JoltField> MSumcheckProver<F> {
    fn new(params: MSumcheckParams<F>, a: Vec<F>) -> Self {
        let eq_m = MultilinearPolynomial::from(EqPolynomial::<F>::evals(&params.r_m));
        Self {
            eq_m,
            a: MultilinearPolynomial::from(a),
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for MSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let mut eval_at_0 = F::zero();
        let mut eval_at_2 = F::zero();
        for g in 0..self.a.len() / 2 {
            let e0 = self.eq_m.get_bound_coeff(2 * g);
            let e1 = self.eq_m.get_bound_coeff(2 * g + 1);
            let a0 = self.a.get_bound_coeff(2 * g);
            let a1 = self.a.get_bound_coeff(2 * g + 1);
            eval_at_0 += e0 * a0;
            eval_at_2 += (e1 + e1 - e0) * (a1 + a1 - a0);
        }
        UniPoly::from_evals_and_hint(previous_claim, &[eval_at_0, eval_at_2])
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_m.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.a.bind_parallel(r_j, BindingOrder::LowToHigh);
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
            OpeningId::new(VirtualPoly::QwenMatMulA, m_sumcheck_id()),
            point,
            self.a.final_claim(),
        );
    }
}

struct MSumcheckVerifier<F: JoltField> {
    params: MSumcheckParams<F>,
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for MSumcheckVerifier<F> {
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
                VirtualPoly::QwenMatMulA,
                m_sumcheck_id(),
            ))
            .1;
        let r = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        EqPolynomial::mle(&self.params.r_m, &r.r) * a
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
            OpeningId::new(VirtualPoly::QwenMatMulA, m_sumcheck_id()),
            point,
        );
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

fn validate_dimensions(params: &MatMulParams) -> Result<()> {
    if params.m == 0 {
        return Err(ProverError::InvalidMatrixDimension { name: "m" });
    }
    if params.k == 0 {
        return Err(ProverError::InvalidMatrixDimension { name: "k" });
    }
    if params.n == 0 {
        return Err(ProverError::InvalidMatrixDimension { name: "n" });
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

fn verify_poly_shape<F: JoltField, C>(
    poly: &Poly<F, C>,
    shape: &Shape,
) -> std::result::Result<(), ProofVerifyError> {
    let expected = shape.padded_power_of_two().numel();
    if poly.data.len() != expected {
        return Err(ProofVerifyError::InvalidInputLength(
            expected,
            poly.data.len(),
        ));
    }
    Ok(())
}

fn verify_dimensions(params: &MatMulParams) -> std::result::Result<(), ProofVerifyError> {
    if params.m == 0 || params.k == 0 || params.n == 0 {
        return Err(ProofVerifyError::InvalidInputLength(1, 0));
    }
    Ok(())
}

fn split_y_point<'a, F: JoltField>(
    point: &'a [F],
    params: &MatMulParams,
) -> Result<(&'a [F], &'a [F])> {
    let row_vars = log2_ceil(params.m);
    let col_vars = log2_ceil(params.n);
    if point.len() != row_vars + col_vars {
        return Err(ProverError::ShapeMismatch {
            name: "Y claim point",
            expected: vec![row_vars + col_vars],
            actual: vec![point.len()],
        });
    }
    Ok(point.split_at(row_vars))
}

fn split_y_point_verify<'a, F: JoltField>(
    point: &'a [F],
    params: &MatMulParams,
) -> std::result::Result<(&'a [F], &'a [F]), ProofVerifyError> {
    let row_vars = log2_ceil(params.m);
    let col_vars = log2_ceil(params.n);
    if point.len() != row_vars + col_vars {
        return Err(ProofVerifyError::InvalidInputLength(
            row_vars + col_vars,
            point.len(),
        ));
    }
    Ok(point.split_at(row_vars))
}

fn fix_a_k<F: JoltField, C>(a_poly: &Poly<F, C>, params: &MatMulParams, r_k: &[F]) -> Vec<F> {
    let m_pad = params.m.next_power_of_two();
    let k_eq = EqPolynomial::<F>::evals(r_k);
    let mut out = vec![F::zero(); m_pad];
    for row in 0..params.m {
        out[row] = (0..params.k)
            .map(|kk| k_eq[kk] * logical_matrix_coeff(a_poly, &params.a_shape(), row, kk))
            .sum();
    }
    out
}

fn eval_w_poly<F: JoltField, C>(
    w_poly: &Poly<F, C>,
    params: &MatMulParams,
    r_k: &[F],
    r_n: &[F],
) -> F {
    let k_eq = EqPolynomial::<F>::evals(r_k);
    let n_eq = EqPolynomial::<F>::evals(r_n);
    let mut out = F::zero();
    for kk in 0..params.k {
        for col in 0..params.n {
            out += k_eq[kk] * n_eq[col] * logical_matrix_coeff(w_poly, &params.w_shape(), kk, col);
        }
    }
    out
}

fn logical_matrix_coeff<F: JoltField, C>(
    poly: &Poly<F, C>,
    shape: &Shape,
    row: usize,
    col: usize,
) -> F {
    debug_assert_eq!(shape.dims().len(), 2);
    let padded = shape.padded_power_of_two();
    let index = row * padded.dims()[1] + col;
    poly.data.get_coeff(index)
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

fn log2_ceil(value: usize) -> usize {
    value.next_power_of_two().trailing_zeros() as usize
}

fn scale_q8<F: JoltField>() -> F {
    F::from_u64(256)
}

fn k_sumcheck_id() -> SumcheckId {
    SumcheckId::NodeExecution(0)
}

fn m_sumcheck_id() -> SumcheckId {
    SumcheckId::NodeExecution(1)
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
    fn proves_matmul_round_from_poly_and_ra() {
        let matmul = MatMulParams::new(2, 2, 2, "A", "W");
        let round = RoundParams::new(vec![2, 2], "acc", "Y");
        let params = MatMulRoundParams::new(round, matmul);

        let a = vec![3, 5, 7, 11];
        let w = vec![13, 17, 19, 23];
        let acc = matmul_acc_for_test(&a, &w, &params.matmul);
        let y = acc.iter().map(|&v| round_q8_to_i32(v)).collect::<Vec<_>>();

        let y_point = vec![Fr::from(3_u64), Fr::from(5_u64)];
        let y_claim = Claim::new(
            poly_from_i32(&y),
            y_point.clone(),
            eval_matrix(&y, 2, 2, &[y_point[0]], &[y_point[1]]),
        );

        let round_ra = round_ra_from_acc(&acc, params.round.shape.padded_power_of_two().numel());
        let mut prover_transcript = Blake2bTranscript::default();
        let (proof, a_claim, w_claim, ra_claims) = prove_matmul_round(
            y_claim,
            poly_from_i32(&a),
            poly_from_i32(&w),
            round_ra,
            &params,
            &mut prover_transcript,
        )
        .unwrap();

        assert_eq!(
            a_claim.point.len(),
            params.matmul.a_shape().padded_power_of_two().point_len()
        );
        assert_eq!(
            w_claim.point.len(),
            params.matmul.w_shape().padded_power_of_two().point_len()
        );
        assert!(!ra_claims.is_empty());

        let round_ra = round_ra_from_acc(&acc, params.round.shape.padded_power_of_two().numel());
        let y_claim = Claim::new(
            poly_from_i32(&y),
            y_point.clone(),
            eval_matrix(&y, 2, 2, &[y_point[0]], &[y_point[1]]),
        );
        let mut verifier_transcript = Blake2bTranscript::default();
        let (verified_a, verified_w, verified_ra) = verify_matmul_round(
            y_claim,
            &proof,
            poly_from_i32(&w),
            round_ra,
            &params,
            &mut verifier_transcript,
        )
        .unwrap();

        assert_eq!(verified_a.point, a_claim.point);
        assert_eq!(verified_a.value, a_claim.value);
        assert_eq!(verified_w.point, w_claim.point);
        assert_eq!(verified_w.value, w_claim.value);
        assert_eq!(verified_ra.len(), ra_claims.len());
    }

    fn poly_from_i32(values: &[i32]) -> Poly<Fr, ()> {
        Poly::new(MultilinearPolynomial::from(values.to_vec()), None)
    }

    fn matmul_acc_for_test(a: &[i32], w: &[i32], params: &MatMulParams) -> Vec<i64> {
        let mut out = vec![0; params.m * params.n];
        for row in 0..params.m {
            for col in 0..params.n {
                out[row * params.n + col] = (0..params.k)
                    .map(|kk| i64::from(a[row * params.k + kk]) * i64::from(w[kk * params.n + col]))
                    .sum();
            }
        }
        out
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

    fn eval_matrix<F: JoltField>(
        values: &[i32],
        rows: usize,
        cols: usize,
        r_row: &[F],
        r_col: &[F],
    ) -> F {
        let row_eq = EqPolynomial::<F>::evals(r_row);
        let col_eq = EqPolynomial::<F>::evals(r_col);
        let mut out = F::zero();
        for row in 0..rows {
            for col in 0..cols {
                out += row_eq[row] * col_eq[col] * F::from_i32(values[row * cols + col]);
            }
        }
        out
    }
}
