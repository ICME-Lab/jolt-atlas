use common::VirtualPoly;
// Design note for future us:
//
// This is the reverse-claim version of a rounded 2D matmul.  The caller
// already has a claim for rounded `Y(r_m, r_n)` from the consumer of the matmul
// output.  The verifier does not receive `A`; `W` is public for this op.  We
// therefore prove the relation in two stages:
//
// 1. Fold the output axes with the incoming claim point and prove
//      Y(r_m, r_n) * 2^8
//        = sum_k A_{r_m}(k) * W_{r_n}(k) + round_bit * 2^8 - rem.
// 2. Expand `A_{r_m}(r_k)` back into a claim on the original `A` tensor:
//      A_{r_m}(r_k) = sum_m eq(m, r_m) * A(m, r_k).
//
// The SHOUT lookup proving `rem -> round_bit` is handled by
// `prove_matmul_round` below.
// `verify_matmul_round_relation` returns the remaining `A` opening claim.  This
// keeps the Qwen layer prover pure and lets a later commitment/opening layer
// discharge the returned claim.  Shapes are logical shapes; each summed
// dimension is padded to the next power of two inside the polynomial domain.
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
    ops::round::{
        ROUND_FRAC_BITS, ROUND_LUT_LEN, RoundLookupProof, RoundParams, RoundWitness,
        padded_lookup_indices, prove_round_lookup, round_lut_table, verify_round_lookup,
    },
    proof::ProveResult,
};

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatMulRoundRelationClaims<F> {
    pub input: Claim<F>,
    pub round_point: Vec<F>,
    pub remainder_opening: F,
    pub round_bit_opening: F,
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

#[derive(Debug, Clone, Default)]
pub struct MatMulRoundWitness {
    pub input: Vec<i32>,
    pub acc: Vec<i64>,
    pub output: Vec<i32>,
    pub frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
}

#[derive(Debug, Clone)]
pub struct MatMulRoundProof<F: JoltField, T: Transcript> {
    pub matmul: MatMulRoundRelationProof<F, T>,
    pub(crate) round_lookup: RoundLookupProof<F, T>,
}

pub fn prove_matmul_round<F, T>(
    y_round_claim: Claim<F>,
    witness: &MatMulRoundWitness,
    w: &[i32],
    params: &MatMulRoundParams,
    transcript: &mut T,
) -> Result<(MatMulRoundProof<F, T>, Claim<F>, Claim<F>)>
where
    F: JoltField,
    T: Transcript,
{
    let round_witness =
        RoundWitness::from_input_output(witness.acc.clone(), witness.output.clone());
    let matmul_result = prove_matmul_round_relation(
        y_round_claim,
        &witness.input,
        w,
        &round_witness.remainder,
        &round_witness.round_bit,
        &params.matmul,
        transcript,
    )?;
    let mut round_accumulator = ProverOpeningAccumulator::new();
    let round_point = OpeningPoint::<BIG_ENDIAN, F>::new(matmul_result.claims.round_point.clone());
    round_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundLut, SumcheckId::NodeExecution(0)),
        (round_point.clone(), matmul_result.claims.round_bit_opening),
    );
    round_accumulator.openings.insert(
        OpeningId::new(
            VirtualPoly::QwenRoundRemainder,
            SumcheckId::NodeExecution(0),
        ),
        (round_point, matmul_result.claims.remainder_opening),
    );
    let round_lookup = prove_round_lookup(
        matmul_result.claims.round_point.clone(),
        matmul_result.claims.round_bit_opening,
        matmul_result.claims.remainder_opening,
        padded_lookup_indices(&round_witness.remainder, &params.round.shape),
        round_lut_table(),
        &mut round_accumulator,
        transcript,
    )?;
    let input = matmul_result.claims.input;
    let round_ra = Claim {
        tensor: TensorId::new(format!("{}_round_ra", params.round.input_tensor.0)),
        logical_shape: Shape::new(vec![
            params.round.shape.padded_power_of_two().numel(),
            ROUND_LUT_LEN,
        ]),
        domain_shape: Shape::new(vec![
            params.round.shape.padded_power_of_two().numel(),
            ROUND_LUT_LEN,
        ]),
        point: round_lookup.ra_point.clone(),
        value: round_lookup.ra_opening,
    };

    Ok((
        MatMulRoundProof {
            matmul: matmul_result.proof,
            round_lookup,
        },
        input,
        round_ra,
    ))
}

pub fn verify_matmul_round<F, T>(
    y_round_claim: Claim<F>,
    proof: &MatMulRoundProof<F, T>,
    w: &[i32],
    params: &MatMulRoundParams,
    transcript: &mut T,
) -> std::result::Result<(Claim<F>, Claim<F>), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    let matmul_claims = verify_matmul_round_relation(
        y_round_claim,
        &proof.matmul,
        w,
        &params.matmul,
        transcript,
    )?;
    let round_lookup = verify_round_lookup(
        params.round.shape.padded_power_of_two().numel(),
        matmul_claims.round_point,
        matmul_claims.round_bit_opening,
        matmul_claims.remainder_opening,
        proof.round_lookup.ra_opening,
        &proof.round_lookup.committed_openings,
        &proof.round_lookup.read_raf,
        &proof.round_lookup.ra_onehot,
        &mut VerifierOpeningAccumulator::new(),
        transcript,
    )?;
    let round_ra = Claim {
        tensor: TensorId::new(format!("{}_round_ra", params.round.input_tensor.0)),
        logical_shape: Shape::new(vec![
            params.round.shape.padded_power_of_two().numel(),
            ROUND_LUT_LEN,
        ]),
        domain_shape: Shape::new(vec![
            params.round.shape.padded_power_of_two().numel(),
            ROUND_LUT_LEN,
        ]),
        point: round_lookup.ra_point,
        value: proof.round_lookup.ra_opening,
    };

    Ok((matmul_claims.input, round_ra))
}

pub fn prove_matmul_round_relation<F, T>(
    y_round_claim: Claim<F>,
    a: &[i32],
    w: &[i32],
    remainder: &[usize],
    round_bit: &[i32],
    params: &MatMulParams,
    transcript: &mut T,
) -> Result<ProveResult<MatMulRoundRelationClaims<F>, MatMulRoundRelationProof<F, T>>>
where
    F: JoltField,
    T: Transcript,
{
    validate_inputs(&y_round_claim, a, w, params)?;
    validate_round_witness(remainder, round_bit, params)?;
    let (r_m, r_n) = split_y_point(&y_round_claim, params)?;

    let partials = MatMulPartials::new(a, w, params, r_m, r_n);
    let round_shape = params.y_shape();
    let padded_remainder = super::round::padded_lookup_indices(remainder, &round_shape);
    let round_bit_as_usize = round_bit
        .iter()
        .map(|&bit| bit as usize)
        .collect::<Vec<_>>();
    let padded_round_bit = super::round::padded_lookup_indices(&round_bit_as_usize, &round_shape);
    let remainder_opening_at_y = eval_flat_usize(&padded_remainder, &y_round_claim.point);
    let round_bit_opening_at_y = eval_flat_usize(&padded_round_bit, &y_round_claim.point);
    let k_params = KSumcheckParams::new(log2_ceil(params.k), y_round_claim.value * scale_q8::<F>());
    let mut k_prover = KRoundSumcheckProver::new(
        k_params,
        partials.a_r,
        partials.w_r,
        remainder_opening_at_y,
        round_bit_opening_at_y,
        y_round_claim.point.clone(),
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

    let a_by_m = fix_a_k(a, params, &r_k);
    let m_params = MSumcheckParams::new(log2_ceil(params.m), a_r_opening, r_m.to_vec());
    let mut m_prover = MSumcheckProver::new(m_params, a_by_m);
    let mut m_accumulator = ProverOpeningAccumulator::new();
    let (m_sumcheck, m_challenges) = Sumcheck::prove(&mut m_prover, &mut m_accumulator, transcript);
    let a_opening = prover_opening(
        &m_accumulator,
        OpeningId::new(VirtualPoly::QwenMatMulA, m_sumcheck_id()),
    )?;

    let m_challenges = m_challenges.into_opening();
    let mut point = normalize_sumcheck_point::<F>(&m_challenges);
    point.extend(r_k);
    let a_claim = Claim {
        tensor: params.a_tensor.clone(),
        logical_shape: params.a_shape(),
        domain_shape: params.a_shape().padded_power_of_two(),
        point,
        value: a_opening,
    };

    Ok(ProveResult::new(
        MatMulRoundRelationClaims {
            input: a_claim,
            round_point: y_round_claim.point,
            remainder_opening,
            round_bit_opening,
        },
        MatMulRoundRelationProof {
            k_sumcheck,
            m_sumcheck,
            a_r_opening,
            w_r_opening,
            a_opening,
            remainder_opening,
            round_bit_opening,
        },
    ))
}

pub fn verify_matmul_round_relation<F, T>(
    y_round_claim: Claim<F>,
    proof: &MatMulRoundRelationProof<F, T>,
    w: &[i32],
    params: &MatMulParams,
    transcript: &mut T,
) -> std::result::Result<MatMulRoundRelationClaims<F>, ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    verify_inputs(&y_round_claim, w, params)?;
    let (r_m, r_n) = split_y_point_verify(&y_round_claim, params)?;

    let k_params = KSumcheckParams::new(log2_ceil(params.k), y_round_claim.value * scale_q8::<F>());
    let k_verifier = KRoundSumcheckVerifier {
        params: k_params,
        y_point: y_round_claim.point.clone(),
    };
    let mut k_accumulator = VerifierOpeningAccumulator::new();
    k_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenMatMulPartialA, k_sumcheck_id()),
        (OpeningPoint::<BIG_ENDIAN, F>::default(), proof.a_r_opening),
    );
    k_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenMatMulPartialW, k_sumcheck_id()),
        (OpeningPoint::<BIG_ENDIAN, F>::default(), proof.w_r_opening),
    );
    k_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundRemainder, k_sumcheck_id()),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.remainder_opening,
        ),
    );
    k_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundLut, k_sumcheck_id()),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.round_bit_opening,
        ),
    );
    let r_k_challenges = Sumcheck::verify(
        &proof.k_sumcheck,
        &k_verifier,
        &mut k_accumulator,
        transcript,
    )?
    .into_opening();
    let r_k = normalize_sumcheck_point::<F>(&r_k_challenges);

    let expected_w = eval_w(w, params, &r_k, r_n);
    if proof.w_r_opening != expected_w {
        return Err(ProofVerifyError::SumcheckVerificationError);
    }

    let m_params = MSumcheckParams::new(log2_ceil(params.m), proof.a_r_opening, r_m.to_vec());
    let m_verifier = MSumcheckVerifier { params: m_params };
    let mut m_accumulator = VerifierOpeningAccumulator::new();
    m_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenMatMulA, m_sumcheck_id()),
        (OpeningPoint::<BIG_ENDIAN, F>::default(), proof.a_opening),
    );
    let r_m_prime_challenges = Sumcheck::verify(
        &proof.m_sumcheck,
        &m_verifier,
        &mut m_accumulator,
        transcript,
    )?
    .into_opening();

    let mut point = normalize_sumcheck_point::<F>(&r_m_prime_challenges);
    point.extend(r_k);
    Ok(MatMulRoundRelationClaims {
        input: Claim {
            tensor: params.a_tensor.clone(),
            logical_shape: params.a_shape(),
            domain_shape: params.a_shape().padded_power_of_two(),
            point,
            value: proof.a_opening,
        },
        round_point: y_round_claim.point,
        remainder_opening: proof.remainder_opening,
        round_bit_opening: proof.round_bit_opening,
    })
}

struct MatMulPartials<F: JoltField> {
    a_r: Vec<F>,
    w_r: Vec<F>,
}

impl<F: JoltField> MatMulPartials<F> {
    fn new(a: &[i32], w: &[i32], params: &MatMulParams, r_m: &[F], r_n: &[F]) -> Self {
        let k_pad = params.k.next_power_of_two();
        let row_eq = EqPolynomial::<F>::evals(r_m);
        let col_eq = EqPolynomial::<F>::evals(r_n);

        let mut a_r = vec![F::zero(); k_pad];
        for (kk, out) in a_r.iter_mut().enumerate().take(params.k) {
            *out = (0..params.m)
                .map(|row| row_eq[row] * F::from_i32(a[row * params.k + kk]))
                .sum();
        }

        let mut w_r = vec![F::zero(); k_pad];
        for (kk, out) in w_r.iter_mut().enumerate().take(params.k) {
            *out = (0..params.n)
                .map(|col| col_eq[col] * F::from_i32(w[kk * params.n + col]))
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

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.input_claim
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

struct KRoundSumcheckProver<F: JoltField> {
    a_r: MultilinearPolynomial<F>,
    w_r: MultilinearPolynomial<F>,
    eq_zero_k: MultilinearPolynomial<F>,
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
        let zero = vec![F::zero(); params.k_vars];
        Self {
            a_r: MultilinearPolynomial::from(a_r),
            w_r: MultilinearPolynomial::from(w_r),
            eq_zero_k: MultilinearPolynomial::from(EqPolynomial::<F>::evals(&zero)),
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
        self.eq_zero_k.bind_parallel(r_j, BindingOrder::LowToHigh);
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

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
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

fn validate_inputs<F: JoltField>(
    y_claim: &Claim<F>,
    a: &[i32],
    w: &[i32],
    params: &MatMulParams,
) -> Result<()> {
    if params.m == 0 {
        return Err(ProverError::InvalidMatrixDimension { name: "m" });
    }
    if params.k == 0 {
        return Err(ProverError::InvalidMatrixDimension { name: "k" });
    }
    if params.n == 0 {
        return Err(ProverError::InvalidMatrixDimension { name: "n" });
    }
    if a.len() != params.m * params.k {
        return Err(ProverError::TensorLenMismatch {
            name: "A",
            shape: params.a_shape().0,
            expected: params.m * params.k,
            actual: a.len(),
        });
    }
    if w.len() != params.k * params.n {
        return Err(ProverError::TensorLenMismatch {
            name: "W",
            shape: params.w_shape().0,
            expected: params.k * params.n,
            actual: w.len(),
        });
    }
    if y_claim.logical_shape != params.y_shape() {
        return Err(ProverError::ShapeMismatch {
            name: "Y claim",
            expected: params.y_shape().0,
            actual: y_claim.logical_shape.0.clone(),
        });
    }
    let expected_domain = params.y_shape().padded_power_of_two();
    if y_claim.domain_shape != expected_domain {
        return Err(ProverError::ShapeMismatch {
            name: "Y claim domain",
            expected: expected_domain.0,
            actual: y_claim.domain_shape.0.clone(),
        });
    }
    let expected_point_len = log2_ceil(params.m) + log2_ceil(params.n);
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
    w: &[i32],
    params: &MatMulParams,
) -> std::result::Result<(), ProofVerifyError> {
    if params.m == 0 || params.k == 0 || params.n == 0 {
        return Err(ProofVerifyError::InvalidInputLength(1, 0));
    }
    if w.len() != params.k * params.n {
        return Err(ProofVerifyError::InvalidInputLength(
            params.k * params.n,
            w.len(),
        ));
    }
    if y_claim.logical_shape != params.y_shape() {
        return Err(ProofVerifyError::InvalidInputLength(
            params.y_shape().numel(),
            y_claim.logical_shape.numel(),
        ));
    }
    if y_claim.domain_shape != params.y_shape().padded_power_of_two() {
        return Err(ProofVerifyError::InvalidInputLength(
            params.y_shape().padded_power_of_two().numel(),
            y_claim.domain_shape.numel(),
        ));
    }
    let expected_point_len = log2_ceil(params.m) + log2_ceil(params.n);
    if y_claim.point.len() != expected_point_len {
        return Err(ProofVerifyError::InvalidInputLength(
            expected_point_len,
            y_claim.point.len(),
        ));
    }
    Ok(())
}

fn validate_round_witness(
    remainder: &[usize],
    round_bit: &[i32],
    params: &MatMulParams,
) -> Result<()> {
    let expected = params.m * params.n;
    if remainder.len() != expected {
        return Err(ProverError::TensorLenMismatch {
            name: "matmul round remainder",
            shape: params.y_shape().0,
            expected,
            actual: remainder.len(),
        });
    }
    if round_bit.len() != expected {
        return Err(ProverError::TensorLenMismatch {
            name: "matmul round bit",
            shape: params.y_shape().0,
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

fn split_y_point<'a, F: JoltField>(
    y_claim: &'a Claim<F>,
    params: &MatMulParams,
) -> Result<(&'a [F], &'a [F])> {
    let row_vars = log2_ceil(params.m);
    let col_vars = log2_ceil(params.n);
    if y_claim.point.len() != row_vars + col_vars {
        return Err(ProverError::ShapeMismatch {
            name: "Y claim point",
            expected: vec![row_vars + col_vars],
            actual: vec![y_claim.point.len()],
        });
    }
    Ok(y_claim.point.split_at(row_vars))
}

fn split_y_point_verify<'a, F: JoltField>(
    y_claim: &'a Claim<F>,
    params: &MatMulParams,
) -> std::result::Result<(&'a [F], &'a [F]), ProofVerifyError> {
    let row_vars = log2_ceil(params.m);
    let col_vars = log2_ceil(params.n);
    if y_claim.point.len() != row_vars + col_vars {
        return Err(ProofVerifyError::InvalidInputLength(
            row_vars + col_vars,
            y_claim.point.len(),
        ));
    }
    Ok(y_claim.point.split_at(row_vars))
}

fn fix_a_k<F: JoltField>(a: &[i32], params: &MatMulParams, r_k: &[F]) -> Vec<F> {
    let m_pad = params.m.next_power_of_two();
    let k_eq = EqPolynomial::<F>::evals(r_k);
    let mut out = vec![F::zero(); m_pad];
    for row in 0..params.m {
        out[row] = (0..params.k)
            .map(|kk| k_eq[kk] * F::from_i32(a[row * params.k + kk]))
            .sum();
    }
    out
}

fn eval_w<F: JoltField>(w: &[i32], params: &MatMulParams, r_k: &[F], r_n: &[F]) -> F {
    let k_eq = EqPolynomial::<F>::evals(r_k);
    let n_eq = EqPolynomial::<F>::evals(r_n);
    let mut out = F::zero();
    for kk in 0..params.k {
        for col in 0..params.n {
            out += k_eq[kk] * n_eq[col] * F::from_i32(w[kk * params.n + col]);
        }
    }
    out
}

fn eval_flat_usize<F: JoltField>(values: &[usize], point: &[F]) -> F {
    let eq = EqPolynomial::<F>::evals(point);
    values
        .iter()
        .enumerate()
        .fold(F::zero(), |acc, (flat, &value)| {
            acc + eq[flat] * F::from_u64(value as u64)
        })
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

fn log2_ceil(value: usize) -> usize {
    value.next_power_of_two().trailing_zeros() as usize
}

fn normalize_sumcheck_point<F: JoltField>(challenges: &[F]) -> Vec<F> {
    OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec())
        .match_endianness::<BIG_ENDIAN>()
        .r
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
    use joltworks::transcripts::Blake2bTranscript;

    use super::*;

    #[test]
    fn proves_and_verifies_matmul_round_non_power_shapes() {
        use crate::ops::round::{ROUND_FRAC_BITS, RoundParams};

        let params = MatMulParams::new(3, 5, 5, "A", "W");
        let round_params = RoundParams::with_frac_bit_tensors(
            vec![params.m, params.n],
            "acc",
            "Y",
            std::array::from_fn(|idx| format!("acc_frac_bit_{idx}")),
        );
        let params = MatMulRoundParams::new(round_params, params);
        let a = vec![3, -5, 2, 7, -9, 4, 1, -3, 6, 8, -2, 5, -7, 9, 11];
        let w = vec![
            2, -1, 4, -3, 5, 7, 6, -8, 1, -4, 3, 9, 10, -2, -6, 8, -7, 12, -11, 13, 14, -15, 16,
            -17, 18,
        ];
        let mut acc = vec![0i64; 15];
        for row in 0..3 {
            for col in 0..5 {
                acc[row * 5 + col] = (0..5)
                    .map(|kk| a[row * 5 + kk] as i64 * w[kk * 5 + col] as i64)
                    .sum::<i64>();
            }
        }
        let y = acc
            .iter()
            .map(|&value| ((value + (1 << (ROUND_FRAC_BITS - 1))) >> ROUND_FRAC_BITS) as i32)
            .collect::<Vec<_>>();
        let r_m = vec![Fr::from(3u64), Fr::from(11u64)];
        let r_n = vec![Fr::from(5u64), Fr::from(13u64), Fr::from(17u64)];
        let y_claim = Claim {
            tensor: TensorId::new("Y"),
            logical_shape: params.round.shape.clone(),
            domain_shape: params.round.shape.padded_power_of_two(),
            point: [r_m.as_slice(), r_n.as_slice()].concat(),
            value: eval_matrix(&y, 3, 5, &r_m, &r_n),
        };
        let witness = MatMulRoundWitness {
            input: a,
            acc,
            output: y,
            frac_bits: std::array::from_fn(|_| Vec::new()),
        };

        let mut prover_transcript = Blake2bTranscript::default();
        let (proof, input, _) = prove_matmul_round::<Fr, _>(
            y_claim.clone(),
            &witness,
            &w,
            &params,
            &mut prover_transcript,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        let (verified_input, _) =
            verify_matmul_round::<Fr, _>(y_claim, &proof, &w, &params, &mut verifier_transcript)
                .unwrap();

        assert_eq!(verified_input, input);
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
