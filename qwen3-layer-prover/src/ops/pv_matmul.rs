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

/// Rounded attention value contraction:
///
///   C[qpos, h, d] = round(sum_kpos P[h, qpos, kpos] * V[kpos, h/2, d] / 2^8)
///
/// The `h/2` mapping is Qwen3's Grouped-Query Attention with group_size=2.
/// This op keeps `h` and `kpos` inside one sumcheck and uses `eq(h, r_h)` as
/// a known selector; evaluating `V` at `drop_lsb(r_h)` directly is not sound
/// for the multilinear extension.

const QWEN3_GQA_GROUP_SIZE: usize = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PvMatmulParams {
    pub seq: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub p_tensor: TensorId,
    pub v_tensor: TensorId,
}

impl PvMatmulParams {
    pub fn new(
        seq: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        p_tensor: impl Into<String>,
        v_tensor: impl Into<String>,
    ) -> Self {
        Self {
            seq,
            q_heads,
            kv_heads,
            head_dim,
            p_tensor: TensorId::new(p_tensor),
            v_tensor: TensorId::new(v_tensor),
        }
    }

    pub fn p_shape(&self) -> Shape {
        Shape::new(vec![self.q_heads, self.seq, self.seq])
    }

    pub fn v_shape(&self) -> Shape {
        Shape::new(vec![self.seq, self.kv_heads, self.head_dim])
    }

    pub fn context_shape(&self) -> Shape {
        Shape::new(vec![self.seq, self.q_heads, self.head_dim])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PvMatmulRoundParams {
    pub round: RoundParams,
    pub pv: PvMatmulParams,
}

impl PvMatmulRoundParams {
    pub fn new(round: RoundParams, pv: PvMatmulParams) -> Self {
        Self { round, pv }
    }
}

#[derive(Debug, Clone)]
pub struct PvMatmulProof<F: JoltField, T: Transcript> {
    pub pv: PvMatmulRoundRelationProof<F, T>,
    pub(crate) round_lookup: RoundLookupProof<F, T>,
}

impl<F: JoltField, T: Transcript> PvMatmulProof<F, T> {
    pub(crate) fn sumcheck_round_count(&self) -> usize {
        self.pv.sumcheck.compressed_polys.len() + self.round_lookup.sumcheck_round_count()
    }

    pub(crate) fn sumcheck_count(&self) -> usize {
        1 + self.round_lookup.sumcheck_count()
    }
}

#[derive(Debug, Clone)]
pub struct PvMatmulRoundRelationProof<F: JoltField, T: Transcript> {
    pub sumcheck: SumcheckInstanceProof<F, T>,
    pub p_opening: F,
    pub v_opening: F,
    pub remainder_opening: F,
    pub round_bit_opening: F,
}

pub fn prove_pv_matmul_round<F, T, C>(
    context_claim: Claim<F, C>,
    p_poly: Poly<F, C>,
    v_poly: Poly<F, C>,
    round_ra: Vec<Poly<F, C>>,
    params: &PvMatmulRoundParams,
    transcript: &mut T,
) -> Result<(
    PvMatmulProof<F, T>,
    Claim<F, C>,
    Claim<F, C>,
    Vec<Claim<F, C>>,
)>
where
    F: JoltField,
    T: Transcript,
    C: Clone,
{
    validate_claim_shape(
        &context_claim,
        &params.pv.context_shape(),
        "PV context claim",
    )?;
    validate_poly_shape(&p_poly, &params.pv.p_shape(), "PV P")?;
    validate_poly_shape(&v_poly, &params.pv.v_shape(), "PV V")?;
    validate_pv_params(&params.pv)?;

    let (remainder_opening, round_bit_opening) =
        round_lookup_openings_from_ra(&round_ra, &context_claim.point, &params.round.shape)?;
    let (relation, p_point, p_value, v_point, v_value, round_point) =
        prove_pv_matmul_round_relation(
            context_claim.point.clone(),
            context_claim.value,
            &p_poly,
            &v_poly,
            remainder_opening,
            round_bit_opening,
            &params.pv,
            transcript,
        )?;

    let mut round_accumulator = ProverOpeningAccumulator::new();
    let round_opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(round_point);
    round_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundLut, pv_sumcheck_id()),
        (round_opening_point.clone(), round_bit_opening),
    );
    round_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundRemainder, pv_sumcheck_id()),
        (round_opening_point, remainder_opening),
    );
    let (round_lookup, round_ra_claims) = prove_round_lookup_from_ra(
        params.round.lookup_site,
        context_claim.point,
        round_ra,
        &params.round.shape,
        round_bit_opening,
        remainder_opening,
        &mut round_accumulator,
        transcript,
    )?;

    Ok((
        PvMatmulProof {
            pv: relation,
            round_lookup,
        },
        Claim::new(p_poly, p_point, p_value),
        Claim::new(v_poly, v_point, v_value),
        round_ra_claims,
    ))
}

pub fn verify_pv_matmul_round<F, T, C>(
    context_claim: Claim<F, C>,
    proof: &PvMatmulProof<F, T>,
    round_ra: Vec<Poly<F, C>>,
    params: &PvMatmulRoundParams,
    transcript: &mut T,
) -> std::result::Result<(Claim<F, C>, Claim<F, C>, Vec<Claim<F, C>>), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
    C: Clone,
{
    verify_claim_shape(&context_claim, &params.pv.context_shape())?;
    verify_pv_params(&params.pv)?;
    let remainder_opening = proof.round_lookup.remainder_opening;
    let round_bit_opening = proof.round_lookup.round_bit_opening;

    let (p_point, v_point, round_point) = verify_pv_matmul_round_relation(
        context_claim.point.clone(),
        context_claim.value,
        proof,
        remainder_opening,
        round_bit_opening,
        &params.pv,
        transcript,
    )?;

    let mut round_accumulator = VerifierOpeningAccumulator::new();
    let round_opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(round_point);
    round_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundLut, pv_sumcheck_id()),
        (round_opening_point.clone(), round_bit_opening),
    );
    round_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundRemainder, pv_sumcheck_id()),
        (round_opening_point, remainder_opening),
    );
    let (_round_lookup, round_ra_claims) = verify_round_lookup_from_ra(
        params.round.lookup_site,
        context_claim.point,
        round_ra,
        &params.round.shape,
        round_bit_opening,
        remainder_opening,
        &proof.round_lookup,
        &mut round_accumulator,
        transcript,
    )?;

    Ok((
        Claim::new(
            dummy_poly(&params.pv.p_shape()),
            p_point,
            proof.pv.p_opening,
        ),
        Claim::new(
            dummy_poly(&params.pv.v_shape()),
            v_point,
            proof.pv.v_opening,
        ),
        round_ra_claims,
    ))
}

fn prove_pv_matmul_round_relation<F, T, C>(
    context_point: Vec<F>,
    context_value: F,
    p_poly: &Poly<F, C>,
    v_poly: &Poly<F, C>,
    remainder_opening: F,
    round_bit_opening: F,
    params: &PvMatmulParams,
    transcript: &mut T,
) -> Result<(
    PvMatmulRoundRelationProof<F, T>,
    Vec<F>,
    F,
    Vec<F>,
    F,
    Vec<F>,
)>
where
    F: JoltField,
    T: Transcript,
{
    let (r_q, r_h, r_d) = split3(&context_point, params.seq, params.q_heads);
    let left = partial_p(p_poly, params, r_q);
    let right = partial_v(v_poly, params, r_d);
    let selector = expanded_head_eq(r_h, params);
    let (relation, r_h_kpos) = prove_pv_round_product(
        context_value * scale_q8::<F>(),
        left,
        right,
        selector,
        remainder_opening,
        round_bit_opening,
        transcript,
    )?;
    let (r_head, r_kpos) = r_h_kpos.split_at(log2_ceil(params.q_heads));
    let kv_h = drop_gqa_lsb_for_dims(r_head, params)?;

    Ok((
        PvMatmulRoundRelationProof {
            p_opening: relation.p_opening,
            v_opening: relation.v_opening,
            remainder_opening: relation.remainder_opening,
            round_bit_opening: relation.round_bit_opening,
            sumcheck: relation.sumcheck,
        },
        [r_head, r_q, r_kpos].concat(),
        relation.p_opening,
        [r_kpos, kv_h, r_d].concat(),
        relation.v_opening,
        context_point,
    ))
}

fn verify_pv_matmul_round_relation<F, T>(
    context_point: Vec<F>,
    context_value: F,
    proof: &PvMatmulProof<F, T>,
    remainder_opening: F,
    round_bit_opening: F,
    params: &PvMatmulParams,
    transcript: &mut T,
) -> std::result::Result<(Vec<F>, Vec<F>, Vec<F>), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    let (r_q, r_h, r_d) = split3(&context_point, params.seq, params.q_heads);
    let r_h_kpos = verify_pv_round_product(
        context_value * scale_q8::<F>(),
        proof,
        r_h,
        remainder_opening,
        round_bit_opening,
        params,
        transcript,
    )?;
    let (r_head, r_kpos) = r_h_kpos.split_at(log2_ceil(params.q_heads));
    let kv_h = drop_gqa_lsb_for_dims_verify(r_head, params)?;
    Ok((
        [r_head, r_q, r_kpos].concat(),
        [r_kpos, kv_h, r_d].concat(),
        context_point,
    ))
}

fn partial_p<F: JoltField, C>(p: &Poly<F, C>, params: &PvMatmulParams, r_q: &[F]) -> Vec<F> {
    let head_pad = params.q_heads.next_power_of_two();
    let seq_pad = params.seq.next_power_of_two();
    let q_eq = EqPolynomial::<F>::evals(r_q);
    let mut out = vec![F::zero(); head_pad * seq_pad];
    for head in 0..params.q_heads {
        for kpos in 0..params.seq {
            out[head * seq_pad + kpos] = (0..params.seq)
                .map(|qpos| q_eq[qpos] * coeff3(p, &params.p_shape(), head, qpos, kpos))
                .sum();
        }
    }
    out
}

fn partial_v<F: JoltField, C>(v: &Poly<F, C>, params: &PvMatmulParams, r_d: &[F]) -> Vec<F> {
    let head_pad = params.q_heads.next_power_of_two();
    let seq_pad = params.seq.next_power_of_two();
    let dim_eq = EqPolynomial::<F>::evals(r_d);
    let mut out = vec![F::zero(); head_pad * seq_pad];
    for head in 0..params.q_heads {
        let kv_head = head / QWEN3_GQA_GROUP_SIZE;
        for kpos in 0..params.seq {
            out[head * seq_pad + kpos] = (0..params.head_dim)
                .map(|d| dim_eq[d] * coeff3(v, &params.v_shape(), kpos, kv_head, d))
                .sum();
        }
    }
    out
}

fn expanded_head_eq<F: JoltField>(r_h: &[F], params: &PvMatmulParams) -> Vec<F> {
    let head_pad = params.q_heads.next_power_of_two();
    let seq_pad = params.seq.next_power_of_two();
    let head_eq = EqPolynomial::<F>::evals(r_h);
    let mut out = vec![F::zero(); head_pad * seq_pad];
    for head in 0..params.q_heads {
        for kpos in 0..seq_pad {
            out[head * seq_pad + kpos] = head_eq[head];
        }
    }
    out
}

fn prove_pv_round_product<F, T>(
    input_claim: F,
    left: Vec<F>,
    right: Vec<F>,
    selector: Vec<F>,
    remainder: F,
    round_bit: F,
    transcript: &mut T,
) -> Result<(PvMatmulRoundRelationProof<F, T>, Vec<F>)>
where
    F: JoltField,
    T: Transcript,
{
    let params = PvProductParams::new(
        left.len().trailing_zeros() as usize,
        input_claim,
        Vec::new(),
    );
    let mut prover = PvRoundProductProver::new(params, left, right, selector, remainder, round_bit);
    let mut accumulator = ProverOpeningAccumulator::new();
    let (sumcheck, challenges) = Sumcheck::prove(&mut prover, &mut accumulator, transcript);
    let p_opening = opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenAttentionLeft, pv_sumcheck_id()),
    )?;
    let v_opening = opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenAttentionRight, pv_sumcheck_id()),
    )?;
    let remainder_opening = opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenRoundRemainder, pv_sumcheck_id()),
    )?;
    let round_bit_opening = opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenRoundLut, pv_sumcheck_id()),
    )?;
    Ok((
        PvMatmulRoundRelationProof {
            sumcheck,
            p_opening,
            v_opening,
            remainder_opening,
            round_bit_opening,
        },
        normalize_sumcheck_point::<F>(&challenges.into_opening()),
    ))
}

fn verify_pv_round_product<F, T>(
    input_claim: F,
    proof: &PvMatmulProof<F, T>,
    output_head_point: &[F],
    remainder_opening: F,
    round_bit_opening: F,
    params: &PvMatmulParams,
    transcript: &mut T,
) -> std::result::Result<Vec<F>, ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    let num_rounds = log2_ceil(params.q_heads) + log2_ceil(params.seq);
    let verifier = PvRoundProductVerifier {
        params: PvProductParams::new(num_rounds, input_claim, output_head_point.to_vec()),
    };
    let mut accumulator = VerifierOpeningAccumulator::new();
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenAttentionLeft, pv_sumcheck_id()),
        (OpeningPoint::<BIG_ENDIAN, F>::default(), proof.pv.p_opening),
    );
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenAttentionRight, pv_sumcheck_id()),
        (OpeningPoint::<BIG_ENDIAN, F>::default(), proof.pv.v_opening),
    );
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundRemainder, pv_sumcheck_id()),
        (OpeningPoint::<BIG_ENDIAN, F>::default(), remainder_opening),
    );
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundLut, pv_sumcheck_id()),
        (OpeningPoint::<BIG_ENDIAN, F>::default(), round_bit_opening),
    );
    let challenges = Sumcheck::verify(&proof.pv.sumcheck, &verifier, &mut accumulator, transcript)?;
    Ok(normalize_sumcheck_point::<F>(&challenges.into_opening()))
}

struct PvProductParams<F: JoltField> {
    num_rounds: usize,
    input_claim: F,
    output_head_point: Vec<F>,
}

impl<F: JoltField> PvProductParams<F> {
    fn new(num_rounds: usize, input_claim: F, output_head_point: Vec<F>) -> Self {
        Self {
            num_rounds,
            input_claim,
            output_head_point,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for PvProductParams<F> {
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

struct PvRoundProductProver<F: JoltField> {
    left: MultilinearPolynomial<F>,
    right: MultilinearPolynomial<F>,
    selector: MultilinearPolynomial<F>,
    eq_zero: MultilinearPolynomial<F>,
    remainder: F,
    round_bit: F,
    params: PvProductParams<F>,
}

impl<F: JoltField> PvRoundProductProver<F> {
    fn new(
        params: PvProductParams<F>,
        left: Vec<F>,
        right: Vec<F>,
        selector: Vec<F>,
        remainder: F,
        round_bit: F,
    ) -> Self {
        let zero = vec![F::zero(); params.num_rounds];
        Self {
            left: MultilinearPolynomial::from(left),
            right: MultilinearPolynomial::from(right),
            selector: MultilinearPolynomial::from(selector),
            eq_zero: MultilinearPolynomial::from(EqPolynomial::<F>::evals(&zero)),
            remainder,
            round_bit,
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for PvRoundProductProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let mut evals = [F::zero(); 3];
        let round_term = self.round_bit * scale_q8::<F>() - self.remainder;
        for g in 0..self.left.len() / 2 {
            let l = [
                self.left.get_bound_coeff(2 * g),
                self.left.get_bound_coeff(2 * g + 1),
            ];
            let r = [
                self.right.get_bound_coeff(2 * g),
                self.right.get_bound_coeff(2 * g + 1),
            ];
            let s = [
                self.selector.get_bound_coeff(2 * g),
                self.selector.get_bound_coeff(2 * g + 1),
            ];
            let z = [
                self.eq_zero.get_bound_coeff(2 * g),
                self.eq_zero.get_bound_coeff(2 * g + 1),
            ];
            for (idx, t) in [F::zero(), F::from_u64(2), F::from_u64(3)]
                .into_iter()
                .enumerate()
            {
                evals[idx] += lerp(l[0], l[1], t) * lerp(r[0], r[1], t) * lerp(s[0], s[1], t)
                    + lerp(z[0], z[1], t) * round_term;
            }
        }
        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.left.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.right.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.selector.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.eq_zero.bind_parallel(r_j, BindingOrder::LowToHigh);
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
            OpeningId::new(VirtualPoly::QwenAttentionLeft, pv_sumcheck_id()),
            point.clone(),
            self.left.final_claim(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenAttentionRight, pv_sumcheck_id()),
            point.clone(),
            self.right.final_claim(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRoundRemainder, pv_sumcheck_id()),
            point.clone(),
            self.remainder,
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRoundLut, pv_sumcheck_id()),
            point,
            self.round_bit,
        );
    }
}

struct PvRoundProductVerifier<F: JoltField> {
    params: PvProductParams<F>,
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for PvRoundProductVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let left = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenAttentionLeft,
                pv_sumcheck_id(),
            ))
            .1;
        let right = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenAttentionRight,
                pv_sumcheck_id(),
            ))
            .1;
        let remainder = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenRoundRemainder,
                pv_sumcheck_id(),
            ))
            .1;
        let round_bit = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenRoundLut,
                pv_sumcheck_id(),
            ))
            .1;
        let point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening())
            .r;
        let head_vars = self.params.output_head_point.len();
        let selector = EqPolynomial::<F>::mle(&self.params.output_head_point, &point[..head_vars]);
        let eq_zero = EqPolynomial::<F>::mle(&vec![F::zero(); self.params.num_rounds], &point);
        left * right * selector + eq_zero * (round_bit * scale_q8::<F>() - remainder)
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
            OpeningId::new(VirtualPoly::QwenAttentionLeft, pv_sumcheck_id()),
            point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenAttentionRight, pv_sumcheck_id()),
            point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRoundRemainder, pv_sumcheck_id()),
            point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRoundLut, pv_sumcheck_id()),
            point,
        );
    }
}

fn validate_pv_params(params: &PvMatmulParams) -> Result<()> {
    if params.seq == 0 || params.q_heads == 0 || params.kv_heads == 0 || params.head_dim == 0 {
        return Err(ProverError::InvalidTensorShape(vec![
            params.seq,
            params.q_heads,
            params.kv_heads,
            params.head_dim,
        ]));
    }
    if params.q_heads / params.kv_heads != QWEN3_GQA_GROUP_SIZE {
        return Err(ProverError::InvalidGqa {
            q_heads: params.q_heads,
            kv_heads: params.kv_heads,
        });
    }
    Ok(())
}

fn verify_pv_params(params: &PvMatmulParams) -> std::result::Result<(), ProofVerifyError> {
    if params.q_heads / params.kv_heads != QWEN3_GQA_GROUP_SIZE {
        return Err(ProofVerifyError::InvalidInputLength(
            QWEN3_GQA_GROUP_SIZE,
            params.q_heads / params.kv_heads,
        ));
    }
    Ok(())
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

fn split3<F>(point: &[F], dim0: usize, dim1: usize) -> (&[F], &[F], &[F]) {
    let v0 = log2_ceil(dim0);
    let v1 = log2_ceil(dim1);
    (&point[..v0], &point[v0..v0 + v1], &point[v0 + v1..])
}

fn drop_gqa_lsb_for_dims<'a, F>(r_h: &'a [F], params: &PvMatmulParams) -> Result<&'a [F]> {
    let q_vars = log2_ceil(params.q_heads);
    let kv_vars = log2_ceil(params.kv_heads);
    if r_h.len() != q_vars || kv_vars + 1 != q_vars {
        return Err(ProverError::InvalidGqa {
            q_heads: params.q_heads,
            kv_heads: params.kv_heads,
        });
    }
    Ok(&r_h[..kv_vars])
}

fn drop_gqa_lsb_for_dims_verify<'a, F>(
    r_h: &'a [F],
    params: &PvMatmulParams,
) -> std::result::Result<&'a [F], ProofVerifyError> {
    let q_vars = log2_ceil(params.q_heads);
    let kv_vars = log2_ceil(params.kv_heads);
    if r_h.len() != q_vars || kv_vars + 1 != q_vars {
        return Err(ProofVerifyError::InvalidInputLength(q_vars, r_h.len()));
    }
    Ok(&r_h[..kv_vars])
}

fn coeff3<F: JoltField, C>(poly: &Poly<F, C>, shape: &Shape, i0: usize, i1: usize, i2: usize) -> F {
    let dims = shape.dims();
    debug_assert_eq!(dims.len(), 3);
    let p1 = dims[1].next_power_of_two();
    let p2 = dims[2].next_power_of_two();
    poly.data.get_bound_coeff((i0 * p1 + i1) * p2 + i2)
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

fn lerp<F: JoltField>(v0: F, v1: F, t: F) -> F {
    v0 + t * (v1 - v0)
}

fn scale_q8<F: JoltField>() -> F {
    F::from_u64(256)
}

fn log2_ceil(value: usize) -> usize {
    value.next_power_of_two().trailing_zeros() as usize
}

fn pv_sumcheck_id() -> SumcheckId {
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
    fn proves_and_verifies_pv_matmul_round_from_poly_and_ra() {
        let pv = PvMatmulParams::new(2, 2, 1, 2, "P", "V");
        let params = PvMatmulRoundParams::new(RoundParams::new(vec![2, 2, 2], "acc", "C"), pv);
        let p = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let v = vec![2, 1, 4, 3];
        let acc = pv_context(&p, &v, &params.pv);
        let y = acc
            .iter()
            .map(|&value| round_q8_to_i32(value))
            .collect::<Vec<_>>();
        let point = vec![Fr::from(3_u64), Fr::from(5_u64), Fr::from(7_u64)];
        let claim = Claim::new(poly_from_i32(&y), point.clone(), eval_flat(&y, &point));
        let round_ra = round_ra_from_acc(
            &acc,
            params.pv.context_shape().padded_power_of_two().numel(),
        );

        let mut prover_transcript = Blake2bTranscript::default();
        let (proof, p_claim, v_claim, ra_claims) = prove_pv_matmul_round(
            claim,
            poly_from_i32(&p),
            poly_from_i32(&v),
            round_ra,
            &params,
            &mut prover_transcript,
        )
        .unwrap();
        assert!(!ra_claims.is_empty());

        let claim = Claim::new(poly_from_i32(&y), point.clone(), eval_flat(&y, &point));
        let round_ra = round_ra_from_acc(
            &acc,
            params.pv.context_shape().padded_power_of_two().numel(),
        );
        let mut verifier_transcript = Blake2bTranscript::default();
        let (verified_p, verified_v, verified_ra) =
            verify_pv_matmul_round(claim, &proof, round_ra, &params, &mut verifier_transcript)
                .unwrap();

        assert_eq!(verified_p.point, p_claim.point);
        assert_eq!(verified_p.value, p_claim.value);
        assert_eq!(verified_v.point, v_claim.point);
        assert_eq!(verified_v.value, v_claim.value);
        assert_eq!(verified_ra.len(), ra_claims.len());
    }

    fn pv_context(p: &[i32], v: &[i32], params: &PvMatmulParams) -> Vec<i64> {
        let mut out = vec![0_i64; params.seq * params.q_heads * params.head_dim];
        for qpos in 0..params.seq {
            for head in 0..params.q_heads {
                let kv_head = head / QWEN3_GQA_GROUP_SIZE;
                for d in 0..params.head_dim {
                    let mut sum = 0_i64;
                    for kpos in 0..params.seq {
                        let p_idx = (head * params.seq + qpos) * params.seq + kpos;
                        let v_idx = (kpos * params.kv_heads + kv_head) * params.head_dim + d;
                        sum += i64::from(p[p_idx]) * i64::from(v[v_idx]);
                    }
                    out[(qpos * params.q_heads + head) * params.head_dim + d] = sum;
                }
            }
        }
        out
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
