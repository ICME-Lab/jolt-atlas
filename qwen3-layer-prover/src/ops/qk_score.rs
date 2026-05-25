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

/// Rounded QK score:
///
///   dot   = round(sum_d Q[qpos, h, d] * K[kpos, h/2, d] / 2^8)
///   score = round(dot * round(2^8 / sqrt(head_dim)) / 2^8)
///
/// The fixed runtime has two explicit rebases, so this proof has two round
/// lookups: one for the raw QK dot product, and one for the head-dim scale.

const QWEN3_GQA_GROUP_SIZE: usize = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QkScoreParams {
    pub seq: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub q_tensor: TensorId,
    pub k_tensor: TensorId,
}

impl QkScoreParams {
    pub fn new(
        seq: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        q_tensor: impl Into<String>,
        k_tensor: impl Into<String>,
    ) -> Self {
        Self {
            seq,
            q_heads,
            kv_heads,
            head_dim,
            q_tensor: TensorId::new(q_tensor),
            k_tensor: TensorId::new(k_tensor),
        }
    }

    pub fn q_shape(&self) -> Shape {
        Shape::new(vec![self.seq, self.q_heads, self.head_dim])
    }

    pub fn k_shape(&self) -> Shape {
        Shape::new(vec![self.seq, self.kv_heads, self.head_dim])
    }

    pub fn score_shape(&self) -> Shape {
        Shape::new(vec![self.q_heads, self.seq, self.seq])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QkScoreRoundParams {
    pub score_round: RoundParams,
    pub dot_round: RoundParams,
    pub qk: QkScoreParams,
}

impl QkScoreRoundParams {
    pub fn new(score_round: RoundParams, dot_round: RoundParams, qk: QkScoreParams) -> Self {
        Self {
            score_round,
            dot_round,
            qk,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QkScoreProof<F: JoltField, T: Transcript> {
    pub dot_opening: F,
    pub(crate) score_round_lookup: RoundLookupProof<F, T>,
    pub qk: QkScoreRoundRelationProof<F, T>,
    pub(crate) dot_round_lookup: RoundLookupProof<F, T>,
}

#[derive(Debug, Clone)]
pub struct QkScoreRoundRelationProof<F: JoltField, T: Transcript> {
    pub sumcheck: SumcheckInstanceProof<F, T>,
    pub q_opening: F,
    pub k_opening: F,
    pub remainder_opening: F,
    pub round_bit_opening: F,
}

pub fn prove_qk_score_round<F, T, C>(
    score_claim: Claim<F, C>,
    q_poly: Poly<F, C>,
    k_poly: Poly<F, C>,
    score_round_ra: Vec<Poly<F, C>>,
    dot_round_ra: Vec<Poly<F, C>>,
    params: &QkScoreRoundParams,
    transcript: &mut T,
) -> Result<(
    QkScoreProof<F, T>,
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
    validate_claim_shape(&score_claim, &params.qk.score_shape(), "QK score claim")?;
    validate_poly_shape(&q_poly, &params.qk.q_shape(), "QK Q")?;
    validate_poly_shape(&k_poly, &params.qk.k_shape(), "QK K")?;
    validate_qk_params(&params.qk)?;

    let (score_remainder, score_round_bit) = round_lookup_openings_from_ra(
        &score_round_ra,
        &score_claim.point,
        &params.score_round.shape,
    )?;
    let inv_sqrt = inv_sqrt_head_dim::<F>(params.qk.head_dim);
    let dot_opening = (score_claim.value * scale_q8::<F>() - score_round_bit * scale_q8::<F>()
        + score_remainder)
        * inv_sqrt.inverse().expect("non-zero inv sqrt");
    transcript.append_scalar(&dot_opening);

    let mut score_round_accumulator = ProverOpeningAccumulator::new();
    let score_round_point = OpeningPoint::<BIG_ENDIAN, F>::new(score_claim.point.clone());
    score_round_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundLut, score_sumcheck_id()),
        (score_round_point.clone(), score_round_bit),
    );
    score_round_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundRemainder, score_sumcheck_id()),
        (score_round_point, score_remainder),
    );
    let (score_round_lookup, score_ra_claims) = prove_round_lookup_from_ra(
        params.score_round.lookup_site,
        score_claim.point.clone(),
        score_round_ra,
        &params.score_round.shape,
        score_round_bit,
        score_remainder,
        &mut score_round_accumulator,
        transcript,
    )?;

    let (dot_remainder, dot_round_bit) =
        round_lookup_openings_from_ra(&dot_round_ra, &score_claim.point, &params.dot_round.shape)?;
    let (relation, q_point, q_value, k_point, k_value, dot_round_point) =
        prove_qk_score_round_relation(
            score_claim.point.clone(),
            dot_opening,
            &q_poly,
            &k_poly,
            dot_remainder,
            dot_round_bit,
            &params.qk,
            transcript,
        )?;

    let mut dot_round_accumulator = ProverOpeningAccumulator::new();
    let dot_round_opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(dot_round_point);
    dot_round_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundLut, qk_sumcheck_id()),
        (dot_round_opening_point.clone(), dot_round_bit),
    );
    dot_round_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundRemainder, qk_sumcheck_id()),
        (dot_round_opening_point, dot_remainder),
    );
    let (dot_round_lookup, dot_ra_claims) = prove_round_lookup_from_ra(
        params.dot_round.lookup_site,
        score_claim.point,
        dot_round_ra,
        &params.dot_round.shape,
        dot_round_bit,
        dot_remainder,
        &mut dot_round_accumulator,
        transcript,
    )?;

    Ok((
        QkScoreProof {
            dot_opening,
            score_round_lookup,
            qk: relation,
            dot_round_lookup,
        },
        Claim::new(q_poly, q_point, q_value),
        Claim::new(k_poly, k_point, k_value),
        score_ra_claims,
        dot_ra_claims,
    ))
}

pub fn verify_qk_score_round<F, T, C>(
    score_claim: Claim<F, C>,
    proof: &QkScoreProof<F, T>,
    score_round_ra: Vec<Poly<F, C>>,
    dot_round_ra: Vec<Poly<F, C>>,
    params: &QkScoreRoundParams,
    transcript: &mut T,
) -> std::result::Result<
    (Claim<F, C>, Claim<F, C>, Vec<Claim<F, C>>, Vec<Claim<F, C>>),
    ProofVerifyError,
>
where
    F: JoltField,
    T: Transcript,
    C: Clone,
{
    verify_claim_shape(&score_claim, &params.qk.score_shape())?;
    verify_qk_params(&params.qk)?;

    let (score_remainder, score_round_bit) = round_lookup_openings_from_ra(
        &score_round_ra,
        &score_claim.point,
        &params.score_round.shape,
    )
    .map_err(|_| ProofVerifyError::SumcheckVerificationError)?;
    let inv_sqrt = inv_sqrt_head_dim::<F>(params.qk.head_dim);
    if score_claim.value * scale_q8::<F>()
        != proof.dot_opening * inv_sqrt + score_round_bit * scale_q8::<F>() - score_remainder
    {
        return Err(ProofVerifyError::SumcheckVerificationError);
    }
    transcript.append_scalar(&proof.dot_opening);

    let mut score_round_accumulator = VerifierOpeningAccumulator::new();
    let score_round_point = OpeningPoint::<BIG_ENDIAN, F>::new(score_claim.point.clone());
    score_round_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundLut, score_sumcheck_id()),
        (score_round_point.clone(), score_round_bit),
    );
    score_round_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundRemainder, score_sumcheck_id()),
        (score_round_point, score_remainder),
    );
    let (_score_lookup, score_ra_claims) = verify_round_lookup_from_ra(
        params.score_round.lookup_site,
        score_claim.point.clone(),
        score_round_ra,
        &params.score_round.shape,
        score_round_bit,
        score_remainder,
        &proof.score_round_lookup,
        &mut score_round_accumulator,
        transcript,
    )?;

    let (dot_remainder, dot_round_bit) =
        round_lookup_openings_from_ra(&dot_round_ra, &score_claim.point, &params.dot_round.shape)
            .map_err(|_| ProofVerifyError::SumcheckVerificationError)?;
    let (q_point, k_point, dot_round_point) = verify_qk_score_round_relation(
        score_claim.point.clone(),
        proof.dot_opening,
        proof,
        dot_remainder,
        dot_round_bit,
        &params.qk,
        transcript,
    )?;

    let mut dot_round_accumulator = VerifierOpeningAccumulator::new();
    let dot_round_opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(dot_round_point);
    dot_round_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundLut, qk_sumcheck_id()),
        (dot_round_opening_point.clone(), dot_round_bit),
    );
    dot_round_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundRemainder, qk_sumcheck_id()),
        (dot_round_opening_point, dot_remainder),
    );
    let (_dot_lookup, dot_ra_claims) = verify_round_lookup_from_ra(
        params.dot_round.lookup_site,
        score_claim.point,
        dot_round_ra,
        &params.dot_round.shape,
        dot_round_bit,
        dot_remainder,
        &proof.dot_round_lookup,
        &mut dot_round_accumulator,
        transcript,
    )?;

    Ok((
        Claim::new(
            dummy_poly(&params.qk.q_shape()),
            q_point,
            proof.qk.q_opening,
        ),
        Claim::new(
            dummy_poly(&params.qk.k_shape()),
            k_point,
            proof.qk.k_opening,
        ),
        score_ra_claims,
        dot_ra_claims,
    ))
}

fn prove_qk_score_round_relation<F, T, C>(
    score_point: Vec<F>,
    dot_value: F,
    q_poly: &Poly<F, C>,
    k_poly: &Poly<F, C>,
    remainder_opening: F,
    round_bit_opening: F,
    params: &QkScoreParams,
    transcript: &mut T,
) -> Result<(
    QkScoreRoundRelationProof<F, T>,
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
    let (r_h, r_q, r_kpos) = split3(&score_point, params.q_heads, params.seq);
    let left = partial_q(q_poly, params, r_q);
    let right = partial_k(k_poly, params, r_kpos);
    let selector = expanded_head_eq(r_h, params);
    let (relation, r_h_d) = prove_qk_round_product(
        dot_value * scale_q8::<F>(),
        left,
        right,
        selector,
        remainder_opening,
        round_bit_opening,
        transcript,
    )?;
    let (r_head, r_d) = r_h_d.split_at(log2_ceil(params.q_heads));
    let kv_h = drop_gqa_lsb_for_dims(r_head, params)?;

    Ok((
        QkScoreRoundRelationProof {
            q_opening: relation.q_opening,
            k_opening: relation.k_opening,
            remainder_opening: relation.remainder_opening,
            round_bit_opening: relation.round_bit_opening,
            sumcheck: relation.sumcheck,
        },
        [r_q, r_head, r_d].concat(),
        relation.q_opening,
        [r_kpos, kv_h, r_d].concat(),
        relation.k_opening,
        score_point,
    ))
}

fn verify_qk_score_round_relation<F, T>(
    score_point: Vec<F>,
    dot_value: F,
    proof: &QkScoreProof<F, T>,
    remainder_opening: F,
    round_bit_opening: F,
    params: &QkScoreParams,
    transcript: &mut T,
) -> std::result::Result<(Vec<F>, Vec<F>, Vec<F>), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    let (r_h, r_q, r_kpos) = split3(&score_point, params.q_heads, params.seq);
    let r_h_d = verify_qk_round_product(
        dot_value * scale_q8::<F>(),
        proof,
        r_h,
        remainder_opening,
        round_bit_opening,
        params,
        transcript,
    )?;
    let (r_head, r_d) = r_h_d.split_at(log2_ceil(params.q_heads));
    let kv_h = drop_gqa_lsb_for_dims_verify(r_head, params)?;
    Ok((
        [r_q, r_head, r_d].concat(),
        [r_kpos, kv_h, r_d].concat(),
        score_point,
    ))
}

fn partial_q<F: JoltField, C>(q: &Poly<F, C>, params: &QkScoreParams, r_q: &[F]) -> Vec<F> {
    let head_pad = params.q_heads.next_power_of_two();
    let dim_pad = params.head_dim.next_power_of_two();
    let seq_eq = EqPolynomial::<F>::evals(r_q);
    let mut out = vec![F::zero(); head_pad * dim_pad];
    for head in 0..params.q_heads {
        for d in 0..params.head_dim {
            out[head * dim_pad + d] = (0..params.seq)
                .map(|pos| seq_eq[pos] * coeff3(q, &params.q_shape(), pos, head, d))
                .sum();
        }
    }
    out
}

fn partial_k<F: JoltField, C>(k: &Poly<F, C>, params: &QkScoreParams, r_kpos: &[F]) -> Vec<F> {
    let head_pad = params.q_heads.next_power_of_two();
    let dim_pad = params.head_dim.next_power_of_two();
    let seq_eq = EqPolynomial::<F>::evals(r_kpos);
    let mut out = vec![F::zero(); head_pad * dim_pad];
    for head in 0..params.q_heads {
        let kv_head = head / QWEN3_GQA_GROUP_SIZE;
        for d in 0..params.head_dim {
            out[head * dim_pad + d] = (0..params.seq)
                .map(|pos| seq_eq[pos] * coeff3(k, &params.k_shape(), pos, kv_head, d))
                .sum();
        }
    }
    out
}

fn expanded_head_eq<F: JoltField>(r_h: &[F], params: &QkScoreParams) -> Vec<F> {
    let head_pad = params.q_heads.next_power_of_two();
    let dim_pad = params.head_dim.next_power_of_two();
    let head_eq = EqPolynomial::<F>::evals(r_h);
    let mut out = vec![F::zero(); head_pad * dim_pad];
    for head in 0..params.q_heads {
        for d in 0..dim_pad {
            out[head * dim_pad + d] = head_eq[head];
        }
    }
    out
}

fn prove_qk_round_product<F, T>(
    input_claim: F,
    left: Vec<F>,
    right: Vec<F>,
    selector: Vec<F>,
    remainder: F,
    round_bit: F,
    transcript: &mut T,
) -> Result<(QkScoreRoundRelationProof<F, T>, Vec<F>)>
where
    F: JoltField,
    T: Transcript,
{
    let params = QkProductParams::new(
        left.len().trailing_zeros() as usize,
        input_claim,
        Vec::new(),
    );
    let mut prover = QkRoundProductProver::new(params, left, right, selector, remainder, round_bit);
    let mut accumulator = ProverOpeningAccumulator::new();
    let (sumcheck, challenges) = Sumcheck::prove(&mut prover, &mut accumulator, transcript);
    let q_opening = opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenAttentionLeft, qk_sumcheck_id()),
    )?;
    let k_opening = opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenAttentionRight, qk_sumcheck_id()),
    )?;
    let remainder_opening = opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenRoundRemainder, qk_sumcheck_id()),
    )?;
    let round_bit_opening = opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenRoundLut, qk_sumcheck_id()),
    )?;
    Ok((
        QkScoreRoundRelationProof {
            sumcheck,
            q_opening,
            k_opening,
            remainder_opening,
            round_bit_opening,
        },
        normalize_sumcheck_point::<F>(&challenges.into_opening()),
    ))
}

fn verify_qk_round_product<F, T>(
    input_claim: F,
    proof: &QkScoreProof<F, T>,
    output_head_point: &[F],
    remainder_opening: F,
    round_bit_opening: F,
    params: &QkScoreParams,
    transcript: &mut T,
) -> std::result::Result<Vec<F>, ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    let num_rounds = log2_ceil(params.q_heads) + log2_ceil(params.head_dim);
    let verifier = QkRoundProductVerifier {
        params: QkProductParams::new(num_rounds, input_claim, output_head_point.to_vec()),
    };
    let mut accumulator = VerifierOpeningAccumulator::new();
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenAttentionLeft, qk_sumcheck_id()),
        (OpeningPoint::<BIG_ENDIAN, F>::default(), proof.qk.q_opening),
    );
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenAttentionRight, qk_sumcheck_id()),
        (OpeningPoint::<BIG_ENDIAN, F>::default(), proof.qk.k_opening),
    );
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundRemainder, qk_sumcheck_id()),
        (OpeningPoint::<BIG_ENDIAN, F>::default(), remainder_opening),
    );
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundLut, qk_sumcheck_id()),
        (OpeningPoint::<BIG_ENDIAN, F>::default(), round_bit_opening),
    );
    let challenges = Sumcheck::verify(&proof.qk.sumcheck, &verifier, &mut accumulator, transcript)?;
    Ok(normalize_sumcheck_point::<F>(&challenges.into_opening()))
}

struct QkProductParams<F: JoltField> {
    num_rounds: usize,
    input_claim: F,
    output_head_point: Vec<F>,
}

impl<F: JoltField> QkProductParams<F> {
    fn new(num_rounds: usize, input_claim: F, output_head_point: Vec<F>) -> Self {
        Self {
            num_rounds,
            input_claim,
            output_head_point,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for QkProductParams<F> {
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

struct QkRoundProductProver<F: JoltField> {
    left: MultilinearPolynomial<F>,
    right: MultilinearPolynomial<F>,
    selector: MultilinearPolynomial<F>,
    eq_zero: MultilinearPolynomial<F>,
    remainder: F,
    round_bit: F,
    params: QkProductParams<F>,
}

impl<F: JoltField> QkRoundProductProver<F> {
    fn new(
        params: QkProductParams<F>,
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

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for QkRoundProductProver<F> {
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
            OpeningId::new(VirtualPoly::QwenAttentionLeft, qk_sumcheck_id()),
            point.clone(),
            self.left.final_claim(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenAttentionRight, qk_sumcheck_id()),
            point.clone(),
            self.right.final_claim(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRoundRemainder, qk_sumcheck_id()),
            point.clone(),
            self.remainder,
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRoundLut, qk_sumcheck_id()),
            point,
            self.round_bit,
        );
    }
}

struct QkRoundProductVerifier<F: JoltField> {
    params: QkProductParams<F>,
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for QkRoundProductVerifier<F> {
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
                qk_sumcheck_id(),
            ))
            .1;
        let right = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenAttentionRight,
                qk_sumcheck_id(),
            ))
            .1;
        let remainder = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenRoundRemainder,
                qk_sumcheck_id(),
            ))
            .1;
        let round_bit = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenRoundLut,
                qk_sumcheck_id(),
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
            OpeningId::new(VirtualPoly::QwenAttentionLeft, qk_sumcheck_id()),
            point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenAttentionRight, qk_sumcheck_id()),
            point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRoundRemainder, qk_sumcheck_id()),
            point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRoundLut, qk_sumcheck_id()),
            point,
        );
    }
}

fn validate_qk_params(params: &QkScoreParams) -> Result<()> {
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

fn verify_qk_params(params: &QkScoreParams) -> std::result::Result<(), ProofVerifyError> {
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

fn drop_gqa_lsb_for_dims<'a, F>(r_h: &'a [F], params: &QkScoreParams) -> Result<&'a [F]> {
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
    params: &QkScoreParams,
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

fn inv_sqrt_head_dim<F: JoltField>(head_dim: usize) -> F {
    let inv = ((1.0 / (head_dim as f64).sqrt()) * 256.0).round() as i32;
    F::from_i32(inv)
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

fn qk_sumcheck_id() -> SumcheckId {
    SumcheckId::NodeExecution(0)
}

fn score_sumcheck_id() -> SumcheckId {
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
    fn proves_and_verifies_qk_score_round_from_poly_and_ra() {
        let qk = QkScoreParams::new(2, 2, 1, 2, "Q", "K");
        let params = QkScoreRoundParams::new(
            RoundParams::new(vec![2, 2, 2], "scaled_acc", "score"),
            RoundParams::new(vec![2, 2, 2], "raw_acc", "dot"),
            qk,
        );
        let q = vec![300, 200, 100, 400, 250, 350, 150, 450];
        let k = vec![120, 220, 320, 420];
        let raw_acc = qk_raw_scores(&q, &k, &params.qk);
        let dot = raw_acc
            .iter()
            .map(|&value| round_q8_to_i32(value))
            .collect::<Vec<_>>();
        let inv_sqrt = ((1.0 / (params.qk.head_dim as f64).sqrt()) * 256.0).round() as i64;
        let scaled_acc = dot
            .iter()
            .map(|&value| i64::from(value) * inv_sqrt)
            .collect::<Vec<_>>();
        let score = scaled_acc
            .iter()
            .map(|&value| round_q8_to_i32(value))
            .collect::<Vec<_>>();
        let point = vec![Fr::from(3_u64), Fr::from(5_u64), Fr::from(7_u64)];
        let score_claim = Claim::new(
            poly_from_i32(&score),
            point.clone(),
            eval_flat(&score, &point),
        );
        let score_ra = round_ra_from_acc(
            &scaled_acc,
            params.qk.score_shape().padded_power_of_two().numel(),
        );
        let dot_ra = round_ra_from_acc(
            &raw_acc,
            params.qk.score_shape().padded_power_of_two().numel(),
        );

        let mut prover_transcript = Blake2bTranscript::default();
        let (proof, q_claim, k_claim, score_ra_claims, dot_ra_claims) = prove_qk_score_round(
            score_claim,
            poly_from_i32(&q),
            poly_from_i32(&k),
            score_ra,
            dot_ra,
            &params,
            &mut prover_transcript,
        )
        .unwrap();
        assert!(!score_ra_claims.is_empty());
        assert!(!dot_ra_claims.is_empty());

        let score_claim = Claim::new(
            poly_from_i32(&score),
            point.clone(),
            eval_flat(&score, &point),
        );
        let score_ra = round_ra_from_acc(
            &scaled_acc,
            params.qk.score_shape().padded_power_of_two().numel(),
        );
        let dot_ra = round_ra_from_acc(
            &raw_acc,
            params.qk.score_shape().padded_power_of_two().numel(),
        );
        let mut verifier_transcript = Blake2bTranscript::default();
        let (verified_q, verified_k, verified_score_ra, verified_dot_ra) = verify_qk_score_round(
            score_claim,
            &proof,
            score_ra,
            dot_ra,
            &params,
            &mut verifier_transcript,
        )
        .unwrap();

        assert_eq!(verified_q.point, q_claim.point);
        assert_eq!(verified_q.value, q_claim.value);
        assert_eq!(verified_k.point, k_claim.point);
        assert_eq!(verified_k.value, k_claim.value);
        assert_eq!(verified_score_ra.len(), score_ra_claims.len());
        assert_eq!(verified_dot_ra.len(), dot_ra_claims.len());
    }

    fn qk_raw_scores(q: &[i32], k: &[i32], params: &QkScoreParams) -> Vec<i64> {
        let mut out = vec![0_i64; params.q_heads * params.seq * params.seq];
        for head in 0..params.q_heads {
            let kv_head = head / QWEN3_GQA_GROUP_SIZE;
            for qpos in 0..params.seq {
                for kpos in 0..params.seq {
                    let mut sum = 0_i64;
                    for d in 0..params.head_dim {
                        let q_idx = (qpos * params.q_heads + head) * params.head_dim + d;
                        let k_idx = (kpos * params.kv_heads + kv_head) * params.head_dim + d;
                        sum += i64::from(q[q_idx]) * i64::from(k[k_idx]);
                    }
                    out[(head * params.seq + qpos) * params.seq + kpos] = sum;
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
