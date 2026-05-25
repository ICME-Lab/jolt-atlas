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
    ops::{
        attention_common::{
            QWEN3_GQA_GROUP_SIZE, drop_gqa_lsb_for_dims, drop_gqa_lsb_for_dims_verify, ensure_len,
            log2_ceil, split3, validate_claim, validate_gqa, verify_claim,
        },
        round::{
            ROUND_FRAC_BITS, ROUND_LUT_LEN, RoundLookupProof, RoundParams, RoundProof,
            RoundWitness, padded_lookup_indices, prove_round, prove_round_lookup, round_lut_table,
            verify_round, verify_round_lookup,
        },
    },
};
use common::VirtualPoly;

// QK score is the attention contraction:
//     raw_acc = sum_d Q[qpos, h, d] * K[kpos, h/2, d]
//     dot     = round(raw_acc / 2^8)
//     S       = round(dot * round(2^8 / sqrt(head_dim)) / 2^8)
//
// The two-step rounding matches the qwen3-awy runtime.  The `dot` tensor is
// not a semantic graph node in Qwen; it exists only to make both fixed-point
// rebases explicit in the proof.
//
// The h/2 mapping is Qwen3's Grouped-Query Attention (GQA) with group_size=2.
// As with PV matmul, the MLE cannot factor the random head point as
// `Q(r_h) * K(drop_lsb(r_h))`.  `Q` and `K` share the head variable, so the
// sumcheck ranges over `h` and `d` and uses `eq(h, r_h)` as a known selector.

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
pub struct QkScoreWitness<'a> {
    pub q: &'a [i32],
    pub k: &'a [i32],
    pub raw_acc: &'a [i64],
    pub dot: &'a [i32],
    pub dot_frac_bits: [&'a [u8]; ROUND_FRAC_BITS],
    pub scale_acc: &'a [i64],
    pub output: &'a [i32],
    pub frac_bits: [&'a [u8]; ROUND_FRAC_BITS],
}

#[derive(Debug, Clone)]
pub struct QkScoreProof<F: JoltField, T: Transcript> {
    pub score_round: RoundProof<F, T>,
    pub dot_opening: F,
    pub(crate) dot_round_lookup: RoundLookupProof<F, T>,
    pub qk: QkScoreRoundRelationProof<F, T>,
}

#[derive(Debug, Clone)]
pub struct QkScoreRoundRelationProof<F: JoltField, T: Transcript> {
    pub sumcheck: SumcheckInstanceProof<F, T>,
    pub left_opening: F,
    pub right_opening: F,
    pub remainder_opening: F,
    pub round_bit_opening: F,
}

pub fn prove_qk_score_round<F, T>(
    score_claim: Claim<F>,
    witness: &QkScoreWitness<'_>,
    params: &QkScoreRoundParams,
    transcript: &mut T,
) -> Result<(QkScoreProof<F, T>, Claim<F>, Claim<F>, Claim<F>, Claim<F>)>
where
    F: JoltField,
    T: Transcript,
{
    let score_round_witness = RoundWitness::from_input_output(witness.scale_acc, witness.output);
    let (score_round_proof, scale_acc_claim, score_ra) = prove_round(
        vec![score_claim],
        &score_round_witness,
        &params.score_round,
        transcript,
    )?;

    let dot_opening = eval_tensor(
        witness.dot,
        &params.qk.score_shape(),
        &scale_acc_claim.point,
    );
    let inv_sqrt = inv_sqrt_head_dim::<F>(params.qk.head_dim);
    if scale_acc_claim.value != dot_opening * inv_sqrt {
        return Err(ProverError::MulMismatch);
    }
    transcript.append_scalar(&dot_opening);

    let dot_claim = Claim {
        tensor: params.dot_round.output_tensor.clone(),
        logical_shape: params.qk.score_shape(),
        domain_shape: params.qk.score_shape().padded_power_of_two(),
        point: scale_acc_claim.point,
        value: dot_opening,
    };

    let dot_round_witness = RoundWitness::from_input_output(witness.raw_acc, witness.dot);
    let (qk_proof, q, k, dot_round_point, dot_round_bit_opening, dot_remainder_opening) =
        prove_qk_score_round_relation(
            dot_claim,
            witness.q,
            witness.k,
            &dot_round_witness.remainder,
            &dot_round_witness.round_bit,
            &params.qk,
            transcript,
        )?;
    let mut dot_round_accumulator = ProverOpeningAccumulator::new();
    let dot_round_opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(dot_round_point.clone());
    dot_round_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundLut, SumcheckId::NodeExecution(0)),
        (dot_round_opening_point.clone(), dot_round_bit_opening),
    );
    dot_round_accumulator.openings.insert(
        OpeningId::new(
            VirtualPoly::QwenRoundRemainder,
            SumcheckId::NodeExecution(0),
        ),
        (dot_round_opening_point, dot_remainder_opening),
    );
    let dot_round_lookup = prove_round_lookup(
        dot_round_point,
        dot_round_bit_opening,
        dot_remainder_opening,
        padded_lookup_indices(&dot_round_witness.remainder, &params.dot_round.shape),
        round_lut_table(),
        &mut dot_round_accumulator,
        transcript,
    )?;
    let dot_ra = Claim {
        tensor: TensorId::new(format!("{}_round_ra", params.dot_round.input_tensor.0)),
        logical_shape: Shape::new(vec![
            params.dot_round.shape.padded_power_of_two().numel(),
            ROUND_LUT_LEN,
        ]),
        domain_shape: Shape::new(vec![
            params.dot_round.shape.padded_power_of_two().numel(),
            ROUND_LUT_LEN,
        ]),
        point: dot_round_lookup.ra_point.clone(),
        value: dot_round_lookup.ra_opening,
    };

    Ok((
        QkScoreProof {
            score_round: score_round_proof,
            dot_opening,
            dot_round_lookup,
            qk: qk_proof,
        },
        q,
        k,
        score_ra,
        dot_ra,
    ))
}

pub fn verify_qk_score_round<F, T>(
    score_claim: Claim<F>,
    proof: &QkScoreProof<F, T>,
    params: &QkScoreRoundParams,
    transcript: &mut T,
) -> std::result::Result<(Claim<F>, Claim<F>, Claim<F>, Claim<F>), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    let (scale_acc_claim, score_ra) = verify_round(
        vec![score_claim],
        &proof.score_round,
        &params.score_round,
        transcript,
    )?;
    transcript.append_scalar(&proof.dot_opening);
    if scale_acc_claim.value != proof.dot_opening * inv_sqrt_head_dim::<F>(params.qk.head_dim) {
        return Err(ProofVerifyError::InvalidInputLength(1, 0));
    }
    let dot_claim = Claim {
        tensor: params.dot_round.output_tensor.clone(),
        logical_shape: params.qk.score_shape(),
        domain_shape: params.qk.score_shape().padded_power_of_two(),
        point: scale_acc_claim.point,
        value: proof.dot_opening,
    };
    let (q, k, dot_round_point, dot_round_bit_opening, dot_remainder_opening) =
        verify_qk_score_round_relation(dot_claim, &proof.qk, &params.qk, transcript)?;
    let dot_round_lookup = verify_round_lookup(
        params.dot_round.shape.padded_power_of_two().numel(),
        dot_round_point,
        dot_round_bit_opening,
        dot_remainder_opening,
        proof.dot_round_lookup.ra_opening,
        &proof.dot_round_lookup.committed_openings,
        &proof.dot_round_lookup.read_raf,
        &proof.dot_round_lookup.ra_onehot,
        &mut VerifierOpeningAccumulator::new(),
        transcript,
    )?;
    let dot_ra = Claim {
        tensor: TensorId::new(format!("{}_round_ra", params.dot_round.input_tensor.0)),
        logical_shape: Shape::new(vec![
            params.dot_round.shape.padded_power_of_two().numel(),
            ROUND_LUT_LEN,
        ]),
        domain_shape: Shape::new(vec![
            params.dot_round.shape.padded_power_of_two().numel(),
            ROUND_LUT_LEN,
        ]),
        point: dot_round_lookup.ra_point,
        value: proof.dot_round_lookup.ra_opening,
    };

    Ok((q, k, score_ra, dot_ra))
}

#[allow(clippy::type_complexity)]
fn prove_qk_score_round_relation<F, T>(
    score_claim: Claim<F>,
    q: &[i32],
    k: &[i32],
    remainder: &[usize],
    round_bit: &[i32],
    params: &QkScoreParams,
    transcript: &mut T,
) -> Result<(
    QkScoreRoundRelationProof<F, T>,
    Claim<F>,
    Claim<F>,
    Vec<F>,
    F,
    F,
)>
where
    F: JoltField,
    T: Transcript,
{
    validate_qk_inputs(&score_claim, q, k, params)?;
    validate_round_witness(remainder, round_bit, &params.score_shape())?;
    let (r_h, r_q, r_kpos) = split3(&score_claim.point, params.q_heads, params.seq, params.seq);
    let left = partial_q(q, params, r_q);
    let right = partial_k(k, params, r_kpos);
    let selector = expanded_head_eq(r_h, params);
    let remainder = eval_round_tensor(remainder, &params.score_shape(), &score_claim.point);
    let round_bit = eval_tensor(round_bit, &params.score_shape(), &score_claim.point);
    let (proof, r_h_d) = prove_qk_round_product(
        score_claim.value * scale_q8::<F>(),
        left,
        right,
        selector,
        remainder,
        round_bit,
        transcript,
    )?;
    let (r_head, r_d) = r_h_d.split_at(log2_ceil(params.q_heads));
    let kv_h = drop_gqa_lsb_for_dims(r_head, params.q_heads, params.kv_heads)?;
    let q_opening = proof.left_opening;
    let k_opening = proof.right_opening;
    let remainder_opening = proof.remainder_opening;
    let round_bit_opening = proof.round_bit_opening;

    Ok((
        proof,
        Claim {
            tensor: params.q_tensor.clone(),
            logical_shape: params.q_shape(),
            domain_shape: params.q_shape().padded_power_of_two(),
            point: [r_q, r_head, r_d].concat(),
            value: q_opening,
        },
        Claim {
            tensor: params.k_tensor.clone(),
            logical_shape: params.k_shape(),
            domain_shape: params.k_shape().padded_power_of_two(),
            point: [r_kpos, kv_h, r_d].concat(),
            value: k_opening,
        },
        score_claim.point,
        round_bit_opening,
        remainder_opening,
    ))
}

#[allow(clippy::type_complexity)]
fn verify_qk_score_round_relation<F, T>(
    score_claim: Claim<F>,
    proof: &QkScoreRoundRelationProof<F, T>,
    params: &QkScoreParams,
    transcript: &mut T,
) -> std::result::Result<(Claim<F>, Claim<F>, Vec<F>, F, F), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    verify_qk_inputs(&score_claim, params)?;
    let (r_h, r_q, r_kpos) = split3(&score_claim.point, params.q_heads, params.seq, params.seq);
    let r_h_d = verify_qk_round_product(
        score_claim.value * scale_q8::<F>(),
        proof,
        r_h,
        params,
        transcript,
    )?;
    let (r_head, r_d) = r_h_d.split_at(log2_ceil(params.q_heads));
    let kv_h = drop_gqa_lsb_for_dims_verify(r_head, params.q_heads, params.kv_heads)?;

    Ok((
        Claim {
            tensor: params.q_tensor.clone(),
            logical_shape: params.q_shape(),
            domain_shape: params.q_shape().padded_power_of_two(),
            point: [r_q, r_head, r_d].concat(),
            value: proof.left_opening,
        },
        Claim {
            tensor: params.k_tensor.clone(),
            logical_shape: params.k_shape(),
            domain_shape: params.k_shape().padded_power_of_two(),
            point: [r_kpos, kv_h, r_d].concat(),
            value: proof.right_opening,
        },
        score_claim.point,
        proof.round_bit_opening,
        proof.remainder_opening,
    ))
}

fn partial_q<F: JoltField>(q: &[i32], params: &QkScoreParams, r_q: &[F]) -> Vec<F> {
    let head_pad = params.q_heads.next_power_of_two();
    let dim_pad = params.head_dim.next_power_of_two();
    let seq_eq = EqPolynomial::<F>::evals(r_q);
    let mut out = vec![F::zero(); head_pad * dim_pad];
    for head in 0..params.q_heads {
        for d in 0..params.head_dim {
            out[head * dim_pad + d] = (0..params.seq)
                .map(|pos| {
                    seq_eq[pos]
                        * F::from_i32(q[(pos * params.q_heads + head) * params.head_dim + d])
                })
                .sum();
        }
    }
    out
}

fn partial_k<F: JoltField>(k: &[i32], params: &QkScoreParams, r_kpos: &[F]) -> Vec<F> {
    let head_pad = params.q_heads.next_power_of_two();
    let dim_pad = params.head_dim.next_power_of_two();
    let seq_eq = EqPolynomial::<F>::evals(r_kpos);
    let mut out = vec![F::zero(); head_pad * dim_pad];
    for head in 0..params.q_heads {
        let kv_head = head / QWEN3_GQA_GROUP_SIZE;
        for d in 0..params.head_dim {
            out[head * dim_pad + d] = (0..params.seq)
                .map(|pos| {
                    seq_eq[pos]
                        * F::from_i32(k[(pos * params.kv_heads + kv_head) * params.head_dim + d])
                })
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
    let left_opening = opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenAttentionLeft, qk_sumcheck_id()),
    )?;
    let right_opening = opening(
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
    let r = normalize_sumcheck_point::<F>(&challenges.into_opening());
    Ok((
        QkScoreRoundRelationProof {
            sumcheck,
            left_opening,
            right_opening,
            remainder_opening,
            round_bit_opening,
        },
        r,
    ))
}

fn verify_qk_round_product<F, T>(
    input_claim: F,
    proof: &QkScoreRoundRelationProof<F, T>,
    output_head_point: &[F],
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
        (OpeningPoint::<BIG_ENDIAN, F>::default(), proof.left_opening),
    );
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenAttentionRight, qk_sumcheck_id()),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.right_opening,
        ),
    );
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundRemainder, qk_sumcheck_id()),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.remainder_opening,
        ),
    );
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundLut, qk_sumcheck_id()),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.round_bit_opening,
        ),
    );
    let challenges = Sumcheck::verify(&proof.sumcheck, &verifier, &mut accumulator, transcript)?;
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
            let l0 = self.left.get_bound_coeff(2 * g);
            let l1 = self.left.get_bound_coeff(2 * g + 1);
            let r0 = self.right.get_bound_coeff(2 * g);
            let r1 = self.right.get_bound_coeff(2 * g + 1);
            let s0 = self.selector.get_bound_coeff(2 * g);
            let s1 = self.selector.get_bound_coeff(2 * g + 1);
            let z0 = self.eq_zero.get_bound_coeff(2 * g);
            let z1 = self.eq_zero.get_bound_coeff(2 * g + 1);
            evals[0] += l0 * r0 * s0 + z0 * round_term;
            evals[1] +=
                (l1 + l1 - l0) * (r1 + r1 - r0) * (s1 + s1 - s0) + (z1 + z1 - z0) * round_term;
            evals[2] += (l1 * F::from_u64(3) - l0 - l0)
                * (r1 * F::from_u64(3) - r0 - r0)
                * (s1 * F::from_u64(3) - s0 - s0)
                + (z1 * F::from_u64(3) - z0 - z0) * round_term;
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
            point,
            self.right.final_claim(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRoundRemainder, qk_sumcheck_id()),
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            self.remainder,
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRoundLut, qk_sumcheck_id()),
            OpeningPoint::<BIG_ENDIAN, F>::default(),
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
            point,
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRoundRemainder, qk_sumcheck_id()),
            OpeningPoint::<BIG_ENDIAN, F>::default(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRoundLut, qk_sumcheck_id()),
            OpeningPoint::<BIG_ENDIAN, F>::default(),
        );
    }
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

fn qk_sumcheck_id() -> SumcheckId {
    SumcheckId::NodeExecution(0)
}

fn validate_qk_inputs<F: JoltField>(
    score_claim: &Claim<F>,
    q: &[i32],
    k: &[i32],
    params: &QkScoreParams,
) -> Result<()> {
    validate_gqa(params.q_heads, params.kv_heads)?;
    ensure_len("Q", &params.q_shape(), q.len())?;
    ensure_len("K", &params.k_shape(), k.len())?;
    validate_claim(score_claim, &params.score_shape())
}

fn verify_qk_inputs<F: JoltField>(
    score_claim: &Claim<F>,
    params: &QkScoreParams,
) -> std::result::Result<(), ProofVerifyError> {
    if params.q_heads / params.kv_heads != QWEN3_GQA_GROUP_SIZE {
        return Err(ProofVerifyError::InvalidInputLength(
            QWEN3_GQA_GROUP_SIZE,
            params.q_heads / params.kv_heads,
        ));
    }
    verify_claim(score_claim, &params.score_shape())
}

fn validate_round_witness(remainder: &[usize], round_bit: &[i32], shape: &Shape) -> Result<()> {
    let expected = shape.numel();
    if remainder.len() != expected {
        return Err(ProverError::TensorLenMismatch {
            name: "qk round remainder",
            shape: shape.0.clone(),
            expected,
            actual: remainder.len(),
        });
    }
    if round_bit.len() != expected {
        return Err(ProverError::TensorLenMismatch {
            name: "qk round bit",
            shape: shape.0.clone(),
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

fn inv_sqrt_head_dim<F: JoltField>(head_dim: usize) -> F {
    let inv = ((1.0 / (head_dim as f64).sqrt()) * 256.0).round() as i32;
    F::from_i32(inv)
}

fn eval_round_tensor<F: JoltField>(values: &[usize], shape: &Shape, point: &[F]) -> F {
    let eq_by_dim = split_point_for_shape(shape, point)
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
        out += weight * F::from_u64(value as u64);
    }
    out
}

fn eval_tensor<F: JoltField>(values: &[i32], shape: &Shape, point: &[F]) -> F {
    let eq_by_dim = split_point_for_shape(shape, point)
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

fn scale_q8<F: JoltField>() -> F {
    F::from_u64(256)
}

fn split_point_for_shape<'a, F: Clone>(shape: &Shape, point: &'a [F]) -> Vec<&'a [F]> {
    let mut offset = 0;
    shape
        .dims()
        .iter()
        .map(|dim| {
            let bits = dim.next_power_of_two().trailing_zeros() as usize;
            let out = &point[offset..offset + bits];
            offset += bits;
            out
        })
        .collect()
}

fn row_major_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; dims.len()];
    for idx in (0..dims.len()).rev().skip(1) {
        strides[idx] = strides[idx + 1] * dims[idx + 1];
    }
    strides
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use joltworks::transcripts::Blake2bTranscript;

    use super::*;

    #[test]
    fn proves_and_verifies_qk_score() {
        let qk_params = QkScoreParams::new(2, 2, 1, 2, "Q", "K");
        let params = QkScoreRoundParams::new(
            RoundParams::new(vec![2, 2, 2], "S_scale_acc", "S"),
            RoundParams::new(vec![2, 2, 2], "S_raw_acc", "S_dot"),
            qk_params.clone(),
        );
        let q = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let k = vec![2, 1, 4, 3];
        let raw_acc = qk_scores(&q, &k, &qk_params);
        let dot = round_outputs(&raw_acc);
        let dot_frac_bits = frac_bits_for(&raw_acc);
        let inv_sqrt = ((1.0 / (qk_params.head_dim as f64).sqrt()) * 256.0).round() as i64;
        let scale_acc = dot
            .iter()
            .map(|&value| i64::from(value) * inv_sqrt)
            .collect::<Vec<_>>();
        let output = round_outputs(&scale_acc);
        let frac_bits = frac_bits_for(&scale_acc);
        let point = vec![Fr::from(3u64), Fr::from(5u64), Fr::from(7u64)];
        let claim = Claim {
            tensor: TensorId::new("S"),
            logical_shape: qk_params.score_shape(),
            domain_shape: qk_params.score_shape().padded_power_of_two(),
            value: eval_tensor(&output, &qk_params.score_shape(), &point),
            point,
        };
        let witness = QkScoreWitness {
            q: &q,
            k: &k,
            raw_acc: &raw_acc,
            dot: &dot,
            dot_frac_bits: dot_frac_bits.each_ref().map(Vec::as_slice),
            scale_acc: &scale_acc,
            output: &output,
            frac_bits: frac_bits.each_ref().map(Vec::as_slice),
        };

        let mut prover_transcript = Blake2bTranscript::default();
        let (proof, q_claim, k_claim, _, _) =
            prove_qk_score_round::<Fr, _>(claim.clone(), &witness, &params, &mut prover_transcript)
                .unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        let (verified_q, verified_k, _, _) =
            verify_qk_score_round::<Fr, _>(claim, &proof, &params, &mut verifier_transcript)
                .unwrap();

        assert_eq!(verified_q, q_claim);
        assert_eq!(verified_k, k_claim);
        assert_eq!(verified_q.tensor.0, "Q");
        assert_eq!(verified_k.tensor.0, "K");
    }

    fn qk_scores(q: &[i32], k: &[i32], params: &QkScoreParams) -> Vec<i64> {
        let mut out = vec![0; params.q_heads * params.seq * params.seq];
        for h in 0..params.q_heads {
            for qpos in 0..params.seq {
                for kpos in 0..params.seq {
                    let kvh = h / QWEN3_GQA_GROUP_SIZE;
                    let mut sum = 0_i64;
                    for d in 0..params.head_dim {
                        sum += i64::from(q[(qpos * params.q_heads + h) * params.head_dim + d])
                            * i64::from(k[(kpos * params.kv_heads + kvh) * params.head_dim + d]);
                    }
                    out[(h * params.seq + qpos) * params.seq + kpos] = sum;
                }
            }
        }
        out
    }

    fn round_outputs(values: &[i64]) -> Vec<i32> {
        values
            .iter()
            .map(|&value| {
                let frac = value.rem_euclid(256);
                ((value + i64::from(frac >= 128) * 256 - frac) / 256) as i32
            })
            .collect()
    }

    fn frac_bits_for(values: &[i64]) -> [Vec<u8>; 8] {
        std::array::from_fn(|bit| {
            values
                .iter()
                .map(|value| ((value.rem_euclid(256) >> bit) & 1) as u8)
                .collect()
        })
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
}
