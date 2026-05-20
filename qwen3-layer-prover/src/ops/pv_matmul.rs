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
            AttentionMatmulProof, QWEN3_GQA_GROUP_SIZE, drop_gqa_lsb_for_dims,
            drop_gqa_lsb_for_dims_verify, ensure_len, log2_ceil, split3, validate_claim,
            validate_gqa, verify_claim,
        },
        round::{
            ROUND_FRAC_BITS, ROUND_LUT_LEN, RoundLookupProof, RoundParams, RoundWitness,
            padded_lookup_indices, prove_round_lookup, round_lut_table, verify_round_lookup,
        },
    },
};
use common::VirtualPoly;

// PV matmul is the attention value contraction:
//     C[qpos, h, d] = sum_kpos P[h, qpos, kpos] * V[kpos, h/2, d].
// The h/2 mapping is Qwen3's Grouped-Query Attention (GQA) with group_size=2.
//
// Do not reduce this to a one-dimensional sum over `kpos` by evaluating
// `V` at `drop_lsb(r_h)`.  In the MLE, `P` and `V` share the same head
// variable, so the correct product sum must keep both `h` and `kpos` inside
// the sumcheck and multiply by the known `eq(h, r_h)` selector.

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

#[derive(Debug, Clone, Default)]
pub struct PvMatmulWitness {
    pub p: Vec<i32>,
    pub v: Vec<i32>,
    pub acc: Vec<i64>,
    pub output: Vec<i32>,
    pub frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
}

#[derive(Debug, Clone)]
pub struct PvMatmulProof<F: JoltField, T: Transcript> {
    pub pv: PvMatmulRoundRelationProof<F, T>,
    pub(crate) round_lookup: RoundLookupProof<F, T>,
}

#[derive(Debug, Clone)]
pub struct PvMatmulRoundRelationProof<F: JoltField, T: Transcript> {
    pub sumcheck: SumcheckInstanceProof<F, T>,
    pub left_opening: F,
    pub right_opening: F,
    pub remainder_opening: F,
    pub round_bit_opening: F,
}

pub fn prove_pv_matmul_round<F, T>(
    context_claim: Claim<F>,
    witness: &PvMatmulWitness,
    params: &PvMatmulRoundParams,
    transcript: &mut T,
) -> Result<(
    PvMatmulProof<F, T>,
    Claim<F>,
    Claim<F>,
    Claim<F>,
)>
where
    F: JoltField,
    T: Transcript,
{
    let round_witness = RoundWitness::from_input_output(witness.acc.clone(), witness.output.clone());
    let (pv_proof, p, v, round_point, round_bit_opening, remainder_opening) =
        prove_pv_matmul_round_relation(
            context_claim,
            &witness.p,
            &witness.v,
            &round_witness.remainder,
            &round_witness.round_bit,
            &params.pv,
            transcript,
        )?;
    let mut round_accumulator = ProverOpeningAccumulator::new();
    let round_opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(round_point.clone());
    round_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundLut, SumcheckId::NodeExecution(0)),
        (round_opening_point.clone(), round_bit_opening),
    );
    round_accumulator.openings.insert(
        OpeningId::new(
            VirtualPoly::QwenRoundRemainder,
            SumcheckId::NodeExecution(0),
        ),
        (round_opening_point, remainder_opening),
    );
    let round_lookup = prove_round_lookup(
        round_point,
        round_bit_opening,
        remainder_opening,
        padded_lookup_indices(&round_witness.remainder, &params.round.shape),
        round_lut_table(),
        &mut round_accumulator,
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
        point: round_lookup.ra_point.clone(),
        value: round_lookup.ra_opening,
    };

    Ok((
        PvMatmulProof {
            pv: pv_proof,
            round_lookup,
        },
        p,
        v,
        round_ra,
    ))
}

pub fn verify_pv_matmul_round<F, T>(
    context_claim: Claim<F>,
    proof: &PvMatmulProof<F, T>,
    params: &PvMatmulRoundParams,
    transcript: &mut T,
) -> std::result::Result<(Claim<F>, Claim<F>, Claim<F>), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    let (p, v, round_point, round_bit_opening, remainder_opening) =
        verify_pv_matmul_round_relation(context_claim, &proof.pv, &params.pv, transcript)?;
    let round_lookup = verify_round_lookup(
        params.round.shape.padded_power_of_two().numel(),
        round_point,
        round_bit_opening,
        remainder_opening,
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

    Ok((p, v, round_ra))
}

#[allow(dead_code)]
fn prove_pv_matmul_acc<F, T>(
    context_claim: Claim<F>,
    p: &[i32],
    v: &[i32],
    params: &PvMatmulParams,
    transcript: &mut T,
) -> Result<(AttentionMatmulProof<F, T>, Claim<F>, Claim<F>)>
where
    F: JoltField,
    T: Transcript,
{
    validate_pv_inputs(&context_claim, p, v, params)?;
    let (r_q, r_h, r_d) = split3(
        &context_claim.point,
        params.seq,
        params.q_heads,
        params.head_dim,
    );
    let left = partial_p(p, params, r_q);
    let right = partial_v(v, params, r_d);
    let selector = expanded_head_eq(r_h, params);
    let (proof, r_h_kpos) =
        prove_pv_product(context_claim.value, left, right, selector, transcript)?;
    let (r_head, r_kpos) = r_h_kpos.split_at(log2_ceil(params.q_heads));
    let kv_h = drop_gqa_lsb_for_dims(r_head, params.q_heads, params.kv_heads)?;
    let p_opening = proof.left_opening;
    let v_opening = proof.right_opening;

    Ok((
        proof,
        Claim {
            tensor: params.p_tensor.clone(),
            logical_shape: params.p_shape(),
            domain_shape: params.p_shape().padded_power_of_two(),
            point: [r_head, r_q, r_kpos].concat(),
            value: p_opening,
        },
        Claim {
            tensor: params.v_tensor.clone(),
            logical_shape: params.v_shape(),
            domain_shape: params.v_shape().padded_power_of_two(),
            point: [r_kpos, kv_h, r_d].concat(),
            value: v_opening,
        },
    ))
}

#[allow(dead_code)]
fn verify_pv_matmul_acc<F, T>(
    context_claim: Claim<F>,
    proof: &AttentionMatmulProof<F, T>,
    params: &PvMatmulParams,
    transcript: &mut T,
) -> std::result::Result<(Claim<F>, Claim<F>), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    verify_pv_inputs(&context_claim, params)?;
    let (r_q, r_h, r_d) = split3(
        &context_claim.point,
        params.seq,
        params.q_heads,
        params.head_dim,
    );
    let r_h_kpos = verify_pv_product(context_claim.value, proof, r_h, params, transcript)?;
    let (r_head, r_kpos) = r_h_kpos.split_at(log2_ceil(params.q_heads));
    let kv_h = drop_gqa_lsb_for_dims_verify(r_head, params.q_heads, params.kv_heads)?;

    Ok((
        Claim {
            tensor: params.p_tensor.clone(),
            logical_shape: params.p_shape(),
            domain_shape: params.p_shape().padded_power_of_two(),
            point: [r_head, r_q, r_kpos].concat(),
            value: proof.left_opening,
        },
        Claim {
            tensor: params.v_tensor.clone(),
            logical_shape: params.v_shape(),
            domain_shape: params.v_shape().padded_power_of_two(),
            point: [r_kpos, kv_h, r_d].concat(),
            value: proof.right_opening,
        },
    ))
}

#[allow(clippy::type_complexity)]
fn prove_pv_matmul_round_relation<F, T>(
    context_claim: Claim<F>,
    p: &[i32],
    v: &[i32],
    remainder: &[usize],
    round_bit: &[i32],
    params: &PvMatmulParams,
    transcript: &mut T,
) -> Result<(PvMatmulRoundRelationProof<F, T>, Claim<F>, Claim<F>, Vec<F>, F, F)>
where
    F: JoltField,
    T: Transcript,
{
    validate_pv_inputs(&context_claim, p, v, params)?;
    validate_round_witness(remainder, round_bit, &params.context_shape())?;
    let (r_q, r_h, r_d) = split3(
        &context_claim.point,
        params.seq,
        params.q_heads,
        params.head_dim,
    );
    let left = partial_p(p, params, r_q);
    let right = partial_v(v, params, r_d);
    let selector = expanded_head_eq(r_h, params);
    let remainder = eval_round_tensor(remainder, &params.context_shape(), &context_claim.point);
    let round_bit = eval_i32_tensor(round_bit, &params.context_shape(), &context_claim.point);
    let (proof, r_h_kpos) = prove_pv_round_product(
        context_claim.value * scale_q8::<F>(),
        left,
        right,
        selector,
        remainder,
        round_bit,
        transcript,
    )?;
    let (r_head, r_kpos) = r_h_kpos.split_at(log2_ceil(params.q_heads));
    let kv_h = drop_gqa_lsb_for_dims(r_head, params.q_heads, params.kv_heads)?;
    let p_opening = proof.left_opening;
    let v_opening = proof.right_opening;
    let remainder_opening = proof.remainder_opening;
    let round_bit_opening = proof.round_bit_opening;

    Ok((
        proof,
        Claim {
            tensor: params.p_tensor.clone(),
            logical_shape: params.p_shape(),
            domain_shape: params.p_shape().padded_power_of_two(),
            point: [r_head, r_q, r_kpos].concat(),
            value: p_opening,
        },
        Claim {
            tensor: params.v_tensor.clone(),
            logical_shape: params.v_shape(),
            domain_shape: params.v_shape().padded_power_of_two(),
            point: [r_kpos, kv_h, r_d].concat(),
            value: v_opening,
        },
        context_claim.point,
        round_bit_opening,
        remainder_opening,
    ))
}

#[allow(clippy::type_complexity)]
fn verify_pv_matmul_round_relation<F, T>(
    context_claim: Claim<F>,
    proof: &PvMatmulRoundRelationProof<F, T>,
    params: &PvMatmulParams,
    transcript: &mut T,
) -> std::result::Result<(Claim<F>, Claim<F>, Vec<F>, F, F), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    verify_pv_inputs(&context_claim, params)?;
    let (r_q, r_h, r_d) = split3(
        &context_claim.point,
        params.seq,
        params.q_heads,
        params.head_dim,
    );
    let r_h_kpos = verify_pv_round_product(
        context_claim.value * scale_q8::<F>(),
        proof,
        r_h,
        params,
        transcript,
    )?;
    let (r_head, r_kpos) = r_h_kpos.split_at(log2_ceil(params.q_heads));
    let kv_h = drop_gqa_lsb_for_dims_verify(r_head, params.q_heads, params.kv_heads)?;

    Ok((
        Claim {
            tensor: params.p_tensor.clone(),
            logical_shape: params.p_shape(),
            domain_shape: params.p_shape().padded_power_of_two(),
            point: [r_head, r_q, r_kpos].concat(),
            value: proof.left_opening,
        },
        Claim {
            tensor: params.v_tensor.clone(),
            logical_shape: params.v_shape(),
            domain_shape: params.v_shape().padded_power_of_two(),
            point: [r_kpos, kv_h, r_d].concat(),
            value: proof.right_opening,
        },
        context_claim.point,
        proof.round_bit_opening,
        proof.remainder_opening,
    ))
}

fn partial_p<F: JoltField>(p: &[i32], params: &PvMatmulParams, r_q: &[F]) -> Vec<F> {
    let head_pad = params.q_heads.next_power_of_two();
    let seq_pad = params.seq.next_power_of_two();
    let q_eq = EqPolynomial::<F>::evals(r_q);
    let mut out = vec![F::zero(); head_pad * seq_pad];
    for head in 0..params.q_heads {
        for kpos in 0..params.seq {
            out[head * seq_pad + kpos] = (0..params.seq)
                .map(|qpos| {
                    q_eq[qpos] * F::from_i32(p[(head * params.seq + qpos) * params.seq + kpos])
                })
                .sum();
        }
    }
    out
}

fn partial_v<F: JoltField>(v: &[i32], params: &PvMatmulParams, r_d: &[F]) -> Vec<F> {
    let head_pad = params.q_heads.next_power_of_two();
    let seq_pad = params.seq.next_power_of_two();
    let dim_eq = EqPolynomial::<F>::evals(r_d);
    let mut out = vec![F::zero(); head_pad * seq_pad];
    for head in 0..params.q_heads {
        let kv_head = head / QWEN3_GQA_GROUP_SIZE;
        for kpos in 0..params.seq {
            out[head * seq_pad + kpos] = (0..params.head_dim)
                .map(|d| {
                    dim_eq[d]
                        * F::from_i32(v[(kpos * params.kv_heads + kv_head) * params.head_dim + d])
                })
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

#[allow(dead_code)]
fn prove_pv_product<F, T>(
    input_claim: F,
    left: Vec<F>,
    right: Vec<F>,
    selector: Vec<F>,
    transcript: &mut T,
) -> Result<(AttentionMatmulProof<F, T>, Vec<F>)>
where
    F: JoltField,
    T: Transcript,
{
    let params = PvProductParams::new(
        left.len().trailing_zeros() as usize,
        input_claim,
        Vec::new(),
    );
    let mut prover = PvProductProver::new(params, left, right, selector);
    let mut accumulator = ProverOpeningAccumulator::new();
    let (sumcheck, challenges) = Sumcheck::prove(&mut prover, &mut accumulator, transcript);
    let left_opening = opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenAttentionLeft, pv_sumcheck_id()),
    )?;
    let right_opening = opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenAttentionRight, pv_sumcheck_id()),
    )?;
    let r = normalize_sumcheck_point::<F>(&challenges.into_opening());
    Ok((
        AttentionMatmulProof {
            sumcheck,
            left_opening,
            right_opening,
        },
        r,
    ))
}

#[allow(dead_code)]
fn verify_pv_product<F, T>(
    input_claim: F,
    proof: &AttentionMatmulProof<F, T>,
    output_head_point: &[F],
    params: &PvMatmulParams,
    transcript: &mut T,
) -> std::result::Result<Vec<F>, ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    let num_rounds = log2_ceil(params.q_heads) + log2_ceil(params.seq);
    let verifier = PvProductVerifier {
        params: PvProductParams::new(num_rounds, input_claim, output_head_point.to_vec()),
    };
    let mut accumulator = VerifierOpeningAccumulator::new();
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenAttentionLeft, pv_sumcheck_id()),
        (OpeningPoint::<BIG_ENDIAN, F>::default(), proof.left_opening),
    );
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenAttentionRight, pv_sumcheck_id()),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.right_opening,
        ),
    );
    let challenges = Sumcheck::verify(&proof.sumcheck, &verifier, &mut accumulator, transcript)?;
    Ok(normalize_sumcheck_point::<F>(&challenges.into_opening()))
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
    let mut prover =
        PvRoundProductProver::new(params, left, right, selector, remainder, round_bit);
    let mut accumulator = ProverOpeningAccumulator::new();
    let (sumcheck, challenges) = Sumcheck::prove(&mut prover, &mut accumulator, transcript);
    let left_opening = opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenAttentionLeft, pv_sumcheck_id()),
    )?;
    let right_opening = opening(
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
    let r = normalize_sumcheck_point::<F>(&challenges.into_opening());
    Ok((
        PvMatmulRoundRelationProof {
            sumcheck,
            left_opening,
            right_opening,
            remainder_opening,
            round_bit_opening,
        },
        r,
    ))
}

fn verify_pv_round_product<F, T>(
    input_claim: F,
    proof: &PvMatmulRoundRelationProof<F, T>,
    output_head_point: &[F],
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
        (OpeningPoint::<BIG_ENDIAN, F>::default(), proof.left_opening),
    );
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenAttentionRight, pv_sumcheck_id()),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.right_opening,
        ),
    );
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundRemainder, pv_sumcheck_id()),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.remainder_opening,
        ),
    );
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundLut, pv_sumcheck_id()),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.round_bit_opening,
        ),
    );
    let challenges = Sumcheck::verify(&proof.sumcheck, &verifier, &mut accumulator, transcript)?;
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

#[allow(dead_code)]
struct PvProductProver<F: JoltField> {
    left: MultilinearPolynomial<F>,
    right: MultilinearPolynomial<F>,
    selector: MultilinearPolynomial<F>,
    params: PvProductParams<F>,
}

impl<F: JoltField> PvProductProver<F> {
    fn new(params: PvProductParams<F>, left: Vec<F>, right: Vec<F>, selector: Vec<F>) -> Self {
        Self {
            left: MultilinearPolynomial::from(left),
            right: MultilinearPolynomial::from(right),
            selector: MultilinearPolynomial::from(selector),
            params,
        }
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
                (l1 + l1 - l0) * (r1 + r1 - r0) * (s1 + s1 - s0)
                    + (z1 + z1 - z0) * round_term;
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
            OpeningPoint::<BIG_ENDIAN, F>::default(),
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

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for PvProductProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let mut evals = [F::zero(); 3];
        for g in 0..self.left.len() / 2 {
            let l0 = self.left.get_bound_coeff(2 * g);
            let l1 = self.left.get_bound_coeff(2 * g + 1);
            let r0 = self.right.get_bound_coeff(2 * g);
            let r1 = self.right.get_bound_coeff(2 * g + 1);
            let s0 = self.selector.get_bound_coeff(2 * g);
            let s1 = self.selector.get_bound_coeff(2 * g + 1);
            evals[0] += l0 * r0 * s0;
            evals[1] += (l1 + l1 - l0) * (r1 + r1 - r0) * (s1 + s1 - s0);
            evals[2] += (l1 * F::from_u64(3) - l0 - l0)
                * (r1 * F::from_u64(3) - r0 - r0)
                * (s1 * F::from_u64(3) - s0 - s0);
        }
        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.left.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.right.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.selector.bind_parallel(r_j, BindingOrder::LowToHigh);
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
            point,
            self.right.final_claim(),
        );
    }
}

#[allow(dead_code)]
struct PvProductVerifier<F: JoltField> {
    params: PvProductParams<F>,
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
            point,
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRoundRemainder, pv_sumcheck_id()),
            OpeningPoint::<BIG_ENDIAN, F>::default(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRoundLut, pv_sumcheck_id()),
            OpeningPoint::<BIG_ENDIAN, F>::default(),
        );
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for PvProductVerifier<F> {
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
        let point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening())
            .r;
        let head_vars = self.params.output_head_point.len();
        let selector = EqPolynomial::<F>::mle(&self.params.output_head_point, &point[..head_vars]);
        left * right * selector
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
            point,
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

fn pv_sumcheck_id() -> SumcheckId {
    SumcheckId::NodeExecution(0)
}

fn validate_pv_inputs<F: JoltField>(
    context_claim: &Claim<F>,
    p: &[i32],
    v: &[i32],
    params: &PvMatmulParams,
) -> Result<()> {
    validate_gqa(params.q_heads, params.kv_heads)?;
    ensure_len("P", &params.p_shape(), p.len())?;
    ensure_len("V", &params.v_shape(), v.len())?;
    validate_claim(context_claim, &params.context_shape())
}

fn verify_pv_inputs<F: JoltField>(
    context_claim: &Claim<F>,
    params: &PvMatmulParams,
) -> std::result::Result<(), ProofVerifyError> {
    if params.q_heads / params.kv_heads != QWEN3_GQA_GROUP_SIZE {
        return Err(ProofVerifyError::InvalidInputLength(
            QWEN3_GQA_GROUP_SIZE,
            params.q_heads / params.kv_heads,
        ));
    }
    verify_claim(context_claim, &params.context_shape())
}

fn validate_round_witness(remainder: &[usize], round_bit: &[i32], shape: &Shape) -> Result<()> {
    let expected = shape.numel();
    if remainder.len() != expected {
        return Err(ProverError::TensorLenMismatch {
            name: "pv round remainder",
            shape: shape.0.clone(),
            expected,
            actual: remainder.len(),
        });
    }
    if round_bit.len() != expected {
        return Err(ProverError::TensorLenMismatch {
            name: "pv round bit",
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

fn eval_i32_tensor<F: JoltField>(values: &[i32], shape: &Shape, point: &[F]) -> F {
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

fn scale_q8<F: JoltField>() -> F {
    F::from_u64(256)
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use joltworks::transcripts::Blake2bTranscript;

    use super::*;

    #[test]
    fn proves_and_verifies_pv_matmul() {
        let pv_params = PvMatmulParams::new(2, 2, 1, 2, "P", "V");
        let params = PvMatmulRoundParams::new(
            RoundParams::new(vec![2, 2, 2], "C_acc", "C"),
            pv_params.clone(),
        );
        let p = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let v = vec![2, 1, 4, 3];
        let context = pv_context(&p, &v, &pv_params);
        let acc = context
            .iter()
            .map(|&value| i64::from(value))
            .collect::<Vec<_>>();
        let output = round_outputs(&acc);
        let frac_bits = frac_bits_for(&acc);
        let point = vec![Fr::from(3u64), Fr::from(5u64), Fr::from(7u64)];
        let claim = Claim {
            tensor: TensorId::new("C"),
            logical_shape: pv_params.context_shape(),
            domain_shape: pv_params.context_shape().padded_power_of_two(),
            value: eval_tensor(&output, &pv_params.context_shape(), &point),
            point,
        };
        let witness = PvMatmulWitness {
            p,
            v,
            acc,
            output,
            frac_bits,
        };

        let mut prover_transcript = Blake2bTranscript::default();
        let (proof, p_claim, v_claim, _) = prove_pv_matmul_round::<Fr, _>(
            claim.clone(),
            &witness,
            &params,
            &mut prover_transcript,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        let (verified_p, verified_v, _) =
            verify_pv_matmul_round::<Fr, _>(claim, &proof, &params, &mut verifier_transcript)
                .unwrap();

        assert_eq!(verified_p, p_claim);
        assert_eq!(verified_v, v_claim);
        assert_eq!(verified_p.tensor.0, "P");
        assert_eq!(verified_v.tensor.0, "V");
    }

    #[test]
    fn proves_and_verifies_pv_matmul_non_power_seq() {
        let pv_params = PvMatmulParams::new(3, 2, 1, 2, "P", "V");
        let params = PvMatmulRoundParams::new(
            RoundParams::new(vec![3, 2, 2], "C_acc", "C"),
            pv_params.clone(),
        );
        let p = (1..=18).collect::<Vec<_>>();
        let v = (2..=7).collect::<Vec<_>>();
        let context = pv_context(&p, &v, &pv_params);
        let acc = context
            .iter()
            .map(|&value| i64::from(value))
            .collect::<Vec<_>>();
        let output = round_outputs(&acc);
        let frac_bits = frac_bits_for(&acc);
        let point = vec![
            Fr::from(3u64),
            Fr::from(5u64),
            Fr::from(7u64),
            Fr::from(11u64),
        ];
        let claim = Claim {
            tensor: TensorId::new("C"),
            logical_shape: pv_params.context_shape(),
            domain_shape: pv_params.context_shape().padded_power_of_two(),
            value: eval_tensor(&output, &pv_params.context_shape(), &point),
            point,
        };
        let witness = PvMatmulWitness {
            p,
            v,
            acc,
            output,
            frac_bits,
        };

        let mut prover_transcript = Blake2bTranscript::default();
        let (proof, p_claim, v_claim, _) = prove_pv_matmul_round::<Fr, _>(
            claim.clone(),
            &witness,
            &params,
            &mut prover_transcript,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        let (verified_p, verified_v, _) =
            verify_pv_matmul_round::<Fr, _>(claim, &proof, &params, &mut verifier_transcript)
                .unwrap();

        assert_eq!(verified_p, p_claim);
        assert_eq!(verified_v, v_claim);
    }

    fn pv_context(p: &[i32], v: &[i32], params: &PvMatmulParams) -> Vec<i32> {
        let mut out = vec![0; params.seq * params.q_heads * params.head_dim];
        for qpos in 0..params.seq {
            for h in 0..params.q_heads {
                for d in 0..params.head_dim {
                    let kvh = h / QWEN3_GQA_GROUP_SIZE;
                    let mut sum = 0;
                    for kpos in 0..params.seq {
                        sum += p[(h * params.seq + qpos) * params.seq + kpos]
                            * v[(kpos * params.kv_heads + kvh) * params.head_dim + d];
                    }
                    out[(qpos * params.q_heads + h) * params.head_dim + d] = sum;
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
