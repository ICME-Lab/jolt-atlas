use common::VirtualPoly;
// Design note for future us:
//
// Qwen3 uses the `rotate_half` RoPE layout.  The head dimension is split into
// two halves, so the 2x2 rotation pairs `(x[d], x[d + head_dim / 2])`:
//     Y[..., d]          = X[..., d]          * cos - X[..., d + half] * sin
//     Y[..., d + half]   = X[..., d + half]   * cos + X[..., d]        * sin
//
// It is tempting to verify one output claim using only
// `X(r_seq,r_head,r_pair,0)`, `X(...,1)`, `cos(r_seq,r_pair)`, and
// `sin(r_seq,r_pair)`.  That is not sound in general because cos/sin are
// position/pair dependent:
//     MLE(X * cos)(r) != MLE(X)(r) * MLE(cos)(r).
//
// Therefore this op uses one sumcheck over `(seq, head, pair)`.  The parity
// coordinate from the incoming output claim is kept as a scalar `r_b`, producing
// known coefficient tables:
//     c_even = (1-r_b) * cos + r_b * sin
//     c_odd  = -(1-r_b) * sin + r_b * cos
//
// The returned claims are on the original input tensor at the sumcheck point,
// with only the parity bit fixed:
//     X(r_seq', r_head', 0, r_pair')
//     X(r_seq', r_head', 1, r_pair')
//
// The parity coordinate is the most significant bit of the final tensor axis:
// `[seq, head, half_selector, pair]`.  This matches the physical
// `[0..half, half..head_dim]` tensor layout used by qwen3-awy's trace.
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
        ROUND_FRAC_BITS, RoundParams, RoundProof, RoundWitness, prove_round, verify_round,
    },
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RopeParams {
    pub seq: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub x_tensor: TensorId,
}

impl RopeParams {
    pub fn new(seq: usize, heads: usize, head_dim: usize, x_tensor: impl Into<String>) -> Self {
        Self {
            seq,
            heads,
            head_dim,
            x_tensor: TensorId::new(x_tensor),
        }
    }

    pub fn tensor_shape(&self) -> Shape {
        Shape::new(vec![self.seq, self.heads, self.head_dim])
    }

    pub fn coeff_shape(&self) -> Shape {
        Shape::new(vec![self.seq, self.head_dim / 2])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RopeRoundParams {
    pub round: RoundParams,
    pub rope: RopeParams,
}

impl RopeRoundParams {
    pub fn new(round: RoundParams, rope: RopeParams) -> Self {
        Self { round, rope }
    }
}

#[derive(Debug, Clone, Default)]
pub struct RopeWitness {
    pub input: Vec<i32>,
    pub acc: Vec<i64>,
    pub output: Vec<i32>,
    pub frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
}

#[derive(Debug, Clone)]
pub struct RopeProof<F: JoltField, T: Transcript> {
    pub round: RoundProof<F, T>,
    pub rope: RopeSumcheckProof<F, T>,
}

#[derive(Debug, Clone)]
pub struct RopeSumcheckProof<F: JoltField, T: Transcript> {
    pub sumcheck: SumcheckInstanceProof<F, T>,
    pub x_even_opening: F,
    pub x_odd_opening: F,
}

pub fn prove_rope_round<F, T>(
    y_claim: Claim<F>,
    witness: &RopeWitness,
    cos: &[i32],
    sin: &[i32],
    params: &RopeRoundParams,
    transcript: &mut T,
) -> Result<(
    RopeProof<F, T>,
    Claim<F>,
    Claim<F>,
    [Claim<F>; ROUND_FRAC_BITS],
)>
where
    F: JoltField,
    T: Transcript,
{
    let round_witness = RoundWitness {
        input: witness.acc.clone(),
        output: witness.output.clone(),
        frac_bits: witness.frac_bits.clone(),
    };
    let (round_proof, acc_claim, frac_bits) =
        prove_round(vec![y_claim], &round_witness, &params.round, transcript)?;
    let (rope_proof, x_even, x_odd) = prove_rope_acc(
        acc_claim,
        &witness.input,
        cos,
        sin,
        &params.rope,
        transcript,
    )?;

    Ok((
        RopeProof {
            round: round_proof,
            rope: rope_proof,
        },
        x_even,
        x_odd,
        frac_bits,
    ))
}

pub fn verify_rope_round<F, T>(
    y_claim: Claim<F>,
    proof: &RopeProof<F, T>,
    cos: &[i32],
    sin: &[i32],
    params: &RopeRoundParams,
    transcript: &mut T,
) -> std::result::Result<(Claim<F>, Claim<F>, [Claim<F>; ROUND_FRAC_BITS]), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    let (acc_claim, frac_bits) =
        verify_round(vec![y_claim], &proof.round, &params.round, transcript)?;
    let (x_even, x_odd) =
        verify_rope_acc(acc_claim, &proof.rope, cos, sin, &params.rope, transcript)?;

    Ok((x_even, x_odd, frac_bits))
}

fn prove_rope_acc<F, T>(
    y_claim: Claim<F>,
    x: &[i32],
    cos: &[i32],
    sin: &[i32],
    params: &RopeParams,
    transcript: &mut T,
) -> Result<(RopeSumcheckProof<F, T>, Claim<F>, Claim<F>)>
where
    F: JoltField,
    T: Transcript,
{
    validate_inputs(&y_claim, x, cos, sin, params)?;
    let (_, _, r_parity, _) = split_rope_point(&y_claim.point, params);
    let sc_params = RopeSumcheckParams::new(
        rope_domain_vars(params),
        y_claim.value,
        y_claim.point.clone(),
        r_parity[0],
        params.clone(),
        cos.to_vec(),
        sin.to_vec(),
    );
    let x_even_poly = rope_x_parity_poly(x, params, 0);
    let x_odd_poly = rope_x_parity_poly(x, params, 1);
    let cos_poly = rope_coeff_poly(cos, params);
    let sin_poly = rope_coeff_poly(sin, params);
    let mut prover =
        RopeSumcheckProver::new(sc_params, x_even_poly, x_odd_poly, cos_poly, sin_poly);
    let mut accumulator = ProverOpeningAccumulator::new();
    let (sumcheck, challenges) = Sumcheck::prove(&mut prover, &mut accumulator, transcript);
    let x_even = opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenRopeXEven, rope_sumcheck_id()),
    )?;
    let x_odd = opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenRopeXOdd, rope_sumcheck_id()),
    )?;
    let rope_point = normalize_sumcheck_point::<F>(&challenges.into_opening());
    let (r_seq, r_head, r_pair) = split_rope_domain_point(&rope_point, params);
    let even_point = [r_seq, r_head, &[F::zero()], r_pair].concat();
    let odd_point = [r_seq, r_head, &[F::one()], r_pair].concat();

    Ok((
        RopeSumcheckProof {
            sumcheck,
            x_even_opening: x_even,
            x_odd_opening: x_odd,
        },
        Claim {
            tensor: params.x_tensor.clone(),
            logical_shape: params.tensor_shape(),
            domain_shape: params.tensor_shape().padded_power_of_two(),
            point: even_point,
            value: x_even,
        },
        Claim {
            tensor: params.x_tensor.clone(),
            logical_shape: params.tensor_shape(),
            domain_shape: params.tensor_shape().padded_power_of_two(),
            point: odd_point,
            value: x_odd,
        },
    ))
}

fn verify_rope_acc<F, T>(
    y_claim: Claim<F>,
    proof: &RopeSumcheckProof<F, T>,
    cos: &[i32],
    sin: &[i32],
    params: &RopeParams,
    transcript: &mut T,
) -> std::result::Result<(Claim<F>, Claim<F>), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    validate_claim_and_coeffs(&y_claim, cos, sin, params)
        .map_err(|err| ProofVerifyError::InvalidOpeningProof(format!("{err}")))?;
    let (_, _, r_parity, _) = split_rope_point(&y_claim.point, params);
    let sc_params = RopeSumcheckParams::new(
        rope_domain_vars(params),
        y_claim.value,
        y_claim.point.clone(),
        r_parity[0],
        params.clone(),
        cos.to_vec(),
        sin.to_vec(),
    );
    let verifier = RopeSumcheckVerifier { params: sc_params };
    let mut accumulator = VerifierOpeningAccumulator::new();
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRopeXEven, rope_sumcheck_id()),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.x_even_opening,
        ),
    );
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRopeXOdd, rope_sumcheck_id()),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.x_odd_opening,
        ),
    );
    let challenges = Sumcheck::verify(&proof.sumcheck, &verifier, &mut accumulator, transcript)?;
    let rope_point = normalize_sumcheck_point::<F>(&challenges.into_opening());
    let (r_seq, r_head, r_pair) = split_rope_domain_point(&rope_point, params);
    let even_point = [r_seq, r_head, &[F::zero()], r_pair].concat();
    let odd_point = [r_seq, r_head, &[F::one()], r_pair].concat();

    Ok((
        Claim {
            tensor: params.x_tensor.clone(),
            logical_shape: params.tensor_shape(),
            domain_shape: params.tensor_shape().padded_power_of_two(),
            point: even_point,
            value: proof.x_even_opening,
        },
        Claim {
            tensor: params.x_tensor.clone(),
            logical_shape: params.tensor_shape(),
            domain_shape: params.tensor_shape().padded_power_of_two(),
            point: odd_point,
            value: proof.x_odd_opening,
        },
    ))
}

struct RopeSumcheckParams<F: JoltField> {
    num_rounds: usize,
    input_claim: F,
    y_point: Vec<F>,
    parity: F,
    op: RopeParams,
    cos: Vec<i32>,
    sin: Vec<i32>,
}

impl<F: JoltField> RopeSumcheckParams<F> {
    fn new(
        num_rounds: usize,
        input_claim: F,
        y_point: Vec<F>,
        parity: F,
        op: RopeParams,
        cos: Vec<i32>,
        sin: Vec<i32>,
    ) -> Self {
        Self {
            num_rounds,
            input_claim,
            y_point,
            parity,
            op,
            cos,
            sin,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for RopeSumcheckParams<F> {
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

struct RopeSumcheckProver<F: JoltField> {
    eq_y: MultilinearPolynomial<F>,
    x_even: MultilinearPolynomial<F>,
    x_odd: MultilinearPolynomial<F>,
    c_even: MultilinearPolynomial<F>,
    c_odd: MultilinearPolynomial<F>,
    params: RopeSumcheckParams<F>,
}

impl<F: JoltField> RopeSumcheckProver<F> {
    fn new(
        params: RopeSumcheckParams<F>,
        x_even: Vec<F>,
        x_odd: Vec<F>,
        cos: Vec<F>,
        sin: Vec<F>,
    ) -> Self {
        let rope_point = rope_domain_point(&params.y_point, &params.op);
        let eq_y = MultilinearPolynomial::from(EqPolynomial::<F>::evals(&rope_point));
        let c_even: Vec<F> = cos
            .iter()
            .zip(&sin)
            .map(|(&cos, &sin)| (F::one() - params.parity) * cos + params.parity * sin)
            .collect();
        let c_odd: Vec<F> = cos
            .iter()
            .zip(&sin)
            .map(|(&cos, &sin)| {
                (F::zero() - (F::one() - params.parity)) * sin + params.parity * cos
            })
            .collect();
        Self {
            eq_y,
            x_even: MultilinearPolynomial::from(x_even),
            x_odd: MultilinearPolynomial::from(x_odd),
            c_even: MultilinearPolynomial::from(c_even),
            c_odd: MultilinearPolynomial::from(c_odd),
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for RopeSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let mut evals = [F::zero(); 3];
        for g in 0..self.x_even.len() / 2 {
            let e = [
                self.eq_y.get_bound_coeff(2 * g),
                self.eq_y.get_bound_coeff(2 * g + 1),
            ];
            let xe = [
                self.x_even.get_bound_coeff(2 * g),
                self.x_even.get_bound_coeff(2 * g + 1),
            ];
            let xo = [
                self.x_odd.get_bound_coeff(2 * g),
                self.x_odd.get_bound_coeff(2 * g + 1),
            ];
            let ce = [
                self.c_even.get_bound_coeff(2 * g),
                self.c_even.get_bound_coeff(2 * g + 1),
            ];
            let co = [
                self.c_odd.get_bound_coeff(2 * g),
                self.c_odd.get_bound_coeff(2 * g + 1),
            ];
            for (idx, t) in [F::zero(), F::from_u64(2), F::from_u64(3)]
                .into_iter()
                .enumerate()
            {
                let e_t = lerp(e[0], e[1], t);
                evals[idx] += e_t
                    * (lerp(xe[0], xe[1], t) * lerp(ce[0], ce[1], t)
                        + lerp(xo[0], xo[1], t) * lerp(co[0], co[1], t));
            }
        }
        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_y.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.x_even.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.x_odd.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.c_even.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.c_odd.bind_parallel(r_j, BindingOrder::LowToHigh);
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
            OpeningId::new(VirtualPoly::QwenRopeXEven, rope_sumcheck_id()),
            point.clone(),
            self.x_even.final_claim(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRopeXOdd, rope_sumcheck_id()),
            point,
            self.x_odd.final_claim(),
        );
    }
}

struct RopeSumcheckVerifier<F: JoltField> {
    params: RopeSumcheckParams<F>,
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for RopeSumcheckVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let x_even = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenRopeXEven,
                rope_sumcheck_id(),
            ))
            .1;
        let x_odd = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenRopeXOdd,
                rope_sumcheck_id(),
            ))
            .1;
        let point = normalize_sumcheck_point::<F>(&sumcheck_challenges.into_opening());
        let rope_y_point = rope_domain_point(&self.params.y_point, &self.params.op);
        let coeff_point = rope_coeff_point(&point, &self.params.op);
        let cos = eval_tensor_at_point(
            &self.params.cos,
            &self.params.op.coeff_shape(),
            &coeff_point,
        );
        let sin = eval_tensor_at_point(
            &self.params.sin,
            &self.params.op.coeff_shape(),
            &coeff_point,
        );
        let c_even = (F::one() - self.params.parity) * cos + self.params.parity * sin;
        let c_odd = (F::zero() - (F::one() - self.params.parity)) * sin + self.params.parity * cos;
        EqPolynomial::mle(&rope_y_point, &point) * (x_even * c_even + x_odd * c_odd)
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
            OpeningId::new(VirtualPoly::QwenRopeXEven, rope_sumcheck_id()),
            point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRopeXOdd, rope_sumcheck_id()),
            point,
        );
    }
}

fn validate_inputs<F: JoltField>(
    y_claim: &Claim<F>,
    x: &[i32],
    cos: &[i32],
    sin: &[i32],
    params: &RopeParams,
) -> Result<()> {
    validate_claim_and_coeffs(y_claim, cos, sin, params)?;
    ensure_len("X", &params.tensor_shape(), x.len())?;
    Ok(())
}

fn validate_claim_and_coeffs<F: JoltField>(
    y_claim: &Claim<F>,
    cos: &[i32],
    sin: &[i32],
    params: &RopeParams,
) -> Result<()> {
    if params.seq == 0 || params.heads == 0 || params.head_dim == 0 || params.head_dim % 2 != 0 {
        return Err(ProverError::InvalidTensorShape(params.tensor_shape().0));
    }
    ensure_len("cos", &params.coeff_shape(), cos.len())?;
    ensure_len("sin", &params.coeff_shape(), sin.len())?;
    if y_claim.logical_shape != params.tensor_shape() {
        return Err(ProverError::ShapeMismatch {
            name: "RoPE Y claim",
            expected: params.tensor_shape().0,
            actual: y_claim.logical_shape.0.clone(),
        });
    }
    let expected_domain = params.tensor_shape().padded_power_of_two();
    if y_claim.domain_shape != expected_domain {
        return Err(ProverError::ShapeMismatch {
            name: "RoPE Y claim domain",
            expected: expected_domain.0,
            actual: y_claim.domain_shape.0.clone(),
        });
    }
    let expected_point_len = y_claim.domain_shape.point_len();
    if y_claim.point.len() != expected_point_len {
        return Err(ProverError::ShapeMismatch {
            name: "RoPE Y claim point",
            expected: vec![expected_point_len],
            actual: vec![y_claim.point.len()],
        });
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

fn rope_x_parity_poly<F: JoltField>(x: &[i32], params: &RopeParams, parity: usize) -> Vec<F> {
    let seq_pad = params.seq.next_power_of_two();
    let heads_pad = params.heads.next_power_of_two();
    let pair_pad = (params.head_dim / 2).next_power_of_two();
    let mut out = vec![F::zero(); seq_pad * heads_pad * pair_pad];
    for s in 0..params.seq {
        for h in 0..params.heads {
            for pair in 0..params.head_dim / 2 {
                let out_idx = (s * heads_pad + h) * pair_pad + pair;
                let x_idx = (s * params.heads + h) * params.head_dim
                    + parity * (params.head_dim / 2)
                    + pair;
                out[out_idx] = F::from_i32(x[x_idx]);
            }
        }
    }
    out
}

fn rope_coeff_poly<F: JoltField>(values: &[i32], params: &RopeParams) -> Vec<F> {
    let seq_pad = params.seq.next_power_of_two();
    let heads_pad = params.heads.next_power_of_two();
    let pair_pad = (params.head_dim / 2).next_power_of_two();
    let mut out = vec![F::zero(); seq_pad * heads_pad * pair_pad];
    for s in 0..params.seq {
        for h in 0..params.heads {
            for pair in 0..params.head_dim / 2 {
                let out_idx = (s * heads_pad + h) * pair_pad + pair;
                let coeff_idx = s * (params.head_dim / 2) + pair;
                out[out_idx] = F::from_i32(values[coeff_idx]);
            }
        }
    }
    out
}

fn rope_domain_point<F: JoltField>(point: &[F], params: &RopeParams) -> Vec<F> {
    let (r_seq, r_head, _, r_pair) = split_rope_point(point, params);
    [r_seq, r_head, r_pair].concat()
}

fn rope_coeff_point<F: JoltField>(rope_point: &[F], params: &RopeParams) -> Vec<F> {
    let (r_seq, _, r_pair) = split_rope_domain_point(rope_point, params);
    [r_seq, r_pair].concat()
}

fn split_rope_domain_point<'a, F>(
    point: &'a [F],
    params: &RopeParams,
) -> (&'a [F], &'a [F], &'a [F]) {
    let seq_vars = log2_ceil(params.seq);
    let head_vars = log2_ceil(params.heads);
    let r_seq = &point[..seq_vars];
    let r_head = &point[seq_vars..seq_vars + head_vars];
    let r_pair = &point[seq_vars + head_vars..];
    (r_seq, r_head, r_pair)
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

fn rope_domain_vars(params: &RopeParams) -> usize {
    log2_ceil(params.seq) + log2_ceil(params.heads) + log2_ceil(params.head_dim / 2)
}

fn rope_sumcheck_id() -> SumcheckId {
    SumcheckId::NodeExecution(0)
}

fn split_rope_point<'a, F>(
    point: &'a [F],
    params: &RopeParams,
) -> (&'a [F], &'a [F], &'a [F], &'a [F]) {
    let seq_vars = log2_ceil(params.seq);
    let head_vars = log2_ceil(params.heads);
    let pair_vars = log2_ceil(params.head_dim / 2);
    let r_seq = &point[..seq_vars];
    let r_head = &point[seq_vars..seq_vars + head_vars];
    let r_parity = &point[seq_vars + head_vars..seq_vars + head_vars + 1];
    let r_pair = &point[seq_vars + head_vars + 1..seq_vars + head_vars + 1 + pair_vars];
    debug_assert_eq!(r_parity.len(), 1);
    (r_seq, r_head, r_parity, r_pair)
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

fn log2_ceil(value: usize) -> usize {
    value.next_power_of_two().trailing_zeros() as usize
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use joltworks::transcripts::Blake2bTranscript;

    use super::*;

    #[test]
    fn proves_and_verifies_rope() {
        let rope_params = RopeParams::new(2, 2, 4, "X");
        let params = RopeRoundParams::new(
            RoundParams::new(vec![2, 2, 4], "Y_acc", "Y"),
            rope_params.clone(),
        );
        let x = vec![
            1, 2, 3, 4, 5, 6, 7, 8, //
            9, 10, 11, 12, 13, 14, 15, 16,
        ];
        let cos = vec![2, 3, 4, 5];
        let sin = vec![7, 11, 13, 17];
        let y = rope_output(&x, &cos, &sin, &rope_params);
        let acc = y.iter().map(|&value| i64::from(value)).collect::<Vec<_>>();
        let output = round_outputs(&acc);
        let frac_bits = frac_bits_for(&acc);
        let point = vec![
            Fr::from(19u64),
            Fr::from(23u64),
            Fr::from(29u64),
            Fr::from(31u64),
        ];
        let y_claim = Claim {
            tensor: TensorId::new("Y"),
            logical_shape: rope_params.tensor_shape(),
            domain_shape: rope_params.tensor_shape().padded_power_of_two(),
            value: eval_tensor_at_point(&output, &rope_params.tensor_shape(), &point),
            point,
        };
        let witness = RopeWitness {
            input: x,
            acc,
            output,
            frac_bits,
        };

        let mut prover_transcript = Blake2bTranscript::default();
        let (proof, x_even, x_odd, _) = prove_rope_round::<Fr, _>(
            y_claim.clone(),
            &witness,
            &cos,
            &sin,
            &params,
            &mut prover_transcript,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        let (verified_even, verified_odd, _) = verify_rope_round::<Fr, _>(
            y_claim,
            &proof,
            &cos,
            &sin,
            &params,
            &mut verifier_transcript,
        )
        .unwrap();

        assert_eq!(verified_even, x_even);
        assert_eq!(verified_odd, x_odd);
        let parity_idx = log2_ceil(rope_params.seq) + log2_ceil(rope_params.heads);
        assert_eq!(verified_even.point.get(parity_idx), Some(&Fr::from(0u64)));
        assert_eq!(verified_odd.point.get(parity_idx), Some(&Fr::from(1u64)));
    }

    fn rope_output(x: &[i32], cos: &[i32], sin: &[i32], params: &RopeParams) -> Vec<i32> {
        let mut y = vec![0; x.len()];
        for s in 0..params.seq {
            for h in 0..params.heads {
                for pair in 0..params.head_dim / 2 {
                    let even = ((s * params.heads + h) * params.head_dim) + pair;
                    let odd = even + params.head_dim / 2;
                    let coeff = s * (params.head_dim / 2) + pair;
                    y[even] = x[even] * cos[coeff] - x[odd] * sin[coeff];
                    y[odd] = x[even] * sin[coeff] + x[odd] * cos[coeff];
                }
            }
        }
        y
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
}
