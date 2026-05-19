use common::VirtualPoly;
// Design note for future us:
//
// This is the reverse-claim version of a plain 2D matmul.  The caller already
// has a claim for `Y(r_m, r_n)` from the consumer of the matmul output.  The
// verifier does not receive `A`; `W` is public for this op.  We therefore
// prove the relation in two stages:
//
// 1. Fold the output axes with the incoming claim point and prove
//      Y(r_m, r_n) = sum_k A_{r_m}(k) * W_{r_n}(k).
// 2. Expand `A_{r_m}(r_k)` back into a claim on the original `A` tensor:
//      A_{r_m}(r_k) = sum_m eq(m, r_m) * A(m, r_k).
//
// `verify_matmul` returns the remaining `A` opening claim.  This keeps the
// Qwen layer prover pure and lets a later commitment/opening layer discharge
// the returned claim.  Shapes are logical shapes; each summed dimension is
// padded to the next power of two inside the polynomial domain.
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
pub struct MatMulProof<F: JoltField, T: Transcript> {
    pub k_sumcheck: SumcheckInstanceProof<F, T>,
    pub m_sumcheck: SumcheckInstanceProof<F, T>,
    pub a_r_opening: F,
    pub w_r_opening: F,
    pub a_opening: F,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatMulClaims<F> {
    pub input: Claim<F>,
}

pub fn prove_matmul<F, T>(
    y_claim: Claim<F>,
    a: &[i32],
    w: &[i32],
    params: &MatMulParams,
    transcript: &mut T,
) -> Result<ProveResult<MatMulClaims<F>, MatMulProof<F, T>>>
where
    F: JoltField,
    T: Transcript,
{
    validate_inputs(&y_claim, a, w, params)?;
    let (r_m, r_n) = split_y_point(&y_claim, params)?;

    let partials = MatMulPartials::new(a, w, params, r_m, r_n);
    let k_params = KSumcheckParams::new(log2_ceil(params.k), y_claim.value);
    let mut k_prover = KSumcheckProver::new(k_params, partials.a_r, partials.w_r);
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
        MatMulClaims { input: a_claim },
        MatMulProof {
            k_sumcheck,
            m_sumcheck,
            a_r_opening,
            w_r_opening,
            a_opening,
        },
    ))
}

pub fn verify_matmul<F, T>(
    y_claim: Claim<F>,
    proof: &MatMulProof<F, T>,
    w: &[i32],
    params: &MatMulParams,
    transcript: &mut T,
) -> std::result::Result<MatMulClaims<F>, ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    verify_inputs(&y_claim, w, params)?;
    let (r_m, r_n) = split_y_point_verify(&y_claim, params)?;

    let k_params = KSumcheckParams::new(log2_ceil(params.k), y_claim.value);
    let k_verifier = KSumcheckVerifier { params: k_params };
    let mut k_accumulator = VerifierOpeningAccumulator::new();
    k_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenMatMulPartialA, k_sumcheck_id()),
        (OpeningPoint::<BIG_ENDIAN, F>::default(), proof.a_r_opening),
    );
    k_accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenMatMulPartialW, k_sumcheck_id()),
        (OpeningPoint::<BIG_ENDIAN, F>::default(), proof.w_r_opening),
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
    Ok(MatMulClaims {
        input: Claim {
            tensor: params.a_tensor.clone(),
            logical_shape: params.a_shape(),
            domain_shape: params.a_shape().padded_power_of_two(),
            point,
            value: proof.a_opening,
        },
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

struct KSumcheckProver<F: JoltField> {
    a_r: MultilinearPolynomial<F>,
    w_r: MultilinearPolynomial<F>,
    params: KSumcheckParams<F>,
}

impl<F: JoltField> KSumcheckProver<F> {
    fn new(params: KSumcheckParams<F>, a_r: Vec<F>, w_r: Vec<F>) -> Self {
        Self {
            a_r: MultilinearPolynomial::from(a_r),
            w_r: MultilinearPolynomial::from(w_r),
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for KSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let mut eval_at_0 = F::zero();
        let mut eval_at_2 = F::zero();
        for g in 0..self.a_r.len() / 2 {
            let a0 = self.a_r.get_bound_coeff(2 * g);
            let a1 = self.a_r.get_bound_coeff(2 * g + 1);
            let w0 = self.w_r.get_bound_coeff(2 * g);
            let w1 = self.w_r.get_bound_coeff(2 * g + 1);
            eval_at_0 += a0 * w0;
            eval_at_2 += (a1 + a1 - a0) * (w1 + w1 - w0);
        }
        UniPoly::from_evals_and_hint(previous_claim, &[eval_at_0, eval_at_2])
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.a_r.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.w_r.bind_parallel(r_j, BindingOrder::LowToHigh);
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
    }
}

struct KSumcheckVerifier<F: JoltField> {
    params: KSumcheckParams<F>,
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for KSumcheckVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        _sumcheck_challenges: &[F::Challenge],
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
        a * w
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
    fn proves_and_verifies_matmul_without_rebase() {
        let params = MatMulParams::new(2, 3, 2, "A", "W");
        let a = vec![1, 2, 3, 4, 5, 6];
        let w = vec![1, 2, 3, 4, 5, 6];
        let r_m = vec![Fr::from(3u64)];
        let r_n = vec![Fr::from(5u64)];
        let y = vec![22, 28, 49, 64];
        let y_claim = Claim {
            tensor: TensorId::new("Y"),
            logical_shape: params.y_shape(),
            domain_shape: params.y_shape().padded_power_of_two(),
            point: [r_m.as_slice(), r_n.as_slice()].concat(),
            value: eval_matrix(&y, params.m, params.n, &r_m, &r_n),
        };

        let mut prover_transcript = Blake2bTranscript::default();
        let result =
            prove_matmul::<Fr, _>(y_claim.clone(), &a, &w, &params, &mut prover_transcript)
                .unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        let claims = verify_matmul::<Fr, _>(
            y_claim,
            &result.proof,
            &w,
            &params,
            &mut verifier_transcript,
        )
        .unwrap();

        assert_eq!(claims.input, result.claims.input);
        assert_eq!(claims.input.tensor.0, "A");
        assert_eq!(
            claims.input.point.len(),
            log2_ceil(params.m) + log2_ceil(params.k)
        );
    }

    #[test]
    fn verifier_rejects_tampered_public_w() {
        let params = MatMulParams::new(2, 3, 2, "A", "W");
        let a = vec![1, 2, 3, 4, 5, 6];
        let w = vec![1, 2, 3, 4, 5, 6];
        let r_m = vec![Fr::from(3u64)];
        let r_n = vec![Fr::from(5u64)];
        let y = vec![22, 28, 49, 64];
        let y_claim = Claim {
            tensor: TensorId::new("Y"),
            logical_shape: params.y_shape(),
            domain_shape: params.y_shape().padded_power_of_two(),
            point: [r_m.as_slice(), r_n.as_slice()].concat(),
            value: eval_matrix(&y, params.m, params.n, &r_m, &r_n),
        };
        let mut prover_transcript = Blake2bTranscript::default();
        let result =
            prove_matmul::<Fr, _>(y_claim.clone(), &a, &w, &params, &mut prover_transcript)
                .unwrap();
        let mut bad_w = w.clone();
        bad_w[0] += 1;

        let mut verifier_transcript = Blake2bTranscript::default();
        let err = verify_matmul::<Fr, _>(
            y_claim,
            &result.proof,
            &bad_w,
            &params,
            &mut verifier_transcript,
        );

        assert!(err.is_err());
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
