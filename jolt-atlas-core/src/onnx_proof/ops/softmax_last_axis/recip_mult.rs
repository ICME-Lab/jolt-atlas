use crate::utils::opening_access::AccOpeningAccessor;
use atlas_onnx_tracer::node::ComputationNode;
use common::VirtualPoly;
use joltworks::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, VerifierOpeningAccumulator,
            BIG_ENDIAN, LITTLE_ENDIAN,
        },
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::math::Math,
};

const DEGREE_BOUND: usize = 3;

/// Shared prover/verifier parameters for softmax reciprocal multiplication proof.
#[derive(Clone)]
pub struct RecipMultParams<F: JoltField> {
    /// Computation node reference.
    pub node: ComputationNode,
    /// Random evaluation point.
    pub r: Vec<F>,
    /// Quantisation scale value S.
    pub S: i32,
    /// `[F, N]` — leading-dim product and last-axis size.
    pub F_N: [usize; 2],
}

impl<F: JoltField> RecipMultParams<F> {
    /// Create new parameters for reciprocal multiplication operation.
    pub fn new(
        node: ComputationNode,
        S: i32,
        F_N: [usize; 2],
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut impl Transcript,
    ) -> Self {
        let r = AccOpeningAccessor::new(accumulator, &node)
            .get_reduced_opening()
            .0
            .r;
        Self { r, S, node, F_N }
    }

    /// Returns `[F, N]`.
    pub fn F_N(&self) -> [usize; 2] {
        self.F_N
    }

    /// Returns the leading-dimension product `F`.
    pub fn F(&self) -> usize {
        self.F_N()[0]
    }

    /// Returns the last-axis size `N`.
    pub fn N(&self) -> usize {
        self.F_N()[1]
    }

    /// Returns log2 of the leading-dimension product `F`.
    fn log_F(&self) -> usize {
        self.F().log_2()
    }

    /// Returns log2 of the last-axis size `N`.
    fn log_N(&self) -> usize {
        self.N().log_2()
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for RecipMultParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let accessor = AccOpeningAccessor::new(accumulator, &self.node);
        let softmax_claim = accessor.get_reduced_opening().1;
        let R_claim = accessor
            .get_advice(VirtualPoly::SoftmaxRecipMultRemainder)
            .1;
        softmax_claim * F::from_i32(self.S) + R_claim
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.r.len()
    }
}

/// Prover state for reciprocal multiplication in softmax.
///
/// Proves Relation A only: `exp_q[k,j] · inv_sum[k] = softmax_q[k,j] · S + R[k,j]`.
/// `inv_sum` is verifier-known (derived from prover-sent `exp_sum_q`), but
/// the prover still needs the polynomial for sumcheck round computation.
pub struct RecipMultProver<F: JoltField> {
    params: RecipMultParams<F>,
    eq: GruenSplitEqPolynomial<F>,
    // log(F * N) variables
    exp_q: MultilinearPolynomial<F>,
    // log(F) variables — verifier-known, but prover needs for round computation
    inv_sum: MultilinearPolynomial<F>,
}

impl<F: JoltField> RecipMultProver<F> {
    /// Constructor for reciprocal multiplication prover.
    pub fn initialize(exp_q: Vec<i32>, inv_sum: Vec<i32>, params: RecipMultParams<F>) -> Self {
        let eq = GruenSplitEqPolynomial::new(&params.r, BindingOrder::LowToHigh);
        let exp_q = MultilinearPolynomial::from(exp_q);
        let inv_sum = MultilinearPolynomial::from(inv_sum);
        Self {
            params,
            eq,
            exp_q,
            inv_sum,
        }
    }

    fn compute_phase_1_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        let m = round + 1;
        let Self {
            eq, exp_q, inv_sum, ..
        } = self;

        // Phase 1: binding j-variables (the low bits of exp_q).
        // Only exp_q depends on the current variable X; inv_sum is constant w.r.t. X.
        //
        //   s(X) = eq_lin(X) * q(X)
        //   q(X) = exp_q(k, X) · inv_sum(k)
        //
        // q is linear in X, so we only need c0 = q(0):
        //   c0 = exp_q(k, 0) · inv_sum(k)
        let [q_constant] = eq.par_fold_out_in_unreduced::<9, 1>(&|kj| {
            let exp_q_0 = exp_q.get_bound_coeff(2 * kj);
            let k = kj >> (self.params.log_N() - m);
            let is_0 = inv_sum.get_bound_coeff(k);
            [exp_q_0 * is_0]
        });
        eq.gruen_poly_deg_2(q_constant, previous_claim)
    }

    fn compute_phase_2_message(&mut self, previous_claim: F) -> UniPoly<F> {
        let Self {
            eq, exp_q, inv_sum, ..
        } = self;

        // Phase 2: binding k-variables (the high bits shared by both polynomials).
        // Both exp_q and inv_sum depend on the current variable X.
        //
        //   s(X) = eq_lin(X) * q(X)
        //   q(X) = exp_q(X) · inv_sum(X)
        //
        // q is quadratic in X (product of two linear-in-X polynomials). Compute:
        //   c0 = q(0) = exp_q(0) · inv_sum(0)
        //   e  = leading coeff of q(X) = exp_q_∞ · inv_sum_∞
        let [q_constant, q_quadratic] = eq.par_fold_out_in_unreduced::<9, 2>(&|g| {
            let exp_q_0 = exp_q.get_bound_coeff(2 * g);
            let exp_q_inf = exp_q.get_bound_coeff(2 * g + 1) - exp_q_0;

            let ise_0 = inv_sum.get_bound_coeff(2 * g);
            let ise_inf = inv_sum.get_bound_coeff(2 * g + 1) - ise_0;

            let c0 = exp_q_0 * ise_0;
            let e = exp_q_inf * ise_inf;
            [c0, e]
        });
        eq.gruen_poly_deg_3(q_constant, q_quadratic, previous_claim)
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for RecipMultProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(name = "RecipMultProver::compute_message", skip_all)]
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        if round < self.params.log_N() {
            self.compute_phase_1_message(round, previous_claim)
        } else {
            self.compute_phase_2_message(previous_claim)
        }
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        self.eq.bind(r_j);
        self.exp_q.bind_parallel(r_j, BindingOrder::LowToHigh);
        if round >= self.params.log_N() {
            self.inv_sum.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        // Only cache exp_q — inv_sum is verifier-known (derived from sent exp_sum_q).
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.node)
            .to_provider(transcript, opening_point);
        provider.append_advice(VirtualPoly::SoftmaxExpQ, self.exp_q.final_claim());
    }
}

/// Verifier for reciprocal multiplication in softmax.
pub struct RecipMultVerifier<F: JoltField> {
    params: RecipMultParams<F>,
    /// Full inv_sum vector (F elements) — used to evaluate at r_sc_leading.
    inv_sum_evals: Vec<F>,
}

impl<F: JoltField> RecipMultVerifier<F> {
    /// Create new verifier for reciprocal multiplication.
    pub fn new(
        node: ComputationNode,
        S: i32,
        F_N: [usize; 2],
        inv_sum_evals: Vec<F>,
        accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = RecipMultParams::new(node, S, F_N, accumulator, transcript);
        Self {
            params,
            inv_sum_evals,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for RecipMultVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        // Only cache exp_q — inv_sum is verifier-known.
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.node)
            .to_provider(transcript, opening_point);
        provider.append_advice(VirtualPoly::SoftmaxExpQ);
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r_sc = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening())
            .r;
        let exp_q_claim = AccOpeningAccessor::new(accumulator, &self.params.node)
            .get_advice(VirtualPoly::SoftmaxExpQ)
            .1;
        // Evaluate inv_sum MLE at the leading part of the sumcheck challenge point.
        let r_sc_leading = &r_sc[..self.params.log_F()];
        let inv_sum_poly = MultilinearPolynomial::from(self.inv_sum_evals.clone());
        let inv_sum_claim = inv_sum_poly.evaluate(r_sc_leading);
        EqPolynomial::mle(&self.params.r, &r_sc) * (exp_q_claim * inv_sum_claim)
    }
}
