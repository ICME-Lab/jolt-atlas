use common::{CommittedPolynomial, VirtualPolynomial};
use joltworks::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
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
    /// Random evaluation point.
    pub r: Vec<F::Challenge>,
    /// Quantisation scale exponent.
    pub S: i32,
    /// Index of the computation node.
    pub computation_node_index: usize,
    /// `[F, N]` — leading-dim product and last-axis size.
    pub F_N: [usize; 2],
    /// Batching challenge drawn from the transcript.
    pub gamma: F,
}

impl<F: JoltField> RecipMultParams<F> {
    /// Create new parameters for reciprocal multiplication operation.
    pub fn new(
        computation_node_index: usize,
        S: i32,
        F_N: [usize; 2],
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let r = accumulator
            .get_node_output_opening(computation_node_index)
            .0
            .r;
        let gamma = transcript.challenge_scalar();
        Self {
            r,
            S,
            computation_node_index,
            F_N,
            gamma,
        }
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

    fn log_F(&self) -> usize {
        self.F().log_2()
    }

    fn log_N(&self) -> usize {
        self.N().log_2()
    }

    #[inline(always)]
    fn S_squared(&self) -> F {
        F::from_i32(self.S * self.S)
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for RecipMultParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let softmax_claim = accumulator
            .get_node_output_opening(self.computation_node_index)
            .1;
        let R_claim = accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::SoftmaxRecipMultRemainder(self.computation_node_index),
                SumcheckId::Execution,
            )
            .1;
        softmax_claim * F::from_i32(self.S) + R_claim
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.r.len()
    }
}

/// Prover state for reciprocal multiplication in softmax.
pub struct RecipMultProver<F: JoltField> {
    params: RecipMultParams<F>,
    eq: GruenSplitEqPolynomial<F>,
    test_eq: MultilinearPolynomial<F>,
    // log(F * N) variables
    exp_q: MultilinearPolynomial<F>,
    // log(F) variables
    inv_sum: MultilinearPolynomial<F>,
    // log(F) variables
    exp_sum: MultilinearPolynomial<F>,
    // log(F) variables
    r_inv: MultilinearPolynomial<F>,
}

impl<F: JoltField> RecipMultProver<F> {
    /// Constructor for reciprocal multiplication prover.
    pub fn initialize(
        exp_q: Vec<i32>,
        inv_sum: Vec<i32>,
        exp_sum: Vec<i32>,
        r_inv: Vec<i32>,
        params: RecipMultParams<F>,
    ) -> Self {
        let eq = GruenSplitEqPolynomial::new(&params.r, BindingOrder::LowToHigh);
        let exp_q = MultilinearPolynomial::from(exp_q);
        let inv_sum = MultilinearPolynomial::from(inv_sum);
        let exp_sum = MultilinearPolynomial::from(exp_sum);
        let r_inv = MultilinearPolynomial::from(r_inv);
        Self {
            test_eq: MultilinearPolynomial::from(EqPolynomial::evals(&params.r)),
            params,
            eq,
            exp_q,
            inv_sum,
            exp_sum,
            r_inv,
        }
    }

    fn compute_phase_1_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        let m = round + 1;
        let Self {
            eq,
            exp_q,
            inv_sum,
            exp_sum,
            r_inv,
            ..
        } = self;

        // Phase 1: binding j-variables (the low bits of exp_q).
        // Only exp_q depends on the current variable X; inv_sum, exp_sum, r_inv are constant w.r.t. X.
        //
        //   s(X) = eq_lin(X) * q(X)
        //   q(X) = exp_q(k, X) · inv_sum(k) + γ · (inv_sum(k) · exp_sum(k) + r_inv(k) - S²)
        //
        // q is linear in X, so we only need c0 = q(0):
        //   c0 = exp_q(k, 0) · inv_sum(k) + γ · (inv_sum(k) · exp_sum(k) + r_inv(k) - S²)
        let [q_constant] = eq.par_fold_out_in_unreduced::<9, 1>(&|kj| {
            let exp_q_0 = exp_q.get_bound_coeff(2 * kj);
            let k = kj >> (self.params.log_N() - m);
            let is_0 = inv_sum.get_bound_coeff(k);
            let es_0 = exp_sum.get_bound_coeff(k);
            let ri_0 = r_inv.get_bound_coeff(k);
            [exp_q_0 * is_0 + self.params.gamma * (is_0 * es_0 + ri_0 - self.params.S_squared())]
        });
        eq.gruen_poly_deg_2(q_constant, previous_claim)
    }

    fn compute_phase_2_message(&mut self, previous_claim: F) -> UniPoly<F> {
        let Self {
            eq,
            exp_q,
            inv_sum,
            exp_sum,
            r_inv,
            ..
        } = self;

        // Phase 2: binding k-variables (the high bits shared by all polynomials).
        // All polynomials now depend on the current variable X.
        //
        //   s(X) = eq_lin(X) * q(X)
        //   q(X) = exp_q(X) · inv_sum(X) + γ · (inv_sum(X) · exp_sum(X) + r_inv(X) - S²)
        //
        // q is quadratic in X (products of two linear-in-X polynomials). Compute:
        //   c0 = q(0) = exp_q(0) · inv_sum(0) + γ · (inv_sum(0) · exp_sum(0) + r_inv(0) - S²)
        //   e  = leading coeff of q(X) = exp_q_∞ · inv_sum_∞ + γ · (inv_sum_∞ · exp_sum_∞)
        //        (product of slopes from each product term; r_inv is linear so contributes no X² term)
        let [q_constant, q_quadratic] = eq.par_fold_out_in_unreduced::<9, 2>(&|g| {
            // exp0, exp_inf = exp_q(0), exp_q(∞)
            let exp_q_0 = exp_q.get_bound_coeff(2 * g);
            let exp_q_inf = exp_q.get_bound_coeff(2 * g + 1) - exp_q_0;

            // inv_sum0, inv_sum_inf = inv_sum(0), inv_sum(∞)
            let ise_0 = inv_sum.get_bound_coeff(2 * g);
            let ise_inf = inv_sum.get_bound_coeff(2 * g + 1) - ise_0;

            // exp_sum0, exp_sum_inf = exp_sum(0), exp_sum(∞)
            let es_0 = exp_sum.get_bound_coeff(2 * g);
            let es_inf = exp_sum.get_bound_coeff(2 * g + 1) - es_0;

            // r_inv0 = r_inv(0)
            let ri_0 = r_inv.get_bound_coeff(2 * g);

            // Compute the constant term c0 = q(0) and leading coeff e of q(X).
            let c0 = exp_q_0 * ise_0
                + self.params.gamma * (ise_0 * es_0 + ri_0 - self.params.S_squared());
            let e = exp_q_inf * ise_inf + self.params.gamma * (ise_inf * es_inf);
            [c0, e]
        });
        eq.gruen_poly_deg_3(q_constant, q_quadratic, previous_claim)
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for RecipMultProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

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
        self.test_eq.bind_parallel(r_j, BindingOrder::LowToHigh);
        if round >= self.params.log_N() {
            self.inv_sum.bind_parallel(r_j, BindingOrder::LowToHigh);
            self.exp_sum.bind_parallel(r_j, BindingOrder::LowToHigh);
            self.r_inv.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        let r_leading: OpeningPoint<BIG_ENDIAN, F> =
            opening_point.r[..self.params.log_F()].to_vec().into();
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::SoftmaxExpQ(self.params.computation_node_index),
            SumcheckId::Execution,
            opening_point.clone(),
            self.exp_q.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::SoftmaxInvSum(self.params.computation_node_index),
            SumcheckId::Execution,
            r_leading.clone(),
            self.inv_sum.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::SoftmaxRInv(self.params.computation_node_index),
            SumcheckId::Execution,
            r_leading.clone(),
            self.r_inv.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::SoftmaxExpSum(self.params.computation_node_index),
            SumcheckId::Execution,
            r_leading.clone(),
            self.exp_sum.final_sumcheck_claim(),
        );
    }
}

/// Verifier for division by sum in softmax.
pub struct RecipMultVerifier<F: JoltField> {
    params: RecipMultParams<F>,
}

impl<F: JoltField> RecipMultVerifier<F> {
    /// Create new verifier for division operation.
    pub fn new(
        computation_node_index: usize,
        S: i32,
        F_N: [usize; 2],
        accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = RecipMultParams::new(computation_node_index, S, F_N, accumulator, transcript);
        Self { params }
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
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::SoftmaxExpQ(self.params.computation_node_index),
            SumcheckId::Execution,
            opening_point.clone(),
        );
        let r_leading: OpeningPoint<BIG_ENDIAN, F> =
            opening_point.r[..self.params.log_F()].to_vec().into();
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::SoftmaxInvSum(self.params.computation_node_index),
            SumcheckId::Execution,
            r_leading.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::SoftmaxRInv(self.params.computation_node_index),
            SumcheckId::Execution,
            r_leading.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::SoftmaxExpSum(self.params.computation_node_index),
            SumcheckId::Execution,
            r_leading.clone(),
        );
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r = self.params.r.clone();
        let r_sc = self.params.normalize_opening_point(sumcheck_challenges).r;
        let exp_q_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxExpQ(self.params.computation_node_index),
                SumcheckId::Execution,
            )
            .1;

        // TODO: Verifier can compute a = inv_sum(r_lead) on its own since:
        // The verifier's final-round check is:
        //   `eq(r, r_sc) · [ exp_q(r_sc) · a
        //                   + α · (a · exp_sum_q(r_lead) + r_inv(r_lead) − S²) ]
        //     = final_claim`
        //
        // This is LINEAR in `a`.  The values `eq(r, r_sc)`, `exp_q(r_sc)`,
        // `exp_sum_q(r_lead)`, `r_inv(r_lead)`, and `final_claim` are all either
        // transcript-derived or independently verified (`exp_q` via Shout,
        // `exp_sum_q` via sum-axis sumcheck, `r_inv` range-checked and its one-hot encoding opening -> PCS).
        // So there is exactly ONE value of `a` that satisfies the equation —
        // the verifier computes it.  The prover doesn't get to "choose" `a`;
        // it falls out of the algebra.
        let inv_sum_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxInvSum(self.params.computation_node_index),
                SumcheckId::Execution,
            )
            .1;
        let exp_sum_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxExpSum(self.params.computation_node_index),
                SumcheckId::Execution,
            )
            .1;
        let r_inv_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxRInv(self.params.computation_node_index),
                SumcheckId::Execution,
            )
            .1;
        EqPolynomial::mle(&r, &r_sc)
            * (exp_q_claim * inv_sum_claim
                + self.params.gamma
                    * (inv_sum_claim * exp_sum_claim + r_inv_claim - self.params.S_squared()))
    }
}
