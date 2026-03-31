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
};

const DEGREE_BOUND: usize = 3;

// TODO: refactor code to use mul sum-check

/// Shared prover/verifier parameters for softmax reciprocal multiplication proof.
#[derive(Clone)]
pub struct MultParams<F: JoltField> {
    /// Random evaluation point.
    pub r: Vec<F::Challenge>,
    /// Quantisation scale exponent.
    pub S: i32,
    /// Index of the computation node.
    pub computation_node_index: usize,
}

impl<F: JoltField> MultParams<F> {
    /// Create new parameters for reciprocal multiplication operation.
    pub fn new(
        computation_node_index: usize,
        S: i32,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        let r = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxExpQ(computation_node_index),
                SumcheckId::Execution,
            )
            .0
            .r;
        Self {
            r,
            S,
            computation_node_index,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for MultParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let exp_q_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxExpQ(self.computation_node_index),
                SumcheckId::Execution,
            )
            .1;
        let R_claim = accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::SoftmaxExpRemainder(self.computation_node_index),
                SumcheckId::Execution,
            )
            .1;
        exp_q_claim * F::from_i32(self.S) + R_claim
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.r.len()
    }
}

/// Prover state for reciprocal multiplication in softmax.
pub struct MultProver<F: JoltField> {
    params: MultParams<F>,
    eq: GruenSplitEqPolynomial<F>,
    // log(F * N) variables
    exp_hi: MultilinearPolynomial<F>,
    // log(F * N) variables
    exp_lo: MultilinearPolynomial<F>,
}

impl<F: JoltField> MultProver<F> {
    /// Constructor for reciprocal multiplication prover.
    pub fn initialize(exp_hi: Vec<i32>, exp_lo: Vec<i32>, params: MultParams<F>) -> Self {
        let eq = GruenSplitEqPolynomial::new(&params.r, BindingOrder::LowToHigh);
        let exp_hi = MultilinearPolynomial::from(exp_hi);
        let exp_lo = MultilinearPolynomial::from(exp_lo);
        Self {
            params,
            eq,
            exp_hi,
            exp_lo,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for MultProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let Self {
            eq, exp_hi, exp_lo, ..
        } = self;
        let [q_constant, q_quadratic] = eq.par_fold_out_in_unreduced::<9, 2>(&|g| {
            let exp_hi_0 = exp_hi.get_bound_coeff(2 * g);
            let exp_hi_inf = exp_hi.get_bound_coeff(2 * g + 1) - exp_hi_0;
            let exp_lo_0 = exp_lo.get_bound_coeff(2 * g);
            let exp_lo_inf = exp_lo.get_bound_coeff(2 * g + 1) - exp_lo_0;
            let c0 = exp_lo_0 * exp_hi_0;
            let e = exp_hi_inf * exp_lo_inf;
            [c0, e]
        });
        eq.gruen_poly_deg_3(q_constant, q_quadratic, previous_claim)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq.bind(r_j);
        self.exp_hi.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.exp_lo.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::SoftmaxExpHi(self.params.computation_node_index),
            SumcheckId::Execution,
            opening_point.clone(),
            self.exp_hi.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::SoftmaxExpLo(self.params.computation_node_index),
            SumcheckId::Execution,
            opening_point.clone(),
            self.exp_lo.final_sumcheck_claim(),
        );
    }
}

/// Verifier for division by sum in softmax.
pub struct MultVerifier<F: JoltField> {
    params: MultParams<F>,
}

impl<F: JoltField> MultVerifier<F> {
    /// Create new verifier for division operation.
    pub fn new(
        computation_node_index: usize,
        S: i32,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = MultParams::new(computation_node_index, S, accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for MultVerifier<F> {
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
            VirtualPolynomial::SoftmaxExpHi(self.params.computation_node_index),
            SumcheckId::Execution,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::SoftmaxExpLo(self.params.computation_node_index),
            SumcheckId::Execution,
            opening_point.clone(),
        );
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r = self.params.r.clone();
        let r_sc = self.params.normalize_opening_point(sumcheck_challenges).r;
        let exp_hi_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxExpHi(self.params.computation_node_index),
                SumcheckId::Execution,
            )
            .1;
        let exp_lo_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxExpLo(self.params.computation_node_index),
                SumcheckId::Execution,
            )
            .1;
        EqPolynomial::mle(&r, &r_sc) * exp_lo_claim * exp_hi_claim
    }
}
