use std::array;

use common::VirtualPolynomial;
use joltworks::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
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
    utils::{index_to_field_bitvector, math::Math, thread::drop_in_background_thread},
};
use rayon::prelude::*;

const DEGREE_BOUND: usize = 3;

/// Shared parameters for the max indicator sumcheck instance.
#[derive(Clone)]
pub struct MaxIndicatorParams<F: JoltField> {
    /// Random evaluation point.
    pub r1_k: Vec<F>,
    /// Index of the computation node.
    pub computation_node_index: usize,
    /// Used to cache the operand claim
    pub operand_node_index: usize,
    /// max_k(r1_k)
    pub input_claim: F,
    /// Softmax dims
    pub F_N: [usize; 2],
    /// Max indicator
    pub e: Vec<u32>,
    /// Argmax index per feature (length F)
    pub argmax_k: Vec<usize>,
}

impl<F: JoltField> MaxIndicatorParams<F> {
    /// Create a new instance of max indicator sumcheck parameters.
    pub fn new(
        computation_node_index: usize,
        operand_node_index: usize,
        F_N: [usize; 2],
        argmax_k: Vec<usize>,
        input_claim: F,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        let [F, N] = F_N;
        let log_f = F.log_2();

        // Get r1 (the point from Stage 1 reciprocal multiplication output)
        let r1 = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxExpQ(computation_node_index),
                SumcheckId::NodeExecution(computation_node_index),
            )
            .0;
        let r1_k = &r1.r[..log_f];

        let mut e: Vec<u32> = vec![0; F * N];
        // Only set the argmax positions — everything else is already 0
        for k in 0..F {
            e[k * N + argmax_k[k]] = 1;
        }
        Self {
            r1_k: r1_k.to_vec(),
            input_claim,
            F_N,
            e,
            argmax_k,
            computation_node_index,
            operand_node_index,
        }
    }

    /// Returns log2 of the leading-dimension product `F`.
    pub fn log_F(&self) -> usize {
        self.F_N[0].log_2()
    }

    /// Returns log2 of the last-axis size `N`.
    pub fn log_N(&self) -> usize {
        self.F_N[1].log_2()
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for MaxIndicatorParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.input_claim
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.F_N.iter().product::<usize>().log_2()
    }
}

/// Prover for the max indicator sumcheck instance.
pub struct MaxIndicatorProver<F: JoltField> {
    params: MaxIndicatorParams<F>,
    eq: Vec<F>,
    gs_eq: Option<GruenSplitEqPolynomial<F>>,
    // log(F * N) variables
    X: MultilinearPolynomial<F>,
    // log(F * N) variables
    e: MultilinearPolynomial<F>,
}

impl<F: JoltField> MaxIndicatorProver<F> {
    /// Constructor for softmax exp multiplication prover.
    pub fn initialize(X: Vec<i32>, mut params: MaxIndicatorParams<F>) -> Self {
        let eq = EqPolynomial::evals(&params.r1_k);
        let X = MultilinearPolynomial::from(X);
        // Take e out of params — the prover never reads params.e again
        let e = MultilinearPolynomial::from(std::mem::take(&mut params.e));
        Self {
            params,
            eq,
            gs_eq: None,
            X,
            e,
        }
    }

    fn compute_phase_1_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        let m = round + 1;
        let Self { X, e, eq, .. } = self;
        let half_poly_len = X.len() / 2;
        let evals: [F; DEGREE_BOUND] = (0..half_poly_len)
            .into_par_iter()
            .map(|kj| {
                let k = kj >> (self.params.log_N() - m);
                let eq_val = eq[k];
                let e_vals = e.sumcheck_evals(kj, DEGREE_BOUND, BindingOrder::LowToHigh);
                let X_vals = X.sumcheck_evals(kj, DEGREE_BOUND, BindingOrder::LowToHigh);
                [
                    eq_val * X_vals[0] * e_vals[0],
                    eq_val * X_vals[1] * e_vals[1],
                    eq_val * X_vals[2] * e_vals[2],
                ]
            })
            .reduce(
                || [F::zero(); DEGREE_BOUND],
                |running, new| array::from_fn(|i| running[i] + new[i]),
            );
        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn compute_phase_2_message(&mut self, previous_claim: F) -> UniPoly<F> {
        let Self {
            gs_eq,
            X,
            e: indicator,
            ..
        } = self;
        let gs_eq = gs_eq.as_ref().unwrap();
        let [q_constant, q_quadratic] = gs_eq.par_fold_out_in_unreduced::<9, 2>(&|g| {
            let i_0 = indicator.get_bound_coeff(2 * g);
            let i_inf = indicator.get_bound_coeff(2 * g + 1) - i_0;

            let X_0 = X.get_bound_coeff(2 * g);
            let X_inf = X.get_bound_coeff(2 * g + 1) - X_0;

            let c0 = i_0 * X_0;
            let e = i_inf * X_inf;
            [c0, e]
        });
        gs_eq.gruen_poly_deg_3(q_constant, q_quadratic, previous_claim)
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for MaxIndicatorProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(name = "MaxIndicatorProver::compute_message", skip_all)]
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        if round < self.params.log_N() {
            self.compute_phase_1_message(round, previous_claim)
        } else {
            self.compute_phase_2_message(previous_claim)
        }
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        self.X.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.e.bind_parallel(r_j, BindingOrder::LowToHigh);

        if round == self.params.log_N() - 1 {
            self.gs_eq = Some(GruenSplitEqPolynomial::new(
                &self.params.r1_k,
                BindingOrder::LowToHigh,
            ));
            drop_in_background_thread(std::mem::take(&mut self.eq));
        }
        if round >= self.params.log_N() {
            self.gs_eq.as_mut().unwrap().bind(r_j);
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

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.operand_node_index),
            SumcheckId::NodeExecution(self.params.computation_node_index),
            opening_point,
            self.X.final_sumcheck_claim(),
        );
    }
}

/// Verifier for the max indicator sumcheck instance.
pub struct MaxIndicatorVerifier<F: JoltField> {
    params: MaxIndicatorParams<F>,
}

impl<F: JoltField> MaxIndicatorVerifier<F> {
    /// Create new verifier for max indicator sumcheck.
    pub fn new(
        computation_node_index: usize,
        operand_node_index: usize,
        F_N: [usize; 2],
        argmax_k: Vec<usize>,
        input_claim: F,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        let params = MaxIndicatorParams::new(
            computation_node_index,
            operand_node_index,
            F_N,
            argmax_k,
            input_claim,
            accumulator,
        );
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for MaxIndicatorVerifier<F> {
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
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.operand_node_index),
            SumcheckId::NodeExecution(self.params.computation_node_index),
            opening_point,
        );
    }

    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::expected_output_claim", skip_all)]
    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r_sc = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening())
            .r;
        let X_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.params.operand_node_index),
                SumcheckId::NodeExecution(self.params.computation_node_index),
            )
            .1;

        // Evaluate e(r_sc) in O(F · log N) by exploiting sparsity:
        // e has exactly F nonzero entries at positions k·N + argmax_k[k],
        // so ẽ(r_k, r_j) = Σ_k eq(r_k, bits(k)) · eq(r_j, bits(argmax_k[k]))
        let r_k = &r_sc[..self.params.log_F()];
        let r_j = &r_sc[self.params.log_F()..];
        let eq_k_evals = EqPolynomial::evals(r_k);
        let log_n = r_j.len();
        let e_claim: F = eq_k_evals
            .iter()
            .zip(self.params.argmax_k.iter())
            .map(|(&eq_k, &argmax_j)| {
                let y = index_to_field_bitvector::<F>(argmax_j as u64, log_n);
                eq_k * EqPolynomial::mle(r_j, &y)
            })
            .sum();
        let r2_k = &r_sc[..self.params.log_F()];
        EqPolynomial::mle(&self.params.r1_k, r2_k) * e_claim * X_claim
    }
}
