use jolt_core::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{BIG_ENDIAN, OpeningPoint},
    },
    transcripts::Transcript,
    utils::math::Math,
};
use rayon::prelude::*;

use crate::jolt::{
    pcs::{SumcheckId, VerifierOpeningAccumulator},
    sumcheck::SumcheckInstance,
    witness::VirtualPolynomial,
};

pub struct ExecutionSumcheck<F: JoltField> {
    prover_state: Option<ExecutionProverState<F>>,
    r_c: Vec<F>,
    rv_claim_c: F,
    index: usize,
    num_rounds: usize,
}

impl<F: JoltField> SumcheckInstance<F> for ExecutionSumcheck<F> {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree(&self) -> usize {
        2
    }

    fn input_claim(&self) -> F {
        self.rv_claim_c
    }

    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        const DEGREE: usize = 2;
        let univariate_poly_evals: [F; DEGREE] = (0..prover_state.a.len() / 2)
            .into_par_iter()
            .map(|i| {
                let a_evals = prover_state
                    .a
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let b_evals = prover_state
                    .B_r
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                [
                    a_evals[0] * b_evals[0], // eval at 0
                    a_evals[1] * b_evals[1], // eval at 2
                ]
            })
            .reduce(
                || [F::zero(); DEGREE],
                |mut running, new| {
                    for i in 0..DEGREE {
                        running[i] += new[i];
                    }
                    running
                },
            );
        univariate_poly_evals.into()
    }

    fn bind(&mut self, r_j: F, _round: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");
        // Bind both polynomials in parallel
        rayon::join(
            || prover_state.a.bind_parallel(r_j, BindingOrder::HighToLow),
            || prover_state.B_r.bind_parallel(r_j, BindingOrder::HighToLow),
        );
    }

    fn cache_openings_prover(
        &self,
        accumulator: std::rc::Rc<std::cell::RefCell<crate::jolt::pcs::ProverOpeningAccumulator<F>>>,
        opening_point: jolt_core::poly::opening_proof::OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        let a_claim = prover_state.a.final_sumcheck_claim();
        let b_claim = prover_state.B_r.final_sumcheck_claim();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::AVec(self.index),
            SumcheckId::MatVecExecution,
            opening_point.clone(),
            a_claim,
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::BMat(self.index),
            SumcheckId::MatVecExecution,
            opening_point.clone(),
            b_claim,
        );
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F],
    ) -> jolt_core::poly::opening_proof::OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn expected_output_claim(
        &self,
        opening_accumulator: Option<std::rc::Rc<std::cell::RefCell<VerifierOpeningAccumulator<F>>>>,
        _r: &[F],
    ) -> F {
        let accumulator = opening_accumulator.as_ref().unwrap();
        let (_, a_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::AVec(self.index),
            SumcheckId::MatVecExecution,
        );
        let (_, b_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::BMat(self.index),
            SumcheckId::MatVecExecution,
        );
        a_claim * b_claim
    }

    fn cache_openings_verifier(
        &self,
        accumulator: std::rc::Rc<
            std::cell::RefCell<crate::jolt::pcs::VerifierOpeningAccumulator<F>>,
        >,
        r_a: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::AVec(self.index),
            SumcheckId::MatVecExecution,
            r_a.clone(),
        );
        let r_b = [self.r_c.clone(), r_a.r.clone()].concat();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::BMat(self.index),
            SumcheckId::MatVecExecution,
            r_b.into(),
        );
    }
}

impl<F: JoltField> ExecutionSumcheck<F> {
    /// Create the prover sum-check instance for matvec precompile
    pub fn new_prover<ProofTranscript: Transcript>(
        a: Vec<i64>,
        b: Vec<i64>,
        r_c: Vec<F>,
        rv_claim_c: F,
        index: usize,
    ) -> Self {
        let eq_r = EqPolynomial::evals(&r_c);
        let k = a.len();
        let n = r_c.len().pow2();
        let a = MultilinearPolynomial::from(a);
        let B_r: Vec<F> = (0..k)
            .into_par_iter()
            .map(|j| (0..n).map(|i| F::from_i64(b[i * k + j]) * eq_r[i]).sum())
            .collect();
        let B_r: MultilinearPolynomial<F> = MultilinearPolynomial::from(B_r);
        debug_assert_eq!(
            rv_claim_c,
            (0..a.len())
                .map(|i| a.get_bound_coeff(i) * B_r.get_bound_coeff(i))
                .sum()
        );
        Self {
            prover_state: Some(ExecutionProverState { a, B_r }),
            r_c,
            rv_claim_c,
            index,
            num_rounds: k.log_2(),
        }
    }

    /// Create the verifier sum-check instance for matvec precompile
    pub fn new_verifier(r_c: Vec<F>, rv_claim_c: F, index: usize, k: usize) -> Self {
        Self {
            prover_state: None,
            r_c,
            rv_claim_c,
            index,
            num_rounds: k.log_2(),
        }
    }
}

/// Stores the "witness" polynomials for the execution sumcheck.
/// These polynomials are virtual
pub struct ExecutionProverState<F: JoltField> {
    a: MultilinearPolynomial<F>,
    B_r: MultilinearPolynomial<F>,
}
