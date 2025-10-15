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
    r_x: Vec<F>,
    rv_claim_res: F,
    index: usize,
    num_rounds: usize,
}

impl<F: JoltField> SumcheckInstance<F> for ExecutionSumcheck<F> {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree(&self) -> usize {
        1
    }

    fn input_claim(&self) -> F {
        self.rv_claim_res
    }

    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        const DEGREE: usize = 1;
        let univariate_poly_evals: [F; DEGREE] = (0..prover_state.a_r.len() / 2)
            .into_par_iter()
            .map(|i| {
                prover_state
                    .a_r
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow)
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
        prover_state.a_r.bind_parallel(r_j, BindingOrder::HighToLow)
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
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::ReduceA(self.index),
            SumcheckId::ReduceSumExecution,
            opening_point.clone(),
            prover_state.a_r.final_sumcheck_claim(),
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
        accumulator
            .borrow()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::ReduceA(self.index),
                SumcheckId::ReduceSumExecution,
            )
            .1
    }

    fn cache_openings_verifier(
        &self,
        accumulator: std::rc::Rc<
            std::cell::RefCell<crate::jolt::pcs::VerifierOpeningAccumulator<F>>,
        >,
        r_y: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let r_b = [self.r_x.clone(), r_y.r.clone()].concat();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::ReduceA(self.index),
            SumcheckId::ReduceSumExecution,
            r_b.into(),
        );
    }
}

impl<F: JoltField> ExecutionSumcheck<F> {
    /// Create the prover sum-check instance for matvec precompile
    pub fn new_prover<ProofTranscript: Transcript>(
        a: Vec<i64>,
        r_x: Vec<F>,
        rv_claim_res: F,
        index: usize,
    ) -> Self {
        let eq_r = EqPolynomial::evals(&r_x);
        // num rows in a
        let m = r_x.len().pow2();
        // num cols in a
        let n = a.len() / m;
        let a_r = (0..n)
            .into_par_iter()
            .map(|y| (0..m).map(|x| F::from_i64(a[x * n + y]) * eq_r[x]).sum())
            .collect::<Vec<_>>();
        debug_assert_eq!(rv_claim_res, a_r.iter().sum());
        Self {
            prover_state: Some(ExecutionProverState {
                a_r: MultilinearPolynomial::from(a_r),
            }),
            r_x,
            rv_claim_res,
            index,
            num_rounds: n.log_2(),
        }
    }

    /// Create the verifier sum-check instance for matvec precompile
    pub fn new_verifier(r_x: Vec<F>, rv_claim_res: F, index: usize, m: usize) -> Self {
        Self {
            prover_state: None,
            r_x,
            rv_claim_res,
            index,
            num_rounds: m.log_2(),
        }
    }
}

/// Stores the "witness" polynomials for the execution sumcheck.
/// These polynomials are virtual
pub struct ExecutionProverState<F: JoltField> {
    a_r: MultilinearPolynomial<F>,
}
