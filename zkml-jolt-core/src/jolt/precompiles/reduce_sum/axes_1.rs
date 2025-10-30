use jolt_core::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{BIG_ENDIAN, OpeningPoint},
    },
    transcripts::Transcript,
    utils::math::Math,
};
use rayon::prelude::*;

use crate::jolt::{
    dag::state_manager::StateManager,
    pcs::{SumcheckId, VerifierOpeningAccumulator},
    sumcheck::SumcheckInstance,
    witness::VirtualPolynomial,
};

pub struct ExecutionSumcheck<F: JoltField> {
    prover_state: Option<ExecutionProverState<F>>,
    r_x: Vec<F>,
    rv_claim_c: F,
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
        self.rv_claim_c
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
        let r_a = [self.r_x.clone(), opening_point.r.clone()].concat();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::PrecompileA(self.index),
            SumcheckId::PrecompileExecution,
            r_a.into(),
            prover_state.a_r.final_sumcheck_claim(),
        );
        let r_b = vec![F::zero()];
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::PrecompileB(self.index),
            SumcheckId::PrecompileExecution,
            r_b.into(),
            F::zero(),
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::PrecompileC(self.index),
            SumcheckId::PrecompileExecution,
            self.r_x.clone().into(),
            self.rv_claim_c,
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
                VirtualPolynomial::PrecompileA(self.index),
                SumcheckId::PrecompileExecution,
            )
            .1
    }

    fn cache_openings_verifier(
        &self,
        accumulator: std::rc::Rc<
            std::cell::RefCell<crate::jolt::pcs::VerifierOpeningAccumulator<F>>,
        >,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let r_a = [self.r_x.clone(), opening_point.r.clone()].concat();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::PrecompileA(self.index),
            SumcheckId::PrecompileExecution,
            r_a.into(),
        );
        let r_b = vec![F::zero()];
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::PrecompileB(self.index),
            SumcheckId::PrecompileExecution,
            r_b.into(),
        );
    }
}

impl<F: JoltField> ExecutionSumcheck<F> {
    /// Create the prover sum-check instance for matvec precompile
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        index: usize,
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        // Get the final memory state (val_final) from the prover
        let final_memory_state = sm.get_val_final();
        let (pp, _, _) = sm.get_prover_data();
        let pp = &pp.shared.precompiles.instances[index];

        // Get the size of the result vector and generate a random challenge
        let m = pp.c_dims[0];
        let r_x: Vec<F> = sm.get_transcript().borrow_mut().challenge_vector(m.log_2());

        // Extract values for operands a and b from memory
        let E = EqPolynomial::evals(&r_x);
        let rv_a = pp.extract_rv(final_memory_state, |m| &m.a_addr);
        let rv_claim_c: F = pp
            .c_addr
            .iter()
            .enumerate()
            .map(|(j, &k)| E[j] * F::from_i64(final_memory_state[k]))
            .sum();
        Self::init_prover(rv_a, r_x, E, rv_claim_c, index)
    }

    /// Create the prover sum-check instance for matvec precompile
    pub fn init_prover(
        a: Vec<i64>,
        r_x: Vec<F>,
        eq_rx: Vec<F>,
        rv_claim_c: F,
        index: usize,
    ) -> Self {
        // num rows in a
        let m = r_x.len().pow2();
        // num cols in a
        let n = a.len() / m;
        let a_r = (0..n)
            .into_par_iter()
            .map(|y| (0..m).map(|x| F::from_i64(a[x * n + y]) * eq_rx[x]).sum())
            .collect::<Vec<_>>();
        debug_assert_eq!(rv_claim_c, a_r.iter().sum());
        Self {
            prover_state: Some(ExecutionProverState {
                a_r: MultilinearPolynomial::from(a_r),
            }),
            r_x,
            rv_claim_c,
            index,
            num_rounds: n.log_2(),
        }
    }

    /// Create the verifier sum-check instance for matvec precompile
    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        index: usize,
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        // Get preprocessing data for this matrix-vector multiplication
        let (pp, _, _) = sm.get_verifier_data();
        let pp = &pp.shared.precompiles.instances[index];

        let m = pp.c_dims[0];
        let n = pp.a_dims[1];

        // Generate the same random challenge as the prover (using the transcript)
        let r_x: Vec<F> = sm.get_transcript().borrow_mut().challenge_vector(m.log_2());

        // cache r_x
        let verifier_accumulator = sm.get_verifier_accumulator();
        verifier_accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::PrecompileC(index),
            SumcheckId::PrecompileExecution,
            r_x.clone().into(),
        );
        let rv_claim_c = sm
            .get_virtual_polynomial_opening(
                VirtualPolynomial::PrecompileC(index),
                SumcheckId::PrecompileExecution,
            )
            .1;
        Self {
            prover_state: None,
            r_x,
            rv_claim_c,
            index,
            num_rounds: n.log_2(),
        }
    }
}

/// Stores the "witness" polynomials for the execution sumcheck.
/// These polynomials are virtual
pub struct ExecutionProverState<F: JoltField> {
    a_r: MultilinearPolynomial<F>,
}
