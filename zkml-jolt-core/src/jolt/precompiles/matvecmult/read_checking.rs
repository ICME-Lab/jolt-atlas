use jolt_core::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{BIG_ENDIAN, OpeningPoint},
    },
    transcripts::Transcript,
    utils::{math::Math, thread::unsafe_allocate_zero_vec},
};
use rayon::prelude::*;

use crate::jolt::{
    dag::state_manager::StateManager, pcs::SumcheckId, sumcheck::SumcheckInstance,
    witness::VirtualPolynomial,
};

pub struct ReadCheckingABCSumcheck<F: JoltField> {
    prover_state: Option<ReadCheckingABCProverState<F>>,
    rv_claim: F,
    gamma: F,
    gamma_sqr: F,
    K: usize,
    index: usize,
}

impl<F: JoltField> SumcheckInstance<F> for ReadCheckingABCSumcheck<F> {
    fn num_rounds(&self) -> usize {
        self.K.log_2()
    }

    fn degree(&self) -> usize {
        2
    }

    fn input_claim(&self) -> F {
        self.rv_claim
    }

    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        const DEGREE: usize = 2;
        let univariate_poly_evals: [F; DEGREE] = (0..prover_state.val_final.len() / 2)
            .into_par_iter()
            .map(|i| {
                let val_evals = prover_state
                    .val_final
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let ra_a_evals = prover_state
                    .ra_a
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let ra_b_evals = prover_state
                    .ra_b
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let ra_c_evals = prover_state
                    .ra_c
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                [
                    val_evals[0]
                        * (ra_a_evals[0]
                            + self.gamma * ra_b_evals[0]
                            + self.gamma_sqr * ra_c_evals[0]), // eval at 0
                    val_evals[1]
                        * (ra_a_evals[1]
                            + self.gamma * ra_b_evals[1]
                            + self.gamma_sqr * ra_c_evals[1]), // eval at 2
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

        rayon::scope(|s| {
            s.spawn(|_| {
                prover_state
                    .val_final
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            });
            s.spawn(|_| {
                prover_state
                    .ra_a
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            });
            s.spawn(|_| {
                prover_state
                    .ra_b
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            });
            s.spawn(|_| {
                prover_state
                    .ra_c
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            });
        });
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
        let val_final_claim = prover_state.val_final.final_sumcheck_claim();
        let ra_a_claim = prover_state.ra_a.final_sumcheck_claim();
        let ra_b_claim = prover_state.ra_b.final_sumcheck_claim();
        let ra_c_claim = prover_state.ra_c.final_sumcheck_claim();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::ValFinal,
            SumcheckId::MatVecReadChecking,
            opening_point.clone(),
            val_final_claim,
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::RaA(self.index),
            SumcheckId::MatVecReadChecking,
            opening_point.clone(),
            ra_a_claim,
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::RaB(self.index),
            SumcheckId::MatVecReadChecking,
            opening_point.clone(),
            ra_b_claim,
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::RaC(self.index),
            SumcheckId::MatVecReadChecking,
            opening_point.clone(),
            ra_c_claim,
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
        opening_accumulator: Option<
            std::rc::Rc<std::cell::RefCell<crate::jolt::pcs::VerifierOpeningAccumulator<F>>>,
        >,
        _r: &[F],
    ) -> F {
        let accumulator = opening_accumulator.as_ref().unwrap();
        let (_, val_final_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::ValFinal,
            SumcheckId::MatVecReadChecking,
        );
        let (_, a_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RaA(self.index),
            SumcheckId::MatVecReadChecking,
        );
        let (_, b_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RaB(self.index),
            SumcheckId::MatVecReadChecking,
        );
        let (_, c_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RaC(self.index),
            SumcheckId::MatVecReadChecking,
        );
        val_final_claim * (a_claim + self.gamma * b_claim + self.gamma_sqr * c_claim)
    }

    fn cache_openings_verifier(
        &self,
        accumulator: std::rc::Rc<
            std::cell::RefCell<crate::jolt::pcs::VerifierOpeningAccumulator<F>>,
        >,
        opening_point: jolt_core::poly::opening_proof::OpeningPoint<BIG_ENDIAN, F>,
    ) {
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::ValFinal,
            SumcheckId::MatVecReadChecking,
            opening_point.clone(),
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::RaA(self.index),
            SumcheckId::MatVecReadChecking,
            opening_point.clone(),
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::RaB(self.index),
            SumcheckId::MatVecReadChecking,
            opening_point.clone(),
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::RaC(self.index),
            SumcheckId::MatVecReadChecking,
            opening_point,
        );
    }
}

impl<F: JoltField> ReadCheckingABCSumcheck<F> {
    pub fn new_prover(
        val_final: &[i64],
        sm: &StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        index: usize,
    ) -> Self {
        let (preprocessing, _, _) = sm.get_prover_data();
        let matvec_pp = &preprocessing.shared.precompiles.matvec_instances[index];
        let K = sm.get_memory_K();
        let gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let gamma_sqr = gamma.square();
        let val_final: MultilinearPolynomial<F> = MultilinearPolynomial::from(val_final.to_vec());
        let (r_a, rv_claim_a) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::AVec(index),
            SumcheckId::MatVecExecution,
        );
        let r_a = r_a.r;
        let (_, rv_claim_b) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::BMat(index),
            SumcheckId::MatVecExecution,
        );
        let (r_c, rv_claim_c) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::CRes(index),
            SumcheckId::MatVecExecution,
        );
        let r_c = r_c.r;
        let r_b = [r_c.clone(), r_a.clone()].concat();
        let rv_claim = rv_claim_a + gamma * rv_claim_b + gamma_sqr * rv_claim_c;
        let E_a = EqPolynomial::evals(&r_a);
        let E_b = EqPolynomial::evals(&r_b);
        let E_c = EqPolynomial::evals(&r_c);
        let mut ra_a = unsafe_allocate_zero_vec::<F>(K);
        let mut ra_b = unsafe_allocate_zero_vec::<F>(K);
        let mut ra_c = unsafe_allocate_zero_vec::<F>(K);
        matvec_pp
            .a
            .iter()
            .enumerate()
            .for_each(|(j, &k)| ra_a[k] += E_a[j]);
        matvec_pp
            .b
            .iter()
            .enumerate()
            .for_each(|(j, &k)| ra_b[k] += E_b[j]);
        matvec_pp
            .c
            .iter()
            .enumerate()
            .for_each(|(j, &k)| ra_c[k] += E_c[j]);
        Self {
            prover_state: Some(ReadCheckingABCProverState {
                ra_a: MultilinearPolynomial::from(ra_a),
                ra_b: MultilinearPolynomial::from(ra_b),
                ra_c: MultilinearPolynomial::from(ra_c),
                val_final,
            }),
            rv_claim,
            gamma,
            gamma_sqr,
            K,
            index,
        }
    }

    pub fn new_verifier(
        index: usize,
        sm: &StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let K = sm.get_memory_K();
        let gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let gamma_sqr = gamma.square();
        let (_, rv_claim_a) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::AVec(index),
            SumcheckId::MatVecExecution,
        );
        let (_, rv_claim_b) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::BMat(index),
            SumcheckId::MatVecExecution,
        );
        let (_, rv_claim_c) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::CRes(index),
            SumcheckId::MatVecExecution,
        );
        let rv_claim = rv_claim_a + gamma * rv_claim_b + gamma_sqr * rv_claim_c;
        Self {
            prover_state: None,
            rv_claim,
            gamma,
            gamma_sqr,
            K,
            index,
        }
    }
}

pub struct ReadCheckingABCProverState<F: JoltField> {
    ra_a: MultilinearPolynomial<F>,
    ra_b: MultilinearPolynomial<F>,
    ra_c: MultilinearPolynomial<F>,
    val_final: MultilinearPolynomial<F>,
}
