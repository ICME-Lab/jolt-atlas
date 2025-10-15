use jolt_core::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{BIG_ENDIAN, OpeningPoint},
    },
    transcripts::Transcript,
    utils::math::Math,
};
use rayon::prelude::*;

use crate::{
    jolt::{
        dag::state_manager::StateManager, pcs::SumcheckId, sumcheck::SumcheckInstance,
        witness::VirtualPolynomial,
    },
    utils::precompile_pp::PrecompilePreprocessingTrait,
};

pub struct ReadCheckingReduceSumcheck<F: JoltField> {
    prover_state: Option<ReadCheckingReduceProverState<F>>,
    rv_claim: F,
    gamma: F,
    K: usize,
    index: usize,
}

impl<F: JoltField> SumcheckInstance<F> for ReadCheckingReduceSumcheck<F> {
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
                let ra_res_evals = prover_state
                    .ra_res
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                [
                    val_evals[0] * (ra_a_evals[0] + self.gamma * ra_res_evals[0]), // eval at 0
                    val_evals[1] * (ra_a_evals[1] + self.gamma * ra_res_evals[1]), // eval at 2
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
                    .ra_res
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
        let ra_res_claim = prover_state.ra_res.final_sumcheck_claim();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::ValFinal,
            SumcheckId::MatVecReadChecking,
            opening_point.clone(),
            val_final_claim,
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::ReduceRaA(self.index),
            SumcheckId::ReduceSumReadChecking,
            opening_point.clone(),
            ra_a_claim,
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::ReduceRaRes(self.index),
            SumcheckId::ReduceSumReadChecking,
            opening_point.clone(),
            ra_res_claim,
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
            VirtualPolynomial::ReduceRaA(self.index),
            SumcheckId::ReduceSumReadChecking,
        );
        let (_, res_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::ReduceRaRes(self.index),
            SumcheckId::ReduceSumReadChecking,
        );
        val_final_claim * (a_claim + self.gamma * res_claim)
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
            VirtualPolynomial::ReduceRaA(self.index),
            SumcheckId::ReduceSumReadChecking,
            opening_point.clone(),
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::ReduceRaRes(self.index),
            SumcheckId::ReduceSumReadChecking,
            opening_point.clone(),
        );
    }
}

impl<F: JoltField> ReadCheckingReduceSumcheck<F> {
    pub fn new_prover(
        val_final: &[i64],
        sm: &StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        index: usize,
    ) -> Self {
        let (preprocessing, _, _) = sm.get_prover_data();
        let reduce_pp = &preprocessing.shared.precompiles.reduce_sum[index];
        let K = sm.get_memory_K();
        let gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let val_final: MultilinearPolynomial<F> = MultilinearPolynomial::from(val_final.to_vec());
        let (r_y, rv_claim_a) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::ReduceA(index),
            SumcheckId::ReduceSumExecution,
        );
        let r_y = r_y.r;
        let (r_x, rv_claim_reduce) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::ReduceRes(index),
            SumcheckId::ReduceSumExecution,
        );
        let r_x = r_x.r;
        let r_a = [r_x.clone(), r_y.clone()].concat();
        let rv_claim = rv_claim_a + gamma * rv_claim_reduce;
        let ra_a = reduce_pp.compute_ra(&r_a, |m| &m.a, K);
        let ra_res = reduce_pp.compute_ra(&r_x, |m| &m.res, K);
        Self {
            prover_state: Some(ReadCheckingReduceProverState {
                ra_a,
                ra_res,
                val_final,
            }),
            rv_claim,
            gamma,
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
        let (_, rv_claim_a) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::ReduceA(index),
            SumcheckId::ReduceSumExecution,
        );
        let (_, rv_claim_res) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::ReduceRes(index),
            SumcheckId::ReduceSumExecution,
        );
        let rv_claim = rv_claim_a + gamma * rv_claim_res;
        Self {
            prover_state: None,
            rv_claim,
            gamma,
            K,
            index,
        }
    }
}

pub struct ReadCheckingReduceProverState<F: JoltField> {
    ra_a: MultilinearPolynomial<F>,
    ra_res: MultilinearPolynomial<F>,
    val_final: MultilinearPolynomial<F>,
}
