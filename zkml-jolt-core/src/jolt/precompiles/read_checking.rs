use crate::jolt::{
    dag::state_manager::StateManager, pcs::SumcheckId, sumcheck::SumcheckInstance,
    witness::VirtualPolynomial,
};
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

pub struct ReadCheckingABCSumcheck<F: JoltField> {
    prover_state: Option<ReadCheckingABCProverState<F>>,
    rv_claim: F,
    gamma_powers: Vec<F>,
    K: usize,
    num_instances: usize,
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
                let ra_evals: [F; DEGREE] = (0..self.num_instances)
                    .map(|index| {
                        let ra_a_evals = prover_state.ra_a[index]
                            .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                        let ra_b_evals = prover_state.ra_b[index]
                            .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                        let ra_c_evals = prover_state.ra_c[index]
                            .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                        [
                            self.gamma_powers[index * 3] * ra_a_evals[0]
                                + self.gamma_powers[index * 3 + 1] * ra_b_evals[0]
                                + self.gamma_powers[index * 3 + 2] * ra_c_evals[0],
                            self.gamma_powers[index * 3] * ra_a_evals[1]
                                + self.gamma_powers[index * 3 + 1] * ra_b_evals[1]
                                + self.gamma_powers[index * 3 + 2] * ra_c_evals[1],
                        ]
                    })
                    .fold([F::zero(); DEGREE], |running, new| {
                        [running[0] + new[0], running[1] + new[1]]
                    });
                [val_evals[0] * ra_evals[0], val_evals[1] * ra_evals[1]]
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
                    .bind_parallel(r_j, BindingOrder::HighToLow);
            });

            // Parallelize all ra_a bindings
            s.spawn(|_| {
                prover_state.ra_a.par_iter_mut().for_each(|poly| {
                    poly.bind_parallel(r_j, BindingOrder::HighToLow);
                });
            });

            // Parallelize all ra_b bindings
            s.spawn(|_| {
                prover_state.ra_b.par_iter_mut().for_each(|poly| {
                    poly.bind_parallel(r_j, BindingOrder::HighToLow);
                });
            });

            // Parallelize all ra_c bindings
            s.spawn(|_| {
                prover_state.ra_c.par_iter_mut().for_each(|poly| {
                    poly.bind_parallel(r_j, BindingOrder::HighToLow);
                });
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
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::ValFinal,
            SumcheckId::PrecompileReadChecking,
            opening_point.clone(),
            prover_state.val_final.final_sumcheck_claim(),
        );
        for index in 0..self.num_instances {
            accumulator.borrow_mut().append_virtual(
                VirtualPolynomial::RaAPrecompile(index),
                SumcheckId::PrecompileReadChecking,
                opening_point.clone(),
                prover_state.ra_a[index].final_sumcheck_claim(),
            );
            accumulator.borrow_mut().append_virtual(
                VirtualPolynomial::RaBPrecompile(index),
                SumcheckId::PrecompileReadChecking,
                opening_point.clone(),
                prover_state.ra_b[index].final_sumcheck_claim(),
            );
            accumulator.borrow_mut().append_virtual(
                VirtualPolynomial::RaCPrecompile(index),
                SumcheckId::PrecompileReadChecking,
                opening_point.clone(),
                prover_state.ra_c[index].final_sumcheck_claim(),
            );
        }
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
            SumcheckId::PrecompileReadChecking,
        );
        let mut ra_claim = F::zero();
        for index in 0..self.num_instances {
            let (_, a_claim) = accumulator.borrow().get_virtual_polynomial_opening(
                VirtualPolynomial::RaAPrecompile(index),
                SumcheckId::PrecompileReadChecking,
            );
            let (_, b_claim) = accumulator.borrow().get_virtual_polynomial_opening(
                VirtualPolynomial::RaBPrecompile(index),
                SumcheckId::PrecompileReadChecking,
            );
            let (_, c_claim) = accumulator.borrow().get_virtual_polynomial_opening(
                VirtualPolynomial::RaCPrecompile(index),
                SumcheckId::PrecompileReadChecking,
            );
            ra_claim += self.gamma_powers[index * 3] * a_claim
                + self.gamma_powers[index * 3 + 1] * b_claim
                + self.gamma_powers[index * 3 + 2] * c_claim;
        }
        val_final_claim * ra_claim
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
            SumcheckId::PrecompileReadChecking,
            opening_point.clone(),
        );
        for index in 0..self.num_instances {
            accumulator.borrow_mut().append_virtual(
                VirtualPolynomial::RaAPrecompile(index),
                SumcheckId::PrecompileReadChecking,
                opening_point.clone(),
            );
            accumulator.borrow_mut().append_virtual(
                VirtualPolynomial::RaBPrecompile(index),
                SumcheckId::PrecompileReadChecking,
                opening_point.clone(),
            );
            accumulator.borrow_mut().append_virtual(
                VirtualPolynomial::RaCPrecompile(index),
                SumcheckId::PrecompileReadChecking,
                opening_point.clone(),
            );
        }
    }
}

impl<F: JoltField> ReadCheckingABCSumcheck<F> {
    pub fn new_prover(
        sm: &StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let K = sm.get_memory_K();
        let pp = sm.get_precompile_preprocessing();
        let num_instances = pp.instances.len();
        let num_polys = 3 * num_instances;
        let gamma: F = sm.get_transcript().borrow_mut().challenge_scalar();
        let mut gamma_powers = vec![F::one()];
        for _ in 0..(num_polys - 1) {
            gamma_powers.push(gamma * gamma_powers.last().unwrap());
        }
        let val_final: MultilinearPolynomial<F> =
            MultilinearPolynomial::from(sm.get_val_final().to_vec());
        let mut rv_claim = F::zero();
        let mut ra_a = Vec::with_capacity(num_instances);
        let mut ra_b = Vec::with_capacity(num_instances);
        let mut ra_c = Vec::with_capacity(num_instances);
        for index in 0..num_instances {
            let (r_a, rv_claim_a) = sm.get_virtual_polynomial_opening(
                VirtualPolynomial::PrecompileA(index),
                SumcheckId::PrecompileExecution,
            );
            let (r_b, rv_claim_b) = sm.get_virtual_polynomial_opening(
                VirtualPolynomial::PrecompileB(index),
                SumcheckId::PrecompileExecution,
            );
            let (r_c, rv_claim_c) = sm.get_virtual_polynomial_opening(
                VirtualPolynomial::PrecompileC(index),
                SumcheckId::PrecompileExecution,
            );
            rv_claim += gamma_powers[index * 3] * rv_claim_a
                + gamma_powers[index * 3 + 1] * rv_claim_b
                + gamma_powers[index * 3 + 2] * rv_claim_c;
            ra_a.push(pp.instances[index].compute_ra(&r_a.r, |m| &m.a_addr, K));
            ra_b.push(pp.instances[index].compute_ra(&r_b.r, |m| &m.b_addr, K));
            ra_c.push(pp.instances[index].compute_ra(&r_c.r, |m| &m.c_addr, K));
        }
        Self {
            prover_state: Some(ReadCheckingABCProverState {
                ra_a,
                ra_b,
                ra_c,
                val_final,
            }),
            rv_claim,
            gamma_powers,
            K,
            num_instances,
        }
    }

    pub fn new_verifier(
        sm: &StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let K = sm.get_memory_K();
        let (pp, _, _) = sm.get_verifier_data();
        let num_instances = pp.shared.precompiles.instances.len();
        let num_polys = 3 * num_instances;
        let gamma: F = sm.get_transcript().borrow_mut().challenge_scalar();
        let mut gamma_powers = vec![F::one()];
        for _ in 0..(num_polys - 1) {
            gamma_powers.push(gamma * gamma_powers.last().unwrap());
        }
        let mut rv_claim = F::zero();
        for index in 0..num_instances {
            let (_, rv_claim_a) = sm.get_virtual_polynomial_opening(
                VirtualPolynomial::PrecompileA(index),
                SumcheckId::PrecompileExecution,
            );
            let (_, rv_claim_b) = sm.get_virtual_polynomial_opening(
                VirtualPolynomial::PrecompileB(index),
                SumcheckId::PrecompileExecution,
            );
            let (_, rv_claim_c) = sm.get_virtual_polynomial_opening(
                VirtualPolynomial::PrecompileC(index),
                SumcheckId::PrecompileExecution,
            );
            rv_claim += gamma_powers[index * 3] * rv_claim_a
                + gamma_powers[index * 3 + 1] * rv_claim_b
                + gamma_powers[index * 3 + 2] * rv_claim_c
        }
        Self {
            prover_state: None,
            rv_claim,
            gamma_powers,
            K,
            num_instances,
        }
    }
}

pub struct ReadCheckingABCProverState<F: JoltField> {
    ra_a: Vec<MultilinearPolynomial<F>>,
    ra_b: Vec<MultilinearPolynomial<F>>,
    ra_c: Vec<MultilinearPolynomial<F>>,
    val_final: MultilinearPolynomial<F>,
}
