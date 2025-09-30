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

use std::{cell::RefCell, rc::Rc};

use crate::jolt::{
    dag::state_manager::StateManager,
    pcs::{ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator},
    sumcheck::SumcheckInstance,
    witness::{CommittedPolynomial, VirtualPolynomial},
};

struct ValFinalSumcheckProverState<F: JoltField> {
    inc: MultilinearPolynomial<F>,
    wa: MultilinearPolynomial<F>,
}

/// This sumcheck virtualizes Val_final(k) as:
/// Val_final(k) = \sum_k Inc(j) * wa(k, j)
///   or equivalently:
/// Val_final(k) = \sum_k Inc(j) * wa(k, j)
/// We feed the output claim Val_final(r_address) from the precompiles
/// into this sumcheck, which reduces it to claims about `Inc` and `wa`.
pub struct ValFinalSumcheck<F: JoltField> {
    T: usize,
    prover_state: Option<ValFinalSumcheckProverState<F>>,
    val_final_claim: F,
}

impl<F: JoltField> ValFinalSumcheck<F> {
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (preprocessing, trace, _) = state_manager.get_prover_data();
        let T = trace.len();

        let r_address = state_manager
            .get_virtual_polynomial_opening(
                VirtualPolynomial::ValFinal,
                SumcheckId::MatVecReadChecking,
            )
            .0
            .r;

        // Compute the size-K table storing all eq(r_address, k) evaluations for
        // k \in {0, 1}^log(K)
        let eq_r_address = EqPolynomial::evals(&r_address);

        let span = tracing::span!(tracing::Level::INFO, "compute wa(r_address, j)");
        let _guard = span.enter();

        // Compute the wa polynomial using the above table
        let wa: Vec<F> = preprocessing
            .bytecode()
            .par_iter()
            .map(|instr| eq_r_address[instr.td as usize])
            .collect();
        let wa = MultilinearPolynomial::from(wa);

        drop(_guard);
        drop(span);

        let inc = CommittedPolynomial::TdIncS.generate_witness(preprocessing, trace);

        let val_final_claim = state_manager
            .get_virtual_polynomial_opening(
                VirtualPolynomial::ValFinal,
                SumcheckId::MatVecReadChecking,
            )
            .1;

        Self {
            T,
            prover_state: Some(ValFinalSumcheckProverState { wa, inc }),
            val_final_claim,
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (_, _, T) = state_manager.get_verifier_data();
        let val_final_claim = state_manager
            .get_virtual_polynomial_opening(
                VirtualPolynomial::ValFinal,
                SumcheckId::MatVecReadChecking,
            )
            .1;
        Self {
            T,
            prover_state: None,
            val_final_claim,
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for ValFinalSumcheck<F> {
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.T.log_2()
    }

    fn input_claim(&self) -> F {
        self.val_final_claim
    }

    #[tracing::instrument(skip_all, name = "ValFinalSumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, _: usize, _previous_claim: F) -> Vec<F> {
        const DEGREE: usize = 2;

        let ValFinalSumcheckProverState { inc, wa, .. } = self.prover_state.as_ref().unwrap();

        let univariate_poly_evals: [F; DEGREE] = (0..inc.len() / 2)
            .into_par_iter()
            .map(|j| {
                let inc_evals = inc.sumcheck_evals_array::<DEGREE>(j, BindingOrder::HighToLow);
                let wa_evals = wa.sumcheck_evals_array::<DEGREE>(j, BindingOrder::HighToLow);
                [inc_evals[0] * wa_evals[0], inc_evals[1] * wa_evals[1]]
            })
            .reduce(
                || [F::zero(); DEGREE],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

        univariate_poly_evals.to_vec()
    }

    #[tracing::instrument(skip_all, name = "ValFinalSumcheck::bind")]
    fn bind(&mut self, r_j: F, _: usize) {
        let ValFinalSumcheckProverState { inc, wa, .. } = self.prover_state.as_mut().unwrap();
        rayon::join(
            || inc.bind_parallel(r_j, BindingOrder::HighToLow),
            || wa.bind_parallel(r_j, BindingOrder::HighToLow),
        );
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        _: &[F],
    ) -> F {
        let accumulator = accumulator.as_ref().unwrap().borrow();
        let inc_claim = accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::TdIncS,
                SumcheckId::PrecompileValFinal,
            )
            .1;
        let wa_claim = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::TdWa, SumcheckId::PrecompileValFinal)
            .1;

        inc_claim * wa_claim
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        r_cycle_prime: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let ValFinalSumcheckProverState { inc, wa, .. } = self.prover_state.as_ref().unwrap();

        let r_address = accumulator
            .borrow()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::ValFinal,
                SumcheckId::MatVecReadChecking,
            )
            .0;
        let wa_opening_point =
            OpeningPoint::new([r_address.r.as_slice(), r_cycle_prime.r.as_slice()].concat());

        accumulator.borrow_mut().append_dense(
            vec![CommittedPolynomial::TdIncS],
            SumcheckId::PrecompileValFinal,
            r_cycle_prime.r,
            &[inc.final_sumcheck_claim()],
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::TdWa,
            SumcheckId::PrecompileValFinal,
            wa_opening_point,
            wa.final_sumcheck_claim(),
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        r_cycle_prime: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let r_address = accumulator
            .borrow()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::ValFinal,
                SumcheckId::MatVecReadChecking,
            )
            .0;
        let wa_opening_point =
            OpeningPoint::new([r_address.r.as_slice(), r_cycle_prime.r.as_slice()].concat());

        accumulator.borrow_mut().append_dense(
            vec![CommittedPolynomial::TdIncS],
            SumcheckId::PrecompileValFinal,
            r_cycle_prime.r,
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::TdWa,
            SumcheckId::PrecompileValFinal,
            wa_opening_point,
        );
    }
}
