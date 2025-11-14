use std::{cell::RefCell, rc::Rc};

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
    dag::state_manager::StateManager,
    pcs::{ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator},
    sumcheck::SumcheckInstance,
    witness::{CommittedPolynomial, VirtualPolynomial},
};

pub struct ValEvaluationProverState<F: JoltField> {
    pub inc: MultilinearPolynomial<F>,
    pub wa: MultilinearPolynomial<F>,
    pub lt: MultilinearPolynomial<F>,
}

pub(crate) struct ValEvaluationSumcheck<F: JoltField> {
    pub r_address: Vec<F>,
    pub input_claim: F,
    pub num_rounds: usize,
    pub r_cycle: Vec<F>,
    pub prover_state: Option<ValEvaluationProverState<F>>,
}

impl<F: JoltField> ValEvaluationSumcheck<F> {
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (preprocessing, trace, _) = state_manager.get_prover_data();
        let K = preprocessing.memory_K();
        let accumulator = state_manager.get_prover_accumulator();

        // Get val_claim from the accumulator (from stage 2 RegistersReadWriteChecking)
        let (opening_point, val_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        );

        // The opening point is r_address || r_cycle
        let r_address_len = K.ilog2() as usize;
        let (r_address_slice, r_cycle_slice) = opening_point.split_at(r_address_len);
        let r_address: Vec<F> = r_address_slice.into();
        let r_cycle: Vec<F> = r_cycle_slice.into();

        let inc = CommittedPolynomial::TdInc.generate_witness(preprocessing, trace);

        let eq_r_address = EqPolynomial::evals(&r_address);
        let wa: Vec<F> = preprocessing
            .bytecode()
            .par_iter()
            .map(|instr| eq_r_address[instr.td as usize])
            .collect();
        let wa = MultilinearPolynomial::from(wa);

        let T = trace.len();
        let mut lt: Vec<F> = unsafe_allocate_zero_vec(T);
        for (i, r) in r_cycle.iter().rev().enumerate() {
            let (evals_left, evals_right) = lt.split_at_mut(1 << i);
            evals_left
                .par_iter_mut()
                .zip(evals_right.par_iter_mut())
                .for_each(|(x, y)| {
                    *y = *x * r;
                    *x += *r - *y;
                });
        }
        let lt = MultilinearPolynomial::from(lt);

        let num_rounds = r_cycle.len().pow2().log_2();
        Self {
            input_claim: val_claim,
            r_address,
            num_rounds,
            r_cycle,
            prover_state: Some(ValEvaluationProverState { inc, wa, lt }),
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (preprocessing, _, trace_length) = state_manager.get_verifier_data();
        let K = preprocessing.memory_K();
        let accumulator = state_manager.get_verifier_accumulator();
        // Get val_claim from the accumulator (from stage 2 RegistersReadWriteChecking)
        let (opening_point, val_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        );

        // The opening point is r_address || r_cycle
        let r_address_len = K.ilog2() as usize;
        let (r_address_slice, r_cycle_slice) = opening_point.split_at(r_address_len);
        let r_address: Vec<F> = r_address_slice.into();
        let r_cycle: Vec<F> = r_cycle_slice.into();

        Self {
            input_claim: val_claim,
            r_address,
            num_rounds: trace_length.log_2(),
            r_cycle,
            prover_state: None,
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for ValEvaluationSumcheck<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    #[tracing::instrument(
        skip_all,
        name = "RegistersValEvaluationSumcheck::compute_prover_message"
    )]
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        const DEGREE: usize = 3;
        let univariate_poly_evals: [F; 3] = (0..prover_state.inc.len() / 2)
            .into_par_iter()
            .map(|i| {
                let inc_evals = prover_state
                    .inc
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let wa_evals = prover_state
                    .wa
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let lt_evals = prover_state
                    .lt
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);

                [
                    inc_evals[0] * wa_evals[0] * lt_evals[0],
                    inc_evals[1] * wa_evals[1] * lt_evals[1],
                    inc_evals[2] * wa_evals[2] * lt_evals[2],
                ]
            })
            .reduce(
                || [F::zero(); 3],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                    ]
                },
            );

        univariate_poly_evals.to_vec()
    }

    #[tracing::instrument(skip_all, name = "RegistersValEvaluationSumcheck::bind")]
    fn bind(&mut self, r_j: F, _round: usize) {
        if let Some(prover_state) = &mut self.prover_state {
            [
                &mut prover_state.inc,
                &mut prover_state.wa,
                &mut prover_state.lt,
            ]
            .par_iter_mut()
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
        }
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F],
    ) -> F {
        // Compute LT(r_cycle', r_cycle)
        let mut lt_eval = F::zero();
        let mut eq_term = F::one();
        for (x, y) in r.iter().zip(self.r_cycle.iter()) {
            lt_eval += (F::one() - x) * y * eq_term;
            eq_term *= F::one() - x - y + *x * y + *x * y;
        }

        let accumulator = accumulator.as_ref().unwrap();
        let (_, inc_claim) = accumulator.borrow().get_committed_polynomial_opening(
            CommittedPolynomial::TdInc,
            SumcheckId::RegistersValEvaluation,
        );
        let (_, wa_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::TdWa,
            SumcheckId::RegistersValEvaluation,
        );

        // Return inc_claim * wa_claim * lt_eval
        inc_claim * wa_claim * lt_eval
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        r_cycle: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let inc_claim = prover_state.inc.final_sumcheck_claim();
        let wa_claim = prover_state.wa.final_sumcheck_claim();

        // Append claims to accumulator
        accumulator.borrow_mut().append_dense(
            vec![CommittedPolynomial::TdInc],
            SumcheckId::RegistersValEvaluation,
            r_cycle.r.clone(),
            &[inc_claim],
        );

        let r = [self.r_address.as_slice(), r_cycle.r.as_slice()].concat();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::TdWa,
            SumcheckId::RegistersValEvaluation,
            OpeningPoint::new(r),
            wa_claim,
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        r_cycle: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        // Append claims to accumulator
        accumulator.borrow_mut().append_dense(
            vec![CommittedPolynomial::TdInc],
            SumcheckId::RegistersValEvaluation,
            r_cycle.r.clone(),
        );

        let r = [self.r_address.as_slice(), r_cycle.r.as_slice()].concat();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::TdWa,
            SumcheckId::RegistersValEvaluation,
            OpeningPoint::new(r),
        );
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    use crate::jolt::{
        JoltProverPreprocessing, JoltSharedPreprocessing, JoltVerifierPreprocessing,
        bytecode::BytecodePreprocessing, dag::state_manager::StateManager, pcs::SumcheckId,
        precompiles::PrecompilePreprocessing, sumcheck::SingleSumcheck, trace::trace,
        witness::VirtualPolynomial,
    };
    use ark_bn254::Fr;
    use ark_std::Zero;
    use jolt_core::{
        poly::{
            commitment::mock::MockCommitScheme,
            eq_poly::EqPolynomial,
            opening_proof::{BIG_ENDIAN, OpeningPoint},
        },
        transcripts::{Blake2bTranscript, Transcript},
        utils::index_to_field_bitvector,
    };
    use onnx_tracer::{ProgramIO, builder, graph::model::Model, model, tensor::Tensor};

    fn evaluate_lt_mle<F: JoltField>(x: &[F], r: &[F]) -> F {
        assert_eq!(x.len(), r.len());
        let mut lt = F::zero();
        let mut eq = F::one();
        for (&x, &r) in x.iter().zip(r.iter()) {
            lt += (F::one() - x) * r * eq;
            eq *= x * r + (F::one() - x) * (F::one() - r);
        }
        lt
    }

    pub fn test_val_evaluation_sumcheck<ModelFunc>(model_fn: ModelFunc, input: Tensor<i32>)
    where
        ModelFunc: Fn() -> Model + Copy,
    {
        let bytecode_preprocessing = BytecodePreprocessing::preprocess(model_fn);
        let shared_preprocessing = JoltSharedPreprocessing {
            bytecode: bytecode_preprocessing,
            precompiles: PrecompilePreprocessing::empty(),
        };

        let (trace, _) = trace(model_fn, &input, &shared_preprocessing.bytecode);

        let log_T = trace.len().ilog2() as usize;
        let log_K = shared_preprocessing.bytecode.memory_K.ilog2() as usize;
        let rounds = log_T + log_K;

        let prover_preprocessing: JoltProverPreprocessing<Fr, MockCommitScheme<Fr>> =
            JoltProverPreprocessing {
                generators: (),
                shared: shared_preprocessing.clone(),
            };

        let verifier_preprocessing: JoltVerifierPreprocessing<Fr, MockCommitScheme<Fr>> =
            JoltVerifierPreprocessing {
                generators: (),
                shared: shared_preprocessing,
            };
        let program_io = ProgramIO {
            input: Tensor::new(None, &[]).unwrap(),
            output: Tensor::new(None, &[]).unwrap(),
        };

        let mut prover_sm = StateManager::<'_, Fr, Blake2bTranscript, _>::new_prover(
            &prover_preprocessing,
            trace.clone(),
            program_io.clone(),
        );
        let mut verifier_sm = StateManager::<'_, Fr, Blake2bTranscript, _>::new_verifier(
            &verifier_preprocessing,
            program_io,
            trace.len(),
            verifier_preprocessing.shared.bytecode.memory_K,
            prover_sm.twist_sumcheck_switch_index,
        );

        let r: Vec<Fr> = prover_sm.transcript.borrow_mut().challenge_vector(rounds);
        let _r: Vec<Fr> = verifier_sm.transcript.borrow_mut().challenge_vector(rounds);
        let (r_address, r_cycle) = r.split_at(log_K);
        let eq_r_address = EqPolynomial::evals(r_address);
        let mut lt_r_cycle: Vec<Fr> = Vec::new(); // Computes the table LT(j, r_cycle) for all j in {0,1}^log_T
        for i in 0..(1 << log_T) {
            let index = index_to_field_bitvector::<Fr>(i, log_T);
            let lt_i_r = evaluate_lt_mle(&index, r_cycle);
            lt_r_cycle.push(lt_i_r);
        }

        let mut increments = Vec::new();
        for (i, (cycle, bytecode)) in trace
            .iter()
            .zip(prover_preprocessing.bytecode())
            .enumerate()
        {
            let write_op = cycle.td_write().1;
            let td = bytecode.td;
            if write_op != 0 {
                increments.push((i, td, write_op));
            }
        }

        let mut val_claim = Fr::zero();
        for (i, td, write_val) in increments.iter() {
            val_claim += eq_r_address[*td as usize] * lt_r_cycle[*i] * Fr::from_u64(*write_val);
        }

        let prover_accumulator = prover_sm.get_prover_accumulator();
        prover_accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
            OpeningPoint::new(r.clone()),
            val_claim,
        );

        let mut prover_sumcheck = ValEvaluationSumcheck::new_prover(&mut prover_sm);

        let mut prover_transcript_ref = prover_sm.transcript.borrow_mut();

        let (proof, r_sumcheck) = SingleSumcheck::prove(
            &mut prover_sumcheck,
            Some(prover_accumulator.clone()),
            &mut *prover_transcript_ref,
        );
        drop(prover_transcript_ref);

        // Take claims
        let prover_acc_borrow = prover_accumulator.borrow();
        let verifier_accumulator = verifier_sm.get_verifier_accumulator();
        let mut verifier_acc_borrow = verifier_accumulator.borrow_mut();

        for (key, (_, value)) in prover_acc_borrow.evaluation_openings().iter() {
            let empty_point = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![]);
            verifier_acc_borrow
                .openings_mut()
                .insert(*key, (empty_point, *value));
        }
        drop(prover_acc_borrow);
        drop(verifier_acc_borrow);

        verifier_accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
            OpeningPoint::new(r.clone()),
        );

        let verifier_sumcheck = ValEvaluationSumcheck::new_verifier(&mut verifier_sm);

        // For round-by-round sumcheck prover claim checking, see MemoryReadWriteChecking's `compute_expected_claims` test function
        let r_sumcheck_verif = SingleSumcheck::verify(
            &verifier_sumcheck,
            &proof,
            Some(verifier_accumulator.clone()),
            &mut *verifier_sm.transcript.borrow_mut(),
        )
        .unwrap();

        assert_eq!(r_sumcheck, r_sumcheck_verif);
    }

    #[test]
    fn test_val_evaluation_rsqrt() {
        let input = Tensor::new(Some(&[1, 1, 1, 1]), &[1, 4]).unwrap();
        test_val_evaluation_sumcheck(builder::rsqrt_model, input);
    }

    #[test]
    fn test_val_evaluation_rsqrt_binary() {
        let input = Tensor::new(Some(&[512]), &[1, 1]).unwrap();
        test_val_evaluation_sumcheck(
            || model(&PathBuf::from("../onnx-tracer/models/rsqrt/network.onnx")),
            input,
        );
    }
}
