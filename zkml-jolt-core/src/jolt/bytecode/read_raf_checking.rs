use crate::jolt::{
    dag::state_manager::StateManager,
    pcs::{ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator},
    sumcheck::SumcheckInstance,
    witness::{CommittedPolynomial, VirtualPolynomial},
};
use itertools::Itertools;
use jolt_core::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        compact_polynomial::SmallScalar,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{BIG_ENDIAN, OpeningPoint},
    },
    transcripts::Transcript,
    utils::{expanding_table::ExpandingTable, math::Math, thread::unsafe_allocate_zero_vec},
};
use onnx_tracer::trace_types::{CircuitFlags, NUM_CIRCUIT_FLAGS};
use rayon::prelude::*;
use std::{cell::RefCell, iter::once, rc::Rc};
use strum::IntoEnumIterator;

/// Number of batched read-checking sumchecks bespokely
const _STAGES: usize = 1;

struct ReadCheckingProverState<F: JoltField> {
    F: MultilinearPolynomial<F>,
    ra: Vec<MultilinearPolynomial<F>>,
    v: Vec<ExpandingTable<F>>,
    eq_poly: MultilinearPolynomial<F>,
    val_gamma: Option<F>,
    pc: Vec<usize>,
}

pub struct ReadRafSumcheck<F: JoltField> {
    log_K: usize,
    log_T: usize,
    log_K_chunk: usize,
    K_chunk: usize,
    d: usize,
    rv_claim: F,
    prover_state: Option<ReadCheckingProverState<F>>,
    val_poly: MultilinearPolynomial<F>,
    r_cycle: Vec<F>,
}

#[derive(Debug, Clone, Copy)]
enum ReadCheckingValType {
    /// Spartan outer sumcheck
    Stage1,
    /// Registers read-write sumcheck
    _Stage2,
    /// Registers val sumcheck wa, PCSumcheck, Instruction Lookups
    _Stage3,
}

impl<F: JoltField> ReadRafSumcheck<F> {
    #[tracing::instrument(skip_all, name = "BytecodeReadRafSumcheck::new_prover")]
    pub fn new_prover(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let K = sm.get_prover_data().0.shared.bytecode.code_size;
        let log_K = K.log_2();
        let d = sm.get_prover_data().0.shared.bytecode.d;
        let log_T = sm.get_prover_data().1.len().log_2();
        let log_K_chunk = log_K.div_ceil(d);
        let K_chunk = 1 << log_K_chunk;
        let (val_1, rv_claim_1, r_cycle_1) = Self::compute_val_rv(sm, ReadCheckingValType::Stage1);
        let rv_claim = rv_claim_1;
        let (preprocessing, trace, _) = sm.get_prover_data();
        let eq_evals = EqPolynomial::evals(&r_cycle_1);
        let T = trace.len();
        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = (T / num_chunks).max(1);
        let F = trace
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_index, trace_chunk)| {
                let mut result_1: Vec<F> = unsafe_allocate_zero_vec(K);
                let mut j = chunk_index * chunk_size;
                for _ in trace_chunk {
                    let pc = preprocessing.shared.bytecode.get_pc(j);
                    result_1[pc] += eq_evals[j];
                    j += 1;
                }
                result_1
            })
            .reduce(
                || unsafe_allocate_zero_vec(K),
                |mut running_1, new_1| {
                    running_1
                        .par_iter_mut()
                        .zip(new_1.into_par_iter())
                        .for_each(|(x, y)| *x += y);
                    running_1
                },
            );
        let eq_poly = MultilinearPolynomial::from(eq_evals);
        let F = MultilinearPolynomial::from(F);
        let mut v = (0..d)
            .map(|_| ExpandingTable::new(K_chunk))
            .collect::<Vec<_>>();
        v.par_iter_mut().for_each(|v| v.reset(F::one()));
        let pc: Vec<usize> = (0..trace.len())
            .into_par_iter()
            .map(|i| preprocessing.shared.bytecode.get_pc(i))
            .collect();

        Self {
            rv_claim,
            log_K,
            log_T,
            d,
            log_K_chunk,
            K_chunk,
            prover_state: Some(ReadCheckingProverState {
                F,
                ra: Vec::with_capacity(d),
                v,
                eq_poly,
                val_gamma: None,
                pc,
            }),
            val_poly: MultilinearPolynomial::from(val_1),
            r_cycle: r_cycle_1,
        }
    }

    pub fn new_verifier(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let K = sm.get_verifier_data().0.shared.bytecode.code_size;
        let log_K = K.log_2();
        let d = sm.get_verifier_data().0.shared.bytecode.d;
        let log_T = sm.get_verifier_data().2.log_2();
        let log_K_chunk = log_K.div_ceil(d);
        let (val_1, rv_claim_1, r_cycle_1) = Self::compute_val_rv(sm, ReadCheckingValType::Stage1);
        let rv_claim = rv_claim_1;
        Self {
            rv_claim,
            log_K,
            K_chunk: 1 << log_K_chunk,
            log_T,
            log_K_chunk,
            d,
            prover_state: None,
            val_poly: MultilinearPolynomial::from(val_1),
            r_cycle: r_cycle_1,
        }
    }

    fn compute_val_rv(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        val_type: ReadCheckingValType,
    ) -> (Vec<F>, F, Vec<F>) {
        match val_type {
            ReadCheckingValType::Stage1 => {
                let gamma: F = sm.get_transcript().borrow_mut().challenge_scalar();
                let mut gamma_powers = vec![F::one()];
                for _ in 0..NUM_CIRCUIT_FLAGS + 2 {
                    gamma_powers.push(gamma * gamma_powers.last().unwrap());
                }
                let (r_cycle, _) = sm.get_virtual_polynomial_opening(
                    VirtualPolynomial::Imm,
                    SumcheckId::SpartanOuter,
                );
                (
                    Self::compute_val_1(sm, &gamma_powers),
                    Self::compute_rv_claim_1(sm, &gamma_powers),
                    r_cycle.r,
                )
            }
            _ => {
                panic!("Not implemented");
            }
        }
    }

    /// Returns a vec of evaluations:
    ///    Val(k) = unexpanded_pc(k) + gamma * imm(k)
    ///             + gamma^2 * circuit_flags[0](k) + gamma^3 * circuit_flags[1](k) + ...
    /// This particular Val virtualizes claims output by Spartan's "outer" sumcheck
    fn compute_val_1(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
    ) -> Vec<F> {
        sm.get_bytecode()
            .par_iter()
            .map(|instruction| {
                let unexpanded_pc = instruction.address;
                let mut linear_combination = F::zero();
                linear_combination += F::from_u64(unexpanded_pc as u64);
                linear_combination += instruction.imm.field_mul(gamma_powers[1]);
                linear_combination += (instruction.td).field_mul(gamma_powers[2]);
                for (flag, gamma_power) in instruction
                    .circuit_flags()
                    .iter()
                    .zip(gamma_powers[3..].iter())
                {
                    if *flag {
                        linear_combination += *gamma_power;
                    }
                }

                linear_combination
            })
            .collect()
    }

    fn compute_rv_claim_1(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
    ) -> F {
        let (_, unexpanded_pc_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanOuter,
        );
        let (_, imm_claim) =
            sm.get_virtual_polynomial_opening(VirtualPolynomial::Imm, SumcheckId::SpartanOuter);
        let (_, rd_claim) =
            sm.get_virtual_polynomial_opening(VirtualPolynomial::Td, SumcheckId::SpartanOuter);
        once(unexpanded_pc_claim)
            .chain(once(imm_claim))
            .chain(once(rd_claim))
            .chain(CircuitFlags::iter().map(|flag| {
                sm.get_virtual_polynomial_opening(
                    VirtualPolynomial::OpFlags(flag),
                    SumcheckId::SpartanOuter,
                )
                .1
            }))
            .zip(gamma_powers)
            .map(|(claim, gamma)| claim * gamma)
            .sum()
    }
}

impl<F: JoltField> SumcheckInstance<F> for ReadRafSumcheck<F> {
    fn degree(&self) -> usize {
        self.d + 1
    }

    fn num_rounds(&self) -> usize {
        self.log_K + self.log_T
    }

    fn input_claim(&self) -> F {
        self.rv_claim
    }

    #[tracing::instrument(skip_all, name = "BytecodeReadRafSumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, round: usize, _previous_claim: F) -> Vec<F> {
        let ps = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        if round < self.log_K {
            const DEGREE: usize = 2;

            let univariate_poly_evals: [F; DEGREE] = (0..self.val_poly.len() / 2)
                .into_par_iter()
                .map(|i| {
                    let ra_evals =
                        ps.F.sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                    let val_evals = self
                        .val_poly
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                    [ra_evals[0] * val_evals[0], ra_evals[1] * val_evals[1]]
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

            univariate_poly_evals.to_vec()
        } else {
            let degree = self.degree();
            (0..ps.ra[0].len() / 2)
                .into_par_iter()
                .map(|i| {
                    let eq_evals = ps
                        .eq_poly
                        .sumcheck_evals(i, degree, BindingOrder::LowToHigh);
                    let val = ps.val_gamma.unwrap();
                    let eq_times_val = eq_evals.into_iter().map(|e| e * val).collect_vec();
                    let ra_evals = ps
                        .ra
                        .iter()
                        .map(|ra| ra.sumcheck_evals(i, degree, BindingOrder::LowToHigh));
                    ra_evals.fold(eq_times_val, |mut running: Vec<F>, new: Vec<F>| {
                        for i in 0..degree {
                            running[i] *= new[i];
                        }
                        running
                    })
                })
                .reduce(
                    || vec![F::zero(); degree],
                    |mut running, new| {
                        for i in 0..degree {
                            running[i] += new[i];
                        }
                        running
                    },
                )
        }
    }

    #[tracing::instrument(skip_all, name = "BytecodeReadRafSumcheck::bind")]
    fn bind(&mut self, r_j: F, round: usize) {
        let ps = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        if round < self.log_K {
            rayon::scope(|s| {
                s.spawn(|_| self.val_poly.bind_parallel(r_j, BindingOrder::HighToLow));
                s.spawn(|_| {
                    ps.F.bind_parallel(r_j, BindingOrder::HighToLow);
                });
                s.spawn(|_| {
                    ps.v[round / self.log_K_chunk].update(r_j);
                });
            });
            if round == self.log_K - 1 {
                self.init_log_t_rounds();
            }
        } else {
            ps.ra
                .par_iter_mut()
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
            ps.eq_poly.bind_parallel(r_j, BindingOrder::LowToHigh)
        }
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F],
    ) -> F {
        let (r_address_prime, r_cycle_prime) = r.split_at(self.log_K);
        // r_cycle is bound LowToHigh, so reverse
        let r_cycle_prime = r_cycle_prime.iter().rev().copied().collect::<Vec<F>>();

        let ra_claims = (0..self.d).map(|i| {
            accumulator
                .as_ref()
                .unwrap()
                .borrow()
                .get_committed_polynomial_opening(
                    CommittedPolynomial::BytecodeRa(i),
                    SumcheckId::BytecodeReadRaf,
                )
                .1
        });

        // We have a separate Val polynomial for each stage
        // Additionally, for stages 1 and 3 we have an Int polynomial for RAF
        // So we would have:
        // Stage 1: gamma^0 * (Val_1 + gamma^3 * Int)
        // Stage 2: gamma^1 * (Val_2)
        // Stage 3: gamma^2 * (Val_3 + gamma^2 * Int)
        // Which matches with the input claim:
        // rv_1 + gamma * rv_2 + gamma^2 * rv_3 + gamma^3 * raf_1 + gamma^4 * raf_3
        let val = self.val_poly.evaluate(r_address_prime)
            * EqPolynomial::mle(&self.r_cycle, &r_cycle_prime);

        ra_claims.fold(val, |running, ra_claim| running * ra_claim)
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        let mut r = opening_point.to_vec();
        r[self.log_K..].reverse();
        OpeningPoint::new(r)
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let ps = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        let (r_address, r_cycle) = opening_point.clone().split_at(self.log_K);

        for i in 0..self.d {
            let r_address = &r_address.r[self.log_K_chunk * i..self.log_K_chunk * (i + 1)];
            accumulator.borrow_mut().append_sparse(
                vec![CommittedPolynomial::BytecodeRa(i)],
                SumcheckId::BytecodeReadRaf,
                r_address.to_vec(),
                r_cycle.clone().into(),
                vec![ps.ra[i].final_sumcheck_claim()],
            );
        }
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let (r_address, r_cycle) = opening_point.split_at(self.log_K);
        (0..self.d).for_each(|i| {
            let r_address = &r_address.r[self.log_K_chunk * i..self.log_K_chunk * (i + 1)];
            accumulator.borrow_mut().append_sparse(
                vec![CommittedPolynomial::BytecodeRa(i)],
                SumcheckId::BytecodeReadRaf,
                [r_address, &r_cycle.r].concat(),
            );
        });
    }
}

impl<F: JoltField> ReadRafSumcheck<F> {
    fn init_log_t_rounds(&mut self) {
        let ps = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        // We have a separate Val polynomial for each stage
        // Additionally, for stages 1 and 3 we have an Int polynomial for RAF
        // So we would have:
        // Stage 1: gamma^0 * (Val_1 + gamma^3 * Int)
        // Stage 2: gamma^1 * (Val_2)
        // Stage 3: gamma^2 * (Val_3 + gamma^2 * Int)
        // Which matches with the input claim:
        // rv_1 + gamma * rv_2 + gamma^2 * rv_3 + gamma^3 * raf_1 + gamma^4 * raf_3
        ps.val_gamma = Some(self.val_poly.final_sumcheck_claim());

        ps.v.par_iter()
            .enumerate()
            .map(|(i, v)| {
                let ra_i: Vec<F> = ps
                    .pc
                    .par_iter()
                    .map(|k| {
                        let k = (k >> (self.log_K_chunk * (self.d - i - 1))) % self.K_chunk;
                        v[k]
                    })
                    .collect();
                MultilinearPolynomial::from(ra_i)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|ra| {
                ps.ra.push(ra);
            });
    }
}
