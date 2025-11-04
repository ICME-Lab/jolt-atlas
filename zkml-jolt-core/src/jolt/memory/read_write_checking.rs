use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_core::{
    field::{JoltField, OptimizedMul},
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{BIG_ENDIAN, LITTLE_ENDIAN, OpeningPoint},
        split_eq_poly::GruenSplitEqPolynomial,
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    utils::{math::Math, thread::unsafe_allocate_zero_vec},
};
use rayon::prelude::*;
use std::{cell::RefCell, rc::Rc};

use crate::jolt::{
    JoltProverPreprocessing,
    dag::state_manager::StateManager,
    pcs::{ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator},
    sumcheck::SumcheckInstance,
    trace::JoltONNXCycle,
    witness::{CommittedPolynomial, VirtualPolynomial},
};

/// A collection of vectors that are used in each of the first log(T / num_chunks)
/// rounds of sumcheck. There is one `DataBuffers` struct per thread/chunk, reused
/// across all log(T / num_chunks) rounds.
struct DataBuffers<F: JoltField> {
    /// Contains
    ///     Val(k, j', 0, ..., 0)
    /// as we iterate over rows j' \in {0, 1}^(log(T) - i)
    val_j_0: Vec<F>,
    /// `val_j_r[0]` contains
    ///     Val(k, j'', 0, r_i, ..., r_1)
    /// `val_j_r[1]` contains
    ///     Val(k, j'', 1, r_i, ..., r_1)
    /// as we iterate over rows j' \in {0, 1}^(log(T) - i)
    val_j_r: [Vec<F>; 2],
    /// `ra[0]` contains
    ///     ra(k, j'', 0, r_i, ..., r_1)
    /// `ra[1]` contains
    ///     ra(k, j'', 1, r_i, ..., r_1)
    /// as we iterate over rows j' \in {0, 1}^(log(T) - i),
    ts1_ra: [Vec<F>; 2],
    ts2_ra: [Vec<F>; 2],
    ts3_ra: [Vec<F>; 2],
    /// `wa[0]` contains
    ///     wa(k, j'', 0, r_i, ..., r_1)
    /// `wa[1]` contains
    ///     wa(k, j'', 1, r_i, ..., r_1)
    /// as we iterate over rows j' \in {0, 1}^(log(T) - i),
    /// where j'' are the higher (log(T) - i - 1) bits of j'
    td_wa: [Vec<F>; 2],
    dirty_indices: Vec<usize>,
}

struct ReadWriteCheckingProverState<F: JoltField> {
    addresses: Vec<(u64, u64, u64, u64)>,
    chunk_size: usize,
    val_checkpoints: Vec<F>,
    data_buffers: Vec<DataBuffers<F>>,
    I: Vec<Vec<(usize, u64, F, F)>>,
    A: Vec<F>,
    gruens_eq_r_prime: GruenSplitEqPolynomial<F>,
    inc_cycle: MultilinearPolynomial<F>,
    // The following polynomials are instantiated after
    // the first phase
    eq_r_prime: Option<MultilinearPolynomial<F>>,
    ts1_ra: Option<MultilinearPolynomial<F>>,
    ts2_ra: Option<MultilinearPolynomial<F>>,
    ts3_ra: Option<MultilinearPolynomial<F>>,
    td_wa: Option<MultilinearPolynomial<F>>,
    val: Option<MultilinearPolynomial<F>>,
}

impl<F: JoltField> ReadWriteCheckingProverState<F> {
    #[tracing::instrument(skip_all, name = "RegistersReadWriteCheckingProverState::initialize")]
    fn initialize<PCS: CommitmentScheme<Field = F>>(
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        trace: &[JoltONNXCycle],
        r_prime: &[F],
    ) -> Self {
        let K = preprocessing.memory_K();
        let T = trace.len();
        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = T / num_chunks;

        let span = tracing::span!(tracing::Level::INFO, "compute deltas");
        let _guard = span.enter();

        let deltas: Vec<Vec<i128>> = trace[..T - chunk_size]
            .par_chunks_exact(chunk_size)
            .zip(preprocessing.bytecode().par_chunks_exact(chunk_size))
            .map(|(trace_chunk, bytecode_chunk)| {
                let mut delta = vec![0; K];
                for (cycle, bytecode) in trace_chunk.iter().zip(bytecode_chunk.iter()) {
                    let k = bytecode.td;
                    let (pre_value, post_value) = cycle.td_write();
                    delta[k as usize] += post_value as i128 - pre_value as i128;
                }
                delta
            })
            .collect();

        drop(_guard);
        drop(span);

        let span = tracing::span!(tracing::Level::INFO, "compute checkpoints");
        let _guard = span.enter();

        // Value in register k before the jth cycle, for j \in {0, chunk_size, 2 * chunk_size, ...}
        let mut checkpoints: Vec<Vec<i128>> = Vec::with_capacity(num_chunks);
        checkpoints.push(vec![0; K]);

        for (chunk_index, delta) in deltas.into_iter().enumerate() {
            let next_checkpoint: Vec<i128> = (0..K)
                .map(|k| checkpoints[chunk_index][k] + delta[k])
                .collect();
            // In RISC-V, the first register is the zero register.
            debug_assert_eq!(next_checkpoint[0], 0);
            checkpoints.push(next_checkpoint);
        }

        // TODO(moodlezoup): could potentially generate these checkpoints in the tracer
        // Generate checkpoints as a flat vector because it will be turned into the
        // materialized Val polynomial after the first half of sumcheck.
        let mut val_checkpoints: Vec<F> = unsafe_allocate_zero_vec(K * num_chunks);
        val_checkpoints
            .par_chunks_mut(K)
            .zip(checkpoints.into_par_iter())
            .for_each(|(val_checkpoint, checkpoint)| {
                val_checkpoint
                    .iter_mut()
                    .zip(checkpoint.iter())
                    .for_each(|(dest, src)| *dest = F::from_i128(*src))
            });

        drop(_guard);
        drop(span);

        // A table that, in round i of sumcheck, stores all evaluations
        //     EQ(x, r_i, ..., r_1)
        // as x ranges over {0, 1}^i.
        // (As described in "Computing other necessary arrays and worst-case
        // accounting", Section 8.2.2)
        let mut A: Vec<F> = unsafe_allocate_zero_vec(chunk_size);
        A[0] = F::one();

        let span = tracing::span!(
            tracing::Level::INFO,
            "compute I (increments data structure)"
        );
        let _guard = span.enter();

        // Data structure described in Equation (72)
        let I: Vec<Vec<(usize, u64, F, F)>> = trace
            .par_chunks(chunk_size)
            .zip(preprocessing.bytecode().par_chunks_exact(chunk_size))
            .enumerate()
            .map(|(chunk_index, (trace_chunk, bytecode_chunk))| {
                // Row index of the I matrix
                let mut j = chunk_index * chunk_size;
                trace_chunk
                    .iter()
                    .zip(bytecode_chunk.iter())
                    .map(|(cycle, bytecode)| {
                        let k = bytecode.td;
                        let (pre_value, post_value) = cycle.td_write();
                        let increment = post_value as i128 - pre_value as i128;
                        let inc = (j, k, F::zero(), F::from_i128(increment));
                        j += 1;
                        inc
                    })
                    .collect()
            })
            .collect();

        drop(_guard);
        drop(span);

        let gruens_eq_r_prime = GruenSplitEqPolynomial::new(r_prime, BindingOrder::LowToHigh);
        let inc_cycle = CommittedPolynomial::TdInc.generate_witness(preprocessing, trace);

        let data_buffers: Vec<DataBuffers<F>> = (0..num_chunks)
            .into_par_iter()
            .map(|_| DataBuffers {
                val_j_0: Vec::with_capacity(K),
                val_j_r: [unsafe_allocate_zero_vec(K), unsafe_allocate_zero_vec(K)],
                ts1_ra: [unsafe_allocate_zero_vec(K), unsafe_allocate_zero_vec(K)],
                ts2_ra: [unsafe_allocate_zero_vec(K), unsafe_allocate_zero_vec(K)],
                ts3_ra: [unsafe_allocate_zero_vec(K), unsafe_allocate_zero_vec(K)],
                td_wa: [unsafe_allocate_zero_vec(K), unsafe_allocate_zero_vec(K)],
                dirty_indices: Vec::with_capacity(K),
            })
            .collect();

        let addresses = preprocessing
            .bytecode()
            .par_iter()
            .map(|instr| (instr.ts1, instr.ts2, instr.ts3, instr.td))
            .collect::<Vec<_>>();

        ReadWriteCheckingProverState {
            addresses,
            chunk_size,
            val_checkpoints,
            data_buffers,
            I,
            A,
            gruens_eq_r_prime,
            inc_cycle,
            eq_r_prime: None,
            ts1_ra: None,
            ts2_ra: None,
            ts3_ra: None,
            td_wa: None,
            val: None,
        }
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone, Default)]
pub struct ReadWriteSumcheckClaims<F: JoltField> {
    pub val_claim: F,
    pub ts1_ra_claim: F,
    pub ts2_ra_claim: F,
    pub ts3_ra_claim: F,
    pub td_wa_claim: F,
    pub inc_claim: F,
}

/// Claims for register read/write values from Spartan
#[derive(Debug, Clone, Default)]
pub struct ReadWriteValueClaims<F: JoltField> {
    pub ts1_rv_claim: F,
    pub ts2_rv_claim: F,
    pub ts3_rv_claim: F,
    pub td_wv_claim: F,
}

pub struct RegistersReadWriteChecking<F: JoltField> {
    K: usize,
    T: usize,
    gamma: F,
    gamma_sqr: F,
    gamma_cube: F,
    sumcheck_switch_index: usize,
    prover_state: Option<ReadWriteCheckingProverState<F>>,
    input_claim: F,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct RegistersReadWriteCheckingProof<F: JoltField, ProofTranscript: Transcript> {
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    sumcheck_switch_index: usize,
}

impl<F: JoltField> RegistersReadWriteChecking<F> {
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (preprocessing, trace, _) = state_manager.get_prover_data();
        let accumulator = state_manager.get_prover_accumulator();

        let (r_cycle, ts1_rv_claim) = accumulator
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::Ts1Value, SumcheckId::SpartanOuter);
        let (_, ts2_rv_claim) = accumulator
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::Ts2Value, SumcheckId::SpartanOuter);
        let (_, ts3_rv_claim) = accumulator
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::Ts3Value, SumcheckId::SpartanOuter);
        let (_, td_wv_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::TdWriteValue,
            SumcheckId::SpartanOuter,
        );

        let transcript = &mut *state_manager.transcript.borrow_mut();
        let gamma: F = transcript.challenge_scalar();
        let gamma_sqr = gamma.square();
        let gamma_cube = gamma_sqr * gamma;
        let input_claim = td_wv_claim
            + gamma * ts1_rv_claim
            + gamma_sqr * ts2_rv_claim
            + gamma_cube * ts3_rv_claim;

        let prover_state =
            ReadWriteCheckingProverState::initialize(preprocessing, trace, &r_cycle.r);

        Self {
            K: preprocessing.memory_K(),
            T: trace.len(),
            gamma,
            gamma_sqr,
            gamma_cube,
            sumcheck_switch_index: state_manager.twist_sumcheck_switch_index,
            prover_state: Some(prover_state),
            input_claim,
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (_, _, trace_length) = state_manager.get_verifier_data();
        let accumulator = state_manager.get_verifier_accumulator();

        let (_, ts1_rv_claim) = accumulator
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::Ts1Value, SumcheckId::SpartanOuter);
        let (_, ts2_rv_claim) = accumulator
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::Ts2Value, SumcheckId::SpartanOuter);
        let (_, ts3_rv_claim) = accumulator
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::Ts3Value, SumcheckId::SpartanOuter);
        let (_, td_wv_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::TdWriteValue,
            SumcheckId::SpartanOuter,
        );

        let transcript = &mut *state_manager.transcript.borrow_mut();
        let gamma: F = transcript.challenge_scalar();
        let gamma_sqr = gamma.square();
        let gamma_cube = gamma_sqr * gamma;
        let input_claim = td_wv_claim
            + gamma * ts1_rv_claim
            + gamma_sqr * ts2_rv_claim
            + gamma_cube * ts3_rv_claim;

        Self {
            K: state_manager.get_memory_K(),
            T: trace_length,
            gamma,
            gamma_sqr,
            gamma_cube,
            sumcheck_switch_index: state_manager.twist_sumcheck_switch_index,
            prover_state: None,
            input_claim,
        }
    }

    fn phase1_compute_prover_message(&mut self, round: usize, previous_claim: F) -> Vec<F> {
        const DEGREE: usize = 3;
        let ReadWriteCheckingProverState {
            addresses,
            I,
            data_buffers,
            A,
            val_checkpoints,
            inc_cycle,
            gruens_eq_r_prime,
            ..
        } = self.prover_state.as_mut().unwrap();

        // Compute quadratic coefficients for Gruen's interpolation
        let quadratic_coeffs: [F; DEGREE - 1] = if gruens_eq_r_prime.E_in_current_len() == 1 {
            // E_in is fully bound, use E_out
            I.par_iter()
                .zip(data_buffers.par_iter_mut())
                .zip(val_checkpoints.par_chunks(self.K))
                .map(|((I_chunk, buffers), checkpoint)| {
                    let mut evals = [F::zero(), F::zero()];

                    let DataBuffers {
                        val_j_0,
                        val_j_r,
                        ts1_ra,
                        ts2_ra,
                        ts3_ra,
                        td_wa,
                        dirty_indices,
                    } = buffers;

                    val_j_0.as_mut_slice().copy_from_slice(checkpoint);

                    // Iterate over I_chunk, two rows at a time.
                    I_chunk
                        .chunk_by(|a, b| a.0 / 2 == b.0 / 2)
                        .for_each(|inc_chunk| {
                            let j_prime = inc_chunk[0].0; // row index

                            for j in j_prime << round..(j_prime + 1) << round {
                                let j_bound = j % (1 << round);

                                let k = addresses[j].0 as usize;
                                dirty_indices.push(k);
                                ts1_ra[0][k] += A[j_bound];

                                let k = addresses[j].1 as usize;
                                dirty_indices.push(k);
                                ts2_ra[0][k] += A[j_bound];

                                let k = addresses[j].2 as usize;
                                dirty_indices.push(k);
                                ts3_ra[0][k] += A[j_bound];

                                let k = addresses[j].3 as usize;
                                dirty_indices.push(k);
                                td_wa[0][k] += A[j_bound];
                            }

                            for j in (j_prime + 1) << round..(j_prime + 2) << round {
                                let j_bound = j % (1 << round);

                                let k = addresses[j].0 as usize;
                                dirty_indices.push(k);
                                ts1_ra[1][k] += A[j_bound];

                                let k = addresses[j].1 as usize;
                                dirty_indices.push(k);
                                ts2_ra[1][k] += A[j_bound];

                                let k = addresses[j].2 as usize;
                                dirty_indices.push(k);
                                ts3_ra[1][k] += A[j_bound];

                                let k = addresses[j].3 as usize;
                                dirty_indices.push(k);
                                td_wa[1][k] += A[j_bound];
                            }

                            for &k in dirty_indices.iter() {
                                val_j_r[0][k] = val_j_0[k];
                            }
                            let mut inc_iter = inc_chunk.iter().peekable();

                            // First of the two rows
                            loop {
                                let (row, col, inc_lt, inc) = inc_iter.next().unwrap();
                                debug_assert_eq!(*row, j_prime);
                                val_j_r[0][*col as usize] += *inc_lt;
                                val_j_0[*col as usize] += *inc;
                                if inc_iter.peek().unwrap().0 != j_prime {
                                    break;
                                }
                            }
                            for &k in dirty_indices.iter() {
                                val_j_r[1][k] = val_j_0[k];
                            }

                            // Second of the two rows
                            for inc in inc_iter {
                                let (row, col, inc_lt, inc) = *inc;
                                debug_assert_eq!(row, j_prime + 1);
                                val_j_r[1][col as usize] += inc_lt;
                                val_j_0[col as usize] += inc;
                            }

                            let eq_r_prime_eval = gruens_eq_r_prime.E_out_current()[j_prime / 2];
                            let inc_cycle_evals = {
                                let inc_cycle_0 = inc_cycle.get_bound_coeff(j_prime);
                                let inc_cycle_1 = inc_cycle.get_bound_coeff(j_prime + 1);
                                let inc_cycle_infty = inc_cycle_1 - inc_cycle_0;
                                [inc_cycle_0, inc_cycle_infty]
                            };

                            let mut td_inner_sum_evals = [F::zero(); DEGREE - 1];
                            let mut ts1_inner_sum_evals = [F::zero(); DEGREE - 1];
                            let mut ts2_inner_sum_evals = [F::zero(); DEGREE - 1];
                            let mut ts3_inner_sum_evals = [F::zero(); DEGREE - 1];

                            for k in dirty_indices.drain(..) {
                                let val_evals = [val_j_r[0][k], val_j_r[1][k] - val_j_r[0][k]];

                                // Check td_wa and compute its contribution if non-zero
                                if !td_wa[0][k].is_zero() || !td_wa[1][k].is_zero() {
                                    let wa_evals = [td_wa[0][k], td_wa[1][k] - td_wa[0][k]];

                                    td_inner_sum_evals[0] += wa_evals[0]
                                        .mul_0_optimized(inc_cycle_evals[0] + val_evals[0]);
                                    td_inner_sum_evals[1] +=
                                        wa_evals[1] * (inc_cycle_evals[1] + val_evals[1]);

                                    td_wa[0][k] = F::zero();
                                    td_wa[1][k] = F::zero();
                                }

                                // Check ts1_ra and compute its contribution if non-zero
                                if !ts1_ra[0][k].is_zero() || !ts1_ra[1][k].is_zero() {
                                    let ra_evals_ts1 = [ts1_ra[0][k], ts1_ra[1][k] - ts1_ra[0][k]];

                                    ts1_inner_sum_evals[0] +=
                                        ra_evals_ts1[0].mul_0_optimized(val_evals[0]);
                                    ts1_inner_sum_evals[1] += ra_evals_ts1[1] * val_evals[1];

                                    ts1_ra[0][k] = F::zero();
                                    ts1_ra[1][k] = F::zero();
                                }

                                // Check ts2_ra and compute its contribution if non-zero
                                if !ts2_ra[0][k].is_zero() || !ts2_ra[1][k].is_zero() {
                                    let ra_evals_ts2 = [ts2_ra[0][k], ts2_ra[1][k] - ts2_ra[0][k]];

                                    ts2_inner_sum_evals[0] +=
                                        ra_evals_ts2[0].mul_0_optimized(val_evals[0]);
                                    ts2_inner_sum_evals[1] += ra_evals_ts2[1] * val_evals[1];

                                    ts2_ra[0][k] = F::zero();
                                    ts2_ra[1][k] = F::zero();
                                }

                                // Check ts3_ra and compute its contribution if non-zero
                                if !ts3_ra[0][k].is_zero() || !ts3_ra[1][k].is_zero() {
                                    let ra_evals_ts3 = [ts3_ra[0][k], ts3_ra[1][k] - ts3_ra[0][k]];

                                    ts3_inner_sum_evals[0] +=
                                        ra_evals_ts3[0].mul_0_optimized(val_evals[0]);
                                    ts3_inner_sum_evals[1] += ra_evals_ts3[1] * val_evals[1];

                                    ts3_ra[0][k] = F::zero();
                                    ts3_ra[1][k] = F::zero();
                                }

                                val_j_r[0][k] = F::zero();
                                val_j_r[1][k] = F::zero();
                            }
                            dirty_indices.clear();

                            evals[0] += eq_r_prime_eval
                                * (td_inner_sum_evals[0]
                                    + self.gamma * ts1_inner_sum_evals[0]
                                    + self.gamma_sqr * ts2_inner_sum_evals[0]
                                    + self.gamma_cube * ts3_inner_sum_evals[0]);
                            evals[1] += eq_r_prime_eval
                                * (td_inner_sum_evals[1]
                                    + self.gamma * ts1_inner_sum_evals[1]
                                    + self.gamma_sqr * ts2_inner_sum_evals[1]
                                    + self.gamma_cube * ts3_inner_sum_evals[1]);
                        });

                    evals
                })
                .reduce(
                    || [F::zero(); DEGREE - 1],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                )
        } else {
            // E_in is not fully bound, handle E_in and E_out
            let num_x_in_bits = gruens_eq_r_prime.E_in_current_len().log_2();
            let x_bitmask = (1 << num_x_in_bits) - 1;

            I.par_iter()
                .zip(data_buffers.par_iter_mut())
                .zip(val_checkpoints.par_chunks(self.K))
                .map(|((I_chunk, buffers), checkpoint)| {
                    let mut evals = [F::zero(), F::zero()];

                    let mut evals_for_current_E_out = [F::zero(), F::zero()];
                    let mut x_out_prev: Option<usize> = None;

                    let DataBuffers {
                        val_j_0,
                        val_j_r,
                        ts1_ra,
                        ts2_ra,
                        ts3_ra,
                        td_wa,
                        dirty_indices,
                    } = buffers;
                    *val_j_0 = checkpoint.to_vec();

                    // Iterate over I_chunk, two rows at a time.
                    I_chunk
                        .chunk_by(|a, b| a.0 / 2 == b.0 / 2)
                        .for_each(|inc_chunk| {
                            let j_prime = inc_chunk[0].0; // row index

                            for j in j_prime << round..(j_prime + 1) << round {
                                let j_bound = j % (1 << round);

                                let k = addresses[j].0;
                                dirty_indices.push(k as usize);
                                ts1_ra[0][k as usize] += A[j_bound];

                                let k = addresses[j].1;
                                dirty_indices.push(k as usize);
                                ts2_ra[0][k as usize] += A[j_bound];

                                let k = addresses[j].2;
                                dirty_indices.push(k as usize);
                                ts3_ra[0][k as usize] += A[j_bound];

                                let k = addresses[j].3;
                                dirty_indices.push(k as usize);
                                td_wa[0][k as usize] += A[j_bound];
                            }

                            for j in (j_prime + 1) << round..(j_prime + 2) << round {
                                let j_bound = j % (1 << round);

                                let k = addresses[j].0;
                                dirty_indices.push(k as usize);
                                ts1_ra[1][k as usize] += A[j_bound];

                                let k = addresses[j].1;
                                dirty_indices.push(k as usize);
                                ts2_ra[1][k as usize] += A[j_bound];

                                let k = addresses[j].2;
                                dirty_indices.push(k as usize);
                                ts2_ra[1][k as usize] += A[j_bound];

                                let k = addresses[j].3;
                                dirty_indices.push(k as usize);
                                td_wa[1][k as usize] += A[j_bound];
                            }

                            for &k in dirty_indices.iter() {
                                val_j_r[0][k] = val_j_0[k];
                            }
                            let mut inc_iter = inc_chunk.iter().peekable();

                            // First of the two rows
                            loop {
                                let (row, col, inc_lt, inc) = inc_iter.next().unwrap();
                                debug_assert_eq!(*row, j_prime);
                                val_j_r[0][*col as usize] += *inc_lt;
                                val_j_0[*col as usize] += *inc;
                                if inc_iter.peek().unwrap().0 != j_prime {
                                    break;
                                }
                            }
                            for &k in dirty_indices.iter() {
                                val_j_r[1][k] = val_j_0[k];
                            }

                            // Second of the two rows
                            for inc in inc_iter {
                                let (row, col, inc_lt, inc) = *inc;
                                debug_assert_eq!(row, j_prime + 1);
                                val_j_r[1][col as usize] += inc_lt;
                                val_j_0[col as usize] += inc;
                            }

                            let x_in = (j_prime / 2) & x_bitmask;
                            let x_out = (j_prime / 2) >> num_x_in_bits;
                            let E_in_eval = gruens_eq_r_prime.E_in_current()[x_in];

                            let inc_cycle_evals = {
                                let inc_cycle_0 = inc_cycle.get_bound_coeff(j_prime);
                                let inc_cycle_1 = inc_cycle.get_bound_coeff(j_prime + 1);
                                let inc_cycle_infty = inc_cycle_1 - inc_cycle_0;
                                [inc_cycle_0, inc_cycle_infty]
                            };

                            // Multiply the running sum by the previous value of E_out_eval when
                            // its value changes and add the result to the total.
                            match x_out_prev {
                                None => {
                                    x_out_prev = Some(x_out);
                                }
                                Some(x) if x_out != x => {
                                    x_out_prev = Some(x_out);

                                    let E_out_eval = gruens_eq_r_prime.E_out_current()[x];
                                    evals[0] += E_out_eval * evals_for_current_E_out[0];
                                    evals[1] += E_out_eval * evals_for_current_E_out[1];

                                    evals_for_current_E_out = [F::zero(), F::zero()];
                                }
                                _ => (),
                            }

                            let mut td_inner_sum_evals = [F::zero(); DEGREE - 1];
                            let mut ts1_inner_sum_evals = [F::zero(); DEGREE - 1];
                            let mut ts2_inner_sum_evals = [F::zero(); DEGREE - 1];
                            let mut ts3_inner_sum_evals = [F::zero(); DEGREE - 1];

                            for k in dirty_indices.drain(..) {
                                let val_evals = [val_j_r[0][k], val_j_r[1][k] - val_j_r[0][k]];

                                // Check td_wa and compute its contribution if non-zero
                                if !td_wa[0][k].is_zero() || !td_wa[1][k].is_zero() {
                                    let wa_evals = [td_wa[0][k], td_wa[1][k] - td_wa[0][k]];

                                    td_inner_sum_evals[0] += wa_evals[0]
                                        .mul_0_optimized(inc_cycle_evals[0] + val_evals[0]);
                                    td_inner_sum_evals[1] +=
                                        wa_evals[1] * (inc_cycle_evals[1] + val_evals[1]);

                                    td_wa[0][k] = F::zero();
                                    td_wa[1][k] = F::zero();
                                }

                                // Check ts1_ra and compute its contribution if non-zero
                                if !ts1_ra[0][k].is_zero() || !ts1_ra[1][k].is_zero() {
                                    let ra_evals_ts1 = [ts1_ra[0][k], ts1_ra[1][k] - ts1_ra[0][k]];

                                    ts1_inner_sum_evals[0] +=
                                        ra_evals_ts1[0].mul_0_optimized(val_evals[0]);
                                    ts1_inner_sum_evals[1] += ra_evals_ts1[1] * val_evals[1];

                                    ts1_ra[0][k] = F::zero();
                                    ts1_ra[1][k] = F::zero();
                                }

                                // Check ts2_ra and compute its contribution if non-zero
                                if !ts2_ra[0][k].is_zero() || !ts2_ra[1][k].is_zero() {
                                    let ra_evals_ts2 = [ts2_ra[0][k], ts2_ra[1][k] - ts2_ra[0][k]];

                                    ts2_inner_sum_evals[0] +=
                                        ra_evals_ts2[0].mul_0_optimized(val_evals[0]);
                                    ts2_inner_sum_evals[1] += ra_evals_ts2[1] * val_evals[1];

                                    ts2_ra[0][k] = F::zero();
                                    ts2_ra[1][k] = F::zero();
                                }

                                // Check ts3_ra and compute its contribution if non-zero
                                if !ts3_ra[0][k].is_zero() || !ts3_ra[1][k].is_zero() {
                                    let ra_evals_ts3 = [ts3_ra[0][k], ts3_ra[1][k] - ts3_ra[0][k]];

                                    ts3_inner_sum_evals[0] +=
                                        ra_evals_ts3[0].mul_0_optimized(val_evals[0]);
                                    ts3_inner_sum_evals[1] += ra_evals_ts3[1] * val_evals[1];

                                    ts3_ra[0][k] = F::zero();
                                    ts3_ra[1][k] = F::zero();
                                }

                                val_j_r[0][k] = F::zero();
                                val_j_r[1][k] = F::zero();
                            }
                            dirty_indices.clear();

                            evals_for_current_E_out[0] += E_in_eval
                                * (td_inner_sum_evals[0]
                                    + self.gamma * ts1_inner_sum_evals[0]
                                    + self.gamma_sqr * ts2_inner_sum_evals[0]
                                    + self.gamma_cube * ts3_inner_sum_evals[0]);
                            evals_for_current_E_out[1] += E_in_eval
                                * (td_inner_sum_evals[1]
                                    + self.gamma * ts1_inner_sum_evals[1]
                                    + self.gamma_sqr * ts2_inner_sum_evals[1]
                                    + self.gamma_cube * ts3_inner_sum_evals[1]);
                        });

                    // Multiply the final running sum by the final value of E_out_eval and add the
                    // result to the total.
                    if let Some(x) = x_out_prev {
                        let E_out_eval = gruens_eq_r_prime.E_out_current()[x];
                        evals[0] += E_out_eval * evals_for_current_E_out[0];
                        evals[1] += E_out_eval * evals_for_current_E_out[1];
                    }
                    evals
                })
                .reduce(
                    || [F::zero(); DEGREE - 1],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                )
        };

        // Convert quadratic coefficients to cubic evaluations
        gruens_eq_r_prime
            .gruen_evals_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], previous_claim)
            .to_vec()
    }

    fn phase2_compute_prover_message(&self) -> Vec<F> {
        const DEGREE: usize = 3;

        let ReadWriteCheckingProverState {
            inc_cycle,
            eq_r_prime,
            ts1_ra,
            ts2_ra,
            ts3_ra,
            td_wa,
            val,
            ..
        } = self.prover_state.as_ref().unwrap();
        let ts1_ra = ts1_ra.as_ref().unwrap();
        let ts2_ra = ts2_ra.as_ref().unwrap();
        let ts3_ra = ts3_ra.as_ref().unwrap();
        let td_wa = td_wa.as_ref().unwrap();
        let val = val.as_ref().unwrap();
        let eq_r_prime = eq_r_prime.as_ref().unwrap();

        let univariate_poly_evals = (0..eq_r_prime.len() / 2)
            .into_par_iter()
            .map(|j| {
                let eq_r_prime_evals =
                    eq_r_prime.sumcheck_evals_array::<DEGREE>(j, BindingOrder::HighToLow);
                let inc_evals =
                    inc_cycle.sumcheck_evals_array::<DEGREE>(j, BindingOrder::HighToLow);

                let inner_sum_evals: [F; DEGREE] = (0..self.K)
                    .into_par_iter()
                    .map(|k| {
                        let index = j * self.K + k;
                        let ts1_ra_evals =
                            ts1_ra.sumcheck_evals_array::<DEGREE>(index, BindingOrder::HighToLow);
                        let ts2_ra_evals =
                            ts2_ra.sumcheck_evals_array::<DEGREE>(index, BindingOrder::HighToLow);
                        let ts3_ra_evals =
                            ts3_ra.sumcheck_evals_array::<DEGREE>(index, BindingOrder::HighToLow);
                        let wa_evals =
                            td_wa.sumcheck_evals_array::<DEGREE>(index, BindingOrder::HighToLow);
                        let val_evals =
                            val.sumcheck_evals_array::<DEGREE>(index, BindingOrder::HighToLow);

                        [
                            wa_evals[0].mul_0_optimized(inc_evals[0] + val_evals[0])
                                + self.gamma * ts1_ra_evals[0].mul_0_optimized(val_evals[0])
                                + self.gamma_sqr * ts2_ra_evals[0].mul_0_optimized(val_evals[0])
                                + self.gamma_cube * ts3_ra_evals[0].mul_0_optimized(val_evals[0]),
                            wa_evals[1].mul_0_optimized(inc_evals[1] + val_evals[1])
                                + self.gamma * ts1_ra_evals[1].mul_0_optimized(val_evals[1])
                                + self.gamma_sqr * ts2_ra_evals[1].mul_0_optimized(val_evals[1])
                                + self.gamma_cube * ts3_ra_evals[1].mul_0_optimized(val_evals[1]),
                            wa_evals[2].mul_0_optimized(inc_evals[2] + val_evals[2])
                                + self.gamma * ts1_ra_evals[2].mul_0_optimized(val_evals[2])
                                + self.gamma_sqr * ts2_ra_evals[2].mul_0_optimized(val_evals[2])
                                + self.gamma_cube * ts3_ra_evals[2].mul_0_optimized(val_evals[2]),
                        ]
                    })
                    .reduce(
                        || [F::zero(); DEGREE],
                        |running, new| {
                            [
                                running[0] + new[0],
                                running[1] + new[1],
                                running[2] + new[2],
                            ]
                        },
                    );

                [
                    eq_r_prime_evals[0] * inner_sum_evals[0],
                    eq_r_prime_evals[1] * inner_sum_evals[1],
                    eq_r_prime_evals[2] * inner_sum_evals[2],
                ]
            })
            .reduce(
                || [F::zero(); DEGREE],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                    ]
                },
            );

        univariate_poly_evals.into()
    }

    fn phase3_compute_prover_message(&self) -> Vec<F> {
        const DEGREE: usize = 3;

        let ReadWriteCheckingProverState {
            inc_cycle,
            eq_r_prime,
            ts1_ra,
            ts2_ra,
            ts3_ra,
            td_wa,
            val,
            ..
        } = self.prover_state.as_ref().unwrap();
        let ts1_ra = ts1_ra.as_ref().unwrap();
        let ts2_ra = ts2_ra.as_ref().unwrap();
        let ts3_ra = ts3_ra.as_ref().unwrap();
        let td_wa = td_wa.as_ref().unwrap();
        let val = val.as_ref().unwrap();

        // Cycle variables are fully bound, so:
        // eq(r', r_cycle) is a constant
        let eq_r_prime_eval = eq_r_prime.as_ref().unwrap().final_sumcheck_claim();
        // ...and Inc(r_cycle) is a constant
        let inc_eval = inc_cycle.final_sumcheck_claim();

        let evals = (0..ts1_ra.len() / 2)
            .into_par_iter()
            .map(|k| {
                let ts1_ra_evals =
                    ts1_ra.sumcheck_evals_array::<DEGREE>(k, BindingOrder::HighToLow);
                let ts2_ra_evals =
                    ts2_ra.sumcheck_evals_array::<DEGREE>(k, BindingOrder::HighToLow);
                let ts3_ra_evals =
                    ts3_ra.sumcheck_evals_array::<DEGREE>(k, BindingOrder::HighToLow);
                let wa_evals = td_wa.sumcheck_evals_array::<DEGREE>(k, BindingOrder::HighToLow);
                let val_evals = val.sumcheck_evals_array::<DEGREE>(k, BindingOrder::HighToLow);

                [
                    wa_evals[0] * (inc_eval + val_evals[0])
                        + self.gamma * ts1_ra_evals[0] * val_evals[0]
                        + self.gamma_sqr * ts2_ra_evals[0] * val_evals[0]
                        + self.gamma_cube * ts3_ra_evals[0] * val_evals[0],
                    wa_evals[1] * (inc_eval + val_evals[1])
                        + self.gamma * ts1_ra_evals[1] * val_evals[1]
                        + self.gamma_sqr * ts2_ra_evals[1] * val_evals[1]
                        + self.gamma_cube * ts3_ra_evals[1] * val_evals[1],
                    wa_evals[2] * (inc_eval + val_evals[2])
                        + self.gamma * ts1_ra_evals[2] * val_evals[2]
                        + self.gamma_sqr * ts2_ra_evals[2] * val_evals[2]
                        + self.gamma_cube * ts3_ra_evals[2] * val_evals[2],
                ]
            })
            .reduce(
                || [F::zero(); DEGREE],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                    ]
                },
            );

        vec![
            eq_r_prime_eval * evals[0],
            eq_r_prime_eval * evals[1],
            eq_r_prime_eval * evals[2],
        ]
    }

    fn phase1_bind(&mut self, r_j: F, round: usize) {
        let ReadWriteCheckingProverState {
            addresses,
            I,
            A,
            inc_cycle,
            gruens_eq_r_prime,
            eq_r_prime,
            chunk_size,
            val_checkpoints,
            ts1_ra,
            ts2_ra,
            ts3_ra,
            td_wa,
            val,
            ..
        } = self.prover_state.as_mut().unwrap();

        let inner_span = tracing::span!(tracing::Level::INFO, "Bind I");
        let _inner_guard = inner_span.enter();

        I.par_iter_mut().for_each(|I_chunk| {
            // Note: A given row in an I_chunk may not be ordered by k after binding
            let mut next_bound_index = 0;
            let mut bound_indices: Vec<Option<usize>> = vec![None; self.K];

            for i in 0..I_chunk.len() {
                let (j_prime, k, inc_lt, inc) = I_chunk[i];
                if let Some(bound_index) = bound_indices[k as usize] {
                    if I_chunk[bound_index].0 == j_prime / 2 {
                        // Neighbor was already processed
                        debug_assert!(j_prime % 2 == 1);
                        I_chunk[bound_index].2 += r_j * inc_lt;
                        I_chunk[bound_index].3 += inc;
                        continue;
                    }
                }
                // First time this k has been encountered
                let bound_value = if j_prime.is_multiple_of(2) {
                    // (1 - r_j) * inc_lt + r_j * inc
                    inc_lt + r_j * (inc - inc_lt)
                } else {
                    r_j * inc_lt
                };

                I_chunk[next_bound_index] = (j_prime / 2, k, bound_value, inc);
                bound_indices[k as usize] = Some(next_bound_index);
                next_bound_index += 1;
            }
            I_chunk.truncate(next_bound_index);
        });

        drop(_inner_guard);
        drop(inner_span);

        gruens_eq_r_prime.bind(r_j);
        inc_cycle.bind_parallel(r_j, BindingOrder::LowToHigh);

        let inner_span = tracing::span!(tracing::Level::INFO, "Update A");
        let _inner_guard = inner_span.enter();

        // Update A for this round (see Equation 55)
        let (A_left, A_right) = A.split_at_mut(1 << round);
        A_left
            .par_iter_mut()
            .zip(A_right.par_iter_mut())
            .for_each(|(x, y)| {
                *y = *x * r_j;
                *x -= *y;
            });

        if round == chunk_size.log_2() - 1 {
            // At this point I has been bound to a point where each chunk contains a single row,
            // so we might as well materialize the full `ra`, `wa`, and `Val` polynomials and perform
            // standard sumcheck directly using those polynomials.

            let span = tracing::span!(tracing::Level::INFO, "Materialize rs1_ra polynomial");
            let _guard = span.enter();

            let num_chunks = addresses.len() / *chunk_size;
            let mut ts1_ra_evals: Vec<F> = unsafe_allocate_zero_vec(self.K * num_chunks);
            ts1_ra_evals
                .par_chunks_mut(self.K)
                .enumerate()
                .for_each(|(chunk_index, ra_chunk)| {
                    for (j_bound, (k, _, _, _)) in addresses
                        [chunk_index * *chunk_size..(chunk_index + 1) * *chunk_size]
                        .iter()
                        .enumerate()
                    {
                        ra_chunk[*k as usize] += A[j_bound];
                    }
                });
            *ts1_ra = Some(MultilinearPolynomial::from(ts1_ra_evals));

            drop(_guard);
            drop(span);

            let span = tracing::span!(tracing::Level::INFO, "Materialize rs2_ra polynomial");
            let _guard = span.enter();

            let num_chunks = addresses.len() / *chunk_size;
            let mut ts2_ra_evals: Vec<F> = unsafe_allocate_zero_vec(self.K * num_chunks);
            ts2_ra_evals
                .par_chunks_mut(self.K)
                .enumerate()
                .for_each(|(chunk_index, ra_chunk)| {
                    for (j_bound, (_, k, _, _)) in addresses
                        [chunk_index * *chunk_size..(chunk_index + 1) * *chunk_size]
                        .iter()
                        .enumerate()
                    {
                        ra_chunk[*k as usize] += A[j_bound];
                    }
                });
            *ts2_ra = Some(MultilinearPolynomial::from(ts2_ra_evals));

            drop(_guard);
            drop(span);

            let span = tracing::span!(tracing::Level::INFO, "Materialize ts2_ra polynomial");
            let _guard = span.enter();

            let num_chunks = addresses.len() / *chunk_size;
            let mut ts3_ra_evals: Vec<F> = unsafe_allocate_zero_vec(self.K * num_chunks);
            ts3_ra_evals
                .par_chunks_mut(self.K)
                .enumerate()
                .for_each(|(chunk_index, ra_chunk)| {
                    for (j_bound, (_, _, k, _)) in addresses
                        [chunk_index * *chunk_size..(chunk_index + 1) * *chunk_size]
                        .iter()
                        .enumerate()
                    {
                        ra_chunk[*k as usize] += A[j_bound];
                    }
                });
            *ts3_ra = Some(MultilinearPolynomial::from(ts3_ra_evals));

            drop(_guard);
            drop(span);

            let span = tracing::span!(tracing::Level::INFO, "Materialize rd_wa polynomial");
            let _guard = span.enter();

            let num_chunks = addresses.len() / *chunk_size;
            let mut td_wa_evals: Vec<F> = unsafe_allocate_zero_vec(self.K * num_chunks);
            td_wa_evals
                .par_chunks_mut(self.K)
                .enumerate()
                .for_each(|(chunk_index, wa_chunk)| {
                    for (j_bound, (_, _, _, k)) in addresses
                        [chunk_index * *chunk_size..(chunk_index + 1) * *chunk_size]
                        .iter()
                        .enumerate()
                    {
                        wa_chunk[*k as usize] += A[j_bound];
                    }
                });
            *td_wa = Some(MultilinearPolynomial::from(td_wa_evals));

            drop(_guard);
            drop(span);

            let span = tracing::span!(tracing::Level::INFO, "Materialize Val polynomial");
            let _guard = span.enter();

            let mut val_evals: Vec<F> = std::mem::take(val_checkpoints);
            val_evals
                .par_chunks_mut(self.K)
                .zip(I.into_par_iter())
                .enumerate()
                .for_each(|(chunk_index, (val_chunk, I_chunk))| {
                    for (j, k, inc_lt, _inc) in I_chunk.iter_mut() {
                        debug_assert_eq!(*j, chunk_index);
                        val_chunk[*k as usize] += *inc_lt;
                    }
                });
            *val = Some(MultilinearPolynomial::from(val_evals));

            drop(_guard);
            drop(span);

            let span = tracing::span!(tracing::Level::INFO, "Materialize eq polynomial");
            let _guard = span.enter();

            let eq_evals: Vec<F> =
                EqPolynomial::evals(&gruens_eq_r_prime.w[..gruens_eq_r_prime.current_index])
                    .par_iter()
                    .map(|x| *x * gruens_eq_r_prime.current_scalar)
                    .collect();
            *eq_r_prime = Some(MultilinearPolynomial::from(eq_evals))
        }
    }

    fn phase2_bind(&mut self, r_j: F) {
        let ReadWriteCheckingProverState {
            ts1_ra,
            ts2_ra,
            ts3_ra,
            td_wa,
            val,
            inc_cycle,
            eq_r_prime,
            ..
        } = self.prover_state.as_mut().unwrap();
        let ts1_ra = ts1_ra.as_mut().unwrap();
        let ts2_ra = ts2_ra.as_mut().unwrap();
        let ts3_ra = ts3_ra.as_mut().unwrap();
        let td_wa = td_wa.as_mut().unwrap();
        let val = val.as_mut().unwrap();
        let eq_r_prime = eq_r_prime.as_mut().unwrap();

        [ts1_ra, ts2_ra, ts3_ra, td_wa, val, inc_cycle, eq_r_prime]
            .into_par_iter()
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
    }

    fn phase3_bind(&mut self, r_j: F) {
        let ReadWriteCheckingProverState {
            ts1_ra,
            ts2_ra,
            ts3_ra,
            td_wa,
            val,
            ..
        } = self.prover_state.as_mut().unwrap();
        let ts1_ra = ts1_ra.as_mut().unwrap();
        let ts2_ra = ts2_ra.as_mut().unwrap();
        let ts3_ra = ts3_ra.as_mut().unwrap();
        let td_wa = td_wa.as_mut().unwrap();
        let val = val.as_mut().unwrap();

        // Note that `eq_r_prime` and `inc` are polynomials over only the cycle
        // variables, so they are not bound here
        [ts1_ra, ts2_ra, ts3_ra, td_wa, val]
            .into_par_iter()
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
    }
}

impl<F: JoltField> SumcheckInstance<F> for RegistersReadWriteChecking<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.K.log_2() + self.T.log_2()
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    #[tracing::instrument(skip_all, name = "RegistersReadWriteChecking::compute_prover_message")]
    fn compute_prover_message(&mut self, round: usize, previous_claim: F) -> Vec<F> {
        let prover_state = self.prover_state.as_ref().unwrap();
        if round < prover_state.chunk_size.log_2() {
            self.phase1_compute_prover_message(round, previous_claim)
        } else if round < self.T.log_2() {
            self.phase2_compute_prover_message()
        } else {
            self.phase3_compute_prover_message()
        }
    }

    #[tracing::instrument(skip_all, name = "RegistersReadWriteChecking::bind")]
    fn bind(&mut self, r_j: F, round: usize) {
        if let Some(prover_state) = self.prover_state.as_ref() {
            if round < prover_state.chunk_size.log_2() {
                self.phase1_bind(r_j, round);
            } else if round < self.T.log_2() {
                self.phase2_bind(r_j);
            } else {
                self.phase3_bind(r_j);
            }
        }
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F],
    ) -> F {
        let accumulator = accumulator.as_ref().unwrap();

        // First `sumcheck_switch_index` rounds bind cycle variables from low to high
        let mut r_cycle = r[..self.sumcheck_switch_index].to_vec();
        // The high-order cycle variables are bound after the switch
        r_cycle.extend(r[self.sumcheck_switch_index..self.T.log_2()].iter().rev());
        let r_cycle = OpeningPoint::<LITTLE_ENDIAN, F>::new(r_cycle);
        let (r_prime, _) = accumulator
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::Ts1Value, SumcheckId::SpartanOuter);

        // eq(r', r_cycle)
        let eq_eval_cycle = EqPolynomial::mle_endian(&r_prime, &r_cycle);

        let (_, val_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, ts1_ra_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::Ts1Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, ts2_ra_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::Ts2Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, ts3_ra_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::Ts3Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, td_wa_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::TdWa,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, inc_claim) = accumulator.borrow().get_committed_polynomial_opening(
            CommittedPolynomial::TdInc,
            SumcheckId::RegistersReadWriteChecking,
        );

        eq_eval_cycle
            * (td_wa_claim * (inc_claim + val_claim)
                + self.gamma * ts1_ra_claim * val_claim
                + self.gamma_sqr * ts2_ra_claim * val_claim
                + self.gamma_cube * ts3_ra_claim * val_claim)
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        // The high-order cycle variables are bound after the switch
        let mut r_cycle = opening_point[self.sumcheck_switch_index..self.T.log_2()].to_vec();
        // First `sumcheck_switch_index` rounds bind cycle variables from low to high
        r_cycle.extend(opening_point[..self.sumcheck_switch_index].iter().rev());
        // Address variables are bound high-to-low
        let r_address = opening_point[self.T.log_2()..].to_vec();

        [r_address, r_cycle].concat().into()
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let val_claim = prover_state.val.as_ref().unwrap().final_sumcheck_claim();
        let ts1_ra_claim = prover_state.ts1_ra.as_ref().unwrap().final_sumcheck_claim();
        let ts2_ra_claim = prover_state.ts2_ra.as_ref().unwrap().final_sumcheck_claim();
        let ts3_ra_claim = prover_state.ts3_ra.as_ref().unwrap().final_sumcheck_claim();
        let td_wa_claim = prover_state.td_wa.as_ref().unwrap().final_sumcheck_claim();
        let inc_claim = prover_state.inc_cycle.final_sumcheck_claim();

        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            val_claim,
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::Ts1Ra,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            ts1_ra_claim,
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::Ts2Ra,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            ts2_ra_claim,
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::Ts3Ra,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            ts3_ra_claim,
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::TdWa,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            td_wa_claim,
        );

        let (_, r_cycle) = opening_point.split_at(self.K.log_2());

        accumulator.borrow_mut().append_dense(
            vec![CommittedPolynomial::TdInc],
            SumcheckId::RegistersReadWriteChecking,
            r_cycle.r,
            &[inc_claim],
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        // Populate opening points for all claims
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::Ts1Ra,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::Ts2Ra,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::TdWa,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
        );

        let (_, r_cycle) = opening_point.split_at(self.K.log_2());

        accumulator.borrow_mut().append_dense(
            vec![CommittedPolynomial::TdInc],
            SumcheckId::RegistersReadWriteChecking,
            r_cycle.r,
        );
    }
}
