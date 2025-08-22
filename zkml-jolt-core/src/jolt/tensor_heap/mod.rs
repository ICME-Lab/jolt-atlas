// pub mod booleanity;
// pub mod hamming_weight;
pub mod output_check;
// pub mod raf_evaluation;
pub mod program_io_polynomial;
pub mod read_write_check;

// #![allow(clippy::needless_range_loop)]
#[cfg(test)]
use crate::jolt::execution_trace::sanity_check_mcc;
use crate::jolt::{
    JoltProverPreprocessing,
    execution_trace::JoltONNXCycle,
    tensor_heap::{
        output_check::{OutputProof, OutputSumcheck},
        read_write_check::ReadWriteCheckingProof,
    },
};
use jolt_core::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator},
        unipoly::{CompressedUniPoly, UniPoly},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{
        errors::ProofVerifyError,
        math::Math,
        thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
        transcript::{AppendToTranscript, Transcript},
    },
};
use onnx_tracer::{ProgramOutput, constants::MAX_TENSOR_SIZE};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct TensorHeapTwistProof<F: JoltField, ProofTranscript: Transcript> {
    pub(crate) K: usize,
    /// Proof for the read-checking and write-checking sumchecks
    /// (steps 3 and 4 of Figure 9).
    read_write_checking_proof: ReadWriteCheckingProof<F, ProofTranscript>,
    /// Proof of the Val-evaluation sumcheck (step 6 of Figure 9).
    val_evaluation_proof: ValEvaluationProof<F, ProofTranscript>,
    /// Proof of the output sumcheck.
    output_proof: OutputProof<F, ProofTranscript>,
}

impl<F: JoltField, ProofTranscript: Transcript> TensorHeapTwistProof<F, ProofTranscript> {
    #[tracing::instrument(skip_all, name = "TensorHeapTwistProof::prove")]
    pub fn prove<PCS: CommitmentScheme<ProofTranscript, Field = F>>(
        preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>,
        trace: &[JoltONNXCycle],
        K: usize,
        _opening_accumulator: &mut ProverOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
        program_output: &ProgramOutput,
    ) -> TensorHeapTwistProof<F, ProofTranscript> {
        #[cfg(test)]
        sanity_check_mcc(trace);
        let log_T = (trace.len() * MAX_TENSOR_SIZE).log_2();

        let r: Vec<F> = transcript.challenge_vector(K.log_2());
        let r_prime: Vec<F> = transcript.challenge_vector(log_T);

        let (read_write_checking_proof, r_address, r_cycle) =
            ReadWriteCheckingProof::prove(trace, r, r_prime, transcript);

        let (val_evaluation_proof, mut r_cycle_prime) = prove_val_evaluation(
            trace,
            r_address.clone(),
            r_cycle,
            read_write_checking_proof.val_claim,
            transcript,
        );
        // Cycle variables are bound from low to high
        r_cycle_prime.reverse();

        let mut final_heap_state = vec![0u32; K];
        trace.iter().for_each(|cycle| {
            cycle
                .td_write()
                .0
                .iter()
                .enumerate()
                .for_each(|(i, &address)| {
                    if address < K {
                        final_heap_state[address] = cycle.td_write().2[i] as u32;
                    }
                });
        });

        let output_proof = OutputSumcheck::prove(
            preprocessing,
            trace,
            &r_address,
            transcript,
            program_output,
            final_heap_state,
        );

        TensorHeapTwistProof {
            K,
            read_write_checking_proof,
            val_evaluation_proof,
            output_proof,
        }
    }

    pub fn verify<PCS: CommitmentScheme<ProofTranscript, Field = F>>(
        &self,
        // commitments: &JoltCommitments<F, PCS, ProofTranscript>,
        T: usize,
        _opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
        program_output: ProgramOutput,
    ) -> Result<(), ProofVerifyError> {
        let log_K = self.K.log_2();
        let log_T = T.log_2();
        let r: Vec<F> = transcript.challenge_vector(log_K);
        let r_prime: Vec<F> = transcript.challenge_vector(log_T);
        let (r_address, r_cycle) = self
            .read_write_checking_proof
            .verify(r, r_prime, transcript);

        let (sumcheck_claim, mut r_cycle_prime) = self.val_evaluation_proof.sumcheck_proof.verify(
            self.read_write_checking_proof.val_claim,
            log_T,
            2,
            transcript,
        )?;
        // Cycle variables are bound from low to high
        r_cycle_prime.reverse();

        // let inc_commitment = &commitments.commitments[CommittedPolynomials::RdInc.to_index()];
        // let r_concat = [r_address.as_slice(), r_cycle_prime.as_slice()].concat();
        // opening_accumulator.append(
        //     &[inc_commitment],
        //     r_concat,
        //     &[self.val_evaluation_proof.inc_claim],
        //     transcript,
        // );

        // Compute LT(r_cycle', r_cycle)
        let mut lt_eval = F::zero();
        let mut eq_term = F::one();
        for (x, y) in r_cycle_prime.iter().zip(r_cycle.iter()) {
            lt_eval += (F::one() - x) * y * eq_term;
            eq_term *= F::one() - x - y + *x * y + *x * y;
        }

        assert_eq!(
            sumcheck_claim,
            lt_eval * self.val_evaluation_proof.inc_claim,
            "Val evaluation sumcheck failed"
        );

        OutputSumcheck::verify(
            &r_address,
            T,
            &self.output_proof,
            transcript,
            program_output,
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ValEvaluationProof<F: JoltField, ProofTranscript: Transcript> {
    /// Sumcheck proof for the Val-evaluation sumcheck (steps 6 of Figure 9).
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    /// The claimed evaluation Inc(r_address, r_cycle') output by the Val-evaluation sumcheck.
    inc_claim: F,
}

/// Implements the sumcheck prover for the Val-evaluation sumcheck described in
/// Section 8.1 and Appendix B of the Twist+Shout paper
/// TODO(moodlezoup): incorporate optimization from Appendix B.2
#[tracing::instrument(skip_all)]
pub fn prove_val_evaluation<F: JoltField, ProofTranscript: Transcript>(
    trace: &[JoltONNXCycle],
    r_address: Vec<F>,
    r_cycle: Vec<F>,
    claimed_evaluation: F,
    transcript: &mut ProofTranscript,
) -> (ValEvaluationProof<F, ProofTranscript>, Vec<F>) {
    let T = r_cycle.len().pow2();

    // Compute the size-K table storing all eq(r_address, k) evaluations for
    // k \in {0, 1}^log(K)
    let eq_r_address = EqPolynomial::evals(&r_address);

    let span = tracing::span!(tracing::Level::INFO, "compute Inc");
    let _guard = span.enter();

    // Compute the Inc polynomial using the above table
    let td_writes: Vec<(usize, u64, u64)> = trace
        .iter()
        .flat_map(|cycle| {
            let (address, pre_value, post_value) = cycle.td_write();
            (0..MAX_TENSOR_SIZE).map(move |i| (address[i], pre_value[i], post_value[i]))
        })
        .collect();
    let inc: Vec<F> = td_writes
        .par_iter()
        .map(|(k, pre_value, post_value)| {
            let increment = *post_value as i64 - *pre_value as i64;
            if increment == 0 {
                F::zero()
            } else {
                eq_r_address[*k] * F::from_i64(increment)
            }
        })
        .collect();
    let mut inc = MultilinearPolynomial::from(inc);

    drop(_guard);
    drop(span);

    let span = tracing::span!(tracing::Level::INFO, "compute LT");
    let _guard = span.enter();

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
    let mut lt = MultilinearPolynomial::from(lt);

    drop(_guard);
    drop(span);

    let num_rounds = T.log_2();
    let mut previous_claim = claimed_evaluation;
    let mut r_cycle_prime: Vec<F> = Vec::with_capacity(num_rounds);

    const DEGREE: usize = 2;

    let span = tracing::span!(tracing::Level::INFO, "Val-evaluation sumcheck");
    let _guard = span.enter();

    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
    for _round in 0..num_rounds {
        #[cfg(test)]
        {
            let expected: F = (0..inc.len())
                .map(|j| inc.get_bound_coeff(j) * lt.get_bound_coeff(j))
                .sum::<F>();
            assert_eq!(
                expected, previous_claim,
                "Sumcheck sanity check failed in round {_round}"
            );
        }

        let inner_span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
        let _inner_guard = inner_span.enter();

        let univariate_poly_evals: [F; 2] = (0..inc.len() / 2)
            .into_par_iter()
            .map(|i| {
                let inc_evals = inc.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                let lt_evals = lt.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

                [inc_evals[0] * lt_evals[0], inc_evals[1] * lt_evals[1]]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

        let univariate_poly = UniPoly::from_evals(&[
            univariate_poly_evals[0],
            previous_claim - univariate_poly_evals[0],
            univariate_poly_evals[1],
        ]);

        drop(_inner_guard);
        drop(inner_span);

        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        r_cycle_prime.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        // Bind polynomials
        rayon::join(
            || inc.bind_parallel(r_j, BindingOrder::LowToHigh),
            || lt.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    let proof = ValEvaluationProof {
        sumcheck_proof: SumcheckInstanceProof::new(compressed_polys),
        inc_claim: inc.final_sumcheck_claim(),
    };

    drop_in_background_thread((inc, eq_r_address, lt));

    (proof, r_cycle_prime)
}
