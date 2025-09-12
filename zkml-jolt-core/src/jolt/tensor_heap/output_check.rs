use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_core::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        range_mask_polynomial::RangeMaskPolynomial,
    },
    subprotocols::{
        sparse_dense_shout::ExpandingTable,
        sumcheck::{BatchableSumcheckInstance, SumcheckInstanceProof},
    },
    utils::{
        errors::ProofVerifyError, math::Math, thread::drop_in_background_thread,
        transcript::Transcript,
    },
};
use onnx_tracer::{
    ProgramIO,
    constants::{INPUT_ADDR, MAX_TENSOR_SIZE, OUTPUT_ADDR},
    trace_types::{index_to_addresses, normalize},
};
use rayon::prelude::*;

use crate::jolt::{
    JoltProverPreprocessing,
    execution_trace::{CommittedPolynomials, JoltONNXCycle, WitnessGenerator},
};

#[derive(Debug, Clone)]
struct OutputSumcheckProverState<F: JoltField> {
    /// The MLE of the final heap state
    val_final: MultilinearPolynomial<F>,
    /// Val_output(k) = Val_final(k) if k is in the "output" region of memory,
    /// and 0 otherwise.
    /// Equivalently, Val_io(k) = Val(k, T) * io_mask(k) for
    /// k \in {0, 1}^log(K)
    val_output: MultilinearPolynomial<F>,
    /// EQ(k, r_address)
    eq_poly: MultilinearPolynomial<F>,
    /// output_mask(k) serves as a "mask" for the output region of memory,
    /// i.e. output_mask(k) = 1 if k is in the "output" region of memory,
    /// and 0 otherwise.
    output_mask: MultilinearPolynomial<F>,
    /// Updated to contain the table of evaluations
    /// EQ(x_1, ..., x_k, r_1, ..., r_k), where r_i is the
    /// random challenge for the i'th round of sumcheck.
    eq_table: ExpandingTable<F>,
}

impl<F: JoltField> OutputSumcheckProverState<F> {
    fn initialize(
        final_heap_state: Vec<u32>,
        r_address: &[F],
        _program_output: &ProgramIO,
    ) -> Self {
        let K = final_heap_state.len();

        debug_assert!(K.is_power_of_two());

        // The heap layout is as follows:
        // - [zero_register, output, input, ..memory]
        // Hence the input-output verification range is [output_start, input_end]
        let output_addresses = index_to_addresses(OUTPUT_ADDR);
        let output_start = output_addresses[0];
        let input_addresses = index_to_addresses(INPUT_ADDR);
        let input_end = *input_addresses.last().unwrap();

        let mut val_output = vec![0; K];
        val_output[output_start..input_end]
            .par_iter_mut()
            .zip(final_heap_state[output_start..input_end].par_iter())
            .for_each(|(dest, src)| *dest = *src);

        // Compute io_mask by setting the relevant coefficients to 1
        let mut output_mask = vec![0u8; K];
        output_mask[output_start..input_end]
            .par_iter_mut()
            .for_each(|k| *k = 1);

        // Initialize the EQ table
        let mut eq_table = ExpandingTable::new(K);
        eq_table.reset(F::one());

        Self {
            val_final: final_heap_state.into(),
            val_output: val_output.into(),
            eq_poly: EqPolynomial::evals(r_address).into(),
            output_mask: output_mask.into(),
            eq_table,
        }
    }
}

#[derive(Debug, Clone)]
struct OutputSumcheckVerifierState<F: JoltField> {
    r_address: Vec<F>,
    program_output: ProgramIO,
}

impl<F: JoltField> OutputSumcheckVerifierState<F> {
    fn initialize(r_address: &[F], program_output: &ProgramIO) -> Self {
        Self {
            r_address: r_address.to_vec(),
            program_output: program_output.clone(),
        }
    }
}

/// Proves that the final ram state is consistent with the claimed
/// program output.
#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct OutputProof<F: JoltField, ProofTranscript: Transcript> {
    output_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    val_final_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    /// Claimed evaluation Val_final(r_address) output by `OutputSumcheck`,
    /// proven using `ValFinalSumcheck`
    val_final_claim: F,
    /// Claimed evaluations Inc(r_cycle) and wa(r_cycle) output by `ValFinalSumcheck`
    output_claims: ValFinalSumcheckClaims<F>,
}

/// Sumcheck for the zero-check
///   0 = \sum_k eq(r_address, k) * io_range(k) * (Val_final(k) - Val_io(k))
/// In plain English: the final memory state (Val_final) should be consistent with
/// the expected program outputs (Val_io) at the indices where the program
/// inputs/outputs are stored (io_range).
#[derive(Debug, Clone)]
pub struct OutputSumcheck<F: JoltField> {
    K: usize,
    verifier_state: Option<OutputSumcheckVerifierState<F>>,
    prover_state: Option<OutputSumcheckProverState<F>>,
    /// Claimed evaluation Val_final(r_address) output by `OutputSumcheck`,
    /// proven using `ValFinalSumcheck`
    val_final_claim: Option<F>,
}

impl<F: JoltField> OutputSumcheck<F> {
    #[tracing::instrument(skip_all, name = "OutputSumcheck")]
    pub fn prove<ProofTranscript: Transcript, PCS: CommitmentScheme<ProofTranscript, Field = F>>(
        preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>,
        trace: &[JoltONNXCycle],
        r_address: &[F],
        transcript: &mut ProofTranscript,
        program_output: &ProgramIO,
        final_heap_state: Vec<u32>,
    ) -> OutputProof<F, ProofTranscript> {
        let K = final_heap_state.len();
        let T = trace.len() * MAX_TENSOR_SIZE;

        let output_sumcheck_prover_state =
            OutputSumcheckProverState::initialize(final_heap_state, r_address, program_output);
        let mut output_sumcheck = OutputSumcheck {
            K,
            verifier_state: None,
            prover_state: Some(output_sumcheck_prover_state),
            val_final_claim: None,
        };
        let (output_sumcheck_proof, _r_address) = output_sumcheck.prove_single(transcript);
        let output_sumcheck_prover_state = output_sumcheck.prover_state.as_ref().unwrap();

        let val_final_prover_state = ValFinalSumcheckProverState::initialize(
            preprocessing,
            trace,
            output_sumcheck_prover_state,
        );
        let mut val_final_sumcheck = ValFinalSumcheck {
            T,
            prover_state: Some(val_final_prover_state),
            val_final_claim: output_sumcheck.val_final_claim.unwrap(),
            output_claims: None,
        };

        let (val_final_sumcheck_proof, _r_cycle) = val_final_sumcheck.prove_single(transcript);
        let output_claims = std::mem::take(val_final_sumcheck.output_claims.as_mut().unwrap());
        let val_final_claim = val_final_sumcheck.val_final_claim;

        drop_in_background_thread((output_sumcheck, val_final_sumcheck));

        OutputProof {
            output_sumcheck_proof,
            val_final_sumcheck_proof,
            val_final_claim,
            output_claims,
        }
    }

    pub fn verify<ProofTranscript: Transcript>(
        // val_init: MultilinearPolynomial<F>,
        r_address: &[F],
        T: usize,
        proof: &OutputProof<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
        program_output: ProgramIO,
    ) -> Result<(), ProofVerifyError> {
        let K = r_address.len().pow2();
        let output_sumcheck_verifier_state =
            OutputSumcheckVerifierState::initialize(r_address, &program_output);

        let output_sumcheck = OutputSumcheck {
            K,
            verifier_state: Some(output_sumcheck_verifier_state),
            prover_state: None,
            val_final_claim: Some(proof.val_final_claim),
        };

        let _r_address_prime =
            output_sumcheck.verify_single(&proof.output_sumcheck_proof, transcript)?;

        let val_final_sumcheck = ValFinalSumcheck {
            T,
            prover_state: None,
            val_final_claim: output_sumcheck.val_final_claim.unwrap(),
            output_claims: Some(proof.output_claims.clone()),
        };
        let _r_cycle_prime =
            val_final_sumcheck.verify_single(&proof.val_final_sumcheck_proof, transcript)?;

        Ok(())
    }
}

impl<F: JoltField, ProofTranscript: Transcript> BatchableSumcheckInstance<F, ProofTranscript>
    for OutputSumcheck<F>
{
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.K.log_2()
    }

    fn input_claim(&self) -> F {
        F::zero()
    }

    #[tracing::instrument(skip_all, name = "OutputSumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, _: usize) -> Vec<F> {
        const DEGREE: usize = 3;
        let OutputSumcheckProverState {
            eq_poly,
            output_mask,
            val_final,
            val_output,
            ..
        } = self.prover_state.as_ref().unwrap();

        let univariate_poly_evals: [F; DEGREE] = (0..eq_poly.len() / 2)
            .into_par_iter()
            .map(|k| {
                let eq_evals = eq_poly.sumcheck_evals(k, DEGREE, BindingOrder::HighToLow);
                let output_mask_evals =
                    output_mask.sumcheck_evals(k, DEGREE, BindingOrder::HighToLow);
                let val_final_evals = val_final.sumcheck_evals(k, DEGREE, BindingOrder::HighToLow);
                let val_io_evals = val_output.sumcheck_evals(k, DEGREE, BindingOrder::HighToLow);
                [
                    eq_evals[0] * output_mask_evals[0] * (val_final_evals[0] - val_io_evals[0]),
                    eq_evals[1] * output_mask_evals[1] * (val_final_evals[1] - val_io_evals[1]),
                    eq_evals[2] * output_mask_evals[2] * (val_final_evals[2] - val_io_evals[2]),
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

        univariate_poly_evals.to_vec()
    }

    #[tracing::instrument(skip_all, name = "OutputSumcheck::bind")]
    fn bind(&mut self, r_j: F, _: usize) {
        // Bind address variable
        let OutputSumcheckProverState {
            // val_init,
            val_final,
            val_output,
            eq_poly,
            output_mask,
            eq_table,
            ..
        } = self.prover_state.as_mut().unwrap();

        // We bind Val_init here despite the fact that it is not used in `compute_prover_message`
        // because we'll need Val_init(r) in `ValFinalSumcheck`
        [val_final, val_output, eq_poly, output_mask]
            .into_par_iter()
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
        eq_table.update(r_j);
    }

    fn cache_openings(&mut self) {
        debug_assert!(self.val_final_claim.is_none());
        let OutputSumcheckProverState { val_final, .. } = self.prover_state.as_ref().unwrap();
        self.val_final_claim = Some(val_final.final_sumcheck_claim());
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let OutputSumcheckVerifierState {
            r_address,
            program_output,
        } = self.verifier_state.as_ref().unwrap();
        // The heap layout is as follows:
        // - [zero_register, output, input, ..memory]
        // Hence the input-output verification range is [output_start, input_end]
        let output_addresses = index_to_addresses(OUTPUT_ADDR);
        let output_start = output_addresses[0];
        let output_end = *output_addresses.last().unwrap();

        let input_addresses = index_to_addresses(INPUT_ADDR);
        let input_start = *input_addresses.first().unwrap();
        let input_end = *input_addresses.last().unwrap();

        let val_final_claim = self.val_final_claim.as_ref().unwrap();

        let r_address_prime = &r[..r_address.len()];

        let output_range = RangeMaskPolynomial::new(output_start as u64, input_end as u64);

        let mut val_output = vec![F::zero(); self.K];
        val_output[output_start..output_end]
            .par_iter_mut()
            .zip(program_output.output.par_iter().map(normalize))
            .for_each(|(dest, src)| *dest = F::from_u64(src));

        val_output[input_start..input_end]
            .par_iter_mut()
            .zip(program_output.input.par_iter().map(normalize))
            .for_each(|(dest, src)| *dest = F::from_u64(src));

        let val_output = MultilinearPolynomial::from(val_output);

        let eq_eval = EqPolynomial::mle(r_address, r_address_prime);
        let output_range_eval = output_range.evaluate_mle(r_address_prime);
        let val_output_eval = val_output.evaluate(r_address_prime);

        // Recall that the sumcheck expression is:
        //   0 = \sum_k eq(r_address, k) * output_range(k) * (Val_final(k) - Val_output(k))
        eq_eval * output_range_eval * (*val_final_claim - val_output_eval)
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone, Default)]
pub struct ValFinalSumcheckClaims<F: JoltField> {
    inc_claim: F,
    wa_claim: F,
}

#[derive(Debug, Clone)]
struct ValFinalSumcheckProverState<F: JoltField> {
    inc: MultilinearPolynomial<F>,
    wa: MultilinearPolynomial<F>,
}

impl<F: JoltField> ValFinalSumcheckProverState<F> {
    fn initialize<
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
    >(
        preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>,
        trace: &[JoltONNXCycle],
        output_sumcheck_prover_state: &OutputSumcheckProverState<F>,
    ) -> Self {
        // For RAM, ra = wa but for the sake of intuition we'll call
        // this `write_addresses`
        let write_addresses: Vec<_> = trace
            .par_iter()
            .flat_map(|cycle| cycle.td_write().0)
            .collect();

        // wa(r_address, j)
        let eq_table = &output_sumcheck_prover_state.eq_table;
        let wa_r_address: Vec<_> = write_addresses.par_iter().map(|k| eq_table[*k]).collect();
        let inc = CommittedPolynomials::TdInc.generate_witness(trace, preprocessing);

        #[cfg(test)]
        {
            let OutputSumcheckProverState {
                // val_init,
                val_final,
                ..
            } = &output_sumcheck_prover_state;
            // Check that wa(r, j), and Inc(j) are consistent with
            // the claim Val_final(r)
            let expected = val_final.final_sumcheck_claim();
            let actual = wa_r_address
                .par_iter()
                .enumerate()
                .map(|(j, wa)| inc.get_coeff(j) * wa)
                .sum::<F>();
            assert_eq!(
                expected, actual,
                "Val_final(r_address) ≠ \\sum_j wa(r_address, j) * Inc(j)"
            );
        }

        Self {
            inc,
            wa: wa_r_address.into(),
        }
    }
}

/// This sumcheck virtualizes Val_final(k) as:
/// Val_final(k) = \sum_j Inc(j) * wa(k, j)
///   or equivalently:
/// Val_final(k) = \sum_j Inc(j) * wa(k, j)
/// We feed the output claim Val_final(r_address) from `OutputSumcheck`
/// into this sumcheck, which reduces it to claims about `Inc` and `wa`.
/// Val_init is zero
#[derive(Debug, Clone)]
pub struct ValFinalSumcheck<F: JoltField> {
    T: usize,
    prover_state: Option<ValFinalSumcheckProverState<F>>,
    val_final_claim: F,
    output_claims: Option<ValFinalSumcheckClaims<F>>,
}

impl<F: JoltField, ProofTranscript: Transcript> BatchableSumcheckInstance<F, ProofTranscript>
    for ValFinalSumcheck<F>
{
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
    fn compute_prover_message(&mut self, _: usize) -> Vec<F> {
        const DEGREE: usize = 2;

        let ValFinalSumcheckProverState { inc, wa, .. } = self.prover_state.as_ref().unwrap();

        let univariate_poly_evals: [F; DEGREE] = (0..inc.len() / 2)
            .into_par_iter()
            .map(|j| {
                let inc_evals = inc.sumcheck_evals(j, DEGREE, BindingOrder::HighToLow);
                let wa_evals = wa.sumcheck_evals(j, DEGREE, BindingOrder::HighToLow);
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

    fn cache_openings(&mut self) {
        debug_assert!(self.output_claims.is_none());
        let ValFinalSumcheckProverState { inc, wa, .. } = self.prover_state.as_mut().unwrap();
        self.output_claims = Some(ValFinalSumcheckClaims {
            inc_claim: inc.final_sumcheck_claim(),
            wa_claim: wa.final_sumcheck_claim(),
        });
    }

    fn expected_output_claim(&self, _: &[F]) -> F {
        let ValFinalSumcheckClaims {
            inc_claim,
            wa_claim,
        } = self.output_claims.as_ref().unwrap();
        *inc_claim * wa_claim
    }
}
