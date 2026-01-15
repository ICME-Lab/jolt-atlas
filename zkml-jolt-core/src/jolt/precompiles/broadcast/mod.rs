use jolt_core::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{BIG_ENDIAN, OpeningPoint},
    },
    transcripts::Transcript,
    utils::math::Math,
};
use onnx_tracer::tensor::Tensor;

use std::{cell::RefCell, rc::Rc};

use crate::jolt::{
    dag::state_manager::StateManager,
    pcs::{ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator},
    sumcheck::SumcheckInstance,
    witness::VirtualPolynomial,
};

/// We treat broadcasting as a tensor product with an identity tensor, with custom equation depending on the number of dimensions, and the dimensions to broadcast.
/// ```text
/// Example:
/// A - [2, 1, 4]               | A_{ij}, (i, j) in [0,2)x[0,4)
/// Broadcast to O - [2, 3, 4]  | O_{ikj}, (i, k, j) in [0,2)x[0,3)x[0,4)
///
/// We use I - [3]              | I_{k}, (k) in [0,3)
/// Equation: "ij,k->ikj"
/// O_{ikj} = A{ij} * I_{k}
/// ```
///
/// For proving.
/// We start with claim about output: O(r)
/// where r = (r_0, r_1, r_2) are the random evaluation points for each dimension i, k and j.
///
/// We want to prove that:
/// O((r_0, r_1, r_2)) = A((r_0, r_2)) * I((r_1)) = A((r_0, r_2))
///
/// Hence the claim about output evaluation at r is just reduced into a claim about A and computing an evaluation of I, with no sumcheck involved
pub struct ExecutionSumcheck {}

impl ExecutionSumcheck {
    pub fn new_prover<F: JoltField>(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        index: usize,
    ) -> Self {
        let final_memory_state = sm.get_val_final();
        let (pp, ..) = sm.get_prover_data();
        let pp = &pp.shared.precompiles.instances[index];

        // Non-padded dimensions, needed to build the broadcasting multilinear polynomial `I`
        let broadcast_dims = pp.b_dims.clone();
        // Padded target dimensions
        let target_dims = pp.c_dims.clone();

        // Extract the input and output addresses, already padded to power-of-two dimensions.
        let input = pp.extract_rv(final_memory_state, |m| &m.a_addr);
        let output = pp.extract_rv(final_memory_state, |m| &m.c_addr);

        Self::init_prover(sm, broadcast_dims, target_dims, input, output, index)
    }

    pub fn init_prover<F: JoltField>(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        broadcast_dims: Vec<usize>,
        target_dims: Vec<usize>,
        input: Vec<i64>,
        output: Vec<i64>,
        index: usize,
    ) -> ExecutionSumcheck {
        // Vars required for multilinear polynomial evaluations for each dimension
        let dim_vars: Vec<usize> = target_dims.iter().map(|dim| dim.log_2()).collect();

        let r_c: Vec<F> = sm
            .get_transcript()
            .borrow_mut()
            .challenge_vector(dim_vars.iter().sum());

        let mut start_idx = 0;
        let mut r_input: Vec<F> = Vec::new();
        let mut r_broadcast: Vec<F> = Vec::new();
        // Allocate challenge variables to input or broadcast polynomials based on whether the dimension is broadcasted
        for (dim_size, vars) in broadcast_dims.iter().zip(dim_vars.iter()) {
            if dim_size == &1 {
                r_input.extend(&r_c[start_idx..start_idx + vars]);
            } else {
                r_broadcast.extend(&r_c[start_idx..start_idx + vars]);
            }
            start_idx += vars;
        }

        // Creating the identity tensor I for broadcasting
        let mut broadcast_tensor = Tensor::new(
            Some(&vec![1i32; broadcast_dims.iter().product()]),
            &broadcast_dims,
        )
        .expect("dims should be correct");
        // Padding the tensor to power-of-two dimensions
        broadcast_tensor.pad_next_power_of_two();

        let broadcast_vals = broadcast_tensor
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<i64>>();

        // Create openings that are inserted in the state manager before creating instances
        let rv_claim_c = MultilinearPolynomial::from(output).evaluate(&r_c);
        sm.get_prover_accumulator().borrow_mut().append_virtual(
            VirtualPolynomial::PrecompileC(index),
            SumcheckId::PrecompileExecution,
            r_c.clone().into(),
            rv_claim_c,
        );

        let rv_claim_a = MultilinearPolynomial::from(input).evaluate(&r_input);
        sm.get_prover_accumulator().borrow_mut().append_virtual(
            VirtualPolynomial::PrecompileA(index),
            SumcheckId::PrecompileExecution,
            r_input.into(),
            rv_claim_a,
        );

        let rv_claim_b = MultilinearPolynomial::from(broadcast_vals).evaluate(&r_broadcast);

        debug_assert_eq!(rv_claim_a * rv_claim_b, rv_claim_c);

        // Dummy value for PrecompileB as read checking needs an opening
        sm.get_prover_accumulator().borrow_mut().append_virtual(
            VirtualPolynomial::PrecompileB(index),
            SumcheckId::PrecompileExecution,
            vec![F::zero()].into(),
            F::zero(),
        );
        Self {}
    }

    pub fn new_verifier<F: JoltField>(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        index: usize,
    ) -> Self {
        let (pp, ..) = sm.get_verifier_data();
        let pp = &pp.shared.precompiles.instances[index];

        let broadcast_dims = pp.b_dims.clone();
        let target_dims = pp.c_dims.clone();

        Self::init_verifier(sm, broadcast_dims, target_dims, index)
    }

    pub fn init_verifier<F: JoltField>(
        sm: &mut StateManager<'_, F, impl Transcript, impl CommitmentScheme<Field = F>>,
        broadcast_dims: Vec<usize>,
        target_dims: Vec<usize>,
        index: usize,
    ) -> Self {
        let dim_vars: Vec<usize> = target_dims.iter().map(|dim| dim.log_2()).collect();
        let num_vars: usize = dim_vars.iter().sum();

        let r_c: Vec<F> = sm.get_transcript().borrow_mut().challenge_vector(num_vars);

        let mut start_idx = 0;
        let mut r_input: Vec<F> = Vec::new();
        let mut r_broadcast: Vec<F> = Vec::new();
        // Allocate challenge variables to input or broadcast polynomials based on whether the dimension is broadcasted
        for (dim_size, vars) in broadcast_dims.iter().zip(dim_vars.iter()) {
            if dim_size == &1 {
                r_input.extend(&r_c[start_idx..start_idx + vars]);
            } else {
                r_broadcast.extend(&r_c[start_idx..start_idx + vars]);
            }
            start_idx += vars;
        }

        // Creating the identity tensor I for broadcasting
        let mut broadcast_tensor = Tensor::new(
            Some(&vec![1i32; broadcast_dims.iter().product()]),
            &broadcast_dims,
        )
        .expect("dims should be correct");
        // Padding the tensor to power-of-two dimensions
        broadcast_tensor.pad_next_power_of_two();

        let broadcast_vals = broadcast_tensor
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<i64>>();

        sm.get_verifier_accumulator().borrow_mut().append_virtual(
            VirtualPolynomial::PrecompileC(index),
            SumcheckId::PrecompileExecution,
            r_c.clone().into(),
        );

        let (_, c_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::PrecompileC(index),
            SumcheckId::PrecompileExecution,
        );

        sm.get_verifier_accumulator().borrow_mut().append_virtual(
            VirtualPolynomial::PrecompileA(index),
            SumcheckId::PrecompileExecution,
            r_input.into(),
        );

        let (_, a_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::PrecompileA(index),
            SumcheckId::PrecompileExecution,
        );

        let b_eval = MultilinearPolynomial::from(broadcast_vals).evaluate(&r_broadcast);

        assert_eq!(a_claim * b_eval, c_claim);

        // Dummy value for PrecompileB as read checking needs an opening
        sm.get_verifier_accumulator().borrow_mut().append_virtual(
            VirtualPolynomial::PrecompileB(index),
            SumcheckId::PrecompileExecution,
            vec![F::zero()].into(),
        );

        Self {}
    }
}

// Dummy implementation so it matches with other precompiles
impl<F: JoltField> SumcheckInstance<F> for ExecutionSumcheck {
    fn degree(&self) -> usize {
        0
    }

    fn num_rounds(&self) -> usize {
        0
    }

    fn input_claim(&self) -> F {
        F::zero()
    }

    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        vec![F::zero()]
    }

    fn bind(&mut self, _r_j: F, _round: usize) {}

    fn expected_output_claim(
        &self,
        _opening_accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        _r: &[F],
    ) -> F {
        F::zero()
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        let mut opening_point = opening_point.to_vec();
        opening_point.reverse();
        opening_point.into()
    }

    fn cache_openings_prover(
        &self,
        _accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        _opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
    }

    fn cache_openings_verifier(
        &self,
        _accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        _opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        jolt::{
            JoltProverPreprocessing, JoltSharedPreprocessing, JoltVerifierPreprocessing,
            bytecode::BytecodePreprocessing, precompiles::PrecompilePreprocessing,
            sumcheck::BatchedSumcheck, trace::JoltONNXCycle,
        },
        utils::precompile_pp::PreprocessingHelper,
    };

    use super::*;
    use ark_bn254::Fr;

    use jolt_core::{poly::commitment::mock::MockCommitScheme, transcripts::Blake2bTranscript};
    use onnx_tracer::{ProgramIO, tensor::Tensor};
    use rand::{Rng, RngCore, SeedableRng, rngs::StdRng};

    // Creates a random `Broadcast` instance.
    pub fn random_broadcast(
        rng: &mut StdRng,
        // Number of dictionary entries to recover
        max_dims: usize,
        // Number of words in the dictionary to gather from
        max_dim_len: usize,
    ) -> (Tensor<i32>, Tensor<i32>) {
        let num_dims = rng.gen_range(1..max_dims);
        let num_bc_dims = rng.gen_range(num_dims..max_dims);

        let dims: Vec<usize> = (0..num_dims)
            .map(|_| {
                if rng.next_u32() % 2 == 0 {
                    1
                } else {
                    rng.gen_range(2..max_dim_len)
                }
            })
            .collect();

        let mut bc_dims: Vec<usize> = (0..num_bc_dims - num_dims)
            .map(|_| rng.gen_range(0..max_dim_len))
            .collect();

        dims.iter().for_each(|&dim| {
            if dim == 1 {
                bc_dims.push(rng.gen_range(0..max_dim_len))
            } else {
                bc_dims.push(dim)
            }
        });

        let input_vals: Vec<i32> = (0..dims.iter().product())
            .map(|_| rng.next_u32() as i32)
            .collect();

        let input = Tensor::new(Some(&input_vals), &dims).unwrap();

        let output = input.expand(&bc_dims).unwrap();

        (input, output)
    }

    #[test]
    fn test_execution_proof() {
        // Number of words to recover from the dictionary
        const NUM_DIMS: usize = 5;
        // Number of words in the dictionary
        const DIM_LEN: usize = 64;

        let mut rng = StdRng::seed_from_u64(42);

        // ----------- State Manager Setup -----------
        let bytecode_pp = BytecodePreprocessing::default();
        let shared_pp = JoltSharedPreprocessing {
            bytecode: bytecode_pp,
            precompiles: PrecompilePreprocessing::empty(),
            fp_lookups: Default::default(),
        };

        let prover_preprocessing: JoltProverPreprocessing<Fr, MockCommitScheme<Fr>> =
            JoltProverPreprocessing {
                generators: (),
                shared: shared_pp.clone(),
            };

        let verifier_preprocessing: JoltVerifierPreprocessing<Fr, MockCommitScheme<Fr>> =
            JoltVerifierPreprocessing {
                generators: (),
                shared: shared_pp,
            };
        let program_io = ProgramIO {
            input: Tensor::new(None, &[]).unwrap(),
            output: Tensor::new(None, &[]).unwrap(),
            min_lookup_input: 0,
            max_lookup_input: 0,
        };

        let trace = vec![JoltONNXCycle::no_op(); 32];

        let mut prover_sm = StateManager::<'_, Fr, Blake2bTranscript, _>::new_prover(
            &prover_preprocessing,
            trace.clone(),
            program_io.clone(),
        );

        let mut verifier_sm = StateManager::<'_, Fr, Blake2bTranscript, _>::new_verifier(
            &verifier_preprocessing,
            program_io,
            trace.len(),
            1 << 8,
            prover_sm.twist_sumcheck_switch_index,
        );
        // ----------- End State Manager Setup -----------

        let mut prover_sumchecks = Vec::new();
        let mut verifier_sumchecks = Vec::new();

        for index in 0..10 {
            let (mut input, mut output) = random_broadcast(&mut rng, NUM_DIMS, DIM_LEN);

            let input_dims = input.dims().to_vec();
            let output_dims = output.dims().to_vec();
            let mut broadcast_dims = output_dims.clone();
            // For each dimension, we set to the dimension size if it is broadcasted, or 1 if not
            broadcast_dims
                .iter_mut()
                .rev()
                .zip(input_dims.iter().rev())
                .for_each(|(dest, in_dim)| {
                    if *in_dim == *dest {
                        *dest = 1;
                    }
                });

            input.pad_next_power_of_two();
            output.pad_next_power_of_two();

            let input_vals: Vec<i64> = input.iter().map(|&x| x as i64).collect();
            let output_vals: Vec<i64> = output.iter().map(|&x| x as i64).collect();

            let exec_sumcheck_prover = ExecutionSumcheck::init_prover(
                &mut prover_sm,
                broadcast_dims.clone(),
                PreprocessingHelper::calculate_padded_dims(&output_dims),
                input_vals,
                output_vals,
                index,
            );

            let prover_acc = prover_sm.get_prover_accumulator();
            let prover_acc_borrow = prover_acc.borrow();
            let verifier_accumulator = verifier_sm.get_verifier_accumulator();
            let mut verifier_acc_borrow = verifier_accumulator.borrow_mut();

            for (key, (_, value)) in prover_acc_borrow.evaluation_openings().iter() {
                let empty_point = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![]);
                verifier_acc_borrow
                    .openings_mut()
                    .insert(*key, (empty_point, *value));
            }
            drop((prover_acc_borrow, verifier_acc_borrow));

            let exec_sumcheck_verifier = ExecutionSumcheck::init_verifier(
                &mut verifier_sm,
                broadcast_dims,
                PreprocessingHelper::calculate_padded_dims(&output_dims),
                index,
            );

            prover_sumchecks.push(exec_sumcheck_prover);
            verifier_sumchecks.push(exec_sumcheck_verifier);
        }

        let prover_sumchecks: Vec<&mut dyn SumcheckInstance<Fr>> = prover_sumchecks
            .iter_mut()
            .map(|sc| &mut *sc as &mut dyn SumcheckInstance<Fr>)
            .collect();
        let verifier_sumchecks: Vec<&dyn SumcheckInstance<Fr>> = verifier_sumchecks
            .iter()
            .map(|sc| &*sc as &dyn SumcheckInstance<Fr>)
            .collect();

        let (proof, _r_sumcheck) = BatchedSumcheck::prove(
            prover_sumchecks,
            Some(prover_sm.get_prover_accumulator()),
            &mut *prover_sm.get_transcript().borrow_mut(),
        );

        let res = BatchedSumcheck::verify(
            &proof,
            verifier_sumchecks,
            Some(verifier_sm.get_verifier_accumulator()),
            &mut *verifier_sm.get_transcript().borrow_mut(),
        );

        assert!(
            res.is_ok(),
            "Sumcheck verification failed with error: {:?}",
            res.err()
        );
    }
}
