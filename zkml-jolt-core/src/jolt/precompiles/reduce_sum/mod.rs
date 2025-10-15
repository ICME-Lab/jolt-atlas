//! # Reduce sum Precompile
//!
//! This module provides specialized proof generation and verification for reduce sum operations within the Jolt proof system. Rather than proving these
//! operations through general VM execution, this precompile uses optimized sumcheck
//! protocols for improved efficiency.
//!
//! ## Overview
//!
//! Reduce sum are common operations in layer norm heads. This precompile handles proving two key aspects:
//!
//! 1. **Execution Correctness**: Proves that Reduce = \sum A was computed correctly according to
//!    the reduce sum algorithm.
//!
//! 2. **Memory Read Correctness**: Proves that the values used in the computation were
//!    correctly read from the claimed memory locations.
//!
//! ## Proof Generation Flow
//!
//! The proof generation follows these steps:
//!
//! 1. Initialize a prover instance with the final memory state
//! 2. Generate an execution proof:
//!    - Sample a random challenge point r_x
//!    - Evaluate the result vector polynomial at r_x
//!    - Prove correctness of the reduce sum
//! 3. Generate a read-checking proof:
//!    - Prove that operands A, result Reduce were correctly accessed from memory
//!
//! ## Verification Flow
//!
//! The verification follows similar steps:
//!
//! 1. Initialize a verifier instance with the claimed result
//! 2. Verify the execution proof
//! 3. Verify the read-checking proof
//!
//! The precompile works together with the `StateManager` to access necessary memory addresses
//! and values for both proof generation and verification.
use std::collections::HashMap;

use crate::{
    jolt::{
        bytecode::BytecodePreprocessing,
        dag::state_manager::StateManager,
        pcs::SumcheckId,
        precompiles::{
            PrecompileTrait,
            reduce_sum::{execution::ExecutionSumcheck, read_checking::ReadCheckingReduceSumcheck},
        },
        sumcheck::SumcheckInstance,
        witness::VirtualPolynomial,
    },
    utils::precompile_pp::{PrecompilePreprocessingTrait, PreprocessingHelper},
};
use jolt_core::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    },
    transcripts::Transcript,
    utils::math::Math,
};
use onnx_tracer::trace_types::ONNXInstr;
use serde::{Deserialize, Serialize};

pub mod execution;
pub mod read_checking;

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Represents preprocessing information for reduce-sum.
///
/// This structure holds memory addresses for operands and results of a
/// reduce-sum operation. These addresses refer to locations in the `Val_final` memory space.
///
/// # Fields
///
/// * `a` - Memory addresses for the matrix operand
/// * `reduce` - Memory addresses for the resulting vector after summation
pub struct ReduceSumPreprocessing {
    /// Memory addresses (in Val_final) for the operand a
    pub a: Vec<usize>,
    /// Memory addresses (in Val_final) for result vector reduce
    pub res: Vec<usize>,
}

impl PrecompilePreprocessingTrait for ReduceSumPreprocessing {}

impl ReduceSumPreprocessing {
    /// Create a new instance of [ReduceSumPreprocessing]
    pub fn new(
        instr: &ONNXInstr,
        td_lookup: &HashMap<usize, &ONNXInstr>,
        bytecode_preprocessing: &BytecodePreprocessing,
    ) -> Self {
        // Get operand instruction
        let a_instr =
            PreprocessingHelper::get_operand_instruction(td_lookup, instr.ts1, "ReduceSum");

        // Extract original dimensions
        let m = a_instr.output_dims[0];
        let n = a_instr.output_dims[1];

        // Calculate padded dimensions
        let (m_padded, n_padded) = PreprocessingHelper::calculate_padded_dimensions(m, n);

        // Collect and process addresses
        let a = PreprocessingHelper::collect_and_pad_matrix(
            a_instr,
            bytecode_preprocessing,
            m,
            m_padded,
            n_padded,
        );
        let res =
            PreprocessingHelper::collect_and_resize_vector(instr, bytecode_preprocessing, m_padded);
        ReduceSumPreprocessing { a, res }
    }
}

#[derive(Debug, Clone)]
/// A reduce-sum precompile instance for the proof system.
///
/// This struct represents a single reduce-sum operation within the precompile DAG.
/// Each instance handles both the execution proof (showing the computation was computed correctly)
/// and read-checking proof (showing that values were correctly read from memory) for one reduce-sum
/// in the trace.
///
/// The precompile optimizes the proof generation and verification for reduce-sums,
/// which are common operations in machine learning models. Instead of proving these operations through
/// the general VM execution, we handle them with specialized sumcheck protocols for better efficiency.
///
/// # Fields
/// * `index` - The index of this precompile instance in the PrecompileDag's collection
pub struct ReduceSumPrecompile {
    /// The index of this precompile instance in the PrecompileDag's collection.
    /// Used to retrieve the appropriate memory addresses for operands (a, b) and results (c)
    /// from the state manager.
    pub index: usize,
}

impl ReduceSumPrecompile {
    /// Creates a new prover instance for a reduce-sum precompile.
    pub fn new_prover(index: usize) -> Self {
        Self { index }
    }

    /// Creates a new verifier instance for a reduce-sum precompile.
    pub fn new_verifier(index: usize) -> Self {
        Self { index }
    }
}

impl PrecompileTrait for ReduceSumPrecompile {
    /// Creates an execution sumcheck instance for this reduce-sum precompile.
    ///
    /// This method constructs a sumcheck instance for the execution phase of the proof generation.
    /// The execution phase proves that the reduce-sum operation was computed
    /// correctly according to the formula C = A * B.
    ///
    /// This method also:
    /// 1. Samples a random point r_c for evaluating the result vector polynomial
    /// 2. Computes the claimed value of the result vector at point r_c
    /// 3. Stores these values in the precompile instance for later use in read-checking
    ///
    /// # Type Parameters
    ///
    /// * `ProofTranscript` - The transcript type implementing the `Transcript` trait
    /// * `PCS` - The polynomial commitment scheme type implementing `CommitmentScheme`
    ///
    /// # Parameters
    ///
    /// * `&mut self` - Mutable reference to this precompile instance
    /// * `sm` - A reference to the state manager containing proof state and data
    ///
    /// # Returns
    ///
    /// A boxed sumcheck instance for the execution phase
    ///
    /// # Implementation Details
    ///
    /// The function:
    /// 1. Retrieves the final memory state and preprocessing data
    /// 2. Generates a random challenge vector r_c using the transcript
    /// 3. Computes the result claim by evaluating the multilinear extension of vector c at r_c
    /// 4. Stores the challenge and claim for later use in read-checking
    /// 5. Extracts the actual values of operands a and b from memory
    /// 6. Creates and returns an ExecutionSumcheck instance with these values
    fn execution_prover_instance<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        &mut self,
        sm: &StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Box<dyn SumcheckInstance<F>> {
        // Get the final memory state (val_final) from the prover
        let final_memory_state = sm.get_val_final();
        let (pp, _, _) = sm.get_prover_data();
        let reduce_pp = &pp.shared.precompiles.reduce_sum[self.index];

        // Get the size of the result vector and generate a random challenge
        let m = reduce_pp.res.len();
        let n = reduce_pp.a.len() / m;
        let r_x: Vec<F> = sm.get_transcript().borrow_mut().challenge_vector(n.log_2());

        // Compute the evaluation of the result vector at the challenge point
        let E = EqPolynomial::evals(&r_x);
        let rv_claim_res: F = reduce_pp
            .res
            .iter()
            .enumerate()
            .map(|(j, &k)| E[j] * F::from_i64(final_memory_state[k]))
            .sum();

        // Verify the computed claim matches the polynomial evaluation (debug check)
        debug_assert_eq!(
            rv_claim_res,
            MultilinearPolynomial::<F>::from(
                reduce_pp.extract_rv(final_memory_state, |m| { &m.res })
            )
            .evaluate(&r_x)
        );

        // cache the claim and challenge for later use in read-checking
        let accumulator = sm.get_prover_accumulator();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::ReduceRes(self.index),
            SumcheckId::ReduceSumExecution,
            r_x.clone().into(),
            rv_claim_res,
        );

        // Extract values for operands a from memory
        let rv_a = reduce_pp.extract_rv(final_memory_state, |m| &m.a);

        // #[cfg(test)]
        // {
        //     use itertools::Itertools;
        //     use onnx_tracer::tensor::Tensor;
        //     let rv_c = reduce_pp.extract_rv(final_memory_state, |m| &m.c);
        //     // The rv_a and rv_b contain the padded values that were actually used in the computation
        //     // We need to reconstruct tensors with the same padded dimensions that were used
        //     let k_padded = reduce_pp.a.len(); // This is the padded k dimension
        //     let n_padded = reduce_pp.c.len(); // This is the padded n dimension
        //     let a = Tensor::new(
        //         Some(&rv_a.iter().map(|&x| x as i32).collect_vec()),
        //         &[1, k_padded],
        //     )
        //     .unwrap();
        //     // Weight matrix with padded dimensions
        //     let b: Tensor<i32> = Tensor::new(
        //         Some(&rv_b.iter().map(|&x| x as i32).collect_vec()),
        //         &[n_padded, k_padded],
        //     )
        //     .unwrap();
        //     let c: onnx_tracer::tensor::Tensor<i32> =
        //         onnx_tracer::tensor::ops::einsum("mk,nk->mn", &[a.clone(), b.clone()]).unwrap();
        //     // Now compare the actual results - both should have the same padded dimensions
        //     assert_eq!(rv_c.iter().map(|&x| x as i32).collect_vec(), c.inner);
        // }

        // Create the execution sumcheck instance
        let execution_check =
            ExecutionSumcheck::new_prover::<ProofTranscript>(rv_a, r_x, rv_claim_res, self.index);
        Box::new(execution_check)
    }

    /// Creates a read-checking sumcheck instance for this reduce-sum precompile.
    ///
    /// This method constructs a sumcheck instance for the read-checking phase of the proof generation.
    /// The read-checking phase proves that the values used in the reduce-sum
    /// (operands a, b and result c) were correctly read from the appropriate memory locations.
    ///
    /// # Type Parameters
    ///
    /// * `ProofTranscript` - The transcript type implementing the `Transcript` trait
    /// * `PCS` - The polynomial commitment scheme type implementing `CommitmentScheme`
    ///
    /// # Parameters
    ///
    /// * `&self` - Reference to this precompile instance
    /// * `sm` - A reference to the state manager containing proof state and data
    ///
    /// # Returns
    ///
    /// A boxed sumcheck instance for the read-checking phase
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The final memory state is not set (should be initialized in new_prover)
    fn read_checking_prover_instance<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        &self,
        sm: &StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Box<dyn SumcheckInstance<F>> {
        let val_final = sm.get_val_final();

        // Create the read-checking sumcheck instance
        let read_check = ReadCheckingReduceSumcheck::new_prover(val_final, sm, self.index);
        Box::new(read_check)
    }

    /// Creates an execution verifier instance for this reduce-sum precompile.
    ///
    /// This method constructs a sumcheck instance for the verifier to check the execution phase
    /// of the proof. The verifier ensures that the claimed reduce-sum result
    /// is consistent with the computation based on the provided claim.
    ///
    /// # Type Parameters
    ///
    /// * `ProofTranscript` - The transcript type implementing the `Transcript` trait
    /// * `PCS` - The polynomial commitment scheme type implementing `CommitmentScheme`
    ///
    /// # Parameters
    ///
    /// * `&mut self` - Mutable reference to this precompile instance
    /// * `sm` - A mutable reference to the state manager containing verification state
    ///
    /// # Returns
    ///
    /// A boxed sumcheck instance for verifying the execution phase
    fn execution_verifier_instance<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        &mut self,
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Box<dyn SumcheckInstance<F>> {
        // Get preprocessing data for this reduce-sum
        let (pp, _, _) = sm.get_verifier_data();
        let reduce_sum = &pp.shared.precompiles.reduce_sum[self.index];

        // Get dimensions of the vector and matrix
        let m = reduce_sum.res.len();
        let n = reduce_sum.a.len() / m;

        // Generate the same random challenge as the prover (using the transcript)
        let r_x: Vec<F> = sm.get_transcript().borrow_mut().challenge_vector(n.log_2());
        // cache r_x
        let verifier_accumulator = sm.get_verifier_accumulator();
        verifier_accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::ReduceRes(self.index),
            SumcheckId::ReduceSumExecution,
            r_x.clone().into(),
        );
        let rv_claim_res = sm
            .get_virtual_polynomial_opening(
                VirtualPolynomial::ReduceRes(self.index),
                SumcheckId::ReduceSumExecution,
            )
            .1;

        // Create the execution verification instance
        Box::new(ExecutionSumcheck::new_verifier(
            r_x,
            rv_claim_res,
            self.index,
            m,
        ))
    }

    /// Creates a read-checking verifier instance for this reduce-sum precompile.
    ///
    /// This method constructs a sumcheck instance for the verifier to check the read-checking phase
    /// of the proof. The verifier ensures that the values used in the reduce-sum
    /// were correctly read from the claimed memory locations.
    ///
    /// # Type Parameters
    ///
    /// * `ProofTranscript` - The transcript type implementing the `Transcript` trait
    /// * `PCS` - The polynomial commitment scheme type implementing `CommitmentScheme`
    ///
    /// # Parameters
    ///
    /// * `&mut self` - Mutable reference to this precompile instance
    /// * `sm` - A reference to the state manager containing verification state
    ///
    /// # Returns
    ///
    /// A boxed sumcheck instance for verifying the read-checking phase
    fn read_checking_verifier_instance<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        &mut self,
        sm: &StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Box<dyn SumcheckInstance<F>> {
        // Create the read-checking verification instance
        let read_check = ReadCheckingReduceSumcheck::new_verifier(self.index, sm);
        Box::new(read_check)
    }
}
