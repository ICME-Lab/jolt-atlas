//! # Matrix-Vector Multiplication Precompile
//!
//! This module provides specialized proof generation and verification for matrix-vector
//! multiplication operations within the Jolt proof system. Rather than proving these
//! operations through general VM execution, this precompile uses optimized sumcheck
//! protocols for improved efficiency.
//!
//! ## Overview
//!
//! Matrix-vector multiplications are common operations in machine learning and scientific
//! computing. This precompile handles proving two key aspects:
//!
//! 1. **Execution Correctness**: Proves that C = A×B was computed correctly according to
//!    the matrix-vector multiplication algorithm.
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
//!    - Sample a random challenge point r_c
//!    - Evaluate the result vector polynomial at r_c
//!    - Prove correctness of the matrix-vector multiplication
//! 3. Generate a read-checking proof:
//!    - Prove that operands A, B and result C were correctly accessed from memory
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
use crate::jolt::{
    dag::state_manager::StateManager,
    pcs::SumcheckId,
    precompiles::matvecmult::{
        execution::ExecutionSumcheck, read_checking::ReadCheckingABCSumcheck,
    },
    sumcheck::SumcheckInstance,
    witness::VirtualPolynomial,
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

pub mod execution;
pub mod read_checking;

#[derive(Debug, Clone)]
/// A matrix-vector multiplication precompile instance for the  proof system.
///
/// This struct represents a single matrix-vector multiplication operation within the precompile DAG.
/// Each instance handles both the execution proof (showing the multiplication was computed correctly)
/// and read-checking proof (showing that values were correctly read from memory) for one matrix-vector
/// multiplication in the computation trace.
///
/// The precompile optimizes the proof generation and verification for matrix-vector multiplications,
/// which are common operations in machine learning models. Instead of proving these operations through
/// the general VM execution, we handle them with specialized sumcheck protocols for better efficiency.
///
/// # Fields
///
/// * `final_memory_state` - The final memory state (Val_final) used by the prover; not needed by verifier
/// * `index` - The index of this precompile instance in the PrecompileDag's collection
pub struct MatVecMultPrecompile {
    /// The index of this precompile instance in the PrecompileDag's collection.
    /// Used to retrieve the appropriate memory addresses for operands (a, b) and results (c)
    /// from the state manager.
    pub index: usize,
}

impl MatVecMultPrecompile {
    /// Creates a new prover instance for a matrix-vector multiplication precompile.
    ///
    /// This constructor initializes a precompile instance for the prover side of the
    /// proof system. It extracts the final memory state from the execution trace to enable the
    /// prover to generate proofs for both the execution and read-checking phases.
    ///
    /// # Type Parameters
    ///
    /// * `ProofTranscript` - The transcript type implementing the `Transcript` trait
    /// * `PCS` - The polynomial commitment scheme type implementing `CommitmentScheme`
    ///
    /// # Parameters
    ///
    /// * `index` - The index of this precompile instance in the PrecompileDag's collection
    /// * `sm` - A reference to the state manager containing the execution trace and preprocessing data
    ///
    /// # Returns
    ///
    /// A new `MatVecMultPrecompile` instance configured for the prover
    ///
    /// # Implementation Details
    ///
    /// The function:
    /// 1. Extracts the memory size K from the state manager
    /// 2. Retrieves preprocessing data and execution trace
    /// 3. Constructs the final memory state by iterating through the trace
    /// 4. Initializes the precompile instance with the final memory state and index
    pub fn new_prover(index: usize) -> Self {
        Self { index }
    }

    /// Creates a new verifier instance for a matrix-vector multiplication precompile.
    ///
    /// This constructor initializes a precompile instance for the verifier side of the
    /// proof system. The verifier doesn't need access to the actual memory state; it only requires
    /// the claimed result value from the proof.
    ///
    /// # Parameters
    ///
    /// * `index` - The index of this precompile instance in the PrecompileDag's collection
    /// * `rv_claim_c` - The claimed result vector value from the proof being verified
    ///
    /// # Returns
    ///
    /// A new `MatVecMultPrecompile` instance configured for the verifier, with the rv_claim_c
    /// field initialized with the provided claim
    pub fn new_verifier(index: usize) -> Self {
        Self { index }
    }
}

impl MatVecMultPrecompile {
    /// Creates an execution sumcheck instance for this matrix-vector multiplication precompile.
    ///
    /// This method constructs a sumcheck instance for the execution phase of the proof generation.
    /// The execution phase proves that the matrix-vector multiplication operation was computed
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
    pub fn execution_prover_instance<
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
        let matvec_pp = &pp.shared.precompiles.matvec_instances[self.index];

        // Get the size of the result vector and generate a random challenge
        let n = matvec_pp.c.len();
        let r_c: Vec<F> = sm.get_transcript().borrow_mut().challenge_vector(n.log_2());

        // Compute the evaluation of the result vector at the challenge point
        let E = EqPolynomial::evals(&r_c);
        let rv_claim_c: F = matvec_pp
            .c
            .iter()
            .enumerate()
            .map(|(j, &k)| E[j] * F::from_i64(final_memory_state[k]))
            .sum();

        // Verify the computed claim matches the polynomial evaluation (debug check)
        debug_assert_eq!(
            rv_claim_c,
            MultilinearPolynomial::<F>::from(
                matvec_pp.extract_rv(final_memory_state, |m| { &m.c })
            )
            .evaluate(&r_c)
        );

        // cache the claim and challenge for later use in read-checking
        let accumulator = sm.get_prover_accumulator();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::CRes(self.index),
            SumcheckId::MatVecExecution,
            r_c.clone().into(),
            rv_claim_c,
        );

        // Extract values for operands a and b from memory
        let rv_a = matvec_pp.extract_rv(final_memory_state, |m| &m.a);
        let rv_b = matvec_pp.extract_rv(final_memory_state, |m| &m.b);

        #[cfg(test)]
        {
            use itertools::Itertools;
            use onnx_tracer::tensor::Tensor;
            let rv_c = matvec_pp.extract_rv(final_memory_state, |m| &m.c);
            // The rv_a and rv_b contain the padded values that were actually used in the computation
            // We need to reconstruct tensors with the same padded dimensions that were used
            let k_padded = matvec_pp.a.len(); // This is the padded k dimension
            let n_padded = matvec_pp.c.len(); // This is the padded n dimension
            let a = Tensor::new(
                Some(&rv_a.iter().map(|&x| x as i32).collect_vec()),
                &[1, k_padded],
            )
            .unwrap();
            // Weight matrix with padded dimensions
            let b: Tensor<i32> = Tensor::new(
                Some(&rv_b.iter().map(|&x| x as i32).collect_vec()),
                &[n_padded, k_padded],
            )
            .unwrap();
            let c: onnx_tracer::tensor::Tensor<i32> =
                onnx_tracer::tensor::ops::einsum("mk,nk->mn", &[a.clone(), b.clone()]).unwrap();
            // Now compare the actual results - both should have the same padded dimensions
            assert_eq!(rv_c.iter().map(|&x| x as i32).collect_vec(), c.inner);
        }

        // Create the execution sumcheck instance
        let execution_check = ExecutionSumcheck::new_prover::<ProofTranscript>(
            rv_a, rv_b, r_c, rv_claim_c, self.index,
        );
        Box::new(execution_check)
    }

    /// Creates a read-checking sumcheck instance for this matrix-vector multiplication precompile.
    ///
    /// This method constructs a sumcheck instance for the read-checking phase of the proof generation.
    /// The read-checking phase proves that the values used in the matrix-vector multiplication
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
    pub fn read_checking_prover_instance<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        &self,
        sm: &StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Box<dyn SumcheckInstance<F>> {
        let val_final = sm.get_val_final();

        // Create the read-checking sumcheck instance
        let read_check = ReadCheckingABCSumcheck::new_prover(val_final, sm, self.index);
        Box::new(read_check)
    }

    /// Creates an execution verifier instance for this matrix-vector multiplication precompile.
    ///
    /// This method constructs a sumcheck instance for the verifier to check the execution phase
    /// of the proof. The verifier ensures that the claimed matrix-vector multiplication result
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
    pub fn execution_verifier_instance<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        &mut self,
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Box<dyn SumcheckInstance<F>> {
        // Get preprocessing data for this matrix-vector multiplication
        let (pp, _, _) = sm.get_verifier_data();
        let matvec_pp = &pp.shared.precompiles.matvec_instances[self.index];

        // Get dimensions of the vector and matrix
        let n = matvec_pp.c.len(); // Size of result vector
        let k = matvec_pp.a.len(); // Size of input vector

        // Generate the same random challenge as the prover (using the transcript)
        let r_c: Vec<F> = sm.get_transcript().borrow_mut().challenge_vector(n.log_2());
        // cache r_c
        let verifier_accumulator = sm.get_verifier_accumulator();
        verifier_accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::CRes(self.index),
            SumcheckId::MatVecExecution,
            r_c.clone().into(),
        );
        let rv_claim_c = sm
            .get_virtual_polynomial_opening(
                VirtualPolynomial::CRes(self.index),
                SumcheckId::MatVecExecution,
            )
            .1;

        // Create the execution verification instance
        Box::new(ExecutionSumcheck::new_verifier(
            r_c, rv_claim_c, self.index, k,
        ))
    }

    /// Creates a read-checking verifier instance for this matrix-vector multiplication precompile.
    ///
    /// This method constructs a sumcheck instance for the verifier to check the read-checking phase
    /// of the proof. The verifier ensures that the values used in the matrix-vector multiplication
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
    pub fn read_checking_verifier_instance<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        &mut self,
        sm: &StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Box<dyn SumcheckInstance<F>> {
        // Create the read-checking verification instance
        let read_check = ReadCheckingABCSumcheck::new_verifier(self.index, sm);
        Box::new(read_check)
    }
}
