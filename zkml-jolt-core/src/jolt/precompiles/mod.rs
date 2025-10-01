//! # Precompile Operations for ZKML
//!
//! This module contains implementations for specialized precompile operations in the Jolt
//! proof system, with a focus on matrix-vector multiplication operations that are crucial for
//! efficient machine learning model execution.
//!
//! ## Core Components
//!
//! * `MatVecPreprocessing` - Stores memory addresses for matrix-vector multiplication operands and results
//! * `PrecompilePreprocessing` - Handles preprocessing of model operations to extract precompile instances
//! * `PrecompileDag` - Manages the directed acyclic graph of precompile operations, handling both prover
//!   and verifier workflows
//! * `MatVecMultPrecompile` - Implements the matrix-vector multiplication precompile (in the matvecmult submodule)
//!
//! ## Workflow
//!
//! The precompile system works in two main phases:
//!
//! 1. **Preprocessing Phase**: Analyzes ONNX model operations to identify precompile operations
//!    (like MatMult) and extracts memory addresses for operands and results.
//!
//! 2. **Proving/Verification Phase**: Handles both execution sum-checks and read-checking sum-checks
//!    to efficiently prove the correctness of precompile operations.
//!
//! The system is designed to optimize performance-critical operations in ML inference by handling
//! them with specialized proof techniques rather than processing them through the general-purpose
//! VM execution proof system.
use crate::jolt::{
    bytecode::{BytecodePreprocessing, tensor_sequence_remaining, zkvm_address},
    dag::state_manager::StateManager,
    precompiles::{matvecmult::MatVecMultPrecompile, val_final::ValFinalSumcheck},
    sumcheck::{BatchedSumcheck, SingleSumcheck, SumcheckInstance},
};
use itertools::Itertools;
use jolt_core::{
    field::JoltField, poly::commitment::commitment_scheme::CommitmentScheme,
    subprotocols::sumcheck::SumcheckInstanceProof, transcripts::Transcript,
    utils::errors::ProofVerifyError,
};
use onnx_tracer::{
    graph::model::Model,
    trace_types::{ONNXInstr, ONNXOpcode},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod matvecmult;
pub mod val_final;

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Represents preprocessing information for matrix-vector multiplication.
///
/// This structure holds memory addresses for operands and results of a matrix-vector
/// multiplication operation. These addresses refer to locations in the `Val_final` memory space.
///
/// # Fields
///
/// * `a` - Memory addresses for the vector operand
/// * `b` - Memory addresses for the matrix operand
/// * `c` - Memory addresses for the resulting vector after multiplication
pub struct MatVecPreprocessing {
    /// Memory addresses (in Val_final) for vector operand a
    pub a: Vec<usize>,
    /// Memory addresses (in Val_final) for Matrix operand b
    pub b: Vec<usize>,
    /// Memory addresses (in Val_final) for result vector c
    pub c: Vec<usize>,
}

impl MatVecPreprocessing {
    /// Extracts read-values from val_final using the stored read-addresses.
    /// Given the val_final lookup table, computes the actual values of operands and results
    /// by indexing into val_final with the preprocessed memory addresses.
    pub fn extract_rv<T>(&self, val_final: &[i64], field_selector: impl Fn(&Self) -> &T) -> Vec<i64>
    where
        T: AsRef<[usize]>,
    {
        field_selector(self)
            .as_ref()
            .iter()
            .map(|&k| val_final[k])
            .collect_vec()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// A structure that holds preprocessing data for precompiled operations.
///
/// This struct contains collections of preprocessing instances for different
/// types of precompiled operations that require preprocessing before execution.
///
/// # Fields
///
/// * `matvec_instances` - A collection of matrix-vector operation preprocessing instances.
pub struct PrecompilePreprocessing {
    pub matvec_instances: Vec<MatVecPreprocessing>,
}

impl PrecompilePreprocessing {
    /// Preprocesses a model to extract matrix-vector multiplication operations for efficient
    /// proving/verification in  proofs.
    ///
    /// This function extracts all matrix multiplication instructions from the provided model,
    /// identifies their memory addresses for both operands and results, and organizes them
    /// into a collection of `MatVecPreprocessing` instances. This preprocessing step is crucial
    /// for enabling efficient proof generation and verification for these operations.
    ///
    /// # Parameters
    ///
    /// * `model` - A function that returns a `Model` instance to be preprocessed
    /// * `bytecode_preprocessing` - Contains address mapping information for tensor elements
    ///
    /// # Returns
    ///
    /// A new `PrecompilePreprocessing` instance containing all extracted matrix-vector operations
    ///
    /// # Implementation Details
    ///
    /// The function:
    /// 1. Decodes the model into bytecode instructions
    /// 2. Builds an O(1) lookup map for instructions by their td value
    /// 3. Iterates through instructions to identify MatMult operations
    /// 4. For each MatMult, collects memory addresses for both operands and the result
    /// 5. Returns a structure containing all identified operations for later proof generation
    ///
    /// # Type Parameters
    ///
    /// * `ModelFunc` - A function type that returns a `Model`
    pub fn preprocess<ModelFunc>(
        model: ModelFunc,
        bytecode_preprocessing: &BytecodePreprocessing,
    ) -> Self
    where
        ModelFunc: Fn() -> Model,
    {
        // Extract MatMult instructions from the model and collect their memory addresses
        // for operands (a, b) and results (c) to enable prover/verifier to efficiently prove/verify the PrecompileProofs
        let bytecode = onnx_tracer::decode_model(model());

        // Build a lookup map for O(1) instruction lookups by td value
        let td_lookup: HashMap<usize, &ONNXInstr> = bytecode
            .iter()
            .filter_map(|instr| instr.td.map(|td| (td, instr)))
            .collect();

        let mut matvec_instances: Vec<MatVecPreprocessing> = Vec::new();

        for instr in bytecode.iter() {
            if instr.opcode == ONNXOpcode::MatMult {
                let ts1 = instr.ts1.unwrap(); // Safe to unwrap - MatMult always has operands
                let ts2 = instr.ts2.unwrap(); // Safe to unwrap - MatMult always has operands

                // O(1) lookup for operand instructions
                let a_instr = td_lookup[&ts1];
                let b_instr = td_lookup[&ts2];

                // Collect addresses for operands and result
                let mut a = Self::collect_addresses(a_instr, bytecode_preprocessing);
                let mut b = Self::collect_addresses(b_instr, bytecode_preprocessing);
                let mut c = Self::collect_addresses(instr, bytecode_preprocessing);

                // Extract original dimensions to match onnx-tracer's padding logic
                // For input a: [m, k] where m=1 (batch), k=input_length
                // For weight b: [n, k] where n=output_length, k=input_length
                // For result c: [m, n] where m=1 (batch), n=output_length

                // a_instr represents the input vector with k elements
                let k = a_instr.output_dims[1]; // k dimension from input vector [1, k]
                // instr represents the result with n elements
                let n = instr.output_dims[1]; // n dimension from output vector [1, n]
                // m is always 1 for our vector case
                let m: usize = 1;

                // Calculate power-of-two dimensions matching onnx-tracer logic
                let m_pow2 = if m.is_power_of_two() {
                    m
                } else {
                    m.next_power_of_two()
                };
                let k_pow2 = if k.is_power_of_two() {
                    k
                } else {
                    k.next_power_of_two()
                };
                let n_pow2 = if n.is_power_of_two() {
                    n
                } else {
                    n.next_power_of_two()
                };

                // Resize to match onnx-tracer's padded dimensions
                a.resize(m_pow2 * k_pow2, 0); // [m_pow2, k_pow2] flattened 
                b.resize(n_pow2 * k_pow2, 0); // [n_pow2, k_pow2] flattened
                c.resize(n_pow2, 0); // n_pow2 to ensure power-of-2 for polynomial construction
                matvec_instances.push(MatVecPreprocessing { a, b, c });
            }
        }
        Self { matvec_instances }
    }

    /// Collects memory addresses for tensor elements based on an instruction.
    ///
    /// This helper method extracts the memory addresses for all active output elements
    /// of a given instruction by querying the bytecode preprocessing map.
    ///
    /// # Parameters
    ///
    /// * `instr` - The ONNX instruction containing tensor information
    /// * `bytecode_preprocessing` - Contains the mapping from virtual addresses to physical addresses
    ///
    /// # Returns
    ///
    /// A vector of memory addresses for the instruction's active output elements
    fn collect_addresses(
        instr: &ONNXInstr,
        bytecode_preprocessing: &BytecodePreprocessing,
    ) -> Vec<usize> {
        (0..instr.active_output_elements)
            .map(|i| {
                bytecode_preprocessing.vt_address_map[&(
                    zkvm_address(instr.td),
                    tensor_sequence_remaining(instr.active_output_elements, i),
                )]
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
/// A directed acyclic graph structure that manages precompiled operations for the ZKML system.
///
/// The `PrecompileDag` serves as the central coordinator for all precompiled operations
/// in the Jolt  proof system, particularly focused on matrix-vector
/// multiplication operations which are common in machine learning models.
///
/// This structure maintains a collection of precompile instances (currently matrix-vector
/// multiplication instances) and provides methods to construct both prover and verifier
/// variants, as well as methods to access execution and read-checking instances for
/// sumcheck protocols.
///
/// # Fields
///
/// * `matvec_instances` - A vector of matrix-vector multiplication precompile instances
pub struct PrecompileDag {
    /// Collection of matrix-vector multiplication precompile instances used in the DAG
    matvec_instances: Vec<MatVecMultPrecompile>,
}

impl PrecompileDag {
    /// Creates a new prover instance of the `PrecompileDag`.
    ///
    /// This constructor initializes a DAG for the prover side of the  proof system.
    /// It extracts matrix-vector multiplication instances from the provided state manager and
    /// initializes prover-specific precompile instances for each one.
    ///
    /// # Type Parameters
    ///
    /// * `F` - The finite field type implementing the `JoltField` trait, used for arithmetic operations
    /// * `ProofTranscript` - The transcript type implementing the `Transcript` trait
    /// * `PCS` - The polynomial commitment scheme type implementing `CommitmentScheme` with field type `F`
    ///
    /// # Parameters
    ///
    /// * `sm` - A reference to the state manager containing preprocessing information and proof state
    ///
    /// # Returns
    ///
    /// A new `PrecompileDag` instance configured for the prover
    pub fn new_prover<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        sm: &StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let num_matvec_instances = sm
            .get_prover_data()
            .0
            .shared
            .precompiles
            .matvec_instances
            .len();
        let matvec_instances = (0..num_matvec_instances)
            .map(MatVecMultPrecompile::new_prover)
            .collect();
        Self { matvec_instances }
    }

    /// Creates a new verifier instance of the `PrecompileDag`.
    ///
    /// This constructor initializes a DAG for the verifier side of the  proof system.
    /// It extracts matrix-vector multiplication instances from the provided state manager and
    /// initializes verifier-specific precompile instances for each one, incorporating claims
    /// from the proof being verified.
    ///
    /// # Type Parameters
    ///
    /// * `ProofTranscript` - The transcript type implementing the `Transcript` trait
    /// * `PCS` - The polynomial commitment scheme type implementing `CommitmentScheme` with field type `F`
    ///
    /// # Parameters
    ///
    /// * `rv_claims_c` - Vector of claims for the result vectors (c) of the matrix-vector multiplications
    /// * `sm` - A reference to the state manager containing preprocessing information and verification state
    ///
    /// # Returns
    ///
    /// A new `PrecompileDag` instance configured for the verifier
    pub fn new_verifier<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        sm: &StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let num_matvec_instances = sm
            .get_verifier_data()
            .0
            .shared
            .precompiles
            .matvec_instances
            .len();
        let matvec_instances = (0..num_matvec_instances)
            .map(MatVecMultPrecompile::new_verifier)
            .collect();
        Self { matvec_instances }
    }
}

impl PrecompileDag {
    /// Gathers all prover instances for execution sum-checks.
    ///
    /// This method collects sumcheck instances from all matrix-vector multiplication precompiles
    /// for the execution phase of the proof generation. The execution phase proves that the
    /// matrix-vector multiplication operations were performed correctly.
    ///
    /// # Type Parameters
    ///
    /// * `F` - The finite field type implementing the `JoltField` trait, used for arithmetic operations
    /// * `ProofTranscript` - The transcript type implementing the `Transcript` trait
    /// * `PCS` - The polynomial commitment scheme type implementing `CommitmentScheme` with field type `F`
    ///
    /// # Parameters
    ///
    /// * `&mut self` - Mutable reference to the precompile DAG
    /// * `sm` - A reference to the state manager containing proof state and data
    ///
    /// # Returns
    ///
    /// A vector of boxed sumcheck instances that implement the `SumcheckInstance<F>` trait
    pub fn execution_prover_instances<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        &mut self,
        sm: &StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        self.matvec_instances
            .iter_mut()
            .map(|instance| instance.execution_prover_instance(sm))
            .collect()
    }

    /// Gathers all prover instances for read-checking sum-checks.
    ///
    /// This method collects sumcheck instances from all matrix-vector multiplication precompiles
    /// for the read-checking phase of the proof generation. The read-checking phase proves that
    /// the values used in the matrix-vector multiplications were correctly read from memory.
    ///
    /// # Type Parameters
    ///
    /// * `F` - The finite field type implementing the `JoltField` trait, used for arithmetic operations
    /// * `ProofTranscript` - The transcript type implementing the `Transcript` trait
    /// * `PCS` - The polynomial commitment scheme type implementing `CommitmentScheme` with field type `F`
    ///
    /// # Parameters
    ///
    /// * `&self` - Reference to the precompile DAG
    /// * `sm` - A reference to the state manager containing proof state and data
    ///
    /// # Returns
    ///
    /// A vector of boxed sumcheck instances that implement the `SumcheckInstance<F>` trait
    pub fn read_checking_prover_instances<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        &self,
        sm: &StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        self.matvec_instances
            .iter()
            .map(|instance| instance.read_checking_prover_instance(sm))
            .collect()
    }

    /// Gets all verifier instance for val_final sum-checks.
    ///
    /// This method collects sumcheck instances from all matrix-vector multiplication precompiles
    /// for the val_final phase of proof verification. These instances are used to verify the
    /// val_final part of the proof, which shows that the values used in the matrix-vector
    /// multiplications were correctly read from memory.
    ///
    /// # Type Parameters
    ///
    /// * `ProofTranscript` - The transcript type implementing the `Transcript` trait
    /// * `PCS` - The polynomial commitment scheme type implementing `CommitmentScheme` with field type `F`
    ///
    /// # Parameters
    ///
    /// * `&mut self` - Mutable reference to the precompile DAG
    /// * `sm` - A reference to the state manager containing verification state and data
    ///
    /// # Returns
    ///
    /// A boxed sumcheck instances that implement the `SumcheckInstance<F>` trait
    pub fn val_final_prover_instance<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        &self,
        sm: &StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Box<dyn SumcheckInstance<F>> {
        Box::new(ValFinalSumcheck::new_prover(sm))
    }

    /// Gets all verifier instances for execution sum-checks.
    ///
    /// This method collects sumcheck instances from all matrix-vector multiplication precompiles
    /// for the execution phase of proof verification. These instances are used to verify the
    /// execution part of the proof, which shows the matrix-vector operations were performed correctly.
    ///
    /// # Type Parameters
    ///
    /// * `F` - The finite field type implementing the `JoltField` trait, used for arithmetic operations
    /// * `ProofTranscript` - The transcript type implementing the `Transcript` trait
    /// * `PCS` - The polynomial commitment scheme type implementing `CommitmentScheme` with field type `F`
    ///
    /// # Parameters
    ///
    /// * `&mut self` - Mutable reference to the precompile DAG
    /// * `sm` - A mutable reference to the state manager containing verification state and data
    ///
    /// # Returns
    ///
    /// A vector of boxed sumcheck instances that implement the `SumcheckInstance<F>` trait
    pub fn execution_verifier_instances<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        &mut self,
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        self.matvec_instances
            .iter_mut()
            .map(|instance| instance.execution_verifier_instance(sm))
            .collect()
    }

    /// Gets all verifier instances for read-checking sum-checks.
    ///
    /// This method collects sumcheck instances from all matrix-vector multiplication precompiles
    /// for the read-checking phase of proof verification. These instances are used to verify the
    /// read-checking part of the proof, which shows that the values used in the matrix-vector
    /// multiplications were correctly read from memory.
    ///
    /// # Type Parameters
    ///
    /// * `ProofTranscript` - The transcript type implementing the `Transcript` trait
    /// * `PCS` - The polynomial commitment scheme type implementing `CommitmentScheme` with field type `F`
    ///
    /// # Parameters
    ///
    /// * `&mut self` - Mutable reference to the precompile DAG
    /// * `sm` - A reference to the state manager containing verification state and data
    ///
    /// # Returns
    ///
    /// A vector of boxed sumcheck instances that implement the `SumcheckInstance<F>` trait
    pub fn read_checking_verifier_instances<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        &mut self,
        sm: &StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        self.matvec_instances
            .iter_mut()
            .map(|instance| instance.read_checking_verifier_instance(sm))
            .collect()
    }

    /// Gets all verifier instance for val_final sum-checks.
    ///
    /// This method collects sumcheck instances from all matrix-vector multiplication precompiles
    /// for the val_final phase of proof verification. These instances are used to verify the
    /// val_final part of the proof, which shows that the values used in the matrix-vector
    /// multiplications were correctly read from memory.
    ///
    /// # Type Parameters
    ///
    /// * `ProofTranscript` - The transcript type implementing the `Transcript` trait
    /// * `PCS` - The polynomial commitment scheme type implementing `CommitmentScheme` with field type `F`
    ///
    /// # Parameters
    ///
    /// * `&mut self` - Mutable reference to the precompile DAG
    /// * `sm` - A reference to the state manager containing verification state and data
    ///
    /// # Returns
    ///
    /// A boxed sumcheck instances that implement the `SumcheckInstance<F>` trait
    pub fn val_final_verifier_instance<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        &self,
        sm: &StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Box<dyn SumcheckInstance<F>> {
        Box::new(ValFinalSumcheck::new_verifier(sm))
    }
}

#[derive(Clone, Debug)]
pub struct PrecompileSNARK<F: JoltField, FS: Transcript> {
    execution_proof: SumcheckInstanceProof<F, FS>,
    read_checking_proof: SumcheckInstanceProof<F, FS>,
    val_final_proof: SumcheckInstanceProof<F, FS>,
}

impl<F: JoltField, FS: Transcript> PrecompileSNARK<F, FS> {
    pub fn prove<'a, PCS: CommitmentScheme<Field = F>>(sm: &StateManager<'a, F, FS, PCS>) -> Self {
        let mut precompile_dag = PrecompileDag::new_prover(sm);

        let execution_proof = Self::prove_execution(&mut precompile_dag, sm);
        let read_checking_proof = Self::prove_read_checking(&precompile_dag, sm);
        let val_final_proof = Self::prove_val_final(&precompile_dag, sm);

        PrecompileSNARK {
            execution_proof,
            read_checking_proof,
            val_final_proof,
        }
    }

    /// Proves the execution phase of matrix-vector multiplication precompiles.
    /// This includes running execution sum-checks and collecting output claims.
    fn prove_execution<'a, PCS: CommitmentScheme<Field = F>>(
        precompile_dag: &mut PrecompileDag,
        sm: &StateManager<'a, F, FS, PCS>,
    ) -> SumcheckInstanceProof<F, FS> {
        // Get all execution sum-checks and batch them
        let mut execution_instances: Vec<_> = precompile_dag.execution_prover_instances(sm);
        let execution_instances_mut: Vec<&mut dyn SumcheckInstance<F>> = execution_instances
            .iter_mut()
            .map(|instance| &mut **instance as &mut dyn SumcheckInstance<F>)
            .collect();

        let transcript = sm.get_transcript();
        let accumulator = sm.get_prover_accumulator();
        let (execution_proof, _r_execution) = BatchedSumcheck::prove(
            execution_instances_mut,
            Some(accumulator.clone()),
            &mut *transcript.borrow_mut(),
        );

        execution_proof
    }

    /// Proves the read-checking phase for matrix-vector multiplication precompiles.
    /// This includes running read-checking sum-checks and collecting memory access claims.
    fn prove_read_checking<'a, PCS: CommitmentScheme<Field = F>>(
        precompile_dag: &PrecompileDag,
        sm: &StateManager<'a, F, FS, PCS>,
    ) -> SumcheckInstanceProof<F, FS> {
        // Get all read-checking sum-checks and batch them
        let mut read_checking_instances: Vec<_> = precompile_dag.read_checking_prover_instances(sm);
        let read_checking_instances_mut: Vec<&mut dyn SumcheckInstance<F>> =
            read_checking_instances
                .iter_mut()
                .map(|instance| &mut **instance as &mut dyn SumcheckInstance<F>)
                .collect();

        let transcript = sm.get_transcript();
        let accumulator = sm.get_prover_accumulator();
        let (read_checking_proof, _r_read_checking) = BatchedSumcheck::prove(
            read_checking_instances_mut,
            Some(accumulator.clone()),
            &mut *transcript.borrow_mut(),
        );

        read_checking_proof
    }

    fn prove_val_final<'a, PCS: CommitmentScheme<Field = F>>(
        precompile_dag: &PrecompileDag,
        sm: &StateManager<'a, F, FS, PCS>,
    ) -> SumcheckInstanceProof<F, FS> {
        let mut val_final_instance = precompile_dag.val_final_prover_instance(sm);
        let transcript = sm.get_transcript();
        let accumulator = sm.get_prover_accumulator();
        let (val_final_proof, _) = SingleSumcheck::prove(
            &mut *val_final_instance,
            Some(accumulator.clone()),
            &mut *transcript.borrow_mut(),
        );
        val_final_proof
    }

    pub fn verify<'a, PCS: CommitmentScheme<Field = F>>(
        &self,
        sm: &mut StateManager<'a, F, FS, PCS>,
    ) -> Result<(), ProofVerifyError> {
        let mut precompile_dag = PrecompileDag::new_verifier(sm);
        let execution_instances: Vec<_> = precompile_dag.execution_verifier_instances(sm);
        let execution_instances_ref: Vec<&dyn SumcheckInstance<F>> = execution_instances
            .iter()
            .map(|instance| &**instance as &dyn SumcheckInstance<F>)
            .collect();
        let transcript = sm.get_transcript();
        let accumulator = sm.get_verifier_accumulator();
        BatchedSumcheck::verify(
            &self.execution_proof,
            execution_instances_ref,
            Some(accumulator.clone()),
            &mut *transcript.borrow_mut(),
        )?;
        let read_checking_instances: Vec<_> = precompile_dag.read_checking_verifier_instances(sm);
        let read_checking_instances_ref: Vec<&dyn SumcheckInstance<F>> = read_checking_instances
            .iter()
            .map(|instance| &**instance as &dyn SumcheckInstance<F>)
            .collect();
        let transcript = sm.get_transcript();
        let accumulator = sm.get_verifier_accumulator();
        BatchedSumcheck::verify(
            &self.read_checking_proof,
            read_checking_instances_ref,
            Some(accumulator.clone()),
            &mut *transcript.borrow_mut(),
        )?;
        let val_final_instance = precompile_dag.val_final_verifier_instance(sm);
        let transcript = sm.get_transcript();
        let accumulator = sm.get_verifier_accumulator();
        SingleSumcheck::verify(
            &*val_final_instance,
            &self.val_final_proof,
            Some(accumulator.clone()),
            &mut *transcript.borrow_mut(),
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use jolt_core::{
        poly::{commitment::mock::MockCommitScheme, opening_proof::OpeningPoint},
        transcripts::KeccakTranscript,
    };
    use onnx_tracer::{builder, tensor::Tensor};

    use crate::jolt::{
        JoltSNARK, dag::state_manager::StateManager, precompiles::PrecompileSNARK, trace::trace,
    };
    type PCS = MockCommitScheme<Fr>;

    /// Helper function to test matrix multiplication models
    fn test_matmult_model_helper<ModelFunc>(model: ModelFunc, input: Tensor<i32>)
    where
        ModelFunc: Fn() -> onnx_tracer::graph::model::Model + Copy,
    {
        // --- Generate proof ---

        let preprocessing =
            JoltSNARK::<Fr, PCS, KeccakTranscript>::prover_preprocess(model, 1 << 10);
        let (trace, program_io) = trace(model, &input, &preprocessing.shared.bytecode);
        let trace_len = trace.len();
        let mut sm: StateManager<'_, Fr, KeccakTranscript, PCS> =
            StateManager::new_prover(&preprocessing, trace, program_io.clone());
        let twist_sumcheck_switch_index = sm.twist_sumcheck_switch_index;
        let K = sm.get_memory_K();
        let proof = PrecompileSNARK::prove(&sm);

        // --- Verify proof ---

        // Verifier StateManager
        let verifier_preprocessing = (&preprocessing).into();
        let mut verifier_sm: StateManager<'_, Fr, KeccakTranscript, PCS> =
            StateManager::new_verifier(
                &verifier_preprocessing,
                program_io,
                trace_len,
                K,
                twist_sumcheck_switch_index,
            );

        let prover_state = sm.prover_state.as_mut().unwrap();
        let openings = std::mem::take(&mut prover_state.accumulator.borrow_mut().openings);
        let opening_accumulator = verifier_sm.get_verifier_accumulator();
        for (key, (_, claim)) in openings.iter() {
            opening_accumulator
                .borrow_mut()
                .openings_mut()
                .insert(*key, (OpeningPoint::default(), *claim));
        }

        assert!(proof.verify(&mut verifier_sm).is_ok());
    }

    #[test]
    fn test_minimal_matmult_non_power_of_two_model() {
        let input = Tensor::new(Some(&[1, 2, 3, 4, 5]), &[1, 5]).unwrap();
        test_matmult_model_helper(builder::minimal_matmult_non_power_of_two_model, input);
    }

    #[test]
    fn test_dual_matmult_non_power_of_two_model() {
        let input = Tensor::new(Some(&[1, 2, 3, 4, 5]), &[1, 5]).unwrap();
        test_matmult_model_helper(builder::dual_matmult_non_power_of_two_model, input);
    }

    #[test]
    fn test_neg_dual_matmult_model() {
        let input = Tensor::new(Some(&[-1, -2, -3, -4]), &[1, 4]).unwrap();
        test_matmult_model_helper(builder::dual_matmult_model, input);
    }

    #[test]
    fn test_dual_matmult_model() {
        let input = Tensor::new(Some(&[1, 2, 3, 4]), &[1, 4]).unwrap();
        test_matmult_model_helper(builder::dual_matmult_model, input);
    }

    #[test]
    fn test_triple_matmult_model() {
        let input = Tensor::new(Some(&[1, 2, 3, 4, 1, 2, 3, 4]), &[1, 8]).unwrap();
        test_matmult_model_helper(builder::triple_matmult_model, input);
    }
}
