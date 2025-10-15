//! # Precompile Operations for ZKML
//!
//! This module contains implementations for specialized precompile operations in the Jolt
//! proof system, with a focus on matrix-vector multiplication operations that are crucial for
//! efficient machine learning model execution.
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
//! them with specialized sum-checks rather than processing them through the general-purpose
//! VM execution proof system.
use std::collections::HashMap;

use crate::jolt::{
    bytecode::BytecodePreprocessing,
    dag::state_manager::StateManager,
    precompiles::{
        matvecmult::{MatVecPrecompile, MatVecPreprocessing},
        reduce_sum::{ReduceSumPrecompile, ReduceSumPreprocessing},
        val_final::ValFinalSumcheck,
    },
    sumcheck::{BatchedSumcheck, SingleSumcheck, SumcheckInstance},
};
use jolt_core::{
    field::JoltField, poly::commitment::commitment_scheme::CommitmentScheme,
    subprotocols::sumcheck::SumcheckInstanceProof, transcripts::Transcript,
    utils::errors::ProofVerifyError,
};
use onnx_tracer::{
    self,
    trace_types::{ONNXInstr, ONNXOpcode},
};
use paste::paste;
use serde::{Deserialize, Serialize};

pub mod matvecmult;
pub mod reduce_sum;
pub mod val_final;

/// Macro to define all precompile types and generate associated code
macro_rules! define_precompiles {
    ($(($variant:ident, $opcode:pat)),* $(,)?) => {
        paste! {
            /// Enum containing all possible precompile instance types
            #[derive(Debug, Clone)]
            pub enum PrecompileInstance {
                $(
                    $variant([<$variant Precompile>]),
                )*
            }

            impl PrecompileInstance {
                /// Creates a new prover instance based on the variant
                pub fn new_prover(variant: PrecompileVariant, index: usize) -> Self {
                    match variant {
                        $(
                            PrecompileVariant::$variant => {
                                Self::$variant([<$variant Precompile>]::new_prover(index))
                            },
                        )*
                    }
                }

                /// Creates a new verifier instance based on the variant
                pub fn new_verifier(variant: PrecompileVariant, index: usize) -> Self {
                    match variant {
                        $(
                            PrecompileVariant::$variant => {
                                Self::$variant([<$variant Precompile>]::new_verifier(index))
                            },
                        )*
                    }
                }
            }

            impl PrecompileTrait for PrecompileInstance {
                fn execution_prover_instance<
                    F: JoltField,
                    ProofTranscript: Transcript,
                    PCS: CommitmentScheme<Field = F>,
                >(
                    &mut self,
                    sm: &StateManager<'_, F, ProofTranscript, PCS>,
                ) -> Box<dyn SumcheckInstance<F>> {
                    match self {
                        $(
                            Self::$variant(instance) => instance.execution_prover_instance(sm),
                        )*
                    }
                }

                fn read_checking_prover_instance<
                    F: JoltField,
                    ProofTranscript: Transcript,
                    PCS: CommitmentScheme<Field = F>,
                >(
                    &self,
                    sm: &StateManager<'_, F, ProofTranscript, PCS>,
                ) -> Box<dyn SumcheckInstance<F>> {
                    match self {
                        $(
                            Self::$variant(instance) => instance.read_checking_prover_instance(sm),
                        )*
                    }
                }

                fn execution_verifier_instance<
                    F: JoltField,
                    ProofTranscript: Transcript,
                    PCS: CommitmentScheme<Field = F>,
                >(
                    &mut self,
                    sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
                ) -> Box<dyn SumcheckInstance<F>> {
                    match self {
                        $(
                            Self::$variant(instance) => instance.execution_verifier_instance(sm),
                        )*
                    }
                }

                fn read_checking_verifier_instance<
                    F: JoltField,
                    ProofTranscript: Transcript,
                    PCS: CommitmentScheme<Field = F>,
                >(
                    &mut self,
                    sm: &StateManager<'_, F, ProofTranscript, PCS>,
                ) -> Box<dyn SumcheckInstance<F>> {
                    match self {
                        $(
                            Self::$variant(instance) => instance.read_checking_verifier_instance(sm),
                        )*
                    }
                }
            }

            /// Enum defining the types of precompile variants available
            #[derive(Debug, Clone, Copy)]
            pub enum PrecompileVariant {
                $(
                    $variant,
                )*
            }

            impl PrecompilePreprocessing {
                /// Helper method to get the count for a specific precompile type
                pub fn get_count(&self, variant: PrecompileVariant) -> usize {
                    match variant {
                        $(
                            PrecompileVariant::$variant => self.[<$variant:snake>].len(),
                        )*
                    }
                }

                /// Generate all precompile instances from preprocessing
                pub fn generate_instances(&self) -> Vec<(PrecompileVariant, usize)> {
                    let mut instances = Vec::new();
                    $(
                        for i in 0..self.[<$variant:snake>].len() {
                            instances.push((PrecompileVariant::$variant, i));
                        }
                    )*
                    instances
                }

                /// Process a single instruction and add it to the appropriate preprocessing collection
                /// This method is generated by the macro to handle all supported opcodes
                fn process_instruction(
                    &mut self,
                    instr: &ONNXInstr,
                    td_lookup: &HashMap<usize, ONNXInstr>,
                    bytecode_preprocessing: &BytecodePreprocessing,
                ) {
                    match instr.opcode {
                        $(
                            $opcode => {
                                self.[<$variant:snake>].push([<$variant Preprocessing>]::new(
                                    instr,
                                    td_lookup,
                                    bytecode_preprocessing,
                                ));
                            },
                        )*
                        _ => {}
                    }
                }
            }
        }
    };
}

define_precompiles!((MatVec, ONNXOpcode::MatMult), (ReduceSum, ONNXOpcode::Sum),);

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
/// A structure that holds preprocessing data for precompiled operations.
///
/// This struct contains collections of preprocessing instances for different
/// types of precompiled operations that require preprocessing before execution.
///
/// # Fields
///
/// * `mat_vec` - A collection of matrix-vector operation preprocessing instances.
pub struct PrecompilePreprocessing {
    pub mat_vec: Vec<MatVecPreprocessing>,
    pub reduce_sum: Vec<ReduceSumPreprocessing>,
}

impl PrecompilePreprocessing {
    /// Preprocesses a model to extract matrix-vector multiplication operations for efficient
    /// proving/verification in  proofs.
    ///
    /// This function extracts all matrix multiplication instructions from the provided model,
    /// identifies their memory addresses for both operands and results, and organizes them
    /// into a collection of `ReduceSumPreprocessing` instances. This preprocessing step is crucial
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
    pub fn preprocess(bytecode_preprocessing: &BytecodePreprocessing) -> Self {
        let mut instance = Self::empty();
        for instr in bytecode_preprocessing.raw_bytecode().iter() {
            instance.process_instruction(
                instr,
                bytecode_preprocessing.td_lookup(),
                bytecode_preprocessing,
            );
        }
        instance
    }

    /// Create an empty instance of PrecompilePreprocessing
    pub fn empty() -> Self {
        Self::default()
    }

    /// Check if there are no precompile instances
    pub fn is_empty(&self) -> bool {
        self.mat_vec.is_empty() && self.reduce_sum.is_empty()
    }
}

pub trait PrecompileTrait {
    fn execution_prover_instance<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        &mut self,
        sm: &StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Box<dyn SumcheckInstance<F>>;

    fn read_checking_prover_instance<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        &self,
        sm: &StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Box<dyn SumcheckInstance<F>>;

    fn execution_verifier_instance<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        &mut self,
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Box<dyn SumcheckInstance<F>>;

    fn read_checking_verifier_instance<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        &mut self,
        sm: &StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Box<dyn SumcheckInstance<F>>;
}

#[derive(Debug, Clone)]
/// A directed acyclic graph structure that manages precompiled operations for the ZKML system.
///
/// The `PrecompileDag` serves as the central coordinator for all precompiled operations
/// in the Jolt  proof system
///
/// This structure maintains a collection of precompile instances and provides methods to
/// construct both prover and verifier variants, as well as methods to access execution
/// and read-checking instances for sumcheck protocols.
///
/// # Fields
///
/// * `instances` - A vector of all precompile instances managed by this DAG
pub struct PrecompileDag {
    /// Collection of all precompile instances used in the Precompile DAG
    instances: Vec<PrecompileInstance>,
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
        let preprocessing = sm.get_precompile_preprocessing();
        let instance_specs = preprocessing.generate_instances();

        let instances = instance_specs
            .into_iter()
            .map(|(variant, index)| PrecompileInstance::new_prover(variant, index))
            .collect();

        Self { instances }
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
        let preprocessing = sm.get_precompile_preprocessing();
        let instance_specs = preprocessing.generate_instances();
        let instances = instance_specs
            .into_iter()
            .map(|(variant, index)| PrecompileInstance::new_verifier(variant, index))
            .collect();
        Self { instances }
    }
}

impl PrecompileDag {
    /// Gathers all prover instances for execution sum-checks.
    ///
    /// This method collects sumcheck instances from all precompiles
    /// for the execution phase of the proof generation. The execution phase proves that the
    /// precompile operations were performed correctly.
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
        self.instances
            .iter_mut()
            .map(|instance| instance.execution_prover_instance(sm))
            .collect()
    }

    /// Gathers all prover instances for read-checking sum-checks.
    ///
    /// This method collects sumcheck instances from all matrix-vector multiplication precompiles
    /// for the read-checking phase of the proof generation. The read-checking phase proves that
    /// the values used in the precompiles were correctly read from memory.
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
        self.instances
            .iter()
            .map(|instance| instance.read_checking_prover_instance(sm))
            .collect()
    }

    /// Gets all verifier instance for val_final sum-checks.
    ///
    /// This method collects sumcheck instances from all precompiles
    /// for the val_final phase of proof verification. These instances are used to verify the
    /// val_final part of the proof, which verifies the val_final claim used in the
    /// read-checking phase.
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
    /// This method collects the sumcheck instances from all precompiles
    /// for the execution phase of proof verification. These instances are used to verify the
    /// execution part of the proof, which shows the precomile operations were performed correctly.
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
        self.instances
            .iter_mut()
            .map(|instance| instance.execution_verifier_instance(sm))
            .collect()
    }

    /// Gets all verifier instances for read-checking sum-checks.
    ///
    /// This method collects sumcheck instances from all the precompiles
    /// for the read-checking phase of proof verification. These instances are used to verify the
    /// read-checking part of the proof, which shows that the values used in the precompiles
    /// were correctly read from memory.
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
        self.instances
            .iter_mut()
            .map(|instance| instance.read_checking_verifier_instance(sm))
            .collect()
    }

    /// Gets all verifier instance for val_final sum-checks.
    ///
    /// This method collects sumcheck instances from all matrix-vector multiplication precompiles
    /// for the val_final phase of proof verification. These instances are used to verify the
    /// val_final part of the proof, which verifies the val_final claim from the read-checking phase of the precompile
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
    #[tracing::instrument(name = "PrecompileSNARK::prove", skip(sm))]
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
    fn test_model_helper<ModelFunc>(model: ModelFunc, input: Tensor<i32>)
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
        test_model_helper(builder::minimal_matmult_non_power_of_two_model, input);
    }

    #[test]
    fn test_dual_matmult_non_power_of_two_model() {
        let input = Tensor::new(Some(&[1, 2, 3, 4, 5]), &[1, 5]).unwrap();
        test_model_helper(builder::dual_matmult_non_power_of_two_model, input);
    }

    #[test]
    fn test_neg_dual_matmult_model() {
        let input = Tensor::new(Some(&[-1, -2, -3, -4]), &[1, 4]).unwrap();
        test_model_helper(builder::dual_matmult_model, input);
    }

    #[test]
    fn test_dual_matmult_model() {
        let input = Tensor::new(Some(&[1, 2, 3, 4]), &[1, 4]).unwrap();
        test_model_helper(builder::dual_matmult_model, input);
    }

    #[test]
    fn test_triple_matmult_model() {
        let input = Tensor::new(Some(&[1, 2, 3, 4, 1, 2, 3, 4]), &[1, 8]).unwrap();
        test_model_helper(builder::triple_matmult_model, input);
    }

    #[test]
    fn test_layernorm_prefix_model() {
        let input = Tensor::new(
            Some(&[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]),
            &[4, 4],
        )
        .unwrap();
        test_model_helper(builder::reduce_mean_model, input);
    }
}
