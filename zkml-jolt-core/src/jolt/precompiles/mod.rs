use crate::{
    jolt::{
        bytecode::BytecodePreprocessing,
        dag::state_manager::StateManager,
        precompiles::{read_checking::ReadCheckingABCSumcheck, val_final::ValFinalSumcheck},
        sumcheck::{BatchedSumcheck, SingleSumcheck, SumcheckInstance},
    },
    utils::precompile_pp::{DimExtractor, EINSUM_REGISTRY, PreprocessingHelper},
};
use jolt_core::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme, eq_poly::EqPolynomial,
        multilinear_polynomial::MultilinearPolynomial,
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, thread::unsafe_allocate_zero_vec},
};
use onnx_tracer::trace_types::{ONNXInstr, ONNXOpcode};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod einsum;
pub mod read_checking;
pub mod reduce_sum;
pub mod val_final;
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PrecompilePreprocessing {
    pub instances: Vec<PreprocessingInstance>,
}

impl PrecompilePreprocessing {
    /// Create a new instance of PrecompilePreprocessing by scanning the bytecode for precompile operations.
    pub fn preprocess(bytecode_preprocessing: &BytecodePreprocessing) -> Self {
        let td_lookup = bytecode_preprocessing.td_lookup();
        let instances = bytecode_preprocessing
            .raw_bytecode()
            .iter()
            .filter_map(|instr| match &instr.opcode {
                ONNXOpcode::Einsum(_) | ONNXOpcode::Sum(_) => Some(PreprocessingInstance::new(
                    instr,
                    td_lookup,
                    bytecode_preprocessing,
                )),
                _ => None,
            })
            .collect();

        PrecompilePreprocessing { instances }
    }

    /// Create an empty PrecompilePreprocessing instance
    pub fn empty() -> Self {
        Self::default()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingInstance {
    /// Memory addresses (in Val_final) for [super::witness::VirtualPolynomial::EinsumA]
    pub a_addr: Vec<usize>,
    /// Memory addresses (in Val_final) for [super::witness::VirtualPolynomial::EinsumB]
    pub b_addr: Vec<usize>,
    /// Memory addresses (in Val_final) for [super::witness::VirtualPolynomial::EinsumC]
    pub c_addr: Vec<usize>,
    /// dims of operand a
    pub a_dims: Vec<usize>,
    /// dims of operand b
    pub b_dims: Vec<usize>,
    /// dims of result c
    pub c_dims: Vec<usize>,
    /// einsum equation string
    pub equation: String,
}

/// Helper to create preprocessing instance with deduplication
struct InstanceBuilder;

impl InstanceBuilder {
    /// Generic instance builder for einsum operations
    fn build_einsum_instance(
        instr: &ONNXInstr,
        td_lookup: &HashMap<usize, ONNXInstr>,
        bytecode_preprocessing: &BytecodePreprocessing,
        equation: &str,
        dims_extractor: DimExtractor,
    ) -> PreprocessingInstance {
        let (a_dims, b_dims, c_dims) = dims_extractor(instr, td_lookup);

        let a_instr = PreprocessingHelper::get_operand_instruction(td_lookup, instr.ts1, equation);
        let b_instr = PreprocessingHelper::get_operand_instruction(td_lookup, instr.ts2, equation);

        let a_addr = PreprocessingHelper::collect_and_pad(a_instr, bytecode_preprocessing, &a_dims);
        let b_addr = PreprocessingHelper::collect_and_pad(b_instr, bytecode_preprocessing, &b_dims);
        let c_addr = PreprocessingHelper::collect_and_pad(instr, bytecode_preprocessing, &c_dims);

        PreprocessingInstance {
            a_addr,
            b_addr,
            c_addr,
            a_dims: PreprocessingHelper::calculate_padded_dims(&a_dims),
            b_dims: PreprocessingHelper::calculate_padded_dims(&b_dims),
            c_dims: PreprocessingHelper::calculate_padded_dims(&c_dims),
            equation: equation.to_string(),
        }
    }

    /// Specialized builder for sum operations
    fn build_sum_instance_axes_1(
        instr: &ONNXInstr,
        td_lookup: &HashMap<usize, ONNXInstr>,
        bytecode_preprocessing: &BytecodePreprocessing,
        _axes: i32,
    ) -> PreprocessingInstance {
        let a_instr = PreprocessingHelper::get_operand_instruction(td_lookup, instr.ts1, "Sum");

        let mut m = a_instr.output_dims[0];
        let mut n = a_instr.output_dims[1];

        if a_instr.output_dims.len() == 3 {
            m = a_instr.output_dims[1];
            n = a_instr.output_dims[2];
        }

        let a_addr = PreprocessingHelper::collect_and_pad(a_instr, bytecode_preprocessing, &[m, n]);
        let b_addr = vec![0, 0]; // Sum has only one operand
        let c_addr = PreprocessingHelper::collect_and_pad(instr, bytecode_preprocessing, &[m]);

        PreprocessingInstance {
            a_addr,
            b_addr,
            c_addr,
            a_dims: PreprocessingHelper::calculate_padded_dims(&[m, n]),
            b_dims: vec![2], // dummy
            c_dims: PreprocessingHelper::calculate_padded_dims(&[m]),
            equation: "sum(1)".to_string(),
        }
    }
}

impl PreprocessingInstance {
    /// Create a new PreprocessingInstance based on the instruction opcode
    pub fn new(
        instr: &ONNXInstr,
        td_lookup: &HashMap<usize, ONNXInstr>,
        bytecode_preprocessing: &BytecodePreprocessing,
    ) -> Self {
        match &instr.opcode {
            ONNXOpcode::Einsum(equation) => {
                // Look up configuration for this equation pattern
                let config = EINSUM_REGISTRY
                    .iter()
                    .find(|(pattern, _)| pattern == &equation.as_str())
                    .map(|(_, config)| config)
                    .unwrap_or_else(|| {
                        panic!("Einsum equation ({equation}) not supported by precompile system")
                    });

                // Special validation for mk,nk->mn case
                if equation == "mk,nk->mn" {
                    assert!(instr.output_dims[0] == 1);
                }

                InstanceBuilder::build_einsum_instance(
                    instr,
                    td_lookup,
                    bytecode_preprocessing,
                    config.equation,
                    config.dims_extractor,
                )
            }
            ONNXOpcode::Sum(axes) => match (axes, instr.output_dims[0], instr.output_dims.len()) {
                (1, _, 2) | (2, 1, 3) => InstanceBuilder::build_sum_instance_axes_1(
                    instr,
                    td_lookup,
                    bytecode_preprocessing,
                    *axes as i32,
                ),

                _ => panic!("Sum operation not supported by precompile system"),
            },
            _ => panic!("Operation not supported by precompile system"),
        }
    }
}

impl PreprocessingInstance {
    /// Extracts read-values from val_final using the stored read-addresses.
    /// Given the val_final lookup table, computes the actual values of operands and results
    /// by indexing into val_final with the preprocessed memory addresses.
    fn extract_rv<T>(&self, val_final: &[i64], field_selector: impl Fn(&Self) -> &T) -> Vec<i64>
    where
        T: AsRef<[usize]>,
    {
        field_selector(self)
            .as_ref()
            .par_iter()
            .map(|&k| val_final[k])
            .collect::<Vec<_>>()
    }

    /// From the addresses compute the binded one-hot poly needed for the read-checking instance
    fn compute_ra<F, T>(
        &self,
        r: &[F],
        field_selector: impl Fn(&Self) -> &T,
        K: usize,
    ) -> MultilinearPolynomial<F>
    where
        T: AsRef<[usize]>,
        F: JoltField,
    {
        let E = EqPolynomial::evals(r);
        let addresses = field_selector(self).as_ref();
        let num_threads = rayon::current_num_threads();
        let chunk_size = addresses.len().div_ceil(num_threads);
        let partial_results: Vec<Vec<F>> = addresses
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let mut local_ra = unsafe_allocate_zero_vec::<F>(K);
                let base_idx = chunk_idx * chunk_size;
                chunk.iter().enumerate().for_each(|(local_j, &k)| {
                    let global_j = base_idx + local_j;
                    local_ra[k] += E[global_j];
                });
                local_ra
            })
            .collect();
        let mut ra = unsafe_allocate_zero_vec::<F>(K);
        for partial in partial_results {
            ra.par_iter_mut()
                .zip(partial.par_iter())
                .for_each(|(dest, &src)| *dest += src);
        }
        MultilinearPolynomial::from(ra)
    }
}

pub struct PrecompileDag {}

impl PrecompileDag {
    /// Creates sumcheck instances for prover/verifier
    fn create_instances<F, ProofTranscript, PCS>(
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
        is_prover: bool,
    ) -> Vec<Box<dyn SumcheckInstance<F>>>
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    {
        let equations: Vec<String> = sm
            .get_precompile_preprocessing()
            .instances
            .iter()
            .map(|pp| pp.equation.clone())
            .collect();

        equations
            .iter()
            .enumerate()
            .map(|(index, equation)| match equation.as_str() {
                "mk,kn->mn" => {
                    (if is_prover {
                        Box::new(einsum::mk_kn_mn::ExecutionSumcheck::new_prover(index, sm))
                    } else {
                        Box::new(einsum::mk_kn_mn::ExecutionSumcheck::new_verifier(index, sm))
                    }) as Box<dyn SumcheckInstance<F>>
                }
                "bmk,kbn->mbn" => {
                    (if is_prover {
                        Box::new(einsum::bmk_kbn_mbn::ExecutionSumcheck::new_prover(
                            index, sm,
                        ))
                    } else {
                        Box::new(einsum::bmk_kbn_mbn::ExecutionSumcheck::new_verifier(
                            index, sm,
                        ))
                    }) as Box<dyn SumcheckInstance<F>>
                }
                "k,nk->n" => {
                    (if is_prover {
                        Box::new(einsum::k_nk_n::ExecutionSumcheck::new_prover(index, sm))
                    } else {
                        Box::new(einsum::k_nk_n::ExecutionSumcheck::new_verifier(index, sm))
                    }) as Box<dyn SumcheckInstance<F>>
                }
                "mbk,nbk->bmn" => {
                    (if is_prover {
                        Box::new(einsum::mbk_nbk_bmn::ExecutionSumcheck::new_prover(
                            index, sm,
                        ))
                    } else {
                        Box::new(einsum::mbk_nbk_bmn::ExecutionSumcheck::new_verifier(
                            index, sm,
                        ))
                    }) as Box<dyn SumcheckInstance<F>>
                }
                "sum(1)" => {
                    (if is_prover {
                        Box::new(reduce_sum::axes_1::ExecutionSumcheck::new_prover(index, sm))
                    } else {
                        Box::new(reduce_sum::axes_1::ExecutionSumcheck::new_verifier(
                            index, sm,
                        ))
                    }) as Box<dyn SumcheckInstance<F>>
                }
                _ => panic!("equation {equation} not supported"),
            })
            .collect()
    }

    pub fn execution_prover_instances<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        Self::create_instances(sm, true)
    }

    pub fn execution_verifier_instances<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        Self::create_instances(sm, false)
    }

    /// Gets the prover instance for the precompile read-checking sum-check.
    ///
    /// This method creates the sumcheck instance for the read-checking phase of [PrecompileSNARK].
    /// Proves polys from execution sum-checks were read correctly from memory.
    ///
    /// # Type Parameters
    ///
    /// * `F` - The field type implementing the `JoltField` trait
    /// * `ProofTranscript` - The transcript type implementing the `Transcript` trait
    /// * `PCS` - The polynomial commitment scheme type implementing `CommitmentScheme` with field type `F`
    ///
    /// # Parameters
    ///
    /// * `sm` - A reference to the state manager containing verification state and data
    ///
    /// # Returns
    ///
    /// A boxed sumcheck instance that implement the `SumcheckInstance<F>` trait
    pub fn read_checking_prover_instance<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        sm: &StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Box<dyn SumcheckInstance<F>> {
        Box::new(ReadCheckingABCSumcheck::new_prover(sm))
    }

    /// Gets the verifier instance for the precompile read-checking sum-check.
    ///
    /// This method creates the verifiers sumcheck instance for the read-checking phase of [PrecompileSNARK].
    /// Verifies polys from execution sum-checks were read correctly from memory.
    ///
    /// # Type Parameters
    ///
    /// * `F` - The field type implementing the `JoltField` trait
    /// * `ProofTranscript` - The transcript type implementing the `Transcript` trait
    /// * `PCS` - The polynomial commitment scheme type implementing `CommitmentScheme` with field type `F`
    ///
    /// # Parameters
    ///
    /// * `sm` - A reference to the state manager containing verification state and data
    ///
    /// # Returns
    ///
    /// A boxed sumcheck instance that implement the `SumcheckInstance<F>` trait
    pub fn read_checking_verifier_instance<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        sm: &StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Box<dyn SumcheckInstance<F>> {
        Box::new(ReadCheckingABCSumcheck::new_verifier(sm))
    }

    /// Gets the prover instance for the val_final sum-check.
    ///
    /// This method creates the prover sumcheck instance for the val_final phase of [PrecompileSNARK]. This instance is used to prove the
    /// val_final eval claim, produced from the read-checking instance
    ///
    /// # Type Parameters
    ///
    /// * `ProofTranscript` - The transcript type implementing the `Transcript` trait
    /// * `PCS` - The polynomial commitment scheme type implementing `CommitmentScheme` with field type `F`
    ///
    /// # Parameters
    ///
    /// * `sm` - A reference to the state manager containing verification state and data
    ///
    /// # Returns
    ///
    /// A boxed sumcheck instance that implement the `SumcheckInstance<F>` trait
    pub fn val_final_prover_instance<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        sm: &StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Box<dyn SumcheckInstance<F>> {
        Box::new(ValFinalSumcheck::new_prover(sm))
    }

    /// Gets the verifier instance for the val_final sum-check.
    ///
    /// This method creates the verifier sumcheck instance for the val_final phase of [PrecompileSNARK]. This instance is used to verify the
    /// val_final eval claim, produced from the read-checking instance
    ///
    /// # Type Parameters
    ///
    /// * `ProofTranscript` - The transcript type implementing the `Transcript` trait
    /// * `PCS` - The polynomial commitment scheme type implementing `CommitmentScheme` with field type `F`
    ///
    /// # Parameters
    ///
    /// * `sm` - A reference to the state manager containing verification state and data
    ///
    /// # Returns
    ///
    /// A boxed sumcheck instance that implement the `SumcheckInstance<F>` trait
    pub fn val_final_verifier_instance<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        sm: &StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Box<dyn SumcheckInstance<F>> {
        Box::new(ValFinalSumcheck::new_verifier(sm))
    }
}

/// A SNARK attesting to the correctness of sum-check precompile operations.
#[derive(Clone, Debug)]
pub struct PrecompileSNARK<F: JoltField, FS: Transcript> {
    execution_proof: SumcheckInstanceProof<F, FS>,
    read_checking_proof: SumcheckInstanceProof<F, FS>,
    val_final_proof: SumcheckInstanceProof<F, FS>,
}

impl<F: JoltField, FS: Transcript> PrecompileSNARK<F, FS> {
    /// Create a new instance of PrecompileSNARK by proving all precompile operations.
    /// This includes execution sum-checks, read-checking sum-checks, and the val_final sum-check
    #[tracing::instrument(name = "PrecompileSNARK::prove", skip(sm))]
    pub fn prove<'a, PCS: CommitmentScheme<Field = F>>(
        sm: &mut StateManager<'a, F, FS, PCS>,
    ) -> Self {
        let execution_proof = Self::prove_execution(sm);
        let read_checking_proof = Self::prove_read_checking(sm);
        let val_final_proof = Self::prove_val_final(sm);
        PrecompileSNARK {
            execution_proof,
            read_checking_proof,
            val_final_proof,
        }
    }

    /// Proves the execution phase of the precompiles.
    /// This includes running execution sum-checks and collecting output claims.
    fn prove_execution<'a, PCS: CommitmentScheme<Field = F>>(
        sm: &mut StateManager<'a, F, FS, PCS>,
    ) -> SumcheckInstanceProof<F, FS> {
        Self::prove_batched_sumchecks(
            &mut PrecompileDag::execution_prover_instances(sm).into_boxed_slice(),
            sm,
        )
    }

    /// Proves the read-checking phase for the precompiles.
    fn prove_read_checking<'a, PCS: CommitmentScheme<Field = F>>(
        sm: &StateManager<'a, F, FS, PCS>,
    ) -> SumcheckInstanceProof<F, FS> {
        let mut read_checking_instance = PrecompileDag::read_checking_prover_instance(sm);
        let transcript = sm.get_transcript();
        let accumulator = sm.get_prover_accumulator();
        let (read_checking_proof, _) = SingleSumcheck::prove(
            &mut *read_checking_instance,
            Some(accumulator.clone()),
            &mut *transcript.borrow_mut(),
        );
        read_checking_proof
    }

    /// At the end of read-checking we need to prove the val_final claim.
    fn prove_val_final<'a, PCS: CommitmentScheme<Field = F>>(
        sm: &StateManager<'a, F, FS, PCS>,
    ) -> SumcheckInstanceProof<F, FS> {
        let mut val_final_instance = PrecompileDag::val_final_prover_instance(sm);
        let transcript = sm.get_transcript();
        let accumulator = sm.get_prover_accumulator();
        let (val_final_proof, _) = SingleSumcheck::prove(
            &mut *val_final_instance,
            Some(accumulator.clone()),
            &mut *transcript.borrow_mut(),
        );
        val_final_proof
    }

    /// Helper function to batch prove execution or read-checking sum-checks
    pub(crate) fn prove_batched_sumchecks<'a, PCS: CommitmentScheme<Field = F>>(
        instances: &mut [Box<dyn SumcheckInstance<F>>],
        sm: &StateManager<'a, F, FS, PCS>,
    ) -> SumcheckInstanceProof<F, FS> {
        let instances_mut: Vec<&mut dyn SumcheckInstance<F>> = instances
            .iter_mut()
            .map(|instance| &mut **instance as &mut dyn SumcheckInstance<F>)
            .collect();
        let transcript = sm.get_transcript();
        let accumulator = sm.get_prover_accumulator();
        let (proof, _r) = BatchedSumcheck::prove(
            instances_mut,
            Some(accumulator.clone()),
            &mut *transcript.borrow_mut(),
        );
        proof
    }

    /// Verifies the PrecompileSNARK proof, ensuring all precompile operations were executed correctly.
    #[tracing::instrument(name = "PrecompileSNARK::verify", skip(self, sm))]
    pub fn verify<'a, PCS: CommitmentScheme<Field = F>>(
        &self,
        sm: &mut StateManager<'a, F, FS, PCS>,
    ) -> Result<(), ProofVerifyError> {
        // Verify execution sum-checks
        Self::verify_batched_sumchecks(
            &self.execution_proof,
            PrecompileDag::execution_verifier_instances(sm),
            sm,
        )?;

        // Verify read-checking sum-check
        Self::verify_single_sumcheck(
            &self.read_checking_proof,
            PrecompileDag::read_checking_verifier_instance(sm),
            sm,
        )?;

        // Verify val_final claim from read-checking sum-checks
        Self::verify_single_sumcheck(
            &self.val_final_proof,
            PrecompileDag::val_final_verifier_instance(sm),
            sm,
        )
    }

    /// Verify val_final sum-check
    fn verify_single_sumcheck<'a, PCS: CommitmentScheme<Field = F>>(
        proof: &SumcheckInstanceProof<F, FS>,
        instance: Box<dyn SumcheckInstance<F>>,
        sm: &StateManager<'a, F, FS, PCS>,
    ) -> Result<(), ProofVerifyError> {
        let transcript = sm.get_transcript();
        let accumulator = sm.get_verifier_accumulator();
        SingleSumcheck::verify(
            &*instance,
            proof,
            Some(accumulator.clone()),
            &mut *transcript.borrow_mut(),
        )?;
        Ok(())
    }

    /// Helper function to verify batched sum-checks
    pub(crate) fn verify_batched_sumchecks<'a, PCS: CommitmentScheme<Field = F>>(
        proof: &SumcheckInstanceProof<F, FS>,
        instances: Vec<Box<dyn SumcheckInstance<F>>>,
        sm: &StateManager<'a, F, FS, PCS>,
    ) -> Result<(), ProofVerifyError> {
        let instances_ref: Vec<&dyn SumcheckInstance<F>> = instances
            .iter()
            .map(|instance| &**instance as &dyn SumcheckInstance<F>)
            .collect();
        let transcript = sm.get_transcript();
        let accumulator = sm.get_verifier_accumulator();
        BatchedSumcheck::verify(
            proof,
            instances_ref,
            Some(accumulator.clone()),
            &mut *transcript.borrow_mut(),
        )?;
        Ok(())
    }
}
