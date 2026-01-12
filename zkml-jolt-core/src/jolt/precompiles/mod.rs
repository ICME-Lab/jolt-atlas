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
use onnx_tracer::trace_types::{AtlasInstr, AtlasOpcode};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    io::{Read, Write},
};

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};

pub mod broadcast;
pub mod einsum;
pub mod gather;
pub mod read_checking;
pub mod reduce_sum;
pub mod val_final;
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PrecompilePreprocessing {
    pub instances: Vec<PreprocessingInstance>,
}

impl PrecompilePreprocessing {
    #[tracing::instrument(name = "PrecompilePreprocessing::preprocess", skip_all)]
    /// Create a new instance of PrecompilePreprocessing by scanning the bytecode for precompile operations.
    pub fn preprocess(bytecode_preprocessing: &BytecodePreprocessing) -> Self {
        let td_lookup = bytecode_preprocessing.td_lookup();
        let instances = bytecode_preprocessing
        .raw_bytecode()
        .iter()
        // TODO(AntoineF4C5): Set a method for each instruction, that only returns a preprocessing instance if required
        .filter_map(|instr| match &instr.opcode {
                    AtlasOpcode::Einsum(_) | AtlasOpcode::Sum(_) | AtlasOpcode::Gather | AtlasOpcode::Broadcast => Some(
                        PreprocessingInstance::new(instr, td_lookup, bytecode_preprocessing),
                    ),
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
        instr: &AtlasInstr,
        td_lookup: &HashMap<usize, AtlasInstr>,
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
        instr: &AtlasInstr,
        td_lookup: &HashMap<usize, AtlasInstr>,
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

    /// Specialized instance builder for Gather operations
    fn build_gather_instance(
        instr: &AtlasInstr,
        td_lookup: &HashMap<usize, AtlasInstr>,
        bytecode_preprocessing: &BytecodePreprocessing,
    ) -> PreprocessingInstance {
        // We set the first input in b, and the second in a.
        // Since in the precompile, it is more straightforward to multiply with the indices-based input on the left. (Avoids transposition)
        let a_instr = PreprocessingHelper::get_operand_instruction(td_lookup, instr.ts2, "Gather");
        let b_instr = PreprocessingHelper::get_operand_instruction(td_lookup, instr.ts1, "Gather");

        let a_dims = get_dimensions(a_instr, 1);
        let b_dims = get_dimensions(b_instr, 2);
        let c_dims = get_dimensions(instr, 2);

        // Dimension compability check
        assert_eq!(a_dims[0], c_dims[0]);
        assert_eq!(b_dims[1], c_dims[1]);

        let a_addr = PreprocessingHelper::collect_and_pad(a_instr, bytecode_preprocessing, &a_dims);
        let b_addr = PreprocessingHelper::collect_and_pad(b_instr, bytecode_preprocessing, &b_dims);
        let c_addr = PreprocessingHelper::collect_and_pad(instr, bytecode_preprocessing, &c_dims);

        PreprocessingInstance {
            a_addr,
            b_addr,
            c_addr,
            a_dims, // We send the unpadded dims for a, since we need it to construct the one-hot encoding
            b_dims: PreprocessingHelper::calculate_padded_dims(&b_dims),
            c_dims: PreprocessingHelper::calculate_padded_dims(&c_dims),
            equation: "Gather".to_string(),
        }
    }

    /// Specialized instance builder for Broadcast operations
    ///
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
    fn build_broadcast_instance(
        instr: &AtlasInstr,
        td_lookup: &HashMap<usize, AtlasInstr>,
        bytecode_preprocessing: &BytecodePreprocessing,
    ) -> PreprocessingInstance {
        let a_instr =
            PreprocessingHelper::get_operand_instruction(td_lookup, instr.ts1, "Broadcast");
        let in_shape = a_instr.output_dims.clone();
        let target_shape = instr.output_dims.clone();

        // Get broadcast dimensions
        let broadcast_dims = get_broadcast_dims(&in_shape, &target_shape)
            .expect("Input shape not broadcastable to target shape");

        let a_addr =
            PreprocessingHelper::collect_and_pad(a_instr, bytecode_preprocessing, &in_shape);
        let c_addr =
            PreprocessingHelper::collect_and_pad(instr, bytecode_preprocessing, &target_shape);

        PreprocessingInstance {
            a_addr,
            b_addr: vec![0, 0], // Broadcast has only one operand, this dummy value is used in read checking
            c_addr,
            a_dims: PreprocessingHelper::calculate_padded_dims(&in_shape),
            b_dims: broadcast_dims, // Not padded, we need raw values to build broadcasting multilinear poly `I`
            c_dims: PreprocessingHelper::calculate_padded_dims(&target_shape),
            equation: "Broadcast".to_string(),
        }
    }
}

/// Gets the required dimensions from an instructions, and panics if the instructions output has less or more dimensions.
/// Why:
/// Sometimes in the tracer, 1-dimension tensors are cast to 2-dimension, with a leading "1" dimension.
/// Ex: [n] tensor get cast to [1, n]
fn get_dimensions(instruction: &AtlasInstr, num_dimensions: usize) -> Vec<usize> {
    let raw_dims = instruction.output_dims.clone();
    assert!(raw_dims.len() >= num_dimensions);
    let mut outdims = Vec::new();

    let mut raw_dims = raw_dims.into_iter().rev();

    // Push dimension to output
    for _ in 0..num_dimensions {
        outdims.insert(0, raw_dims.next().unwrap());
    }

    // Assert all subsequent dims are equal to 1
    for dims in raw_dims {
        assert_eq!(dims, 1, "Dimension mismatch");
    }

    outdims
}

/// Computes the broadcast dimensions
/// # Returns
/// An array of dimensions, where each dimensions is either 1 if no broadcast is needed in that dimension, or the target dimension otherwise.
///
/// Returns None if the input shape is not broadcastable to the target shape.
fn get_broadcast_dims(input_dims: &[usize], target_dims: &[usize]) -> Option<Vec<usize>> {
    assert!(input_dims.len() <= target_dims.len());

    let mut broadcast_dims = target_dims.to_vec();
    for ((i, &target_dim), &input_dim) in target_dims
        .iter()
        .enumerate()
        .rev()
        .zip(input_dims.iter().rev())
    {
        if input_dim == target_dim {
            broadcast_dims[i] = 1;
        } else if input_dim != 1 {
            return None;
        }
    }
    Some(broadcast_dims)
}

impl PreprocessingInstance {
    /// Create a new PreprocessingInstance based on the instruction opcode
    pub fn new(
        instr: &AtlasInstr,
        td_lookup: &HashMap<usize, AtlasInstr>,
        bytecode_preprocessing: &BytecodePreprocessing,
    ) -> Self {
        match &instr.opcode {
            AtlasOpcode::Einsum(equation) => {
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
            AtlasOpcode::Sum(axes) => match (axes, instr.output_dims[0], instr.output_dims.len()) {
                (1, _, 2) | (2, 1, 3) => InstanceBuilder::build_sum_instance_axes_1(
                    instr,
                    td_lookup,
                    bytecode_preprocessing,
                    *axes as i32,
                ),

                _ => panic!("Sum operation not supported by precompile system"),
            },
            AtlasOpcode::Gather => {
                InstanceBuilder::build_gather_instance(instr, td_lookup, bytecode_preprocessing)
            }
            AtlasOpcode::Broadcast => {
                InstanceBuilder::build_broadcast_instance(instr, td_lookup, bytecode_preprocessing)
            }
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
                "Gather" => {
                    (if is_prover {
                        Box::new(gather::ExecutionSumcheck::new_prover(sm, index))
                    } else {
                        Box::new(gather::ExecutionSumcheck::new_verifier(sm, index))
                    }) as Box<dyn SumcheckInstance<F>>
                }
                "Broadcast" => {
                    (if is_prover {
                        Box::new(broadcast::ExecutionSumcheck::new_prover(sm, index))
                    } else {
                        Box::new(broadcast::ExecutionSumcheck::new_verifier(sm, index))
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

    /// Gets the prover instance for the precompile Hamming Booleanity sum-check.
    ///
    /// This method creates the sumcheck instance for the `optional` Hamming Weight phase of [PrecompileSNARK].
    /// Proves the correct construction of claimed `hw`, which is used to prove Hamming Booleanity in `Gather` instances
    /// This allows to assert correctness of padding for one-hot encoded vectors
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
    /// A vec of boxed sumcheck instance that implement the `SumcheckInstance<F>` trait
    pub fn hamming_weight_prover_instances<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        let equations: Vec<String> = sm
            .get_precompile_preprocessing()
            .instances
            .iter()
            .map(|pp| pp.equation.clone())
            .collect();

        equations
            .iter()
            .enumerate()
            .filter_map(|(index, equation)| match equation.as_str() {
                "Gather" => Some(
                    Box::new(gather::HammingWeightSumcheck::new_prover(sm, index))
                        as Box<dyn SumcheckInstance<F>>,
                ),
                _ => None,
            })
            .collect()
    }

    pub fn hamming_weight_verifier_instances<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        let equations: Vec<String> = sm
            .get_precompile_preprocessing()
            .instances
            .iter()
            .map(|pp| pp.equation.clone())
            .collect();

        equations
            .iter()
            .enumerate()
            .filter_map(|(index, equation)| match equation.as_str() {
                "Gather" => Some(
                    Box::new(gather::HammingWeightSumcheck::new_verifier(sm, index))
                        as Box<dyn SumcheckInstance<F>>,
                ),
                _ => None,
            })
            .collect()
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
    pub(crate) execution_proof: SumcheckInstanceProof<F, FS>,
    pub(crate) hamming_weight_proof: Option<SumcheckInstanceProof<F, FS>>,
    pub(crate) read_checking_proof: SumcheckInstanceProof<F, FS>,
    pub(crate) val_final_proof: SumcheckInstanceProof<F, FS>,
}

impl<F: JoltField, FS: Transcript> CanonicalSerialize for PrecompileSNARK<F, FS> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.execution_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.hamming_weight_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.read_checking_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.val_final_proof
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.execution_proof.serialized_size(compress)
            + self.hamming_weight_proof.serialized_size(compress)
            + self.read_checking_proof.serialized_size(compress)
            + self.val_final_proof.serialized_size(compress)
    }
}

impl<F: JoltField, FS: Transcript> Valid for PrecompileSNARK<F, FS> {
    fn check(&self) -> Result<(), SerializationError> {
        self.execution_proof.check()?;
        self.hamming_weight_proof.check()?;
        self.read_checking_proof.check()?;
        self.val_final_proof.check()?;
        Ok(())
    }
}

impl<F: JoltField, FS: Transcript> CanonicalDeserialize for PrecompileSNARK<F, FS> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let execution_proof =
            SumcheckInstanceProof::deserialize_with_mode(&mut reader, compress, validate)?;
        let hamming_weight_proof = Option::<SumcheckInstanceProof<F, FS>>::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
        )?;
        let read_checking_proof =
            SumcheckInstanceProof::deserialize_with_mode(&mut reader, compress, validate)?;
        let val_final_proof =
            SumcheckInstanceProof::deserialize_with_mode(&mut reader, compress, validate)?;
        Ok(Self {
            execution_proof,
            hamming_weight_proof,
            read_checking_proof,
            val_final_proof,
        })
    }
}

impl<F: JoltField, FS: Transcript> PrecompileSNARK<F, FS> {
    /// Create a new instance of PrecompileSNARK by proving all precompile operations.
    /// This includes execution sum-checks, read-checking sum-checks, and the val_final sum-check
    #[tracing::instrument(name = "PrecompileSNARK::prove", skip(sm))]
    pub fn prove<'a, PCS: CommitmentScheme<Field = F>>(
        sm: &mut StateManager<'a, F, FS, PCS>,
    ) -> Self {
        let has_gather = sm
            .get_precompile_preprocessing()
            .instances
            .iter()
            .any(|instance| matches!(instance.equation.as_str(), "Gather"));

        let execution_proof = Self::prove_execution(sm);
        let hamming_weight_proof = if has_gather {
            Some(Self::prove_hamming_weight(sm))
        } else {
            None
        };
        let read_checking_proof = Self::prove_read_checking(sm);
        let val_final_proof = Self::prove_val_final(sm);
        PrecompileSNARK {
            execution_proof,
            hamming_weight_proof,
            read_checking_proof,
            val_final_proof,
        }
    }

    #[tracing::instrument(name = "PrecompileSNARK::prove_execution", skip(sm))]
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

    #[tracing::instrument(name = "PrecompileSNARK::prove_hamming_weight", skip(sm))]
    /// Proves the execution phase of the precompiles.
    /// This includes running execution sum-checks and collecting output claims.
    fn prove_hamming_weight<'a, PCS: CommitmentScheme<Field = F>>(
        sm: &mut StateManager<'a, F, FS, PCS>,
    ) -> SumcheckInstanceProof<F, FS> {
        Self::prove_batched_sumchecks(
            &mut PrecompileDag::hamming_weight_prover_instances(sm).into_boxed_slice(),
            sm,
        )
    }

    #[tracing::instrument(name = "PrecompileSNARK::prove_read_checking", skip(sm))]
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

    #[tracing::instrument(name = "PrecompileSNARK::prove_val_final", skip(sm))]
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
        let has_gather = sm
            .get_precompile_preprocessing()
            .instances
            .iter()
            .any(|instance| matches!(instance.equation.as_str(), "Gather"));

        Self::verify_batched_sumchecks(
            &self.execution_proof,
            PrecompileDag::execution_verifier_instances(sm),
            sm,
        )?;

        if has_gather {
            Self::verify_batched_sumchecks(
                self.hamming_weight_proof
                    .as_ref()
                    .expect("Should hold a proof"),
                PrecompileDag::hamming_weight_verifier_instances(sm),
                sm,
            )?;
        }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_dims() {
        let in_shape = vec![1, 4]; //   [1, 4]
        let target = vec![3, 4]; //     [3, 4] Compatible shapes
        assert_eq!(get_broadcast_dims(&in_shape, &target), Some(vec![3, 1]));

        let in_shape = vec![3, 4]; //   [   3, 4]
        let target = vec![2, 3, 4]; //  [2, 3, 4] Compatible shapes
        assert_eq!(get_broadcast_dims(&in_shape, &target), Some(vec![2, 1, 1]));

        let in_shape = vec![2, 1, 4]; //    [2, 1, 4]
        let target = vec![2, 3, 4]; //      [2, 3, 4] Compatible shapes
        assert_eq!(get_broadcast_dims(&in_shape, &target), Some(vec![1, 3, 1]));

        let in_shape = vec![2, 3]; //   [2, 3]
        let target = vec![2, 4]; //     [2, 4] Not compatible shapes
        assert!(get_broadcast_dims(&in_shape, &target).is_none());

        let in_shape = vec![3, 4]; //   [   3, 4]
        let target = vec![3, 4, 2]; //  [3, 4, 2] Not compatible shapes
        assert!(get_broadcast_dims(&in_shape, &target).is_none());
    }
}
