//! Fixed-point activation functions via lookup tables.
//!
//! This module proves activation functions (erf, tanh) using precomputed
//! lookup tables. Instead of computing these transcendental functions in-circuit,
//! we verify that output values match a lookup table indexed by quantized inputs.
//!
//! We employ shout for small tables's to prove correctness of activations efficiently.

use crate::{
    jolt::{
        bytecode::BytecodePreprocessing,
        dag::state_manager::StateManager,
        pcs::{ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator},
        sumcheck::{BatchedSumcheck, SumcheckInstance},
        witness::VirtualPolynomial,
    },
    utils::precompile_pp::PreprocessingHelper,
};
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use jolt_core::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        compact_polynomial::SmallScalar,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{BIG_ENDIAN, OpeningPoint},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    utils::{
        errors::ProofVerifyError,
        math::Math,
        thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
    },
};
use onnx_tracer::{tensor::Tensor, trace_types::AtlasOpcode};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{cell::RefCell, collections::HashMap, rc::Rc};

pub mod erf;
pub mod tanh;

// Maximum allowed LUT size exponent (2^16 = 65536 entries)
pub const MAX_LOG_FP_LOOKUP_TABLE_SIZE: usize = 16;

// Default scale factor for activation outputs (maps [-1, 1] to [-128, 128])
pub const DEFAULT_ACTIVATION_SCALE: f64 = 128.0;

/// Types of activation functions supported by the fp_lookups module
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActivationType {
    Erf,
    Tanh,
}

impl ActivationType {
    /// Returns the opcode variant for this activation type
    pub fn matches_opcode(&self, opcode: &AtlasOpcode) -> bool {
        match self {
            ActivationType::Erf => matches!(opcode, AtlasOpcode::Erf),
            ActivationType::Tanh => matches!(opcode, AtlasOpcode::Tanh),
        }
    }

    /// Creates ActivationType from an AtlasOpcode
    pub fn from_opcode(opcode: &AtlasOpcode) -> Option<Self> {
        match opcode {
            AtlasOpcode::Erf => Some(ActivationType::Erf),
            AtlasOpcode::Tanh => Some(ActivationType::Tanh),
            _ => None,
        }
    }
}

/// Enum wrapper for activation lookup tables.
/// This allows us to use trait methods on concrete types while maintaining extensibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationLookupTable {
    Erf(erf::ErfTable),
    Tanh(tanh::TanhTable),
}

/// Macro to delegate method calls to the appropriate variant in ActivationLookupTable
macro_rules! delegate_activation_method {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            ActivationLookupTable::Erf(table) => table.$method($($arg),*),
            ActivationLookupTable::Tanh(table) => table.$method($($arg),*),
        }
    };
}

impl ActivationLookupTable {
    /// Create an ActivationLookupTable from an ActivationType
    pub fn from_type(activation_type: ActivationType) -> Self {
        match activation_type {
            ActivationType::Erf => ActivationLookupTable::Erf(erf::ErfTable),
            ActivationType::Tanh => ActivationLookupTable::Tanh(tanh::TanhTable),
        }
    }

    /// Get the activation type for this table
    pub fn activation_type(&self) -> ActivationType {
        delegate_activation_method!(self, activation_type)
    }

    /// Materialize the lookup table
    pub fn materialize(&self, log_table_size: usize, scale: f64) -> Vec<i32> {
        delegate_activation_method!(self, materialize, log_table_size, scale)
    }

    /// Materialize as tensor
    pub fn materialize_tensor(&self, log_table_size: usize, scale: f64) -> Tensor<i32> {
        delegate_activation_method!(self, materialize_tensor, log_table_size, scale)
    }

    /// Evaluate the MLE at a given point
    pub fn evaluate_mle<F: JoltField>(&self, r: &[F], log_table_size: usize, scale: f64) -> F {
        delegate_activation_method!(self, evaluate_mle, r, log_table_size, scale)
    }
}

/// A single fp lookup activation instance found during preprocessing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FpLookupInstance {
    /// The type of activation function
    pub activation_type: ActivationType,
    /// Index of this activation in the bytecode (td address)
    pub td_address: usize,
    /// Memory addresses for the input operand
    pub input_addr: Vec<usize>,
    /// Memory addresses for the output
    pub output_addr: Vec<usize>,
    /// Output dimensions of the activation
    pub output_dims: Vec<usize>,
}

/// Preprocessing data for fp lookup activations
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FpLookupPreprocessing {
    /// All activation instances found in the model
    pub instances: Vec<FpLookupInstance>,
}

impl FpLookupPreprocessing {
    /// Create preprocessing by scanning bytecode for activation opcodes (Erf, Tanh)
    #[tracing::instrument(name = "FpLookupPreprocessing::preprocess", skip_all)]
    pub fn preprocess(bytecode_preprocessing: &BytecodePreprocessing) -> Self {
        let td_lookup = bytecode_preprocessing.td_lookup();
        let instances = bytecode_preprocessing
            .raw_bytecode()
            .iter()
            .filter_map(|instr| {
                ActivationType::from_opcode(&instr.opcode).map(|activation_type| {
                    FpLookupInstance::new(instr, activation_type, td_lookup, bytecode_preprocessing)
                })
            })
            .collect();

        FpLookupPreprocessing { instances }
    }

    /// Create an empty preprocessing instance
    pub fn empty() -> Self {
        Self::default()
    }

    /// Check if there are any fp lookup activations
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Get the number of activation instances
    pub fn num_instances(&self) -> usize {
        self.instances.len()
    }
}

impl FpLookupInstance {
    /// Create a new FpLookupInstance from an instruction
    pub fn new(
        instr: &onnx_tracer::trace_types::AtlasInstr,
        activation_type: ActivationType,
        td_lookup: &HashMap<usize, onnx_tracer::trace_types::AtlasInstr>,
        bytecode_preprocessing: &BytecodePreprocessing,
    ) -> Self {
        // Get input operand instruction
        let input_instr = PreprocessingHelper::get_operand_instruction(
            td_lookup,
            instr.ts1,
            &format!("{activation_type:?} operand"),
        );

        // Collect memory addresses
        let input_addr = PreprocessingHelper::collect_and_pad(
            input_instr,
            bytecode_preprocessing,
            &input_instr.output_dims,
        );
        let output_addr =
            PreprocessingHelper::collect_and_pad(instr, bytecode_preprocessing, &instr.output_dims);

        FpLookupInstance {
            activation_type,
            td_address: instr.address,
            input_addr,
            output_addr,
            output_dims: instr.output_dims.clone(),
        }
    }
}

/// Compute the log2 of the LUT size needed to cover a given range
/// The table covers signed integers from -2^(n-1) to 2^(n-1)-1
pub fn compute_log_table_size(min_val: i32, max_val: i32) -> usize {
    let abs_max = min_val.abs().max(max_val.abs()) as usize;
    // We need n bits where 2^(n-1) > abs_max, i.e., n > log2(abs_max) + 1
    let log_size = if abs_max == 0 {
        1
    } else {
        (abs_max.next_power_of_two().trailing_zeros() as usize) + 1
    };
    // Clamp to maximum allowed size
    log_size.min(MAX_LOG_FP_LOOKUP_TABLE_SIZE)
}

/// Generate a LUT tensor for the given activation type and table parameters.
///
/// This is a convenience function that uses the ActivationTable trait internally.
/// For more control, use `get_activation_table(activation_type).materialize_tensor()`.
pub fn generate_lut_tensor(
    activation_type: ActivationType,
    log_table_size: usize,
    scale: f64,
) -> Tensor<i32> {
    get_activation_table(activation_type).materialize_tensor(log_table_size, scale)
}

/// Generate a multilinear polynomial from a LUT tensor.
///
/// This converts a Tensor<i32> (as returned by `generate_lut_tensor`) into a
/// multilinear polynomial.
pub fn generate_lut_polynomial<F: JoltField>(val_tensor: &Tensor<i32>) -> MultilinearPolynomial<F> {
    let val: Vec<i64> = val_tensor
        .data()
        .to_vec()
        .into_iter()
        .map(|v| v as i64)
        .collect();
    MultilinearPolynomial::from(val)
}

/// Sumcheck instance for a single activation lookup.
///
/// Implements the shout protocol for small tables proving that activation outputs.
/// Each instance handles one activation node.
pub struct FpLookupSumcheck<F: JoltField> {
    prover_state: Option<FpLookupProver<F>>,
    input_claim: F,
    num_rounds: usize,
    r_cycle: Vec<F>,
    z: F,
    /// Index of this instance in the batch (used for VirtualPolynomial addressing)
    instance_index: usize,
    /// The activation lookup table for this instance
    activation_table: ActivationLookupTable,
    /// Log size of the LUT table
    log_table_size: usize,
}

impl<F: JoltField> SumcheckInstance<F> for FpLookupSumcheck<F> {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree(&self) -> usize {
        2
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        const DEGREE: usize = 2;
        let univariate_poly_evals: [F; DEGREE] = (0..prover_state.F.len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra = prover_state
                    .F
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let val = prover_state
                    .val
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                [
                    ra[0] * (val[0] + self.z), // eval at 0
                    ra[1] * (val[1] + self.z), // eval at 2
                ]
            })
            .reduce(
                || [F::zero(); DEGREE],
                |mut running, new| {
                    for i in 0..DEGREE {
                        running[i] += new[i];
                    }
                    running
                },
            );
        univariate_poly_evals.into()
    }

    fn bind(&mut self, r_j: F, _round: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");
        // Bind both polynomials in parallel
        rayon::join(
            || prover_state.F.bind_parallel(r_j, BindingOrder::HighToLow),
            || prover_state.val.bind_parallel(r_j, BindingOrder::HighToLow),
        );
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
        let r = [self.r_cycle.clone(), opening_point.r.clone()].concat();
        // Use indexed VirtualPolynomial for batched execution
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::FpLookupRa(self.instance_index),
            SumcheckId::FpLookup,
            r.into(),
            prover_state.F.final_sumcheck_claim(),
        );

        // cache prev claim (rv_claim)
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::FpLookupRv(self.instance_index),
            SumcheckId::FpLookup,
            self.r_cycle.clone().into(),
            self.input_claim - self.z,
        );
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let r = [self.r_cycle.clone(), opening_point.r.clone()].concat();
        // Use indexed VirtualPolynomial for batched verification
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::FpLookupRa(self.instance_index),
            SumcheckId::FpLookup,
            r.into(),
        );
    }

    fn expected_output_claim(
        &self,
        opening_accumulator: Option<std::rc::Rc<std::cell::RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F],
    ) -> F {
        let accumulator = opening_accumulator.as_ref().unwrap();
        // Use indexed VirtualPolynomial to get the correct instance's ra_claim
        let (_, ra_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::FpLookupRa(self.instance_index),
            SumcheckId::FpLookup,
        );
        // Evaluate the LUT MLE using the ActivationLookupTable
        let val_r =
            self.activation_table
                .evaluate_mle(r, self.log_table_size, DEFAULT_ACTIVATION_SCALE);
        ra_claim * (val_r + self.z)
    }
}

/// Prover state for fp lookup sumcheck.
///
/// Contains the polynomials needed to prove lookup correctness:
pub struct FpLookupProver<F: JoltField> {
    F: MultilinearPolynomial<F>,
    val: MultilinearPolynomial<F>,
    input_claim: F,
    r_cycle: Vec<F>,
    z: F,
}

impl<F: JoltField> FpLookupProver<F> {
    /// Generate prover state for a single fp lookup instance
    pub fn generate<'a, PCS: CommitmentScheme<Field = F>, FS: Transcript>(
        sm: &mut StateManager<'a, F, FS, PCS>,
        instance: &FpLookupInstance,
        log_table_size: usize,
    ) -> (Self, Vec<usize>, Vec<F>) {
        let val_hashmap = sm.get_val_final();
        let transcript = sm.get_transcript();
        let bytecode_pp = sm.get_bytecode_pp();
        let table_size = 1 << log_table_size;

        // Find the activation instruction for this instance
        let activation_instr = bytecode_pp
            .raw_bytecode()
            .iter()
            .find(|&instr| instr.address == instance.td_address)
            .expect("Activation instruction not found for instance");

        // Generate LUT based on activation type
        let val_tensor = generate_lut_tensor(
            instance.activation_type,
            log_table_size,
            DEFAULT_ACTIVATION_SCALE,
        );
        let val: MultilinearPolynomial<F> = generate_lut_polynomial(&val_tensor);

        // Compute rv(tau)
        let mut rv = bytecode_pp.get_rv(activation_instr, val_hashmap);
        rv.resize(rv.len().next_power_of_two(), val_tensor.data()[0] as i64);
        let rv: MultilinearPolynomial<F> = MultilinearPolynomial::from(rv);
        let T = rv.len();
        let n = T.log_2();
        let r_cycle: Vec<F> = transcript.borrow_mut().challenge_vector(n);
        let z: F = transcript.borrow_mut().challenge_scalar();
        let rv_claim = rv.evaluate(&r_cycle);

        let a_instr = PreprocessingHelper::get_operand_instruction(
            bytecode_pp.td_lookup(),
            activation_instr.ts1,
            &format!("{:?} operand", instance.activation_type),
        );

        // Read addresses for the input operand
        let read_addresses = bytecode_pp.get_rv(a_instr, val_hashmap);
        let mut read_addresses: Vec<usize> = read_addresses
            .iter()
            .map(|&x| n_bits_to_usize(x as i32, log_table_size))
            .collect();
        read_addresses.resize(T, 0);

        #[cfg(test)]
        {
            // rv(i) = Val(raf(i))
            let rv_int = bytecode_pp.get_rv(activation_instr, val_hashmap);
            for i in 0..rv_int.len() {
                assert_eq!(
                    rv_int[i] as i32, val_tensor[read_addresses[i]],
                    "Mismatch at index {}: rv = {}, val = {}",
                    i, rv_int[i] as i32, val_tensor[read_addresses[i]],
                )
            }

            // Check poly version of rv(i) = Val(raf(i))
            for i in 0..rv.len() {
                assert_eq!(
                    rv.get_bound_coeff(i),
                    val.get_bound_coeff(read_addresses[i]),
                    "Mismatch at index {}: rv = {}, val = {}",
                    i,
                    rv.get_bound_coeff(i),
                    val.get_bound_coeff(read_addresses[i])
                )
            }
        }

        let E = EqPolynomial::evals(&r_cycle);
        let F_coeffs: Vec<F> = read_addresses
            .iter()
            .enumerate()
            .collect::<Vec<_>>()
            .par_iter()
            .fold(
                || unsafe_allocate_zero_vec(table_size),
                |mut local_F, &(j, &k)| {
                    local_F[k] += E[j];
                    local_F
                },
            )
            .reduce(
                || unsafe_allocate_zero_vec(table_size),
                |mut a, b| {
                    for i in 0..table_size {
                        a[i] += b[i];
                    }
                    a
                },
            );
        let F = MultilinearPolynomial::from(F_coeffs.clone());

        #[cfg(test)]
        {
            let expected_claim: F = (0..F.len())
                .map(|i| F.get_bound_coeff(i) * val.get_bound_coeff(i))
                .sum();
            assert_eq!(expected_claim, rv_claim)
        }
        (
            FpLookupProver::new(F, val, rv_claim + z, r_cycle, z),
            read_addresses,
            F_coeffs,
        )
    }

    pub fn new(
        F: MultilinearPolynomial<F>,
        val: MultilinearPolynomial<F>,
        input_claim: F,
        r_cycle: Vec<F>,
        z: F,
    ) -> Self {
        FpLookupProver {
            F,
            val,
            input_claim,
            r_cycle,
            z,
        }
    }
}

/// Proof that activation function outputs match lookup table values.
///
/// Contains a batched sumcheck proof covering all activation instances.
/// Multiple activations (e.g., 3 GELU layers) are proven together.
#[derive(Clone, Debug)]
pub struct FpLookupProof<F: JoltField, FS: Transcript> {
    sumcheck_proof: SumcheckInstanceProof<F, FS>,
}

impl<F: JoltField, FS: Transcript> FpLookupProof<F, FS> {
    /// Prove all fp lookup instances using batched sumcheck.
    #[tracing::instrument(name = "FpLookupProof::prove", skip_all)]
    pub fn prove<'a, PCS: CommitmentScheme<Field = F>>(
        sm: &mut StateManager<'a, F, FS, PCS>,
    ) -> FpLookupProof<F, FS> {
        // Clone preprocessing data to avoid borrow conflicts
        let fp_lookup_instances = sm.get_fp_lookup_pp().instances.clone();
        debug_assert!(!fp_lookup_instances.is_empty());

        // Get log table size from program I/O (observed lookup input range)
        let log_table_size = sm.program_io.log_lookup_table_size();

        // Create prover instances for each fp lookup
        let mut instances: Vec<Box<dyn SumcheckInstance<F>>> = fp_lookup_instances
            .iter()
            .enumerate()
            .flat_map(|(idx, instance)| {
                let (prover, read_addresses, F) =
                    FpLookupProver::generate(sm, instance, log_table_size);
                let input_claim = prover.input_claim;
                let r_cycle = prover.r_cycle.clone();
                let z = prover.z;
                let booleanity = BooleanitySumcheck::new_prover(
                    sm,
                    read_addresses,
                    r_cycle.clone(),
                    F,
                    r_cycle.len(),
                    idx,
                );
                [
                    Box::new(FpLookupSumcheck {
                        prover_state: Some(prover),
                        input_claim,
                        num_rounds: log_table_size,
                        r_cycle,
                        z,
                        instance_index: idx,
                        activation_table: ActivationLookupTable::from_type(
                            instance.activation_type,
                        ),
                        log_table_size,
                    }) as Box<dyn SumcheckInstance<F>>,
                    Box::new(booleanity) as Box<dyn SumcheckInstance<F>>,
                ]
            })
            .collect();

        // Batch prove all instances
        let instances_mut: Vec<&mut dyn SumcheckInstance<F>> = instances
            .iter_mut()
            .map(|instance| &mut **instance as &mut dyn SumcheckInstance<F>)
            .collect();

        let (sumcheck_proof, _r) = BatchedSumcheck::prove(
            instances_mut,
            Some(sm.get_prover_accumulator().clone()),
            &mut *sm.get_transcript().borrow_mut(),
        );

        Self { sumcheck_proof }
    }

    /// Verify all fp lookup instances using batched sumcheck
    #[tracing::instrument(name = "FpLookupProof::verify", skip_all)]
    pub fn verify<'a, PCS: CommitmentScheme<Field = F>>(
        &self,
        sm: &mut StateManager<'a, F, FS, PCS>,
    ) -> Result<(), ProofVerifyError> {
        // Clone preprocessing data to avoid borrow conflicts
        let fp_lookup_instances = sm.get_fp_lookup_pp().instances.clone();
        if fp_lookup_instances.is_empty() {
            return Ok(());
        }

        // Get log table size from program I/O (observed lookup input range)
        let log_table_size = sm.program_io.log_lookup_table_size();

        // Create verifier instances for each fp lookup
        let instances: Vec<Box<dyn SumcheckInstance<F>>> = fp_lookup_instances
            .iter()
            .enumerate()
            .flat_map(|(idx, instance)| {
                // Generate tau from transcript
                let T = instance
                    .output_dims
                    .iter()
                    .product::<usize>()
                    .next_power_of_two();
                let log_T = T.log_2();
                let r_cycle: Vec<F> = sm.get_transcript().borrow_mut().challenge_vector(log_T);
                let z: F = sm.get_transcript().borrow_mut().challenge_scalar();

                // Register the claim for this instance
                let verifier_accumulator = sm.get_verifier_accumulator();
                verifier_accumulator.borrow_mut().append_virtual(
                    VirtualPolynomial::FpLookupRv(idx),
                    SumcheckId::FpLookup,
                    r_cycle.clone().into(),
                );

                // Get the input claim
                let input_claim =
                    sm.get_virtual_polynomial_opening(
                        VirtualPolynomial::FpLookupRv(idx),
                        SumcheckId::FpLookup,
                    )
                    .1 + z;

                let booleanity = BooleanitySumcheck::new_verifier(sm, r_cycle.clone(), log_T, idx);

                [
                    Box::new(FpLookupSumcheck {
                        prover_state: None,
                        input_claim,
                        num_rounds: log_table_size,
                        r_cycle,
                        z,
                        instance_index: idx,
                        activation_table: ActivationLookupTable::from_type(
                            instance.activation_type,
                        ),
                        log_table_size,
                    }) as Box<dyn SumcheckInstance<F>>,
                    Box::new(booleanity) as Box<dyn SumcheckInstance<F>>,
                ]
            })
            .collect();

        // Batch verify all instances
        let instances_ref: Vec<&dyn SumcheckInstance<F>> = instances
            .iter()
            .map(|instance| &**instance as &dyn SumcheckInstance<F>)
            .collect();

        BatchedSumcheck::verify(
            &self.sumcheck_proof,
            instances_ref,
            Some(sm.get_verifier_accumulator().clone()),
            &mut *sm.get_transcript().borrow_mut(),
        )?;

        Ok(())
    }
}

impl<F: JoltField, FS: Transcript> CanonicalSerialize for FpLookupProof<F, FS> {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.sumcheck_proof.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.sumcheck_proof.serialized_size(compress)
    }
}

impl<F: JoltField, FS: Transcript> Valid for FpLookupProof<F, FS> {
    fn check(&self) -> Result<(), SerializationError> {
        self.sumcheck_proof.check()
    }
}

impl<F: JoltField, FS: Transcript> CanonicalDeserialize for FpLookupProof<F, FS> {
    fn deserialize_with_mode<R: std::io::Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let sumcheck_proof =
            SumcheckInstanceProof::deserialize_with_mode(reader, compress, validate)?;
        Ok(Self { sumcheck_proof })
    }
}

/// Converts a usize index to an n-bit signed integer (two's complement).
///
/// This function interprets a usize in the range [0, 2^n) as a signed n-bit integer
/// using two's complement representation. Values in [0, 2^(n-1)) map to positive
/// integers, while values in [2^(n-1), 2^n) map to negative integers.
///
/// # Arguments
/// * `i` - The unsigned index to convert (must be < 2^n)
/// * `n` - The bit width for the signed representation
///
/// # Returns
/// An i32 representing the signed value in the range [-2^(n-1), 2^(n-1))
///
/// # Examples
/// ```
/// use zkml_jolt_core::jolt::fp_lookups::usize_to_n_bits;
///
/// // 4-bit signed integers range from -8 to 7
/// assert_eq!(usize_to_n_bits(0, 4), 0);    // 0000 -> 0
/// assert_eq!(usize_to_n_bits(7, 4), 7);    // 0111 -> 7
/// assert_eq!(usize_to_n_bits(8, 4), -8);   // 1000 -> -8
/// assert_eq!(usize_to_n_bits(15, 4), -1);  // 1111 -> -1
///
/// // 8-bit example
/// assert_eq!(usize_to_n_bits(127, 8), 127);   // 01111111 -> 127
/// assert_eq!(usize_to_n_bits(128, 8), -128);  // 10000000 -> -128
/// assert_eq!(usize_to_n_bits(255, 8), -1);    // 11111111 -> -1
/// ```
pub fn usize_to_n_bits(i: usize, n: usize) -> i32 {
    if i >= 1 << (n - 1) {
        i as i32 - (1 << n)
    } else {
        i as i32
    }
}

/// Converts an n-bit signed integer to its usize index representation.
///
/// This is the inverse of `usize_to_n_bits`. It converts a signed n-bit integer
/// in two's complement representation to its corresponding unsigned index in [0, 2^n).
/// Negative values are mapped to indices in [2^(n-1), 2^n), and positive values
/// to indices in [0, 2^(n-1)).
///
/// # Arguments
/// * `i` - The signed integer to convert (must be in range [-2^(n-1), 2^(n-1)))
/// * `n` - The bit width for the representation
///
/// # Returns
/// A usize index in the range [0, 2^n)
///
/// # Examples
/// ```
/// use zkml_jolt_core::jolt::fp_lookups::n_bits_to_usize;
///
/// // 4-bit signed integers
/// assert_eq!(n_bits_to_usize(0, 4), 0);    // 0 -> 0000
/// assert_eq!(n_bits_to_usize(7, 4), 7);    // 7 -> 0111
/// assert_eq!(n_bits_to_usize(-8, 4), 8);   // -8 -> 1000
/// assert_eq!(n_bits_to_usize(-1, 4), 15);  // -1 -> 1111
///
/// // 8-bit example
/// assert_eq!(n_bits_to_usize(127, 8), 127);   // 127 -> 01111111
/// assert_eq!(n_bits_to_usize(-128, 8), 128);  // -128 -> 10000000
/// assert_eq!(n_bits_to_usize(-1, 8), 255);    // -1 -> 11111111
/// ```
///
/// # Round-trip property
/// ```
/// use zkml_jolt_core::jolt::fp_lookups::{usize_to_n_bits, n_bits_to_usize};
///
/// // For any valid n-bit index, converting to signed and back yields the original
/// for i in 0..16 {
///     assert_eq!(n_bits_to_usize(usize_to_n_bits(i, 4), 4), i);
/// }
///
/// // For any valid n-bit signed value, converting to index and back yields the original
/// for i in -8..8 {
///     assert_eq!(usize_to_n_bits(n_bits_to_usize(i, 4), 4), i);
/// }
/// ```
pub fn n_bits_to_usize(i: i32, n: usize) -> usize {
    if i < 0 {
        (i + (1 << n)) as usize
    } else {
        i as usize
    }
}

/// Trait for activation function lookup tables used in fp_lookups.
///
/// This trait abstracts over different activation functions (Erf, Tanh, etc.)
/// that can be computed via lookup tables in the proof system.
///
/// # Design
/// Each activation table:
/// - Has a default scale factor that determines quantization precision
/// - Can materialize a full lookup table for a given bit-width
/// - Can evaluate a single input value
/// - Can compute the multilinear extension (MLE) polynomial
pub trait ActivationTable: Send + Sync {
    /// readable name for this activation type
    fn name(&self) -> &'static str;

    /// Returns the activation type enum variant
    fn activation_type(&self) -> ActivationType;

    /// The default scale factor for this activation (typically 128.0 for [-1, 1] outputs)
    fn default_scale(&self) -> f64;

    /// Materialize the full lookup table as i32 values.
    ///
    /// The table has `2^log_table_size` entries, indexed from 0 to 2^log_table_size - 1.
    /// Index i represents the signed value `usize_to_n_bits(i, log_table_size)`.
    ///
    /// # Arguments
    /// * `log_table_size` - Log2 of the table size (e.g., 10 for 1024 entries)
    /// * `scale` - Scale factor for the activation output
    fn materialize(&self, log_table_size: usize, scale: f64) -> Vec<i32>;

    /// Materialize the table as a Tensor<i32>
    fn materialize_tensor(&self, log_table_size: usize, scale: f64) -> Tensor<i32> {
        let table = self.materialize(log_table_size, scale);
        Tensor::new(Some(&table), &[1, table.len()]).unwrap()
    }
}

/// Extension trait for ActivationTable that provides MLE evaluation.
/// This is separate from ActivationTable to keep ActivationTable object-safe.
pub trait ActivationTableExt: ActivationTable {
    /// Evaluate the multilinear extension (MLE) of the lookup table at a given point.
    ///
    /// This method computes the MLE of the lookup table and evaluates it at the point `r`.
    /// The default implementation materializes the full table and converts it to a polynomial,
    /// but implementations may override this for more efficient evaluation.
    ///
    /// # Arguments
    /// * `r` - The evaluation point (must have length equal to `log_table_size`)
    /// * `log_table_size` - Log2 of the table size
    /// * `scale` - Scale factor for the activation output
    ///
    /// # Returns
    /// The MLE evaluation at point `r`
    fn evaluate_mle<F: JoltField>(&self, r: &[F], log_table_size: usize, scale: f64) -> F {
        let table = self.materialize(log_table_size, scale);
        let table_i64: Vec<i64> = table.into_iter().map(|v| v as i64).collect();
        let poly: MultilinearPolynomial<F> = MultilinearPolynomial::from(table_i64);
        poly.evaluate(r)
    }
}

// Blanket implementation: any ActivationTable also implements ActivationTableExt
impl<T: ActivationTable> ActivationTableExt for T {}

/// Get an ActivationTable implementation for the given activation type
pub fn get_activation_table(activation_type: ActivationType) -> Box<dyn ActivationTable> {
    match activation_type {
        ActivationType::Erf => Box::new(erf::ErfTable),
        ActivationType::Tanh => Box::new(tanh::TanhTable),
    }
}

struct BooleanityProverState<F: JoltField> {
    read_addresses: Vec<usize>,
    B: MultilinearPolynomial<F>,
    D: MultilinearPolynomial<F>,
    G: Vec<F>,
    H: Option<MultilinearPolynomial<F>>,
    F: Vec<F>,
    eq_r_r: F,
}

pub struct BooleanitySumcheck<F: JoltField> {
    log_T: usize,
    log_K: usize,
    prover_state: Option<BooleanityProverState<F>>,
    r_cycle: Vec<F>,
    r_address: Vec<F>,
    instance_index: usize,
}

impl<F: JoltField> BooleanitySumcheck<F> {
    #[tracing::instrument(skip_all, name = "BytecodeBooleanitySumcheck::new_prover")]
    pub fn new_prover(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        read_addresses: Vec<usize>,
        r_cycle: Vec<F>,
        G: Vec<F>,
        log_T: usize,
        instance_index: usize,
    ) -> Self {
        let log_K = sm.program_io.log_lookup_table_size();
        let r_address: Vec<F> = sm.transcript.borrow_mut().challenge_vector(log_K);
        Self {
            prover_state: Some(BooleanityProverState::new(
                read_addresses,
                EqPolynomial::evals(&r_cycle),
                G,
                &r_address,
                log_K,
            )),
            log_T,
            log_K,
            r_address,
            r_cycle,
            instance_index,
        }
    }

    pub fn new_verifier(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        r_cycle: Vec<F>,
        log_T: usize,
        instance_index: usize,
    ) -> Self {
        let log_K = sm.program_io.log_lookup_table_size();
        let r_address: Vec<F> = sm.transcript.borrow_mut().challenge_vector(log_K);
        Self {
            prover_state: None,
            log_T,
            log_K,
            r_address,
            r_cycle,
            instance_index,
        }
    }
}

impl<F: JoltField> BooleanityProverState<F> {
    fn new(
        read_addresses: Vec<usize>,
        eq_r_cycle: Vec<F>,
        G: Vec<F>,
        r_address: &[F],
        log_K: usize,
    ) -> Self {
        let B = MultilinearPolynomial::from(EqPolynomial::evals(r_address));

        let mut F_vec: Vec<F> = unsafe_allocate_zero_vec(log_K.pow2());
        F_vec[0] = F::one();

        let D = MultilinearPolynomial::from(eq_r_cycle);

        BooleanityProverState {
            read_addresses,
            B,
            D,
            H: None,
            G,
            F: F_vec,
            eq_r_r: F::zero(),
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for BooleanitySumcheck<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.log_K + self.log_T
    }

    fn input_claim(&self) -> F {
        F::zero()
    }

    #[tracing::instrument(skip_all, name = "BytecodeBooleanitySumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, round: usize, _previous_claim: F) -> Vec<F> {
        if round < self.log_K {
            // Phase 1: First log(K_chunk) rounds
            self.compute_phase1_message(round)
        } else {
            // Phase 2: Last log(T) rounds
            self.compute_phase2_message()
        }
    }

    #[tracing::instrument(skip_all, name = "BytecodeBooleanitySumcheck::bind")]
    fn bind(&mut self, r_j: F, round: usize) {
        let ps = self.prover_state.as_mut().unwrap();

        if round < self.log_K {
            // Phase 1: Bind B and update F
            ps.B.bind_parallel(r_j, BindingOrder::LowToHigh);

            // Update F for this round (see Equation 55)
            let (F_left, F_right) = ps.F.split_at_mut(1 << round);
            F_left
                .par_iter_mut()
                .zip(F_right.par_iter_mut())
                .for_each(|(x, y)| {
                    *y = *x * r_j;
                    *x -= *y;
                });

            // If transitioning to phase 2, prepare H
            if round == self.log_K - 1 {
                let mut read_addresses = std::mem::take(&mut ps.read_addresses);
                let f_ref = &ps.F;
                ps.H = Some({
                    let coeffs: Vec<F> = std::mem::take(&mut read_addresses)
                        .into_par_iter()
                        .map(|j| f_ref[j])
                        .collect();
                    MultilinearPolynomial::from(coeffs)
                });
                ps.eq_r_r = ps.B.final_sumcheck_claim();

                // Drop G arrays, F array, and read_addresses as they're no longer needed in phase 2
                let g = std::mem::take(&mut ps.G);
                drop_in_background_thread(g);

                let f = std::mem::take(&mut ps.F);
                drop_in_background_thread(f);

                drop_in_background_thread(read_addresses);
            }
        } else {
            let H = ps.H.as_mut().unwrap();
            rayon::join(
                || H.bind_parallel(r_j, BindingOrder::LowToHigh),
                || ps.D.bind_parallel(r_j, BindingOrder::LowToHigh),
            );
        }
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F],
    ) -> F {
        let accumulator = accumulator.as_ref().unwrap();
        let (_, ra_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::FpLookupRa(self.instance_index),
            SumcheckId::FpLookupBooleanity,
        );

        EqPolynomial::mle(
            r,
            &self
                .r_address
                .iter()
                .cloned()
                .rev()
                .chain(self.r_cycle.iter().cloned().rev())
                .collect::<Vec<F>>(),
        ) * (ra_claim.square() - ra_claim)
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        let mut opening_point = opening_point.to_vec();
        opening_point[..self.log_K].reverse();
        opening_point[self.log_K..].reverse();
        opening_point.into()
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let ps = self.prover_state.as_ref().unwrap();
        let ra_claim = ps.H.as_ref().unwrap().final_sumcheck_claim();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::FpLookupRa(self.instance_index),
            SumcheckId::FpLookupBooleanity,
            opening_point,
            ra_claim,
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::FpLookupRa(self.instance_index),
            SumcheckId::FpLookupBooleanity,
            opening_point,
        );
    }
}

impl<F: JoltField> BooleanitySumcheck<F> {
    fn compute_phase1_message(&self, round: usize) -> Vec<F> {
        let p = self.prover_state.as_ref().unwrap();
        let m = round + 1;
        const DEGREE: usize = 3;

        // EQ(k_m, c) for k_m \in {0, 1} and c \in {0, 2, 3}
        const EQ_KM_C: [[i8; 3]; 2] = [
            [
                1,  // eq(0, 0) = 0 * 0 + (1 - 0) * (1 - 0)
                -1, // eq(0, 2) = 0 * 2 + (1 - 0) * (1 - 2)
                -2, // eq(0, 3) = 0 * 3 + (1 - 0) * (1 - 3)
            ],
            [
                0, // eq(1, 0) = 1 * 0 + (1 - 1) * (1 - 0)
                2, // eq(1, 2) = 1 * 2 + (1 - 1) * (1 - 2)
                3, // eq(1, 3) = 1 * 3 + (1 - 1) * (1 - 3)
            ],
        ];

        // EQ(k_m, c)^2 for k_m \in {0, 1} and c \in {0, 2, 3}
        const EQ_KM_C_SQUARED: [[u8; 3]; 2] = [[1, 1, 4], [0, 4, 9]];

        let univariate_poly_evals: [F; 3] = (0..p.B.len() / 2)
            .into_par_iter()
            .map(|k_prime| {
                // Get B evaluations at points 0, 2, 3
                let B_evals =
                    p.B.sumcheck_evals_array::<DEGREE>(k_prime, BindingOrder::LowToHigh);

                let inner_sum = (0..1 << m)
                    .into_par_iter()
                    .map(|k| {
                        // Since we're binding variables from low to high, k_m is the high bit
                        let k_m = k >> (m - 1);
                        // We then index into F using (k_{m-1}, ..., k_1)
                        let F_k = p.F[k % (1 << (m - 1))];
                        // G_times_F := G[k] * F[k_1, ...., k_{m-1}]
                        let k_G = (k_prime << m) + k;
                        let G_times_F = p.G[k_G] * F_k;
                        // For c \in {0, 2, 3} compute:
                        //    G[k] * (F[k_1, ...., k_{m-1}, c]^2 - F[k_1, ...., k_{m-1}, c])
                        //    = G_times_F * (eq(k_m, c)^2 * F[k_1, ...., k_{m-1}] - eq(k_m, c))
                        [
                            G_times_F
                                * (EQ_KM_C_SQUARED[k_m][0].field_mul(F_k)
                                    - F::from_i64(EQ_KM_C[k_m][0] as i64)),
                            G_times_F
                                * (EQ_KM_C_SQUARED[k_m][1].field_mul(F_k)
                                    - F::from_i64(EQ_KM_C[k_m][1] as i64)),
                            G_times_F
                                * (EQ_KM_C_SQUARED[k_m][2].field_mul(F_k)
                                    - F::from_i64(EQ_KM_C[k_m][2] as i64)),
                        ]
                    })
                    .reduce(
                        || [F::zero(); 3],
                        |running, new| {
                            [
                                running[0] + new[0],
                                running[1] + new[1],
                                running[2] + new[2],
                            ]
                        },
                    );

                [
                    B_evals[0] * inner_sum[0],
                    B_evals[1] * inner_sum[1],
                    B_evals[2] * inner_sum[2],
                ]
            })
            .reduce(
                || [F::zero(); 3],
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

    fn compute_phase2_message(&self) -> Vec<F> {
        let p = self.prover_state.as_ref().unwrap();
        const DEGREE: usize = 3;

        let univariate_poly_evals: [F; 3] = (0..p.D.len() / 2)
            .into_par_iter()
            .map(|i| {
                // Get D and H evaluations at points 0, 2, 3
                let D_evals =
                    p.D.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let H = p.H.as_ref().unwrap();
                let H_evals = H.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let evals = [
                    H_evals[0].square() - H_evals[0],
                    H_evals[1].square() - H_evals[1],
                    H_evals[2].square() - H_evals[2],
                ];
                [
                    D_evals[0] * evals[0],
                    D_evals[1] * evals[1],
                    D_evals[2] * evals[2],
                ]
            })
            .reduce(
                || [F::zero(); 3],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                    ]
                },
            );

        vec![
            p.eq_r_r * univariate_poly_evals[0],
            p.eq_r_r * univariate_poly_evals[1],
            p.eq_r_r * univariate_poly_evals[2],
        ]
    }
}
