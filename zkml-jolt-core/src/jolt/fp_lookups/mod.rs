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
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{BIG_ENDIAN, OpeningPoint},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math, thread::unsafe_allocate_zero_vec},
};
use onnx_tracer::{tensor::Tensor, trace_types::AtlasOpcode};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{cell::RefCell, collections::HashMap, rc::Rc};

pub mod erf;
pub mod tanh;

// Maximum allowed LUT size exponent (2^16 = 65536 entries)
pub const MAX_LOG_FP_LOOKUP_TABLE_SIZE: usize = 16;

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
            &instr.output_dims,
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
    tau: Vec<F>,
    /// Index of this instance in the batch (used for VirtualPolynomial addressing)
    instance_index: usize,
    /// The activation type for this instance
    activation_type: ActivationType,
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
                let a_evals = prover_state
                    .F
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let b_evals = prover_state
                    .val
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                [
                    a_evals[0] * b_evals[0], // eval at 0
                    a_evals[1] * b_evals[1], // eval at 2
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
        let r = [self.tau.clone(), opening_point.r.clone()].concat();
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
            self.tau.clone().into(),
            self.input_claim,
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
        let r = [self.tau.clone(), opening_point.r.clone()].concat();
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
        // Evaluate the LUT MLE using the ActivationTable trait
        let activation_table = get_activation_table(self.activation_type);
        let val_tensor = activation_table.materialize_tensor(self.log_table_size, 128.0);
        let val_poly: MultilinearPolynomial<F> = generate_lut_polynomial(&val_tensor);
        ra_claim * val_poly.evaluate(r)
    }
}

/// Prover state for fp lookup sumcheck.
///
/// Contains the polynomials needed to prove lookup correctness:
pub struct FpLookupProver<F: JoltField> {
    F: MultilinearPolynomial<F>,
    val: MultilinearPolynomial<F>,
    rv_claim: F,
    tau: Vec<F>,
}

impl<F: JoltField> FpLookupProver<F> {
    /// Generate prover state for a single fp lookup instance
    pub fn generate_for_instance<'a, PCS: CommitmentScheme<Field = F>, FS: Transcript>(
        sm: &mut StateManager<'a, F, FS, PCS>,
        instance: &FpLookupInstance,
        log_table_size: usize,
    ) -> Self {
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
        let val_tensor = generate_lut_tensor(instance.activation_type, log_table_size, 128.0);
        let val: MultilinearPolynomial<F> = generate_lut_polynomial(&val_tensor);

        // Compute rv(tau)
        let mut rv = bytecode_pp.get_rv(activation_instr, val_hashmap);
        rv.resize(rv.len().next_power_of_two(), val_tensor.data()[0] as i64);
        let rv: MultilinearPolynomial<F> = MultilinearPolynomial::from(rv);
        let T = rv.len();
        let n = T.log_2();
        let tau: Vec<F> = transcript.borrow_mut().challenge_vector(n);
        let rv_claim = rv.evaluate(&tau);

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

        let E = EqPolynomial::evals(&tau);
        let F: Vec<F> = read_addresses
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
        let F = MultilinearPolynomial::from(F);

        #[cfg(test)]
        {
            let expected_claim: F = (0..F.len())
                .map(|i| F.get_bound_coeff(i) * val.get_bound_coeff(i))
                .sum();
            assert_eq!(expected_claim, rv_claim)
        }
        FpLookupProver::new(F, val, rv_claim, tau)
    }

    pub fn new(
        F: MultilinearPolynomial<F>,
        val: MultilinearPolynomial<F>,
        rv_claim: F,
        tau: Vec<F>,
    ) -> Self {
        FpLookupProver {
            F,
            val,
            rv_claim,
            tau,
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
            .map(|(idx, instance)| {
                let prover = FpLookupProver::generate_for_instance(sm, instance, log_table_size);
                let input_claim = prover.rv_claim;
                let tau = prover.tau.clone();
                Box::new(FpLookupSumcheck {
                    prover_state: Some(prover),
                    input_claim,
                    num_rounds: log_table_size,
                    tau,
                    instance_index: idx,
                    activation_type: instance.activation_type,
                    log_table_size,
                }) as Box<dyn SumcheckInstance<F>>
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
            .map(|(idx, instance)| {
                // Generate tau from transcript
                let n = instance
                    .output_dims
                    .iter()
                    .product::<usize>()
                    .next_power_of_two()
                    .log_2();
                let tau: Vec<F> = sm.get_transcript().borrow_mut().challenge_vector(n);

                // Register the claim for this instance
                let verifier_accumulator = sm.get_verifier_accumulator();
                verifier_accumulator.borrow_mut().append_virtual(
                    VirtualPolynomial::FpLookupRv(idx),
                    SumcheckId::FpLookup,
                    tau.clone().into(),
                );

                // Get the input claim
                let input_claim = sm
                    .get_virtual_polynomial_opening(
                        VirtualPolynomial::FpLookupRv(idx),
                        SumcheckId::FpLookup,
                    )
                    .1;

                Box::new(FpLookupSumcheck {
                    prover_state: None,
                    input_claim,
                    num_rounds: log_table_size,
                    tau,
                    instance_index: idx,
                    activation_type: instance.activation_type,
                    log_table_size,
                }) as Box<dyn SumcheckInstance<F>>
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

/// Converts a usize to an n-bit signed integer represented as i32
pub fn usize_to_n_bits(i: usize, n: usize) -> i32 {
    if i >= 1 << (n - 1) {
        i as i32 - (1 << n)
    } else {
        i as i32
    }
}

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

/// Extension trait for ActivationTable that provides polynomial operations.
/// This is separate from ActivationTable to allow ActivationTable to be object-safe.
pub trait ActivationTableExt: ActivationTable {
    /// Generate the multilinear polynomial representation of the lookup table.
    fn materialize_polynomial<F: JoltField>(
        &self,
        log_table_size: usize,
        scale: f64,
    ) -> MultilinearPolynomial<F> {
        let table = self.materialize(log_table_size, scale);
        let table_i64: Vec<i64> = table.into_iter().map(|v| v as i64).collect();
        MultilinearPolynomial::from(table_i64)
    }

    /// Evaluate the multilinear extension of the LUT at a given point.
    fn evaluate_mle<F: JoltField>(&self, r: &[F], log_table_size: usize, scale: f64) -> F {
        let poly: MultilinearPolynomial<F> = self.materialize_polynomial(log_table_size, scale);
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
