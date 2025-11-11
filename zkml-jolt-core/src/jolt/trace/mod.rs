//! # ONNX Execution Trace Module
//!
//! This module provides functionality for tracing ONNX model execution and converting
//! raw ONNX execution traces into Jolt-compatible instruction traces. It serves as a bridge
//! between ONNX operations and Jolt's zero-knowledge virtual machine (zkVM) representation.
//!
//! ## Overview
//!
//! The module handles the following key responsibilities:
//! - Converting ONNX execution traces to Jolt instruction cycles
//! - Managing memory operations and tensor value extraction
//! - Creating lookup queries for different ONNX operations
//! - Providing a unified interface for instruction lookups
//!
//! ## Key Components
//!
//! - [`trace`]: Main entry point for generating execution traces from ONNX models
//! - [`JoltONNXCycle`]: Represents a single execution cycle in the Jolt zkVM
//! - [`LookupFunction`]: Enum encapsulating different instruction types
//! - [`MemoryOps`]: Structure holding memory operation values
//!
//! ## Supported ONNX Operations
//!
//! The module currently supports the following ONNX operations:
//! - Add: Element-wise addition
//! - Sub: Element-wise subtraction  
//! - Mul: Element-wise multiplication
//! - Constant: Constant value operations
//! - Relu: Rectified Linear Unit activation
//! - MatMult: Matrix multiplication (special handling)
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use onnx_tracer::tensor::Tensor;
//!
//! let input = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
//! let preprocessing = BytecodePreprocessing::new(/* ... */);
//! let (trace, program_io) = trace(|| model, &input, &preprocessing);
//! ```

use crate::{
    jolt::{
        bytecode::{BytecodePreprocessing, JoltONNXBytecode},
        executor::instructions::{
            InstructionLookup, VirtualInstructionSequence, add::AddInstruction,
            beq::BeqInstruction, broadcast::BroadCastInstruction, div::DivInstruction,
            gte::GteInstruction, mul::MulInstruction, relu::ReluInstruction,
            reshape::ReshapeInstruction, softmax::SoftmaxInstruction, sub::SubInstruction,
            virtual_advice::AdviceInstruction,
            virtual_assert_valid_div0::AssertValidDiv0Instruction,
            virtual_assert_valid_signed_remainder::AssertValidSignedRemainderInstruction,
            virtual_const::ConstInstruction, virtual_move::MoveInstruction,
            virtual_pow2::Pow2Instruction,
        },
        lookup_table::LookupTables,
    },
    utils::tensor_to_u64s,
};
use jolt_core::zkvm::instruction::LookupQuery;
use onnx_tracer::{
    ProgramIO,
    graph::model::Model,
    tensor::Tensor,
    trace_types::{ONNXCycle, ONNXOpcode},
};
use serde::{Deserialize, Serialize};

/// The word size used for all instruction operations in the Jolt zkVM.
/// This constant defines the bit width for all arithmetic and memory operations.
pub const WORD_SIZE: usize = 32;

#[tracing::instrument(skip_all, name = "trace")]
/// Generates an execution trace for an ONNX model with the given input.
///
/// This is the main entry point for tracing ONNX model execution. It takes a model
/// factory function, input tensor, and bytecode preprocessing information to produce
/// a complete execution trace compatible with the Jolt zkVM.
///
/// # Arguments
///
/// * `model` - A closure that returns the ONNX model to execute
/// * `input` - The input tensor containing the data to process
/// * `preprocessing` - Bytecode preprocessing information that specifies the expected
///   trace structure and code size
///
/// # Returns
///
/// A tuple containing:
/// - `Vec<JoltONNXCycle>`: The complete execution trace as Jolt-compatible cycles
/// - `ProgramIO`: Input/output information from the program execution
///
/// # Type Parameters
///
/// * `ModelFunc` - A function type that returns a Model when called
///
/// # Example
///
/// ```rust,ignore
/// let (trace, io) = trace(model, &input_tensor, &preprocessing);
/// ```
pub fn trace<ModelFunc>(
    model: ModelFunc,
    input: &Tensor<i32>,
    preprocessing: &BytecodePreprocessing,
) -> (Vec<JoltONNXCycle>, ProgramIO)
where
    ModelFunc: Fn() -> Model,
{
    // Execute the ONNX model to get the raw execution trace
    let (raw_trace, program_io) = onnx_tracer::execution_trace(model(), input);
    let expanded_raw_trace = expand_virtual_traces(raw_trace, preprocessing.max_td);
    // Convert the raw ONNX trace to Jolt-compatible format
    let trace = inline_tensor_trace(expanded_raw_trace, preprocessing);
    (trace, program_io)
}

#[tracing::instrument(skip_all, name = "expand_virtual_traces")]
pub fn expand_virtual_traces(raw_trace: Vec<ONNXCycle>, max_td: usize) -> Vec<ONNXCycle> {
    raw_trace
        .into_iter()
        .flat_map(|cycle| match cycle.instr.opcode {
            ONNXOpcode::Div => DivInstruction::<32>::virtual_trace(cycle, max_td),
            ONNXOpcode::Softmax => SoftmaxInstruction::virtual_trace(cycle, max_td),
            // TODO(AntoineF4C5): Add back after stage 2 sum-check works
            // ONNXOpcode::Rsqrt => RsqrtInstruction::<32>::virtual_trace(cycle, max_td),
            _ => vec![cycle],
        })
        .collect()
}

#[tracing::instrument(skip_all, name = "inline_tensor_trace")]
/// Converts a raw ONNX execution trace into a Jolt-compatible instruction trace.
///
/// This function processes the raw trace from ONNX execution and inlines tensor operations
/// according to the preprocessed bytecode specification. Each ONNX operation may produce
/// multiple Jolt cycles depending on the number of active output elements.
///
/// # Arguments
///
/// * `raw_trace` - The raw execution trace from ONNX model execution
/// * `preprocessing` - Bytecode preprocessing that contains the expected instruction sequence
///   and final code size
///
/// # Returns
///
/// A vector of `JoltONNXCycle` representing the complete execution trace, padded to
/// the specified code size with no-op cycles if necessary.
///
/// # Implementation Details
///
/// The function:
/// 1. Starts with a no-op cycle at position 0
/// 2. For each raw ONNX cycle, generates multiple Jolt cycles based on active output elements
/// 3. Advances the program counter by the number of active output elements
/// 4. Pads the final trace to match the expected code size
///
/// # Note
///
/// The bytecode preprocessing specifies the bytecode trace since we don't prove sub-graphs.
/// This allows for deterministic trace generation that matches the expected program structure.
pub fn inline_tensor_trace(
    raw_trace: Vec<ONNXCycle>,
    preprocessing: &BytecodePreprocessing,
) -> Vec<JoltONNXCycle> {
    TraceInliner::new(preprocessing).inline(raw_trace)
}

/// Coordinates the conversion of raw ONNX cycles into scalar Jolt cycles.
struct TraceInliner<'a> {
    preprocessing: &'a BytecodePreprocessing,
    next_pc: usize,
    trace: Vec<JoltONNXCycle>,
}

impl<'a> TraceInliner<'a> {
    fn new(preprocessing: &'a BytecodePreprocessing) -> Self {
        Self {
            preprocessing,
            next_pc: 1,
            trace: vec![JoltONNXCycle::no_op()],
        }
    }

    fn inline(mut self, raw_trace: Vec<ONNXCycle>) -> Vec<JoltONNXCycle> {
        for raw_cycle in raw_trace.iter() {
            self.append_cycle(raw_cycle);
        }
        self.finish()
    }

    fn append_cycle(&mut self, raw_cycle: &ONNXCycle) {
        let element_count = raw_cycle.num_output_elements();
        let bytecode_slice =
            &self.preprocessing.bytecode[self.next_pc..self.next_pc + element_count];
        let assembler = CycleAssembler::new(raw_cycle, bytecode_slice);
        self.trace.extend(assembler.assemble());
        self.next_pc += element_count;
    }

    fn finish(mut self) -> Vec<JoltONNXCycle> {
        self.trace
            .resize(self.preprocessing.code_size, JoltONNXCycle::no_op());
        self.trace
    }
}

/// Builds the Jolt cycle sequence for a single ONNX cycle.
struct CycleAssembler<'a> {
    instructions: &'a [JoltONNXBytecode],
    values: CycleValueCache,
}

impl<'a> CycleAssembler<'a> {
    fn new(raw_cycle: &ONNXCycle, instructions: &'a [JoltONNXBytecode]) -> Self {
        let element_count = raw_cycle.num_output_elements();
        assert_eq!(
            instructions.len(),
            element_count,
            "Bytecode slice should align with the number of output elements",
        );
        Self {
            instructions,
            values: CycleValueCache::from_cycle(raw_cycle, element_count),
        }
    }

    fn assemble(&self) -> Vec<JoltONNXCycle> {
        let mut cycles = Vec::with_capacity(self.instructions.len());
        for index in 0..self.instructions.len() {
            cycles.push(self.build_cycle(index));
        }
        cycles
    }

    fn build_cycle(&self, index: usize) -> JoltONNXCycle {
        let memory_ops = self.values.memory_ops(index);
        let advice_value = self.values.advice(index);
        let lookup = JoltONNXCycle::create_lookup_function(
            &self.instructions[index],
            &memory_ops,
            advice_value,
        );
        JoltONNXCycle::new(lookup, memory_ops)
    }
}

/// Convenience wrapper retained for tests and tooling that operate on a single cycle at a time.
pub fn inline_tensor_cycle(
    raw_cycle: &ONNXCycle,
    instrs: &[JoltONNXBytecode],
) -> Vec<JoltONNXCycle> {
    CycleAssembler::new(raw_cycle, instrs).assemble()
}

/// Helper structure for extracting and organizing tensor values from an ONNX cycle.
///
/// This struct provides a convenient way to extract tensor values from different
/// sources within an ONNX cycle and organize them by element index for easy access
/// during Jolt cycle generation.
struct CycleValueCache {
    /// Source tensor 1 values for each active element
    ts1_vals: Vec<u64>,
    /// Source tensor 2 values for each active element  
    ts2_vals: Vec<u64>,
    /// Source tensor 3 values for each active element  
    ts3_vals: Vec<u64>,
    /// Destination tensor pre-operation values for each active element
    td_pre_vals: Vec<u64>,
    /// Destination tensor post-operation values for each active element
    td_post_vals: Vec<u64>,
    /// Advice values for each active element (if applicable)
    advice_vals: Option<Vec<u64>>,
}

impl CycleValueCache {
    /// Extracts tensor values from an ONNX cycle with proper handling for different operation types.
    ///
    /// # Arguments
    ///
    /// * `raw_cycle` - The ONNX cycle containing tensor operation data
    /// * `size` - The number of active output elements to extract
    ///
    /// # Returns
    ///
    /// A `CycleValueCache` with vectors of values for each tensor type.
    ///
    /// # Special Handling
    ///
    /// Einsum and Sum operations reuse the zero register because they are handled
    /// by specialized sum-check precompiles rather than element-wise lookups.
    fn from_cycle(raw_cycle: &ONNXCycle, size: usize) -> Self {
        let (ts1_vals, ts2_vals, ts3_vals) = match raw_cycle.instr.opcode {
            ONNXOpcode::Einsum(_) | ONNXOpcode::Sum(_) => {
                (vec![0; size], vec![0; size], vec![0; size])
            }
            ONNXOpcode::Broadcast => {
                // broadcast ts1
                let mut ts1 = raw_cycle
                    .memory_state
                    .ts1_val
                    .clone()
                    .expect("Broadcast ts1 should be set");
                ts1 = ts1
                    .expand(&raw_cycle.instr.output_dims)
                    .expect("Expand should always work for broadcast cycles");
                (tensor_to_u64s(&ts1), vec![0; size], vec![0; size])
            }
            _ => (
                raw_cycle.ts1_vals().unwrap_or_else(|| vec![0; size]),
                raw_cycle.ts2_vals().unwrap_or_else(|| vec![0; size]),
                raw_cycle.ts3_vals().unwrap_or_else(|| vec![0; size]),
            ),
        };

        Self {
            ts1_vals,
            ts2_vals,
            ts3_vals,
            td_pre_vals: raw_cycle.td_pre_vals().unwrap_or_else(|| vec![0; size]),
            td_post_vals: raw_cycle.td_post_vals().unwrap_or_else(|| vec![0; size]),
            advice_vals: raw_cycle.advice_value(),
        }
    }

    fn memory_ops(&self, index: usize) -> MemoryOps {
        MemoryOps::new(
            self.ts1_vals[index],
            self.ts2_vals[index],
            self.ts3_vals[index],
            self.td_pre_vals[index],
            self.td_post_vals[index],
        )
    }

    fn advice(&self, index: usize) -> Option<u64> {
        self.advice_vals.as_ref().map(|values| values[index])
    }
}

/// Represents a single execution cycle in the Jolt zkVM for ONNX operations.
///
/// Each `JoltONNXCycle` corresponds to one instruction execution in the Jolt virtual machine.
/// It contains the lookup function (operation to perform) and the memory operations
/// (register reads and writes) associated with that instruction.
///
/// These cycles are paired with preprocessed bytecode trace cycles to ensure
/// deterministic execution
#[derive(Debug, Clone)]
pub struct JoltONNXCycle {
    /// The lookup function specifying the operation to perform.
    /// None indicates we do not constrain the operation via lookup.
    pub lookup: Option<LookupFunction>,
    /// Memory operations including register reads and writes
    pub memory_ops: MemoryOps,
}

impl JoltONNXCycle {
    /// Creates a new JoltONNXCycle with the specified lookup function and memory operations.
    ///
    /// # Arguments
    ///
    /// * `lookup` - Optional lookup function specifying the operation to perform
    /// * `memory_ops` - Memory operations including register values
    pub fn new(lookup: Option<LookupFunction>, memory_ops: MemoryOps) -> Self {
        Self { lookup, memory_ops }
    }

    /// Creates a no-op cycle with default memory operations.
    ///
    /// No-op cycles are used for padding traces to the required code size
    /// and represent instructions that don't perform any meaningful computation.
    pub fn no_op() -> Self {
        Self {
            lookup: None,
            memory_ops: MemoryOps::default(),
        }
    }

    /// Creates the appropriate lookup function for the given instruction and memory operations.
    ///
    /// # Arguments
    ///
    /// * `instr` - The instruction containing the opcode and immediate value
    /// * `memory_ops` - The memory operations containing operand values
    ///
    /// # Returns
    ///
    /// An optional `LookupFunction` that corresponds to the instruction's operation.
    pub fn create_lookup_function(
        instr: &JoltONNXBytecode,
        memory_ops: &MemoryOps,
        advice_value: Option<u64>,
    ) -> Option<LookupFunction> {
        match instr.opcode {
            ONNXOpcode::Add => Some(LookupFunction::Add(AddInstruction::<WORD_SIZE>(
                memory_ops.ts1_val,
                memory_ops.ts2_val,
            ))),
            ONNXOpcode::Broadcast => {
                Some(LookupFunction::BroadCast(
                    BroadCastInstruction::<WORD_SIZE>(memory_ops.ts1_val),
                ))
            }
            ONNXOpcode::Constant => Some(LookupFunction::Const(ConstInstruction::<WORD_SIZE>(
                instr.imm,
            ))),
            ONNXOpcode::Eq => Some(LookupFunction::Eq(BeqInstruction(
                memory_ops.ts1_val,
                memory_ops.ts2_val,
            ))),
            ONNXOpcode::Gte => Some(LookupFunction::Gte(GteInstruction(
                memory_ops.ts1_val,
                memory_ops.ts2_val,
            ))),
            ONNXOpcode::Mul => Some(LookupFunction::Mul(MulInstruction::<WORD_SIZE>(
                memory_ops.ts1_val,
                memory_ops.ts2_val,
            ))),
            ONNXOpcode::Relu => Some(LookupFunction::Relu(ReluInstruction::<WORD_SIZE>(
                memory_ops.ts1_val,
            ))),
            ONNXOpcode::Reshape => Some(LookupFunction::Reshape(ReshapeInstruction::<WORD_SIZE>(
                memory_ops.ts1_val,
            ))),
            ONNXOpcode::Sub => Some(LookupFunction::Sub(SubInstruction::<WORD_SIZE>(
                memory_ops.ts1_val,
                memory_ops.ts2_val,
            ))),
            ONNXOpcode::VirtualAdvice => Some(LookupFunction::Advice(
                AdviceInstruction::<WORD_SIZE>(advice_value.expect("Advice value should be set")),
            )),
            ONNXOpcode::VirtualAssertEq => Some(LookupFunction::Eq(BeqInstruction::<WORD_SIZE>(
                memory_ops.ts1_val,
                memory_ops.ts2_val,
            ))),
            ONNXOpcode::VirtualAssertValidDiv0 => Some(LookupFunction::VirtualAssertValidDiv0(
                AssertValidDiv0Instruction::<WORD_SIZE>(memory_ops.ts1_val, memory_ops.ts2_val),
            )),
            ONNXOpcode::VirtualAssertValidSignedRemainder => {
                Some(LookupFunction::VirtualAssertValidSignedRemainder(
                    AssertValidSignedRemainderInstruction::<WORD_SIZE>(
                        memory_ops.ts1_val,
                        memory_ops.ts2_val,
                    ),
                ))
            }
            ONNXOpcode::VirtualConst => Some(LookupFunction::Const(ConstInstruction::<WORD_SIZE>(
                instr.imm,
            ))),
            ONNXOpcode::VirtualMove => {
                Some(LookupFunction::VirtualMove(MoveInstruction::<WORD_SIZE>(
                    memory_ops.ts1_val,
                )))
            }
            ONNXOpcode::VirtualPow2 => {
                Some(LookupFunction::VirtualPow2(Pow2Instruction::<WORD_SIZE>(
                    memory_ops.ts1_val,
                )))
            }
            // Other opcodes (like MatMult) don't have lookup functions
            _ => None,
        }
    }

    /// Generates a random JoltONNXCycle for testing purposes.
    ///
    /// Creates a cycle with random memory values and constructs the appropriate
    /// lookup query for the given opcode.
    ///
    /// # Arguments
    ///
    /// * `opcode` - The ONNX opcode to create a cycle for
    /// * `rng` - Random number generator for creating random values
    ///
    /// # Returns
    ///
    /// A randomly generated `JoltONNXCycle` with the specified opcode.
    pub fn random(opcode: ONNXOpcode, rng: &mut rand::rngs::StdRng) -> Self {
        use rand::RngCore;

        // Generate random memory operation values
        let memory_ops = MemoryOps::random(rng);

        // Create a random bytecode instruction
        let jolt_onnx_bytecode = JoltONNXBytecode {
            opcode,
            imm: rng.next_u64(),
            ..JoltONNXBytecode::no_op()
        };

        // Create the cycle with the appropriate lookup function
        let lookup = Self::create_lookup_function(&jolt_onnx_bytecode, &memory_ops, None);
        Self::new(lookup, memory_ops)
    }

    /// Returns the value read from the first source tensor register (ts1).
    pub fn ts1_read(&self) -> u64 {
        self.memory_ops.ts1_val
    }

    /// Returns the value read from the second source tensor register (ts2).
    pub fn ts2_read(&self) -> u64 {
        self.memory_ops.ts2_val
    }

    /// Returns the value read from the third source tensor register (ts3).
    pub fn ts3_read(&self) -> u64 {
        self.memory_ops.ts3_val
    }

    /// Returns the destination tensor write values.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - `pre_val`: The value in the destination register before the operation
    /// - `post_val`: The value in the destination register after the operation
    pub fn td_write(&self) -> (u64, u64) {
        (self.memory_ops.td_pre_val, self.memory_ops.td_post_val)
    }
}

/// Implementation of `LookupQuery` trait for `JoltONNXCycle`.
///
/// This implementation allows JoltONNXCycle to participate in the Jolt zkVM's
/// lookup argument system, which is used to prove the correctness of instruction
/// executions through cryptographic lookup tables.
impl LookupQuery<WORD_SIZE> for JoltONNXCycle {
    /// Converts the cycle's lookup function to instruction inputs.
    ///
    /// # Returns
    ///
    /// A tuple of (u64, i64) representing the instruction inputs,
    /// or (0, 0) if no lookup function is present.
    fn to_instruction_inputs(&self) -> (u64, i64) {
        self.lookup.as_ref().map_or((0, 0), |lookup_function| {
            lookup_function.to_instruction_inputs()
        })
    }

    /// Returns the lookup table index for this cycle's operation.
    ///
    /// The index identifies which lookup table should be used for
    /// proving this instruction's correctness.
    fn to_lookup_index(&self) -> u64 {
        self.lookup
            .as_ref()
            .map_or(0, |lookup_function| lookup_function.to_lookup_index())
    }

    /// Returns the operands used for the lookup table query.
    ///
    /// # Returns
    ///
    /// A tuple of (u64, u64) representing the lookup operands,
    /// or (0, 0) if no lookup function is present.
    fn to_lookup_operands(&self) -> (u64, u64) {
        self.lookup.as_ref().map_or((0, 0), |lookup_function| {
            lookup_function.to_lookup_operands()
        })
    }

    /// Returns the expected output from the lookup table query.
    ///
    /// This value is used to verify that the instruction was executed correctly.
    fn to_lookup_output(&self) -> u64 {
        self.lookup
            .as_ref()
            .map_or(0, |lookup_function| lookup_function.to_lookup_output())
    }
}

/// Implementation of `InstructionLookup` trait for `JoltONNXCycle`.
///
/// This implementation provides access to the lookup tables required for
/// proving instruction correctness in the Jolt zkVM.
///
/// # Note
///
/// TODO: This implementation may be redundant since `JoltONNXBytecode` already
/// implements this trait. Consider refactoring to eliminate duplication.
impl InstructionLookup<WORD_SIZE> for JoltONNXCycle {
    /// Returns the lookup table associated with this cycle's operation.
    ///
    /// # Returns
    ///
    /// An optional `LookupTables` instance containing the cryptographic
    /// lookup table for this instruction, or `None` if no lookup is required.
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        self.lookup
            .as_ref()
            .and_then(|lookup_function| lookup_function.lookup_table())
    }
}

/// Represents the memory operations for a single instruction cycle.
///
/// This structure holds the values for all register operations that occur
/// during the execution of one instruction. It includes reads from source
/// tensors and writes to destination tensors.
///
/// # Memory Model
///
/// The Jolt zkVM uses a register-based memory model where:
/// - `ts1` and `ts2` are source tensor registers (read-only for this instruction)
/// - `td` is the destination tensor register (read before, written after)
///
/// The pre and post values for the destination register enable verification
/// that the instruction was executed correctly by comparing the expected
/// output with the actual result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct MemoryOps {
    /// Value read from the first source tensor register (ts1)
    ts1_val: u64,
    /// Value read from the second source tensor register (ts2)  
    ts2_val: u64,
    /// Value read from the third source tensor register (ts3)
    ts3_val: u64,
    /// Value in the destination tensor register before the operation
    td_pre_val: u64,
    /// Value in the destination tensor register after the operation
    td_post_val: u64,
}

impl MemoryOps {
    /// Creates a new MemoryOps with the specified values.
    ///
    /// # Arguments
    ///
    /// * `ts1_val` - Value for the first source tensor register
    /// * `ts2_val` - Value for the second source tensor register
    /// * `td_pre_val` - Value in destination register before operation
    /// * `td_post_val` - Value in destination register after operation
    pub fn new(
        ts1_val: u64,
        ts2_val: u64,
        ts3_val: u64,
        td_pre_val: u64,
        td_post_val: u64,
    ) -> Self {
        Self {
            ts1_val,
            ts2_val,
            ts3_val,
            td_pre_val,
            td_post_val,
        }
    }

    /// Creates a MemoryOps with random values for testing.
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator to use for value generation
    ///
    /// # Returns
    ///
    /// A new `MemoryOps` instance with random values for all fields.
    pub fn random(rng: &mut rand::rngs::StdRng) -> Self {
        use rand::RngCore;
        Self::new(
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        )
    }
}

/// Macro for defining the LookupFunction enum and its trait implementations.
///
/// This macro generates a comprehensive enum that encapsulates all supported
/// instruction types and automatically implements the required traits for
/// lookup table operations.
///
/// # Generated Implementations
///
/// The macro generates implementations for:
/// - `LookupQuery<WORD_SIZE>`: Enables participation in lookup arguments
/// - `InstructionLookup<WORD_SIZE>`: Provides access to lookup tables
/// - `Clone`, `Debug`: Standard Rust traits
/// - `Serialize`, `Deserialize`: For serialization support
///
/// # Parameters
///
/// - `enum_name`: Name of the generated enum
/// - `word_size`: Constant representing the word size for instructions
/// - `variant: type` pairs: Each supported instruction variant and its type
macro_rules! define_lookup_enum {
    (
        enum $enum_name:ident,
        const $word_size:ident,
        $($variant:ident : $inner:ty),+ $(,)?
    ) => {
        #[derive(Clone, Debug, Serialize, Deserialize)]
        pub enum $enum_name {
            $(
                $variant($inner),
            )+
        }

        impl LookupQuery<$word_size> for $enum_name {
            fn to_instruction_inputs(&self) -> (u64, i64) {
                match self {
                    $(
                        $enum_name::$variant(inner) => inner.to_instruction_inputs(),
                    )+
                }
            }

            fn to_lookup_index(&self) -> u64 {
                match self {
                    $(
                        $enum_name::$variant(inner) => inner.to_lookup_index(),
                    )+
                }
            }

            fn to_lookup_operands(&self) -> (u64, u64) {
                match self {
                    $(
                        $enum_name::$variant(inner) => inner.to_lookup_operands(),
                    )+
                }
            }

            fn to_lookup_output(&self) -> u64 {
                match self {
                    $(
                        $enum_name::$variant(inner) => inner.to_lookup_output(),
                    )+
                }
            }
        }

        impl InstructionLookup<$word_size> for $enum_name {
            fn lookup_table(&self) -> Option<LookupTables<$word_size>> {
                match self {
                    $(
                        $enum_name::$variant(inner) => inner.lookup_table(),
                    )+
                }
            }
        }
    };
}

// Generate the LookupFunction enum with all supported instruction types
define_lookup_enum!(
    enum LookupFunction,
    const WORD_SIZE,
    Add: AddInstruction<WORD_SIZE>,
    Advice: AdviceInstruction<WORD_SIZE>,
    BroadCast: BroadCastInstruction<WORD_SIZE>,
    Const: ConstInstruction<WORD_SIZE>,
    Eq: BeqInstruction<WORD_SIZE>,
    Gte: GteInstruction<WORD_SIZE>,
    Mul: MulInstruction<WORD_SIZE>,
    Relu: ReluInstruction<WORD_SIZE>,
    Reshape: ReshapeInstruction<WORD_SIZE>,
    Sub: SubInstruction<WORD_SIZE>,
    VirtualAssertValidSignedRemainder: AssertValidSignedRemainderInstruction<WORD_SIZE>,
    VirtualAssertValidDiv0: AssertValidDiv0Instruction<WORD_SIZE>,
    VirtualConst: ConstInstruction<WORD_SIZE>,
    VirtualMove: MoveInstruction<WORD_SIZE>,
    VirtualPow2: Pow2Instruction<WORD_SIZE>,
);
