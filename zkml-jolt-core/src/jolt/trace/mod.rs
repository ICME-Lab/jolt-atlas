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

use crate::jolt::{
    bytecode::{BytecodePreprocessing, JoltONNXBytecode},
    executor::instructions::{
        InstructionLookup, VirtualInstructionSequence, abs::AbsInstruction, add::AddInstruction,
        beq::BeqInstruction, div::DivInstruction, mul::MulInstruction, relu::ReluInstruction,
        sub::SubInstruction, virtual_advice::AdviceInstruction,
        virtual_assert_valid_div0::AssertValidDiv0Instruction,
        virtual_assert_valid_signed_remainder::AssertValidSignedRemainderInstruction,
        virtual_const::ConstInstruction, virtual_move::MoveInstruction,
    },
    lookup_table::LookupTables,
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
/// let (trace, io) = trace(|| create_model(), &input_tensor, &preprocessing);
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

    let raw_trace = expand_raw_trace(raw_trace, preprocessing.max_td);
    // Convert the raw ONNX trace to Jolt-compatible format
    let trace = inline_tensor_trace(raw_trace, preprocessing);
    (trace, program_io)
}

pub fn expand_raw_trace(raw_trace: Vec<ONNXCycle>, max_td: usize) -> Vec<ONNXCycle> {
    raw_trace
        .into_iter()
        .flat_map(|cycle| match cycle.instr.opcode {
            ONNXOpcode::Div => DivInstruction::<32>::virtual_trace(cycle, max_td),
            _ => vec![cycle],
        })
        .collect()
}

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
    // Initialize trace with a no-op cycle at the beginning
    let mut trace = vec![JoltONNXCycle::no_op()];
    let mut current_pc = 1;

    // Process each raw ONNX cycle
    for raw_cycle in raw_trace.iter() {
        let current_active_output_els = raw_cycle.instr.active_output_elements;

        // Generate Jolt cycles for this ONNX operation
        let sequence = inline_tensor_cycle(
            raw_cycle,
            &preprocessing.bytecode[current_pc..current_pc + current_active_output_els],
        );

        // Add the generated cycles to the trace
        trace.extend(sequence);

        // Advance program counter by the number of elements processed
        current_pc += current_active_output_els;
    }

    // Pad the trace to the expected code size with no-op cycles
    trace.resize(preprocessing.code_size, JoltONNXCycle::no_op());
    trace
}

/// Converts a single ONNX cycle into multiple Jolt execution cycles.
///
/// This function takes a raw ONNX cycle and creates individual Jolt cycles for each
/// active output element. Each element gets its own cycle with appropriate memory
/// operations and lookup queries.
///
/// # Arguments
///
/// * `raw_cycle` - The raw ONNX cycle containing tensor operations and values
/// * `instrs` - A slice of Jolt bytecode instructions corresponding to this cycle
///
/// # Returns
///
/// A vector of `JoltONNXCycle` where each cycle corresponds to one active output element.
///
/// # Implementation Details
///
/// For each active output element:
/// 1. Extracts tensor values (ts1, ts2, td_pre, td_post) at the corresponding index
/// 2. Creates memory operations structure
/// 3. Constructs a Jolt cycle with the memory operations
/// 4. Builds the appropriate lookup query based on the instruction opcode
pub fn inline_tensor_cycle(
    raw_cycle: &ONNXCycle,
    instrs: &[JoltONNXBytecode],
) -> Vec<JoltONNXCycle> {
    let active_output_elements = raw_cycle.instr.active_output_elements;
    assert_eq!(
        instrs.len(),
        active_output_elements,
        "Instruction count must match active output elements"
    );

    // Extract tensor values for all active elements
    let tensor_values = TensorValues::from_cycle(raw_cycle, active_output_elements);

    // Create a Jolt cycle for each active output element
    (0..active_output_elements)
        .map(|i| create_jolt_cycle(&tensor_values, i, &instrs[i]))
        .collect()
}

/// Creates a single Jolt cycle from tensor values and instruction.
///
/// # Arguments
///
/// * `tensor_values` - The extracted tensor values for all elements
/// * `element_index` - The index of the element to create a cycle for
/// * `instr` - The instruction for this element
///
/// # Returns
///
/// A `JoltONNXCycle` for the specified element.
fn create_jolt_cycle(
    tensor_values: &TensorValues,
    element_index: usize,
    instr: &JoltONNXBytecode,
) -> JoltONNXCycle {
    // Create memory operations for this element
    let memory_ops = MemoryOps::new(
        tensor_values.ts1_vals[element_index],
        tensor_values.ts2_vals[element_index],
        tensor_values.td_pre_vals[element_index],
        tensor_values.td_post_vals[element_index],
    );

    // Create the cycle with the appropriate lookup function
    let lookup = JoltONNXCycle::create_lookup_function(
        instr,
        &memory_ops,
        tensor_values.advice_vals.as_ref().map(|v| v[element_index]),
    );
    JoltONNXCycle::new(lookup, memory_ops)
}

/// Helper structure for extracting and organizing tensor values from an ONNX cycle.
///
/// This struct provides a convenient way to extract tensor values from different
/// sources within an ONNX cycle and organize them by element index for easy access
/// during Jolt cycle generation.
struct TensorValues {
    /// Source tensor 1 values for each active element
    ts1_vals: Vec<u64>,
    /// Source tensor 2 values for each active element  
    ts2_vals: Vec<u64>,
    /// Destination tensor pre-operation values for each active element
    td_pre_vals: Vec<u64>,
    /// Destination tensor post-operation values for each active element
    td_post_vals: Vec<u64>,
    /// Advice values for each active element (if applicable)
    advice_vals: Option<Vec<u64>>,
}

impl TensorValues {
    /// Extracts tensor values from an ONNX cycle with proper handling for different operation types.
    ///
    /// # Arguments
    ///
    /// * `raw_cycle` - The ONNX cycle containing tensor operation data
    /// * `size` - The number of active output elements to extract
    ///
    /// # Returns
    ///
    /// A `TensorValues` struct with vectors of values for each tensor type.
    ///
    /// # Special Handling
    ///
    /// For MatMult operations, ts1 and ts2 values are set to zero because MatMult
    /// operations use a different memory consistency check (MCC) pattern and are
    /// handled separately from other element-wise operations.
    fn from_cycle(raw_cycle: &ONNXCycle, size: usize) -> Self {
        let (ts1_vals, ts2_vals) = if Self::is_non_elementwise_op(&raw_cycle.instr.opcode) {
            // Non-elementwise operations don't use ts1/ts2 in the same way
            (vec![0; size], vec![0; size])
        } else {
            // Extract tensor values, defaulting to zero if not present
            (
                raw_cycle.ts1_vals().unwrap_or_else(|| vec![0; size]),
                raw_cycle.ts2_vals().unwrap_or_else(|| vec![0; size]),
            )
        };

        Self {
            ts1_vals,
            ts2_vals,
            td_pre_vals: raw_cycle.td_pre_vals().unwrap_or_else(|| vec![0; size]),
            td_post_vals: raw_cycle.td_post_vals().unwrap_or_else(|| vec![0; size]),
            advice_vals: raw_cycle.advice_value(),
        }
    }

    /// Checks if an operation is non-elementwise and requires special handling.
    ///
    /// # Arguments
    ///
    /// * `opcode` - The ONNX operation code to check
    ///
    /// # Returns
    ///
    /// `true` if the operation is non-elementwise, `false` otherwise.
    fn is_non_elementwise_op(opcode: &ONNXOpcode) -> bool {
        matches!(opcode, ONNXOpcode::MatMult)
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
            ONNXOpcode::Sub => Some(LookupFunction::Sub(SubInstruction::<WORD_SIZE>(
                memory_ops.ts1_val,
                memory_ops.ts2_val,
            ))),
            ONNXOpcode::Mul => Some(LookupFunction::Mul(MulInstruction::<WORD_SIZE>(
                memory_ops.ts1_val,
                memory_ops.ts2_val,
            ))),
            ONNXOpcode::Constant => Some(LookupFunction::Const(ConstInstruction::<WORD_SIZE>(
                instr.imm,
            ))),
            ONNXOpcode::Relu => Some(LookupFunction::Relu(ReluInstruction::<WORD_SIZE>(
                memory_ops.ts1_val,
            ))),
            ONNXOpcode::VirtualConst => Some(LookupFunction::Const(ConstInstruction::<WORD_SIZE>(
                instr.imm,
            ))),
            ONNXOpcode::VirtualMove => {
                Some(LookupFunction::VirtualMove(MoveInstruction::<WORD_SIZE>(
                    memory_ops.ts1_val,
                )))
            }
            ONNXOpcode::VirtualAdvice => Some(LookupFunction::Advice(
                AdviceInstruction::<WORD_SIZE>(advice_value.expect("Advice value should be set")),
            )),
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
            ONNXOpcode::VirtualAssertEq => {
                Some(LookupFunction::VirtualAssertEq(
                    BeqInstruction::<WORD_SIZE>(memory_ops.ts1_val, memory_ops.ts2_val),
                ))
            }
            // Other opcodes (like MatMult) don't have lookup functions
            ONNXOpcode::Abs => Some(LookupFunction::Abs(AbsInstruction::<WORD_SIZE>(
                memory_ops.ts1_val,
            ))),
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
    pub fn new(ts1_val: u64, ts2_val: u64, td_pre_val: u64, td_post_val: u64) -> Self {
        Self {
            ts1_val,
            ts2_val,
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
    Sub: SubInstruction<WORD_SIZE>,
    Mul: MulInstruction<WORD_SIZE>,
    Const: ConstInstruction<WORD_SIZE>,
    Relu: ReluInstruction<WORD_SIZE>,
    Abs: AbsInstruction<WORD_SIZE>,
    Advice: AdviceInstruction<WORD_SIZE>,
    VirtualAssertValidSignedRemainder: AssertValidSignedRemainderInstruction<WORD_SIZE>,
    VirtualAssertValidDiv0: AssertValidDiv0Instruction<WORD_SIZE>,
    VirtualAssertEq: BeqInstruction<WORD_SIZE>,
    VirtualMove: MoveInstruction<WORD_SIZE>,
    VirtualConst: ConstInstruction<WORD_SIZE>,
);

/// This function validates that the execution trace matches the expected memory
/// operations by simulating the execution and checking that reads and writes
/// occur at the correct memory addresses with the expected values.
///
/// # Purpose
///
/// This validation is crucial because runtime sometimes converts operands to
/// floating point for intermediate calculations, which can cause mismatches
/// between expected outputs and actual trace values. This function helps catch
/// such discrepancies during testing.
///
/// # Arguments
///
/// * `bytecode` - The sequence of Jolt ONNX bytecode instructions
/// * `execution_trace` - The corresponding execution trace cycles
/// * `memory_size` - The size of the memory space (number of memory addresses)
///
/// # Validation Process
///
/// For each instruction-cycle pair:
/// 1. Validates that ts1 and ts2 reads match the current memory state
/// 2. Validates that td pre-write value matches current memory at td address
/// 3. Updates memory with the td post-write value
///
/// # Panics
///
/// This function will panic with detailed error information if any memory
/// operation doesn't match the expected values, including:
/// - The cycle number where the error occurred
/// - The instruction and cycle details
/// - Expected vs actual values
/// - The memory address involved
///
/// # Example Error Output
///
/// ```text
/// TS1 READ error at cycle_42: <instruction> <cycle>; Expected: 123, got: 456 at address 78
/// ```
#[cfg(test)]
pub fn sanity_check_mcc(
    bytecode: &[JoltONNXBytecode],
    execution_trace: &[JoltONNXCycle],
    memory_size: usize,
) {
    assert_eq!(
        bytecode.len(),
        execution_trace.len(),
        "Bytecode and execution trace must have the same length"
    );

    // Initialize memory with all zeros
    let mut memory = vec![0u64; memory_size];

    // Validate each cycle against its corresponding instruction
    for (cycle_index, (cycle, instr)) in execution_trace.iter().zip(bytecode.iter()).enumerate() {
        validate_memory_operation(cycle_index, cycle, instr, &mut memory);
    }
}

/// Validates a single memory operation against the current memory state.
///
/// # Arguments
///
/// * `cycle_index` - The index of the current cycle (for error reporting)
/// * `cycle` - The execution cycle to validate
/// * `instr` - The corresponding instruction
/// * `memory` - The current memory state (will be updated with td post-write value)
#[cfg(test)]
fn validate_memory_operation(
    cycle_index: usize,
    cycle: &JoltONNXCycle,
    instr: &JoltONNXBytecode,
    memory: &mut [u64],
) {
    // Extract memory addresses from the instruction
    let addresses = MemoryAddresses::from_instruction(instr);

    // Validate reads from source registers
    validate_read(
        "TS1",
        cycle_index,
        addresses.ts1,
        cycle.ts1_read(),
        memory,
        instr,
        cycle,
    );
    validate_read(
        "TS2",
        cycle_index,
        addresses.ts2,
        cycle.ts2_read(),
        memory,
        instr,
        cycle,
    );

    // Validate destination register pre-write state
    let (td_pre, td_post) = cycle.td_write();
    validate_read(
        "TD",
        cycle_index,
        addresses.td,
        td_pre,
        memory,
        instr,
        cycle,
    );

    // Update memory with the post-write value
    memory[addresses.td] = td_post;
}

/// Helper struct to hold memory addresses for cleaner code.
#[cfg(test)]
struct MemoryAddresses {
    ts1: usize,
    ts2: usize,
    td: usize,
}

#[cfg(test)]
impl MemoryAddresses {
    fn from_instruction(instr: &JoltONNXBytecode) -> Self {
        Self {
            ts1: instr.ts1 as usize,
            ts2: instr.ts2 as usize,
            td: instr.td as usize,
        }
    }
}

/// Validates a single read operation.
///
/// # Arguments
///
/// * `register_name` - Name of the register being validated (for error messages)
/// * `cycle_index` - The cycle index (for error reporting)
/// * `address` - The memory address being read
/// * `actual_value` - The value that was read according to the trace
/// * `memory` - The current memory state
/// * `instr` - The instruction (for error reporting)
/// * `cycle` - The cycle (for error reporting)
#[cfg(test)]
fn validate_read(
    register_name: &str,
    cycle_index: usize,
    address: usize,
    actual_value: u64,
    memory: &[u64],
    instr: &JoltONNXBytecode,
    cycle: &JoltONNXCycle,
) {
    let expected_value = memory[address];
    assert_eq!(
        expected_value,
        actual_value,
        "{} READ error at cycle_{}: {:#?} {:#?}; Expected: {}, got: {} at address {}",
        register_name,
        cycle_index,
        instr,
        cycle,
        expected_value as u32 as i32,
        actual_value as u32 as i32,
        address
    );
}
