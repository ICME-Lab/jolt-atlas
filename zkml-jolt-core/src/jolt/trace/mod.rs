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
    jolt::bytecode::{BytecodePreprocessing, JoltONNXBytecode},
    utils::tensor_to_u64s,
};
use onnx_tracer::{
    ProgramIO,
    graph::model::Model,
    tensor::Tensor,
    trace_types::{AtlasCycle, AtlasOpcode, ONNXCycle},
    utils::VirtualSlotCounter,
};

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
pub fn expand_virtual_traces(raw_trace: Vec<ONNXCycle>, max_td: usize) -> Vec<AtlasCycle> {
    let mut virtual_slot_counter = VirtualSlotCounter::new(max_td + 1);
    raw_trace
        .into_iter()
        .flat_map(|cycle| cycle.virtual_trace(&mut virtual_slot_counter))
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
    raw_trace: Vec<AtlasCycle>,
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

    fn inline(mut self, raw_trace: Vec<AtlasCycle>) -> Vec<JoltONNXCycle> {
        for raw_cycle in raw_trace.iter() {
            self.append_cycle(raw_cycle);
        }
        self.finish()
    }

    fn append_cycle(&mut self, raw_cycle: &AtlasCycle) {
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
    fn new(raw_cycle: &AtlasCycle, instructions: &'a [JoltONNXBytecode]) -> Self {
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

        JoltONNXCycle::new(self.instructions[index].clone(), memory_ops, advice_value)
    }
}

/// Convenience wrapper retained for tests and tooling that operate on a single cycle at a time.
pub fn inline_tensor_cycle(
    raw_cycle: &AtlasCycle,
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
    // TODO(AntoineF4C5): Refactor so that the cycle values extraction logic is defined in each instruction
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
    fn from_cycle(raw_cycle: &AtlasCycle, size: usize) -> Self {
        let (ts1_vals, ts2_vals, ts3_vals) = match raw_cycle.instr.opcode {
            AtlasOpcode::Einsum(_) | AtlasOpcode::Sum(_) => {
                (vec![0; size], vec![0; size], vec![0; size])
            }
            AtlasOpcode::Broadcast => {
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
    // TODO(AntoineF4C5): Might just retrieve few information from the instr such as maybe just the Add(x, y) fed with relevant values)
    /// The lookup function specifying the operation to perform.
    /// None indicates we do not constrain the operation via lookup.
    pub instr: JoltONNXBytecode,
    /// Memory operations including register reads and writes
    pub memory_ops: MemoryOps,
    pub advice_value: Option<u64>,
}

impl JoltONNXCycle {
    /// Creates a new JoltONNXCycle with the specified lookup function and memory operations.
    ///
    /// # Arguments
    ///
    /// * `lookup` - Optional lookup function specifying the operation to perform
    /// * `memory_ops` - Memory operations including register values
    pub fn new(
        instruction: JoltONNXBytecode,
        memory_ops: MemoryOps,
        advice_value: Option<u64>,
    ) -> Self {
        Self {
            instr: instruction,
            memory_ops,
            advice_value,
        }
    }

    /// Creates a no-op cycle with default memory operations.
    ///
    /// No-op cycles are used for padding traces to the required code size
    /// and represent instructions that don't perform any meaningful computation.
    pub fn no_op() -> Self {
        Self {
            instr: JoltONNXBytecode::no_op(),
            memory_ops: MemoryOps::default(),
            advice_value: None,
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
    pub fn random(opcode: AtlasOpcode, rng: &mut rand::rngs::StdRng) -> Self {
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
        Self::new(jolt_onnx_bytecode, memory_ops, None)
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
