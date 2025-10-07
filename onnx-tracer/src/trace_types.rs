//! Type library used to build the execution trace.
//! Used to format the bytecode and define each instr flags and memory access patterns.
//! Used by the runtime to generate an execution trace for ONNX runtime execution.

use crate::tensor::Tensor;
use rand::{rngs::StdRng, RngCore};
use serde::{Deserialize, Serialize};
use std::ops::{Index, IndexMut};
use strum::EnumCount;
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

/// Represents a step in the execution trace, where an execution trace is a `Vec<ONNXCycle>`.
/// Records what the VM did at a cycle of execution.
/// Constructed at each step in the VM execution cycle, documenting instr, reads & state changes (writes).

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ONNXCycle {
    pub instr: ONNXInstr,
    pub memory_state: MemoryState,
    pub advice_value: Option<Tensor<i32>>,
}

impl ONNXCycle {
    pub fn no_op() -> Self {
        ONNXCycle {
            instr: ONNXInstr::no_op(),
            memory_state: MemoryState::default(),
            advice_value: None,
        }
    }

    pub fn random(opcode: ONNXOpcode, rng: &mut StdRng) -> Self {
        ONNXCycle {
            instr: ONNXInstr::dummy(opcode),
            memory_state: MemoryState::random(rng),
            advice_value: Some(Tensor::from((0..1).map(|_| rng.next_u64() as u32 as i32))),
        }
    }

    pub fn td(&self) -> Option<usize> {
        self.instr.td
    }

    pub fn ts1(&self) -> Option<usize> {
        self.instr.ts1
    }

    pub fn ts2(&self) -> Option<usize> {
        self.instr.ts2
    }

    pub fn ts3(&self) -> Option<usize> {
        self.instr.ts3
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Default, Serialize, Deserialize, PartialOrd, Ord)]
pub struct MemoryState {
    pub ts1_val: Option<Tensor<i32>>,
    pub ts2_val: Option<Tensor<i32>>,
    pub ts3_val: Option<Tensor<i32>>,
    pub td_pre_val: Option<Tensor<i32>>,
    pub td_post_val: Option<Tensor<i32>>,
}

impl MemoryState {
    pub fn random(rng: &mut StdRng) -> Self {
        MemoryState {
            ts1_val: Some(Tensor::new(Some(&[rng.next_u64() as u32 as i32]), &[1]).unwrap()),
            ts2_val: Some(Tensor::new(Some(&[rng.next_u64() as u32 as i32]), &[1]).unwrap()),
            ts3_val: Some(Tensor::new(Some(&[rng.next_u64() as u32 as i32]), &[1]).unwrap()),
            td_pre_val: Some(Tensor::new(Some(&[rng.next_u64() as u32 as i32]), &[1]).unwrap()),
            td_post_val: Some(Tensor::new(Some(&[rng.next_u64() as u32 as i32]), &[1]).unwrap()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
/// Represents a single ONNX instruction parsed from the model.
/// Represents a single ONNX instruction in the program code.
///
/// Each `ONNXInstr` contains the program counter address, the operation code,
/// and up to two input tensor operands that specify the sources
/// of tensor data in the computation graph. The operands are optional and may
/// be `None` if the instruction requires fewer than two inputs.
///
/// # Fields
/// - `address`: The program counter (PC) address of this instruction in the bytecode.
/// - `opcode`: The operation code (opcode) that defines the instruction's function.
/// - `ts1`: The first input tensor operand, specified as an `Option<usize>`, representing the index of a node in the computation graph. Analogous to the `rs1` register in RISC-V.
/// - `ts2`: The second input tensor operand, specified as an `Option<usize>`, representing the index of a node in the computation graph. Analogous to the `rs2` register in RISC-V.
///
/// The ONNX model is converted into a sequence of [`ONNXInstr`]s, forming the program code.
/// During runtime, the program counter (PC) is used to fetch the next instruction from this read-only memory storing the program bytecode.
pub struct ONNXInstr {
    /// The program counter (PC) address of this instruction in the bytecode.
    pub address: usize,
    /// The operation code (opcode) that defines the instruction's function.
    pub opcode: ONNXOpcode,
    /// The first input tensor operand, specified as the index of a node in the computation graph.
    /// This index (`node_idx`) identifies which node's output tensor will be used as input for this instruction.
    /// Since each node produces only one output tensor in this simplified ISA, the index is sufficient.
    /// If the instruction requires fewer than two inputs, this will be `None`.
    /// Conceptually, `ts1` is analogous to the `rs1` register specifier in RISC-V,
    /// as both indicate the source location (address or index) of an operand.
    pub ts1: Option<usize>,
    /// The second input tensor operand, also specified as the index of a node in the computation graph.
    /// Like `ts1`, this index identifies another node whose output tensor will be used as input.
    /// If the instruction requires only one or zero inputs, this will be `None`.
    /// This field is analogous to the `rs2` register specifier in RISC-V,
    /// serving to specify the address or index of the second operand.
    pub ts2: Option<usize>,
    /// Special opcodes like IFF/Where/Select may use a third operand, which is the index of the condition tensor.
    pub ts3: Option<usize>,
    /// The destination tensor index, which is the index of the node in the computation graph
    /// where the result of this instruction will be stored.
    /// This is analogous to the `rd` register specifier in RISC-V, indicating
    /// where the result of the operation should be written.
    pub td: Option<usize>,
    pub imm: Option<Tensor<i32>>, // Immediate value, if applicable
    /// If this instruction is part of a "virtual sequence" (see Section 6.2 of the
    /// Jolt paper), then this contains the number of virtual instructions after this
    /// one in the sequence. I.e. if this is the last instruction in the sequence,
    /// `virtual_sequence_remaining` will be Some(0); if this is the penultimate instruction
    /// in the sequence, `virtual_sequence_remaining` will be Some(1); etc.
    pub virtual_sequence_remaining: Option<usize>,
    pub output_dims: [usize; 2], // TODO: Scale system for higher rank tensors
    /// Number of active elements in the output (useful since we pad the output to `MAX_TENSOR_SIZE`).
    pub active_output_elements: usize,
}

#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum MemoryOp {
    Read(u64, u64),       // (address, value)
    Write(u64, u64, u64), // (address, old_value, new_value)
}

impl MemoryOp {
    pub fn noop_read() -> Self {
        Self::Read(0, 0)
    }

    pub fn noop_write() -> Self {
        Self::Write(0, 0, 0)
    }

    pub fn address(&self) -> u64 {
        match self {
            MemoryOp::Read(a, _) => *a,
            MemoryOp::Write(a, _, _) => *a,
        }
    }
}

impl ONNXCycle {
    pub fn ts1_vals(&self) -> Option<Vec<u64>> {
        self.build_vals(self.memory_state.ts1_val.as_ref())
    }

    pub fn ts2_vals(&self) -> Option<Vec<u64>> {
        self.build_vals(self.memory_state.ts2_val.as_ref())
    }

    pub fn ts3_vals(&self) -> Option<Vec<u64>> {
        self.build_vals(self.memory_state.ts3_val.as_ref())
    }

    pub fn td_post_vals(&self) -> Option<Vec<u64>> {
        self.build_vals(self.memory_state.td_post_val.as_ref())
    }

    /// Returns a zero-filled Vec<u64> for pre-execution values of td.
    ///
    /// Currently always zeros; may change for const opcodes.
    pub fn td_pre_vals(&self) -> Option<Vec<u64>> {
        self.build_vals(self.memory_state.td_pre_val.as_ref())
    }

    fn build_vals(&self, tensor_opt: Option<&Tensor<i32>>) -> Option<Vec<u64>> {
        tensor_opt.map(|tensor| tensor.inner.iter().map(normalize).collect())
    }

    /// Returns the optional tensor for ts1 (unmodified).
    pub fn ts1_val_raw(&self) -> Option<&Tensor<i32>> {
        self.memory_state.ts1_val.as_ref()
    }

    /// Returns the optional tensor for ts2 (unmodified).
    pub fn ts2_val_raw(&self) -> Option<&Tensor<i32>> {
        self.memory_state.ts2_val.as_ref()
    }

    /// Returns the optional tensor for ts3 (unmodified).
    pub fn ts3_val_raw(&self) -> Option<&Tensor<i32>> {
        self.memory_state.ts3_val.as_ref()
    }

    /// Returns the optional tensor for td_post (unmodified).
    pub fn td_post_val_raw(&self) -> Option<&Tensor<i32>> {
        self.memory_state.td_post_val.as_ref()
    }

    /// Returns the optional tensor for advice.
    /// # Note normalizes the advice value to u64 and pads it to `MAX_TENSOR_SIZE`.
    /// # Panics if the advice value's length exceeds `MAX_TENSOR_SIZE`.
    pub fn advice_value(&self) -> Option<Vec<u64>> {
        self.advice_value
            .as_ref()
            .map(|tensor| tensor.inner.iter().map(normalize).collect())
    }

    pub fn imm(&self) -> Option<Vec<u64>> {
        self.instr.imm()
    }
}

// converts a i32 to a u64 preserving sign-bit
// Used in the zkVM to convert raw trace values into the zkVM's 64 bit container type
pub fn normalize(value: &i32) -> u64 {
    *value as u32 as u64
}

/// Boolean flags used in Jolt's R1CS constraints (`opflags` in the Jolt paper).
/// Note that the flags below deviate somewhat from those described in Appendix A.1
/// of the Jolt paper.
#[derive(Clone, Copy, Debug, PartialEq, EnumCountMacro, EnumIter, Eq, Hash, PartialOrd, Ord)]
pub enum CircuitFlags {
    /// 1 if the first instruction operand is TS1 value; 0 otherwise.
    LeftOperandIsTs1Value,
    /// 1 if the first instruction operand is TS2 value; 0 otherwise.
    RightOperandIsTs2Value,
    /// 1 if the second instruction operand is `imm`; 0 otherwise.
    RightOperandIsImm,
    /// 1 if the first lookup operand is the sum of the two instruction operands.
    AddOperands,
    /// 1 if the first lookup operand is the difference between the two instruction operands.
    SubtractOperands,
    /// 1 if the first lookup operand is the product of the two instruction operands.
    MultiplyOperands,
    /// 1 if the lookup output is to be stored in `td` at the end of the step.
    WriteLookupOutputToTD,
    /// 1 if the instruction is "inline", as defined in Section 6.1 of the Jolt paper.
    InlineSequenceInstruction,
    /// 1 if the instruction is an assert, as defined in Section 6.1.1 of the Jolt paper.
    Assert,
    /// Used in virtual sequences; the program counter should be the same for the full sequence.
    DoNotUpdateUnexpandedPC,
    /// Is (virtual) advice instruction
    Advice,
    /// 1 if this is constant instruction; 0 otherwise.
    Const,
    /// Is noop instruction
    IsNoop,
}

pub const NUM_CIRCUIT_FLAGS: usize = CircuitFlags::COUNT;

pub trait InterleavedBitsMarker {
    fn is_interleaved_operands(&self) -> bool;
}

impl InterleavedBitsMarker for [bool; NUM_CIRCUIT_FLAGS] {
    fn is_interleaved_operands(&self) -> bool {
        !self[CircuitFlags::AddOperands]
            && !self[CircuitFlags::SubtractOperands]
            && !self[CircuitFlags::MultiplyOperands]
            && !self[CircuitFlags::Advice]
            && !self[CircuitFlags::Const]
    }
}

impl Index<CircuitFlags> for [bool; NUM_CIRCUIT_FLAGS] {
    type Output = bool;
    fn index(&self, index: CircuitFlags) -> &bool {
        &self[index as usize]
    }
}

impl IndexMut<CircuitFlags> for [bool; NUM_CIRCUIT_FLAGS] {
    fn index_mut(&mut self, index: CircuitFlags) -> &mut bool {
        &mut self[index as usize]
    }
}

impl ONNXInstr {
    pub fn no_op() -> Self {
        ONNXInstr {
            address: 0,
            opcode: ONNXOpcode::Noop,
            ts1: None,
            ts2: None,
            ts3: None,
            td: None,
            imm: None,
            virtual_sequence_remaining: None,
            active_output_elements: 0,
            output_dims: [0, 0],
        }
    }

    pub fn output_node(last_node: &ONNXInstr) -> Self {
        ONNXInstr {
            address: last_node.address + 1,
            opcode: ONNXOpcode::Output,
            ts1: last_node.td,
            ..ONNXInstr::no_op()
        }
    }

    pub fn dummy(opcode: ONNXOpcode) -> Self {
        ONNXInstr {
            address: 0,
            opcode,
            ts1: None,
            ts2: None,
            ts3: None,
            td: None,
            imm: None,
            virtual_sequence_remaining: None,
            active_output_elements: 0,
            output_dims: [0, 0],
        }
    }

    pub fn imm(&self) -> Option<Vec<u64>> {
        self.imm
            .as_ref()
            .map(|imm| imm.inner.iter().map(normalize).collect())
    }
}

// TODO: Expand the instruction set architecture (ISA):
//       For phase 1, we focus on supporting text-classification models.
//       This reduced ISA currently includes only the opcodes commonly used in such models.
//       Future phases should extend this set to support a broader range of ONNX operations.

#[derive(Clone, Hash, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
/// Operation code uniquely identifying each ONNX instruction's function
pub enum ONNXOpcode {
    Noop,
    Constant,
    Input,
    Output,
    Abs,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Relu,
    MatMult,
    Gather,
    Transpose,
    Sqrt,
    /// Used for the ReduceMean operator, which is internally converted to a
    /// combination of Sum and Div operations.
    Sum,
    MeanOfSquares,
    Sigmoid,
    Softmax,
    RebaseScale(Box<ONNXOpcode>),
    Gte,
    Reshape,
    ArgMax,
    ReduceMax,
    Select,
    Broadcast,
    AddressedNoop,

    // Virtual instructions
    VirtualAdvice,
    VirtualAssertValidSignedRemainder,
    VirtualAssertValidDiv0,
    VirtualMove,
    VirtualAssertEq,
    VirtualConst,
}
