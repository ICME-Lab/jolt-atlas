//! Type library used to build the execution trace.
//! Used to format the bytecode and define each instr flags and memory access patterns.
//! Used by the runtime to generate an execution trace for ONNX runtime execution.

use crate::{tensor::Tensor, utils::normalize};
use rand::{rngs::StdRng, RngCore};
use serde::{Deserialize, Serialize};

/// Represents a step in the execution trace, where an execution trace is a `Vec<ONNXCycle>`.
/// Records what the VM did at a cycle of execution.
/// Constructed at each step in the VM execution cycle, documenting instr, reads & state changes (writes).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
// TODO(AntoineF4C5): Rename AtlasCycle, a cycle should be a single vm operation (hence on 1-element entries)
pub struct Cycle<T: Default> {
    pub instr: Instr<T>,
    pub memory_state: MemoryState,
    pub advice_value: Option<Tensor<i32>>,
}

impl<T: Default> Cycle<T> {
    pub fn no_op() -> Self {
        Cycle {
            instr: Instr::<T>::no_op(),
            memory_state: MemoryState::default(),
            advice_value: None,
        }
    }

    pub fn random(opcode: T, rng: &mut StdRng) -> Self {
        Cycle {
            instr: Instr::dummy(opcode),
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

    pub fn num_output_elements(&self) -> usize {
        self.instr.num_output_elements()
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

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
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
pub struct Instr<T: Default> {
    /// The program counter (PC) address of this instruction in the bytecode.
    pub address: usize,
    /// The operation code (opcode) that defines the instruction's function.
    pub opcode: T,
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
    pub output_dims: Vec<usize>,
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
impl<T: Default> Cycle<T> {
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

impl<T: Default> Instr<T> {
    pub fn no_op() -> Self {
        Self::default()
    }

    // pub fn output_node(last_node: &AtlasInstr) -> Self {
    //     AtlasInstr {
    //         address: last_node.address + 1,
    //         opcode: AtlasOpcode::Output,
    //         ts1: last_node.td,
    //         ..AtlasInstr::no_op()
    //     }
    // }

    pub fn dummy(opcode: T) -> Self {
        Instr {
            opcode,
            ..Self::no_op()
        }
    }

    pub fn imm(&self) -> Option<Vec<u64>> {
        self.imm
            .as_ref()
            .map(|imm| imm.inner.iter().map(normalize).collect())
    }

    pub fn num_output_elements(&self) -> usize {
        self.output_dims.iter().product()
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
/// Operation code uniquely identifying each ONNX instruction's function
pub enum ONNXOpcode {
    #[default]
    Noop,
    Constant,
    Input,
    // Output,
    Add,
    Sub,
    Mul,
    Div,
    DivI,
    Relu,
    Rsqrt,
    Einsum(String),
    Sum(usize),
    Sra,
    /// Used for the ReduceMean operator, which is internally converted to a
    /// combination of ReduceSum and Div operations.
    // MeanOfSquares,
    // Sigmoid,
    Softmax,
    Gte,
    Eq,
    Erf,
    Tanh,
    Reshape,
    Select,
    Broadcast,
    AddressedNoop,
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
/// Operation code uniquely identifying each VM instruction's function
pub enum AtlasOpcode {
    #[default]
    Noop,
    Constant,
    Input,
    // Output,
    Add,
    Sub,
    Mul,
    Relu,
    Einsum(String),
    Sum(usize),
    Gte,
    Eq,
    Reshape,
    Select,
    Broadcast,
    AddressedNoop,
    Erf,
    Tanh,

    // Virtual instructions
    VirtualAdvice,
    VirtualAssertEq,
    VirtualAssertValidDiv0,
    VirtualAssertValidSignedRemainder,
    VirtualMove,
    VirtualPow2,
    VirtualSaturatingSum,
    VirtualShiftRightBitmask,
    VirtualSra,
}

// Aliases for structures holding information for instructions from ONNX
pub type ONNXCycle = Cycle<ONNXOpcode>;
pub type ONNXInstr = Instr<ONNXOpcode>;

// Aliases for structures holding information for instructions in the Atlas VM
pub type AtlasCycle = Cycle<AtlasOpcode>;
pub type AtlasInstr = Instr<AtlasOpcode>;

// Default mapping for ONNX opcodes that are directly supported by the zkVM
impl TryFrom<ONNXOpcode> for AtlasOpcode {
    type Error = &'static str;

    fn try_from(value: ONNXOpcode) -> Result<Self, Self::Error> {
        let atlas_opcode = match value {
            ONNXOpcode::Noop => AtlasOpcode::Noop,
            ONNXOpcode::Add => AtlasOpcode::Add,
            ONNXOpcode::Broadcast => AtlasOpcode::Broadcast,
            ONNXOpcode::Constant => AtlasOpcode::Constant,
            ONNXOpcode::Eq => AtlasOpcode::Eq,
            ONNXOpcode::Erf => AtlasOpcode::Erf,
            ONNXOpcode::Tanh => AtlasOpcode::Tanh,
            ONNXOpcode::Gte => AtlasOpcode::Gte,
            ONNXOpcode::Input => AtlasOpcode::Input,
            ONNXOpcode::Mul => AtlasOpcode::Mul,
            ONNXOpcode::Relu => AtlasOpcode::Relu,
            ONNXOpcode::Reshape => AtlasOpcode::Reshape,
            ONNXOpcode::Sub => AtlasOpcode::Sub,
            // Those are treated by specialized sum_check precompiles
            ONNXOpcode::Einsum(subscripts) => AtlasOpcode::Einsum(subscripts),
            ONNXOpcode::Sum(axis) => AtlasOpcode::Sum(axis),
            // Those are treated with circuit flags
            ONNXOpcode::Select => AtlasOpcode::Select,
            _ => return Err("Opcode is not directly supported by the zkvm and should be virtualized with a sequence of vm-compatible opcodes.")
        };
        Ok(atlas_opcode)
    }
}

// Default mapping for ONNX instructions that are directly supported by the zkVM
impl TryFrom<Instr<ONNXOpcode>> for Instr<AtlasOpcode> {
    type Error = &'static str;

    fn try_from(value: Instr<ONNXOpcode>) -> Result<Self, Self::Error> {
        let atlas_opcode: AtlasOpcode = value.opcode.try_into()?;
        Ok(Instr {
            opcode: atlas_opcode,
            address: value.address,
            ts1: value.ts1,
            ts2: value.ts2,
            ts3: value.ts3,
            td: value.td,
            imm: value.imm,
            virtual_sequence_remaining: value.virtual_sequence_remaining,
            output_dims: value.output_dims,
        })
    }
}

// Default mapping for ONNX cycles that are directly supported by the zkVM
impl TryFrom<Cycle<ONNXOpcode>> for Cycle<AtlasOpcode> {
    type Error = &'static str;

    fn try_from(value: Cycle<ONNXOpcode>) -> Result<Self, Self::Error> {
        let atlas_instr: Instr<AtlasOpcode> = value.instr.try_into()?;
        Ok(Cycle {
            instr: atlas_instr,
            memory_state: value.memory_state,
            advice_value: None, // value.advice_value,
        })
    }
}
