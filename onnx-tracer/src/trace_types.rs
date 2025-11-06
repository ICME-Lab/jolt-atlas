//! Type library used to build the execution trace.
//! Used to format the bytecode and define each instr flags and memory access patterns.
//! Used by the runtime to generate an execution trace for ONNX runtime execution.

use crate::tensor::Tensor;
use rand::{rngs::StdRng, RngCore};
use serde::{Deserialize, Serialize};
use std::fmt;
use tabled::Tabled;

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
    pub output_dims: Vec<usize>,
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

impl ONNXInstr {
    pub fn no_op() -> Self {
        Self::default()
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
            opcode,
            ..Self::no_op()
        }
    }

    pub fn imm(&self) -> Option<Vec<u64>> {
        self.imm
            .as_ref()
            .map(|imm| imm.inner.iter().map(normalize).collect())
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
/// Operation code uniquely identifying each ONNX instruction's function
pub enum ONNXOpcode {
    #[default]
    Noop,
    Constant,
    Input,
    Output,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Relu,
    Rsqrt,
    MatVec,
    MatMult,
    BatchedMatMult,
    Einsum(String),
    Sum(usize),
    Gather,
    Transpose,
    Sqrt,
    /// Used for the ReduceMean operator, which is internally converted to a
    /// combination of ReduceSum and Div operations.
    ReduceSum,
    MeanOfSquares,
    Sigmoid,
    Softmax,
    RebaseScale(Box<ONNXOpcode>),
    Gte,
    Eq,
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
    VirtualPow2,
    VirtualSaturatingSum,
}

/// Helper function to format optional values for display
fn display_option<T: fmt::Display>(opt: &Option<T>) -> String {
    match opt {
        Some(val) => val.to_string(),
        None => String::new(),
    }
}

/// Helper function to format immediate tensors with truncation for large values
fn display_imm(imm: &Option<Tensor<i32>>) -> String {
    match imm {
        None => String::new(),
        Some(tensor) => {
            const MAX_DISPLAY: usize = 6;
            const SHOW_EACH_SIDE: usize = 2;
            let len = tensor.inner.len();
            if len <= MAX_DISPLAY {
                format!("{:?}", tensor.inner)
            } else {
                let start: String = tensor
                    .inner
                    .iter()
                    .take(SHOW_EACH_SIDE)
                    .map(|n| format!("{n}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let end: String = tensor
                    .inner
                    .iter()
                    .skip(len - SHOW_EACH_SIDE)
                    .map(|n| format!("{n}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("[{start}...{end}] ({len})")
            }
        }
    }
}

/// Helper function to format inputs as a compact string
fn display_inputs(ts1: &Option<usize>, ts2: &Option<usize>, ts3: &Option<usize>) -> String {
    let mut inputs = Vec::new();
    if let Some(t1) = ts1 {
        inputs.push(format!("ts1={t1}"));
    }
    if let Some(t2) = ts2 {
        inputs.push(format!("ts2={t2}"));
    }
    if let Some(t3) = ts3 {
        inputs.push(format!("ts3={t3}"));
    }
    if inputs.is_empty() {
        String::new()
    } else {
        inputs.join(", ")
    }
}

impl fmt::Debug for ONNXInstr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ONNXInstr")
            .field("address", &self.address)
            .field("opcode", &self.opcode)
            .field("ts1", &self.ts1)
            .field("ts2", &self.ts2)
            .field("ts3", &self.ts3)
            .field("td", &self.td)
            .field("imm", &display_imm(&self.imm))
            .field("virtual_seq_remaining", &self.virtual_sequence_remaining)
            .field("output_dims", &self.output_dims)
            .field("active_output_elements", &self.active_output_elements)
            .finish()
    }
}

impl Tabled for ONNXInstr {
    const LENGTH: usize = 7;

    fn headers() -> Vec<std::borrow::Cow<'static, str>> {
        vec![
            std::borrow::Cow::Borrowed("address"),
            std::borrow::Cow::Borrowed("opcode"),
            std::borrow::Cow::Borrowed("inputs"),
            std::borrow::Cow::Borrowed("td"),
            std::borrow::Cow::Borrowed("imm"),
            std::borrow::Cow::Borrowed("output_dims"),
            std::borrow::Cow::Borrowed("active_elems"),
        ]
    }

    fn fields(&self) -> Vec<std::borrow::Cow<'_, str>> {
        vec![
            std::borrow::Cow::Owned(self.address.to_string()),
            std::borrow::Cow::Owned(format!("{:?}", self.opcode)),
            std::borrow::Cow::Owned(display_inputs(&self.ts1, &self.ts2, &self.ts3)),
            std::borrow::Cow::Owned(display_option(&self.td)),
            std::borrow::Cow::Owned(display_imm(&self.imm)),
            std::borrow::Cow::Owned(format!("{:?}", self.output_dims)),
            std::borrow::Cow::Owned(self.active_output_elements.to_string()),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_display_imm() {
        let small_imm = Tensor::new(Some(&[1, 2, 3]), &[1, 3]).ok();
        let large_imm = Tensor::new(Some(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), &[1, 10]).ok();
        let none_imm: Option<Tensor<i32>> = None;
        assert_eq!(display_imm(&small_imm), "[1, 2, 3]");
        assert_eq!(display_imm(&large_imm), "[0, 1...8, 9] (10)");
        assert_eq!(display_imm(&none_imm), "");
    }
}
