#![allow(non_snake_case)]

use crate::{
    instructions::{
        add::Add, broadcast::Broadcast, constant::Constant, div::Div, divi::DivI, einsum::Einsum,
        eq::Eq, gte::Gte, input::Input, mul::Mul, relu::Relu, reshape::Reshape, rsqrt::Rsqrt,
        select::Select, softmax::Softmax, sra::Sra, sub::Sub, sum::Sum,
    },
    trace_types::{AtlasCycle, AtlasInstr, MemoryState, ONNXCycle, ONNXInstr, ONNXOpcode},
    utils::VirtualSlotCounter,
};

pub mod add;
pub mod broadcast;
pub mod constant;
pub mod div;
pub mod divi;
pub mod einsum;
pub mod eq;
pub mod gte;
pub mod input;
pub mod mul;
pub mod relu;
pub mod reshape;
pub mod rsqrt;
pub mod select;
pub mod softmax;
pub mod sra;
pub mod sub;
pub mod sum;
pub mod virtuals;

#[cfg(test)]
pub mod test;

pub const WORD_SIZE: usize = 32;

// Trait for element-wise instructions
pub trait VirtualInstructionSequence {
    const SEQUENCE_LENGTH: usize = 1;
    fn virtual_sequence(
        instr: ONNXInstr,
        virtual_slot: &mut VirtualSlotCounter,
    ) -> Vec<AtlasInstr> {
        let dummy_cycle = ONNXCycle {
            instr,
            memory_state: MemoryState::default(),
            advice_value: None,
        };
        Self::virtual_trace(dummy_cycle, virtual_slot)
            .into_iter()
            .map(|cycle| cycle.instr)
            .collect()
    }
    fn virtual_trace(cycle: ONNXCycle, virtual_slot: &mut VirtualSlotCounter) -> Vec<AtlasCycle>;
    fn sequence_output(x: &[u64], y: &[u64]) -> Vec<u64>;
}

// Trait for special instructions that are treated with specific sumcheck
// TODO(AntoineF4C5): Goal is that each instruction can have its implementation so that we call the method
pub trait PrecompileInstruction {
    fn virtual_sequence(instr: ONNXInstr, _virtual_slot: &mut usize) -> Vec<AtlasInstr> {
        vec![instr
            .try_into()
            .expect("Expected a precompile-treated instruction")]
    }

    fn virtual_trace(cycle: ONNXCycle, _virtual_slot: &mut usize) -> Vec<AtlasCycle> {
        vec![cycle
            .try_into()
            .expect("Expected a precompile-treated instruction")]
    }
}

// Helper to treat both variants of ONNXOpcode with and without inner type
macro_rules! expand_op_var {
    (($instr:ident)) => {
        ONNXOpcode::$instr
    };
    (($instr:ident$type:ty)) => {
        ONNXOpcode::$instr(_)
    };
}

macro_rules! define_onnx_structs {
    ($($instr:ident$(($type:ty))?),* $(,)?) => {
        impl ONNXInstr {
            pub fn virtual_sequence(self: Self, virtual_slot: &mut VirtualSlotCounter) -> Vec<AtlasInstr> {
                let dummy_cycle = ONNXCycle {
                    instr: self,
                    memory_state: MemoryState::default(),
                    advice_value: None,
                };
                Self::virtual_trace(dummy_cycle, virtual_slot)
                    .into_iter()
                    .map(|cycle| cycle.instr)
                    .collect()
                }

            fn virtual_trace(cycle: ONNXCycle, virtual_slot: &mut VirtualSlotCounter) -> Vec<AtlasCycle> {
                match cycle.instr.opcode {
                    ONNXOpcode::Noop => panic!("Unsupported opcode: Noop"),
                    ONNXOpcode::AddressedNoop => panic!("Unsupported opcode: AddressedNoop"),
                    $(
                        expand_op_var!(($instr$($type)?)) => <$instr as VirtualInstructionSequence>::virtual_trace(cycle, virtual_slot),
                    )*
                }
            }
        }

        impl ONNXCycle {
            pub fn virtual_trace(self: Self, virtual_slot: &mut VirtualSlotCounter) -> Vec<AtlasCycle> {
                match self.instr.opcode {
                    ONNXOpcode::Noop => panic!("Unsupported opcode: Noop"),
                    ONNXOpcode::AddressedNoop => panic!("Unsupported opcode: AddressedNoop"),
                    $(
                        expand_op_var!(($instr$($type)?)) => <$instr as VirtualInstructionSequence>::virtual_trace(self, virtual_slot),
                    )*

                }
            }
        }
    }
}

define_onnx_structs!(
    Add,
    Broadcast,
    Constant,
    Div,
    DivI,
    Einsum(String),
    Eq,
    Gte,
    Input,
    Mul,
    Relu,
    Reshape,
    Rsqrt,
    Select,
    Softmax,
    Sra,
    Sub,
    Sum(usize)
);

pub trait ElementWise {
    fn exec(x: u64, y: u64) -> u64;
}

// Defines a onnx instruction as an element-wise operation on two inputs
// TODO(AntoineF4C5): Differentiate between elemen-wise instructions and tensor-wise, allow for 3-elements element-wise (Select)
macro_rules! declare_onnx_instr {
    (name   = $name:ident) => {
        #[derive(
            Copy,
            Clone,
            Debug,
            Eq,
            PartialEq,
            Hash,
            Ord,
            PartialOrd,
            serde::Serialize,
            serde::Deserialize,
        )]
        pub struct $name(pub u64, pub u64);
    };
}

pub(crate) use declare_onnx_instr;
