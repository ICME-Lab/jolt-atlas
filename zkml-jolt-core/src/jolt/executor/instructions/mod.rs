use jolt_core::utils::interleave_bits;
use onnx_tracer::trace_types::AtlasOpcode;

use crate::jolt::{lookup_table::LookupTables, trace::JoltONNXCycle};
use onnx_tracer::instructions::{
    add::Add,
    broadcast::Broadcast,
    constant::Constant,
    einsum::Einsum,
    eq::Eq,
    gte::Gte,
    input::Input,
    mul::Mul,
    relu::Relu,
    reshape::Reshape,
    select::Select,
    sub::Sub,
    sum::Sum,
    virtuals::{
        VirtualAdvice, VirtualAssertEq, VirtualAssertValidDiv0, VirtualAssertValidSignedRemainder,
        VirtualMove, VirtualPow2, VirtualSaturatingSum, VirtualShiftRightBitmask, VirtualSra,
    },
};

pub mod add;
pub mod broadcast;
pub mod constant;
pub mod einsum;
pub mod eq;
pub mod gte;
pub mod input;
pub mod mul;
pub mod relu;
pub mod reshape;
pub mod select;
pub mod sub;
pub mod sum;
pub mod virtual_advice;
pub mod virtual_assert_eq;
pub mod virtual_assert_valid_div0;
pub mod virtual_assert_valid_signed_remainder;
pub mod virtual_move;
pub mod virtual_pow2;
pub mod virtual_saturating_sum;
pub mod virtual_shift_right_bitmask;
pub mod virtual_sra;

#[cfg(test)]
pub mod test;

const WORD_SIZE: usize = 32;

pub trait InstructionLookup<const WORD_SIZE: usize> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>>;
}

pub trait LookupQuery<const WORD_SIZE: usize> {
    /// Returns a tuple of the instruction's inputs. If the instruction has only one input,
    /// one of the tuple values will be 0.
    fn to_instruction_inputs(&self) -> (u64, i64);

    /// Returns a tuple of the instruction's lookup operands. By default, these are the
    /// same as the instruction inputs returned by `to_instruction_inputs`, but in some cases
    /// (e.g. ADD, MUL) the instruction inputs are combined to form a single lookup operand.
    fn to_lookup_operands(&self) -> (u64, u64) {
        let (x, y) = self.to_instruction_inputs();
        (x, y as u64)
    }

    /// Converts this instruction's operands into a lookup index (as used in sparse-dense Shout).
    /// By default, interleaves the two bits of the two operands together.
    fn to_lookup_index(&self) -> u64 {
        let (x, y) = LookupQuery::<WORD_SIZE>::to_lookup_operands(self);
        interleave_bits(x as u32, y as u32)
    }

    /// Computes the output lookup entry for this instruction as a u64.
    fn to_lookup_output(&self) -> u64;
}

impl JoltONNXCycle {
    pub fn get_left_input(&self) -> u64 {
        match self.instr.opcode {
            AtlasOpcode::Noop => 0,
            AtlasOpcode::Constant => self.instr.imm,
            AtlasOpcode::VirtualAdvice => self
                .advice_value
                .expect("Expected advice value for virtual advice instruction"),
            _ => self.ts1_read(),
        }
    }

    pub fn get_right_input(&self) -> u64 {
        match self.instr.opcode {
            AtlasOpcode::Noop => 0,
            _ => self.ts2_read(),
        }
    }
}

// Helper to treat both variants of ONNXOpcode with and without inner type
macro_rules! expand_op_var {
    ($instr:ident) => {
        AtlasOpcode::$instr
    };
    ($instr:ident$type:ty) => {
        AtlasOpcode::$instr(_)
    };
}

macro_rules! define_onnx_trait_impls {
    ($($instr:ident$(($type:ty))?),* $(,)?) => {
        impl InstructionLookup<WORD_SIZE> for JoltONNXCycle {
            fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
                match self.instr.opcode {
                    AtlasOpcode::Noop => None,
                    AtlasOpcode::AddressedNoop => None,
                    $(
                        expand_op_var!($instr$($type)?) => $instr(0, 0).lookup_table(),
                    )*
                }
            }
        }

        impl LookupQuery<WORD_SIZE> for JoltONNXCycle {
            fn to_instruction_inputs(&self) -> (u64, i64) {
                let x = self.get_left_input();
                let y = self.get_right_input();
                match self.instr.opcode {
                    AtlasOpcode::Noop => (0, 0),
                    AtlasOpcode::AddressedNoop => (0, 0),
                    $(
                        expand_op_var!($instr$($type)?) => LookupQuery::<WORD_SIZE>::to_instruction_inputs(&$instr(x, y)),
                    )*
                }
            }

            fn to_lookup_operands(&self) -> (u64, u64) {
                let x = self.get_left_input();
                let y = self.get_right_input();
                match self.instr.opcode {
                    AtlasOpcode::Noop => (0, 0),
                    AtlasOpcode::AddressedNoop => (0, 0),
                    $(
                        expand_op_var!($instr$($type)?) => LookupQuery::<WORD_SIZE>::to_lookup_operands(&$instr(x, y)),
                    )*
                }
            }

            fn to_lookup_index(&self) -> u64 {
                let x = self.get_left_input();
                let y = self.get_right_input();
                match self.instr.opcode {
                    AtlasOpcode::Noop => 0,
                    AtlasOpcode::AddressedNoop => 0,
                    $(
                        expand_op_var!($instr$($type)?) => LookupQuery::<WORD_SIZE>::to_lookup_index(&$instr(x, y)),
                    )*
                }
            }

            fn to_lookup_output(&self) -> u64 {
                let x = self.get_left_input();
                let y = self.get_right_input();
                match self.instr.opcode {
                    AtlasOpcode::Noop => 0,
                    AtlasOpcode::AddressedNoop => 0,
                    $(
                        expand_op_var!($instr$($type)?) => LookupQuery::<WORD_SIZE>::to_lookup_output(&$instr(x, y)),
                    )*
                }
            }
        }
    }
}

define_onnx_trait_impls!(
    Add,
    Broadcast,
    Constant,
    Einsum(String),
    Eq,
    Gte,
    Input,
    Mul,
    Relu,
    Reshape,
    Select,
    Sub,
    Sum(usize),
    VirtualAdvice,
    VirtualAssertEq,
    VirtualAssertValidDiv0,
    VirtualAssertValidSignedRemainder,
    VirtualMove,
    VirtualPow2,
    VirtualSaturatingSum,
    VirtualShiftRightBitmask,
    VirtualSra
);
