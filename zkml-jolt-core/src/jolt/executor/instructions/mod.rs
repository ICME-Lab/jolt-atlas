use crate::jolt::lookup_table::LookupTables;
use onnx_tracer::trace_types::{MemoryState, ONNXCycle, ONNXInstr, ONNXOpcode};

pub mod add;
pub mod beq;
pub mod broadcast;
pub mod div;
pub mod gte;
pub mod mul;
pub mod relu;
pub mod reshape;
pub mod rsqrt;
pub mod softmax;
pub mod sub;
pub mod virtual_advice;
pub mod virtual_assert_valid_div0;
pub mod virtual_assert_valid_signed_remainder;
pub mod virtual_const;
pub mod virtual_move;
pub mod virtual_pow2;

#[cfg(test)]
pub mod test;

pub trait InstructionLookup<const WORD_SIZE: usize> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>>;
}

pub trait VirtualInstructionSequence {
    const SEQUENCE_LENGTH: usize;
    fn virtual_sequence(instr: ONNXInstr, K: usize) -> Vec<ONNXInstr> {
        let dummy_cycle = ONNXCycle {
            instr,
            memory_state: MemoryState::default(),
            advice_value: None,
        };
        Self::virtual_trace(dummy_cycle, K)
            .into_iter()
            .map(|cycle| cycle.instr)
            .collect()
    }
    fn virtual_trace(cycle: ONNXCycle, K: usize) -> Vec<ONNXCycle>;
    fn sequence_output(x: Vec<u64>, y: Vec<u64>, inner: Option<ONNXOpcode>) -> Vec<u64>;
}
