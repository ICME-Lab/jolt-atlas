use onnx_tracer::trace_types::{MemoryState, ONNXCycle, ONNXInstr, ONNXOpcode};

pub mod add;
pub mod argmax;
pub mod beq;
pub mod div;
pub mod ge;
pub mod mul;
pub mod output;
pub mod rebase_scale;
pub mod reduce_max;
pub mod reduce_sum;
pub mod relu;
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

pub trait VirtualInstructionSequence {
    const SEQUENCE_LENGTH: usize;
    fn virtual_sequence(instr: ONNXInstr) -> Vec<ONNXInstr> {
        let dummy_cycle = ONNXCycle {
            instr,
            memory_state: MemoryState::default(),
            advice_value: None,
        };
        Self::virtual_trace(dummy_cycle)
            .into_iter()
            .map(|cycle| cycle.instr)
            .collect()
    }
    fn virtual_trace(cycle: ONNXCycle) -> Vec<ONNXCycle>;
    fn sequence_output(x: Vec<u64>, y: Vec<u64>, inner: Option<ONNXOpcode>) -> Vec<u64>;
}
