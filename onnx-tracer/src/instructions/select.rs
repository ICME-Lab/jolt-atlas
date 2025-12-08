use crate::{
    instructions::{declare_onnx_instr, VirtualInstructionSequence},
    trace_types::{AtlasCycle, ONNXOpcode},
    utils::VirtualSlotCounter,
};

// Element-wise
declare_onnx_instr!(name = Select);

impl Select {
    #[allow(unused)]
    fn exec(cond: u64, x: u64, y: u64) -> u64 {
        if cond != 0 {
            x
        } else {
            y
        }
    }
}

impl VirtualInstructionSequence for Select {
    fn virtual_trace(
        cycle: crate::trace_types::ONNXCycle,
        _K: &mut VirtualSlotCounter,
    ) -> Vec<AtlasCycle> {
        debug_assert!(matches!(cycle.instr.opcode, ONNXOpcode::Select));
        vec![cycle.try_into().unwrap()]
    }

    fn sequence_output(_x: &[u64], _y: &[u64]) -> Vec<u64> {
        unimplemented!("Proven by specialized sum-check")
    }
}
