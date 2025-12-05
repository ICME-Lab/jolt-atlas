use crate::{
    instructions::{declare_onnx_instr, ElementWise, VirtualInstructionSequence},
    trace_types::{AtlasCycle, ONNXOpcode},
    utils::VirtualSlotCounter,
};

// Element-wise
declare_onnx_instr!(name = Input);

// Input value is fed in right operand
impl ElementWise for Input {
    fn exec(_x: u64, y: u64) -> u64 {
        y
    }
}

impl VirtualInstructionSequence for Input {
    fn virtual_trace(
        cycle: crate::trace_types::ONNXCycle,
        _K: &mut VirtualSlotCounter,
    ) -> Vec<AtlasCycle> {
        assert_eq!(cycle.instr.opcode, ONNXOpcode::Input);
        vec![cycle.try_into().unwrap()]
    }

    fn sequence_output(_x: &[u64], y: &[u64]) -> Vec<u64> {
        y.iter().map(|&y| Self::exec(0, y)).collect()
    }
}

#[cfg(test)]
mod test {
    use crate::instructions::test::jolt_virtual_sequence_test;

    use super::*;

    #[test]
    fn virtual_sequence_32() {
        jolt_virtual_sequence_test::<Input>(ONNXOpcode::Input, 16);
    }
}
