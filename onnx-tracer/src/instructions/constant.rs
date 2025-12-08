use crate::{
    instructions::{declare_onnx_instr, ElementWise, VirtualInstructionSequence},
    trace_types::{AtlasCycle, ONNXCycle, ONNXOpcode},
    utils::VirtualSlotCounter,
};

// Element-wise
declare_onnx_instr!(name = Constant);

// Constant value is fed in right operand
impl ElementWise for Constant {
    fn exec(_x: u64, y: u64) -> u64 {
        y
    }
}

impl VirtualInstructionSequence for Constant {
    fn virtual_trace(cycle: ONNXCycle, _K: &mut VirtualSlotCounter) -> Vec<AtlasCycle> {
        debug_assert_eq!(cycle.instr.opcode, ONNXOpcode::Constant);
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
        jolt_virtual_sequence_test::<Constant>(ONNXOpcode::Constant, 16);
    }
}
