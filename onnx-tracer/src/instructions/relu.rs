use crate::{
    instructions::{declare_onnx_instr, ElementWise, VirtualInstructionSequence, WORD_SIZE},
    trace_types::{AtlasCycle, ONNXOpcode},
    utils::VirtualSlotCounter,
};

// Element-wise
declare_onnx_instr!(name = Relu);

impl ElementWise for Relu {
    fn exec(x: u64, _y: u64) -> u64 {
        let is_neg = match WORD_SIZE {
            32 => (x as i32) < 0,
            64 => (x as i64) < 0,
            _ => panic!("Unsupported WORD_SIZE: {WORD_SIZE}"),
        };
        if is_neg {
            0
        } else {
            x
        }
    }
}

impl VirtualInstructionSequence for Relu {
    fn virtual_trace(
        cycle: crate::trace_types::ONNXCycle,
        _K: &mut VirtualSlotCounter,
    ) -> Vec<AtlasCycle> {
        debug_assert_eq!(cycle.instr.opcode, ONNXOpcode::Relu);
        vec![cycle.try_into().unwrap()]
    }

    fn sequence_output(x: &[u64], _y: &[u64]) -> Vec<u64> {
        x.iter().map(|&x| Self::exec(x, 0)).collect()
    }
}

#[cfg(test)]
mod test {
    use crate::instructions::test::jolt_virtual_sequence_test;

    use super::*;

    #[test]
    fn virtual_sequence_32() {
        jolt_virtual_sequence_test::<Relu>(ONNXOpcode::Relu, 16);
    }
}
