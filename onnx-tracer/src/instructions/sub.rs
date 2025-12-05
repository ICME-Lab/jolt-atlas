use crate::{
    instructions::{declare_onnx_instr, ElementWise, VirtualInstructionSequence, WORD_SIZE},
    trace_types::{AtlasCycle, ONNXOpcode},
    utils::VirtualSlotCounter,
};

// Element-wise
declare_onnx_instr!(name = Sub);

impl ElementWise for Sub {
    fn exec(x: u64, y: u64) -> u64 {
        match WORD_SIZE {
            32 => (x as i32).wrapping_sub(y as i32) as u32 as u64,
            64 => x.wrapping_sub(y),
            _ => panic!("Unsupported WORD_SIZE: {WORD_SIZE}"),
        }
    }
}

impl VirtualInstructionSequence for Sub {
    fn virtual_trace(
        cycle: crate::trace_types::ONNXCycle,
        _K: &mut VirtualSlotCounter,
    ) -> Vec<AtlasCycle> {
        assert_eq!(cycle.instr.opcode, ONNXOpcode::Sub);
        vec![cycle.try_into().unwrap()]
    }

    fn sequence_output(x: &[u64], y: &[u64]) -> Vec<u64> {
        assert_eq!(x.len(), y.len());
        x.iter()
            .zip(y.iter())
            .map(|(&x, &y)| Self::exec(x, y))
            .collect()
    }
}

#[cfg(test)]
mod test {
    use crate::instructions::test::jolt_virtual_sequence_test;

    use super::*;

    #[test]
    fn virtual_sequence_32() {
        jolt_virtual_sequence_test::<Sub>(ONNXOpcode::Sub, 16);
    }
}
