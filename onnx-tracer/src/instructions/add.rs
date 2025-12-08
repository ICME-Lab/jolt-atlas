use crate::{
    instructions::{declare_onnx_instr, ElementWise, VirtualInstructionSequence, WORD_SIZE},
    trace_types::{AtlasCycle, ONNXCycle, ONNXOpcode},
    utils::VirtualSequenceCounter,
};

// Element-wise
declare_onnx_instr!(name = Add);

impl ElementWise for Add {
    fn exec(x: u64, y: u64) -> u64 {
        match WORD_SIZE {
            32 => (x as u32).wrapping_add(y as u32) as u64,
            64 => x.wrapping_add(y),
            _ => panic!("Unsupported WORD_SIZE: {WORD_SIZE}"),
        }
    }
}

// TODO(AntoineF4C5): Could build a macro to impl the trait the same way on all ElementWise instructions
impl VirtualInstructionSequence for Add {
    fn virtual_trace(cycle: ONNXCycle, _K: &mut VirtualSequenceCounter) -> Vec<AtlasCycle> {
        debug_assert_eq!(cycle.instr.opcode, ONNXOpcode::Add);
        vec![cycle.try_into().unwrap()]
    }

    fn sequence_output(x: &[u64], y: &[u64]) -> Vec<u64> {
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
        jolt_virtual_sequence_test::<Add>(ONNXOpcode::Add, 16);
    }
}
