use crate::{
    instructions::{declare_onnx_instr, ElementWise, VirtualInstructionSequence, WORD_SIZE},
    trace_types::{AtlasCycle, ONNXOpcode},
    utils::VirtualSlotCounter,
};

// Element-wise
declare_onnx_instr!(name = Gte);

impl ElementWise for Gte {
    fn exec(x: u64, y: u64) -> u64 {
        match WORD_SIZE {
            32 => (x as i32 >= y as i32).into(),
            64 => (x as i64 >= y as i64).into(),
            _ => panic!("Unsupported WORD_SIZE: {WORD_SIZE}"),
        }
    }
}

impl VirtualInstructionSequence for Gte {
    fn virtual_trace(
        cycle: crate::trace_types::ONNXCycle,
        _K: &mut VirtualSlotCounter,
    ) -> Vec<AtlasCycle> {
        assert_eq!(cycle.instr.opcode, ONNXOpcode::Gte);
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
        jolt_virtual_sequence_test::<Gte>(ONNXOpcode::Gte, 16);
    }
}
