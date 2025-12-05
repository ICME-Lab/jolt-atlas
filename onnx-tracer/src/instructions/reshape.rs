use crate::{
    instructions::{declare_onnx_instr, VirtualInstructionSequence},
    tensor::Tensor,
    trace_types::{AtlasCycle, ONNXOpcode},
    utils::VirtualSlotCounter,
};

// Tensor-spanning
declare_onnx_instr!(name = Reshape);

impl Reshape {
    #[allow(unused)]
    fn exec(mut tensor: Tensor<i32>, shape: &[usize]) -> Tensor<i32> {
        tensor.reshape(shape).unwrap();
        tensor
    }
}

impl VirtualInstructionSequence for Reshape {
    fn virtual_trace(
        cycle: crate::trace_types::ONNXCycle,
        _K: &mut VirtualSlotCounter,
    ) -> Vec<AtlasCycle> {
        assert_eq!(cycle.instr.opcode, ONNXOpcode::Reshape);
        vec![cycle.try_into().unwrap()]
    }

    fn sequence_output(x: &[u64], _y: &[u64]) -> Vec<u64> {
        x.to_vec()
    }
}

#[cfg(test)]
mod test {
    use crate::instructions::test::jolt_virtual_sequence_test;

    use super::*;

    #[test]
    fn virtual_sequence_32() {
        jolt_virtual_sequence_test::<Reshape>(ONNXOpcode::Reshape, 16);
    }
}
