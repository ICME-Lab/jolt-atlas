use crate::{
    instructions::{declare_onnx_instr, VirtualInstructionSequence},
    tensor::Tensor,
    trace_types::{AtlasCycle, ONNXCycle, ONNXOpcode},
    utils::VirtualSequenceCounter,
};

// Tensor-spanning
declare_onnx_instr!(name = Broadcast);

impl Broadcast {
    #[allow(unused)]
    fn exec(tensor: Tensor<i32>, shape: &[usize]) -> Tensor<i32> {
        tensor.expand(shape).unwrap()
    }
}

impl VirtualInstructionSequence for Broadcast {
    fn virtual_trace(cycle: ONNXCycle, _K: &mut VirtualSequenceCounter) -> Vec<AtlasCycle> {
        assert_eq!(cycle.instr.opcode, ONNXOpcode::Broadcast);
        vec![cycle.try_into().unwrap()]
    }

    fn sequence_output(_x: &[u64], _y: &[u64]) -> Vec<u64> {
        unimplemented!("Proven by specialized sum-check")
    }
}
