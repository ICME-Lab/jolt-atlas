use crate::{
    instructions::{declare_onnx_instr, VirtualInstructionSequence},
    tensor::{self, Tensor},
    trace_types::{AtlasCycle, ONNXOpcode},
    utils::VirtualSlotCounter,
};

// Tensor-spanning

declare_onnx_instr!(name = Einsum);

impl Einsum {
    #[allow(unused)]
    fn exec(subscripts: &str, inputs: &[Tensor<i32>]) -> Tensor<i32> {
        tensor::ops::einsum(subscripts, inputs).unwrap()
    }
}

impl VirtualInstructionSequence for Einsum {
    fn virtual_trace(
        cycle: crate::trace_types::ONNXCycle,
        _K: &mut VirtualSlotCounter,
    ) -> Vec<AtlasCycle> {
        matches!(cycle.instr.opcode, ONNXOpcode::Einsum(_));
        vec![cycle.try_into().unwrap()]
    }

    fn sequence_output(_x: &[u64], _y: &[u64]) -> Vec<u64> {
        unimplemented!("Proven by specialized sum-check")
    }
}
