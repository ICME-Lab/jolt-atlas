use crate::{
    instructions::{declare_onnx_instr, VirtualInstructionSequence},
    tensor::{self, Tensor},
    trace_types::{AtlasCycle, ONNXOpcode},
    utils::VirtualSlotCounter,
};

// Tensor-spanning
declare_onnx_instr!(name = Sum);

impl Sum {
    #[allow(unused)]
    fn exec(tensor: Tensor<i32>, axes: &[usize]) -> Tensor<i32> {
        tensor::ops::sum_axes(&tensor, axes).unwrap()
    }
}

// TODO(AntoineF4C5); Create new trait for precompile-verified operations

impl VirtualInstructionSequence for Sum {
    fn virtual_trace(
        cycle: crate::trace_types::ONNXCycle,
        _K: &mut VirtualSlotCounter,
    ) -> Vec<AtlasCycle> {
        matches!(cycle.instr.opcode, ONNXOpcode::Sum(_));
        vec![cycle.try_into().unwrap()]
    }

    fn sequence_output(_x: &[u64], _y: &[u64]) -> Vec<u64> {
        unimplemented!("Proven by specialized sum-check")
    }
}
