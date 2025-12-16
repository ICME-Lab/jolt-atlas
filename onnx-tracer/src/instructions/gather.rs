use crate::{
    instructions::{declare_onnx_instr, VirtualInstructionSequence},
    tensor::Tensor,
    trace_types::{AtlasCycle, ONNXOpcode},
    utils::VirtualSlotCounter,
};

// Tensor-spanning
declare_onnx_instr!(name = Gather);

impl Gather {
    #[allow(unused)]
    fn exec(indexes: Tensor<i32>, B: Tensor<i32>) -> Tensor<i32> {
        todo!()
        // tensor::ops::gather(&B, &indexes, 0).unwrap()
    }
}

// TODO(AntoineF4C5); Create new trait for precompile-verified operations

impl VirtualInstructionSequence for Gather {
    fn virtual_trace(
        cycle: crate::trace_types::ONNXCycle,
        _K: &mut VirtualSlotCounter,
    ) -> Vec<AtlasCycle> {
        debug_assert!(matches!(cycle.instr.opcode, ONNXOpcode::Gather));
        vec![cycle.try_into().unwrap()]
    }

    fn sequence_output(_x: &[u64], _y: &[u64]) -> Vec<u64> {
        unimplemented!("Proven by specialized sum-check")
    }
}
