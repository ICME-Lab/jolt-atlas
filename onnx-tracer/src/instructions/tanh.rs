use crate::{
    instructions::{declare_onnx_instr, VirtualInstructionSequence},
    tensor::Tensor,
    trace_types::{AtlasCycle, ONNXOpcode},
    utils::VirtualSlotCounter,
};

// Element-wise (lookup-based activation function)

declare_onnx_instr!(name = Tanh);

impl Tanh {
    #[allow(unused)]
    fn exec(subscripts: &str, inputs: &[Tensor<i32>]) -> Tensor<i32> {
        todo!()
    }
}

impl VirtualInstructionSequence for Tanh {
    fn virtual_trace(
        cycle: crate::trace_types::ONNXCycle,
        _K: &mut VirtualSlotCounter,
    ) -> Vec<AtlasCycle> {
        debug_assert!(matches!(cycle.instr.opcode, ONNXOpcode::Tanh));
        vec![cycle.try_into().unwrap()]
    }

    fn sequence_output(_x: &[u64], _y: &[u64]) -> Vec<u64> {
        unimplemented!("Proven by specialized sum-check (fp_lookups)")
    }
}
