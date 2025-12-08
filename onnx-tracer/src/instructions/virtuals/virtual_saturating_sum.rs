use crate::{instructions::declare_onnx_instr, tensor::Tensor};

// Tensor-spanning
declare_onnx_instr!(name = VirtualSaturatingSum);

// TODO(AntoineF4C5): Build instruction
impl VirtualSaturatingSum {
    #[allow(unused)]
    fn exec(tensor: Tensor<i32>) -> Tensor<i32> {
        let sum = tensor
            .iter()
            .fold(0i32, |acc, &current| acc.saturating_add(current));
        Tensor::from(vec![sum].into_iter())
    }
}
