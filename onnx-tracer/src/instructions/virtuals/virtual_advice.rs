use crate::instructions::{declare_onnx_instr, ElementWise};

// Virtual
declare_onnx_instr!(name = VirtualAdvice);

impl ElementWise for VirtualAdvice {
    fn exec(x: u64, _y: u64) -> u64 {
        x
    }
}

// TODO(AntoineF4C5); Create new trait for virtual operations
impl VirtualAdvice {
    pub fn sequence_output(x: &[u64], _y: &[u64]) -> Vec<u64> {
        x.iter().map(|&x| Self::exec(x, 0)).collect()
    }
}
