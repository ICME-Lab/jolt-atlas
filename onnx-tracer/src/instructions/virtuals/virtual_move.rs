use crate::instructions::{declare_onnx_instr, ElementWise};

// Virtual
declare_onnx_instr!(name = VirtualMove);

impl ElementWise for VirtualMove {
    fn exec(x: u64, _y: u64) -> u64 {
        x
    }
}

impl VirtualMove {
    pub fn sequence_output(x: &[u64], _y: &[u64]) -> Vec<u64> {
        x.iter().map(|&x| Self::exec(x, 0)).collect()
    }
}
