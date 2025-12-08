use crate::instructions::{declare_onnx_instr, ElementWise, WORD_SIZE};

// Virtual
declare_onnx_instr!(name = VirtualPow2);

impl ElementWise for VirtualPow2 {
    fn exec(x: u64, _y: u64) -> u64 {
        match WORD_SIZE {
            32 => 1u64 << (x % 32),
            64 => 1u64 << (x % 64),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}

impl VirtualPow2 {
    pub fn sequence_output(x: &[u64], _y: &[u64]) -> Vec<u64> {
        x.iter().map(|&x| Self::exec(x, 0)).collect()
    }
}
