use crate::instructions::{declare_onnx_instr, ElementWise, WORD_SIZE};

// Virtual
declare_onnx_instr!(name = VirtualShiftRightBitmask);

impl ElementWise for VirtualShiftRightBitmask {
    fn exec(x: u64, _y: u64) -> u64 {
        match WORD_SIZE {
            32 => {
                let shift = x % 32;
                let ones = (1u64 << (32 - shift)) - 1;
                ones << shift
            }
            64 => {
                let shift = x % 64;
                let ones = (1u128 << (64 - shift)) - 1;
                (ones << shift) as u64
            }
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}
impl VirtualShiftRightBitmask {
    pub fn sequence_output(x: &[u64], _y: &[u64]) -> Vec<u64> {
        x.iter().map(|&x| Self::exec(x, 0)).collect()
    }
}
