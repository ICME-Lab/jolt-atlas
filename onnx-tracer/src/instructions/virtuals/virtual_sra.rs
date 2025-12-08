use crate::instructions::{declare_onnx_instr, ElementWise, WORD_SIZE};

// Virtual
declare_onnx_instr!(name = VirtualSra);

impl ElementWise for VirtualSra {
    fn exec(x: u64, y: u64) -> u64 {
        match WORD_SIZE {
            32 => {
                let shift = WORD_SIZE as u32 - (!(y as u32)).leading_zeros();
                ((x as i32) >> shift) as u32 as u64
            }
            64 => {
                let shift = WORD_SIZE as u32 - (!y).leading_zeros();
                ((x as i64) >> shift) as u64
            }
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}

impl VirtualSra {
    pub fn sequence_output(x: &[u64], y: &[u64]) -> Vec<u64> {
        x.iter()
            .zip(y.iter())
            .map(|(&x, &y)| Self::exec(x, y))
            .collect()
    }
}
