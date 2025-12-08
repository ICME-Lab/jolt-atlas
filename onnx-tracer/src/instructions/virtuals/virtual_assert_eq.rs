use crate::instructions::{declare_onnx_instr, ElementWise, WORD_SIZE};

// Virtual
declare_onnx_instr!(name = VirtualAssertEq);

impl ElementWise for VirtualAssertEq {
    fn exec(x: u64, y: u64) -> u64 {
        match WORD_SIZE {
            32 => (x as u32 == y as u32).into(),
            64 => (x == y).into(),
            _ => panic!("Unsupported WORD_SIZE: {WORD_SIZE}"),
        }
    }
}

impl VirtualAssertEq {
    pub fn sequence_output(x: &[u64], y: &[u64]) -> Vec<u64> {
        x.iter()
            .zip(y.iter())
            .map(|(&x, &y)| Self::exec(x, y))
            .collect()
    }
}
