use crate::instructions::{declare_onnx_instr, ElementWise, WORD_SIZE};

// Virtual
declare_onnx_instr!(name = VirtualAssertValidDiv0);

impl ElementWise for VirtualAssertValidDiv0 {
    fn exec(divisor: u64, quotient: u64) -> u64 {
        match WORD_SIZE {
            32 => {
                let quotient = quotient as i32;
                if divisor == 0 {
                    (quotient == -1).into()
                } else {
                    1
                }
            }
            64 => {
                let quotient = quotient as i64;
                if divisor == 0 {
                    (quotient == -1).into()
                } else {
                    1
                }
            }
            _ => panic!("Unsupported WORD_SIZE: {WORD_SIZE}"),
        }
    }
}

impl VirtualAssertValidDiv0 {
    pub fn sequence_output(x: &[u64], y: &[u64]) -> Vec<u64> {
        x.iter()
            .zip(y.iter())
            .map(|(&x, &y)| Self::exec(x, y))
            .collect()
    }
}
