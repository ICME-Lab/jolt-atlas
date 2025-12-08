use crate::instructions::{declare_onnx_instr, ElementWise, WORD_SIZE};

// Virtual
declare_onnx_instr!(name = VirtualAssertValidSignedRemainder);

impl ElementWise for VirtualAssertValidSignedRemainder {
    fn exec(remainder: u64, divisor: u64) -> u64 {
        match WORD_SIZE {
            32 => {
                let remainder = remainder as i32;
                let divisor = divisor as i32;
                let is_remainder_zero = remainder == 0;
                let is_divisor_zero = divisor == 0;

                if is_remainder_zero || is_divisor_zero {
                    1
                } else {
                    let remainder_sign = remainder >> 31;
                    let divisor_sign = divisor >> 31;
                    (remainder.unsigned_abs() < divisor.unsigned_abs()
                        && remainder_sign == divisor_sign)
                        .into()
                }
            }
            64 => {
                let remainder = remainder as i64;
                let divisor = divisor as i64;
                let is_remainder_zero = remainder == 0;
                let is_divisor_zero = divisor == 0;

                if is_remainder_zero || is_divisor_zero {
                    1
                } else {
                    let remainder_sign = remainder >> 63;
                    let divisor_sign = divisor >> 63;
                    (remainder.unsigned_abs() < divisor.unsigned_abs()
                        && remainder_sign == divisor_sign)
                        .into()
                }
            }
            _ => panic!("Unsupported WORD_SIZE: {WORD_SIZE}"),
        }
    }
}

impl VirtualAssertValidSignedRemainder {
    pub fn sequence_output(x: &[u64], y: &[u64]) -> Vec<u64> {
        x.iter()
            .zip(y.iter())
            .map(|(&x, &y)| Self::exec(x, y))
            .collect()
    }
}
