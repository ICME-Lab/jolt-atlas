use crate::{
    instructions::{declare_onnx_instr, ElementWise, WORD_SIZE},
    tensor::Tensor,
    trace_types::{AtlasCycle, AtlasInstr, AtlasOpcode, MemoryState, ONNXCycle, ONNXOpcode},
    utils::{u64_vec_to_i32_iter, VirtualSequenceCounter, VirtualSlotCounter},
};

use super::{
    add::Add, eq::Eq, mul::Mul, virtuals::VirtualAdvice, virtuals::VirtualAssertValidDiv0,
    virtuals::VirtualAssertValidSignedRemainder, VirtualInstructionSequence,
};

// Expandable
declare_onnx_instr!(name = Div);

impl ElementWise for Div {
    fn exec(x: u64, y: u64) -> u64 {
        match WORD_SIZE {
            32 => {
                let x = x as i32;
                let y = y as i32;
                if y == 0 {
                    return -1i32 as u32 as u64;
                }
                let mut quotient = x / y;
                let remainder = x % y;
                if (remainder < 0 && y > 0) || (remainder > 0 && y < 0) {
                    quotient -= 1;
                }
                quotient as u32 as u64
            }
            64 => {
                let x = x as i64;
                let y = y as i64;
                if y == 0 {
                    return -1i64 as u64;
                }
                let mut quotient = x / y;
                let remainder = x % y;
                if (remainder < 0 && y > 0) || (remainder > 0 && y < 0) {
                    quotient -= 1;
                }
                quotient as u64
            }
            _ => panic!("Unsupported WORD_SIZE: {WORD_SIZE}"),
        }
    }
}

impl VirtualInstructionSequence for Div {
    const SEQUENCE_LENGTH: usize = 8;

    fn virtual_trace(cycle: ONNXCycle, virtual_slot: &mut VirtualSlotCounter) -> Vec<AtlasCycle> {
        assert_eq!(cycle.instr.opcode, ONNXOpcode::Div);
        let num_outputs = cycle.instr.num_output_elements();

        // If DIV is part of a longer virtual sequence, recover the counter to continue decrementing it
        let remaining = cycle
            .instr
            .virtual_sequence_remaining
            .unwrap_or(Self::SEQUENCE_LENGTH);
        assert!(
            remaining >= Self::SEQUENCE_LENGTH,
            "Not enough remaining virtual sequence steps"
        );

        let mut counter = VirtualSequenceCounter::new(remaining);

        // DIV source registers
        let r_x = cycle.instr.ts1;
        let r_y = cycle.instr.ts2;
        let td = cycle.instr.td;

        let v_q = Some(virtual_slot.inc());
        let v_r = Some(virtual_slot.inc());
        let v_qy = Some(virtual_slot.inc());
        let v_0 = Some(virtual_slot.inc());

        // DIV operands
        let x = cycle.ts1_vals().unwrap_or(vec![0; num_outputs]);
        let y = cycle.ts2_vals().unwrap_or(vec![0; num_outputs]);
        let mut virtual_trace = vec![];

        let (quotient, remainder) = {
            let mut quotient_tensor = vec![0; num_outputs];
            let mut remainder_tensor = vec![0; num_outputs];
            for i in 0..num_outputs {
                let x = x[i];
                let y = y[i];
                let (quotient, remainder) = match WORD_SIZE {
                    32 => {
                        if y == 0 {
                            (u32::MAX as u64, x)
                        } else {
                            let mut quotient = x as i32 / y as i32;
                            let mut remainder = x as i32 % y as i32;
                            if (remainder < 0 && (y as i32) > 0)
                                || (remainder > 0 && (y as i32) < 0)
                            {
                                remainder += y as i32;
                                quotient -= 1;
                            }
                            (quotient as u32 as u64, remainder as u32 as u64)
                        }
                    }
                    64 => {
                        if y == 0 {
                            (u64::MAX, x)
                        } else {
                            let mut quotient = x as i64 / y as i64;
                            let mut remainder = x as i64 % y as i64;
                            if (remainder < 0 && (y as i64) > 0)
                                || (remainder > 0 && (y as i64) < 0)
                            {
                                remainder += y as i64;
                                quotient -= 1;
                            }
                            (quotient as u64, remainder as u64)
                        }
                    }
                    _ => panic!("Unsupported WORD_SIZE: {WORD_SIZE}",),
                };
                quotient_tensor[i] = quotient;
                remainder_tensor[i] = remainder;
            }
            (quotient_tensor, remainder_tensor)
        };

        let q = VirtualAdvice::sequence_output(&quotient, &[]);
        virtual_trace.push(AtlasCycle {
            instr: AtlasInstr {
                address: cycle.instr.address,
                opcode: AtlasOpcode::VirtualAdvice,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_q,
                imm: None,
                virtual_sequence_remaining: Some(counter.dec()),
                output_dims: cycle.instr.output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: None,
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&q))),
            },
            advice_value: Some(Tensor::from(u64_vec_to_i32_iter(&quotient))),
        });

        let r = VirtualAdvice::sequence_output(&remainder, &[]);
        virtual_trace.push(AtlasCycle {
            instr: AtlasInstr {
                address: cycle.instr.address,
                opcode: AtlasOpcode::VirtualAdvice,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_r,
                imm: None,
                virtual_sequence_remaining: Some(counter.dec()),
                output_dims: cycle.instr.output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: None,
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&r))),
            },
            advice_value: Some(Tensor::from(u64_vec_to_i32_iter(&remainder))),
        });

        let is_valid: Vec<u64> = VirtualAssertValidSignedRemainder::sequence_output(&r, &y);
        is_valid.iter().for_each(|&valid| {
            assert_eq!(valid, 1, "Invalid signed remainder detected");
        });
        virtual_trace.push(AtlasCycle {
            instr: AtlasInstr {
                address: cycle.instr.address,
                opcode: AtlasOpcode::VirtualAssertValidSignedRemainder,
                ts1: v_r,
                ts2: r_y,
                ts3: None,
                td: None,
                imm: None,
                virtual_sequence_remaining: Some(counter.dec()),
                output_dims: cycle.instr.output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&r))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&y))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: None,
            },
            advice_value: None,
        });

        let is_valid: Vec<u64> = VirtualAssertValidDiv0::sequence_output(&y, &q);
        is_valid.iter().for_each(|&valid| {
            assert_eq!(valid, 1, "Invalid division by zero detected");
        });
        virtual_trace.push(AtlasCycle {
            instr: AtlasInstr {
                address: cycle.instr.address,
                opcode: AtlasOpcode::VirtualAssertValidDiv0,
                ts1: r_y,
                ts2: v_q,
                ts3: None,
                td: None,
                imm: None,
                virtual_sequence_remaining: Some(counter.dec()),
                output_dims: cycle.instr.output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&y))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&q))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: None,
            },
            advice_value: None,
        });

        let q_y = Mul::sequence_output(&q, &y);
        virtual_trace.push(AtlasCycle {
            instr: AtlasInstr {
                address: cycle.instr.address,
                opcode: AtlasOpcode::Mul,
                ts1: v_q,
                ts2: r_y,
                ts3: None,
                td: v_qy,
                imm: None,
                virtual_sequence_remaining: Some(counter.dec()),
                output_dims: cycle.instr.output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&q))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&y))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&q_y))),
            },
            advice_value: None,
        });

        let add_0 = Add::sequence_output(&q_y, &r);
        virtual_trace.push(AtlasCycle {
            instr: AtlasInstr {
                address: cycle.instr.address,
                opcode: AtlasOpcode::Add,
                ts1: v_qy,
                ts2: v_r,
                ts3: None,
                td: v_0,
                imm: None,
                virtual_sequence_remaining: Some(counter.dec()),
                output_dims: cycle.instr.output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&q_y))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&r))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&add_0))),
            },
            advice_value: None,
        });

        let _assert_eq = Eq::sequence_output(&add_0, &x);
        virtual_trace.push(AtlasCycle {
            instr: AtlasInstr {
                address: cycle.instr.address,
                opcode: AtlasOpcode::VirtualAssertEq,
                ts1: v_0,
                ts2: r_x,
                ts3: None,
                td: None,
                imm: None,
                virtual_sequence_remaining: Some(counter.dec()),
                output_dims: cycle.instr.output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&add_0))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&x))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: None,
            },
            advice_value: None,
        });

        virtual_trace.push(AtlasCycle {
            instr: AtlasInstr {
                address: cycle.instr.address,
                opcode: AtlasOpcode::VirtualMove,
                ts1: v_q,
                ts2: None,
                ts3: None,
                td,
                imm: None,
                virtual_sequence_remaining: Some(counter.dec()),
                output_dims: cycle.instr.output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&q))),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: cycle.memory_state.td_pre_val.clone(),
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&q))),
            },
            advice_value: None,
        });

        assert_eq!(
            virtual_trace.len(),
            Self::SEQUENCE_LENGTH,
            "Incorrect virtual trace length"
        );

        virtual_trace
    }

    fn sequence_output(x: &[u64], y: &[u64]) -> Vec<u64> {
        x.iter()
            .zip(y.iter())
            .map(|(&x, &y)| Self::exec(x, y))
            .collect()
    }
}

#[cfg(test)]
mod test {
    use crate::instructions::test::jolt_virtual_sequence_test;

    use super::*;

    #[test]
    fn virtual_sequence_32() {
        jolt_virtual_sequence_test::<Div>(ONNXOpcode::Div, 16);
    }
}
