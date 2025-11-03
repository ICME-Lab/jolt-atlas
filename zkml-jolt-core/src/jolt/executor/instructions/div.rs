use itertools::Itertools;
use jolt_core::zkvm::instruction::LookupQuery;
use onnx_tracer::{
    constants::virtual_tensor_index,
    tensor::Tensor,
    trace_types::{MemoryState, ONNXCycle, ONNXInstr, ONNXOpcode},
};

use crate::{
    jolt::executor::instructions::{
        VirtualInstructionSequence, add::AddInstruction, beq::BeqInstruction, mul::MulInstruction,
        virtual_advice::AdviceInstruction, virtual_assert_valid_div0::AssertValidDiv0Instruction,
        virtual_assert_valid_signed_remainder::AssertValidSignedRemainderInstruction,
    },
    utils::{VirtualSequenceCounter, u64_vec_to_i32_iter},
};

/// Perform signed division and return the result
pub struct DivInstruction<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> VirtualInstructionSequence for DivInstruction<WORD_SIZE> {
    const SEQUENCE_LENGTH: usize = 9;

    fn virtual_trace(cycle: ONNXCycle, K: usize) -> Vec<ONNXCycle> {
        assert_eq!(cycle.instr.opcode, ONNXOpcode::Div);
        let num_outputs = cycle.instr.active_output_elements;

        // If DIV is part of a longer virtual sequence, recover the counter to continue decrementing it
        let virtual_sequence_remaining =
            if let Some(remaining) = cycle.instr.virtual_sequence_remaining {
                assert!(
                    remaining >= Self::SEQUENCE_LENGTH,
                    "Not enough remaining virtual sequence steps"
                );
                remaining
            } else {
                Self::SEQUENCE_LENGTH
            };

        let mut vseq_counter = VirtualSequenceCounter::new(virtual_sequence_remaining);

        // DIV source registers
        let r_x = cycle.instr.ts1;

        // Virtual registers used in sequence
        let v_c = Some(virtual_tensor_index(0, K, cycle.instr.td.unwrap()));
        let v_q = Some(virtual_tensor_index(1, K, cycle.instr.td.unwrap()));
        let v_r = Some(virtual_tensor_index(2, K, cycle.instr.td.unwrap()));
        let v_qy = Some(virtual_tensor_index(3, K, cycle.instr.td.unwrap()));
        let v_0 = Some(virtual_tensor_index(4, K, cycle.instr.td.unwrap()));

        // DIV operands
        let x = cycle.ts1_vals().unwrap_or(vec![0; num_outputs]);
        let y = cycle.imm().unwrap_or(vec![0; num_outputs]);
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

        // const
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualConst,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_c,
                imm: cycle.instr.imm.clone(),
                virtual_sequence_remaining: Some(vseq_counter.dec()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: None,
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: cycle.instr.imm.clone(),
            },
            advice_value: None,
        });

        let q = (0..num_outputs)
            .map(|i| AdviceInstruction::<WORD_SIZE>(quotient[i]).to_lookup_output())
            .collect_vec();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualAdvice,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_q,
                imm: None,
                virtual_sequence_remaining: Some(vseq_counter.dec()),
                active_output_elements: cycle.instr.active_output_elements,
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

        let r = (0..num_outputs)
            .map(|i| AdviceInstruction::<WORD_SIZE>(remainder[i]).to_lookup_output())
            .collect_vec();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualAdvice,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_r,
                imm: None,
                virtual_sequence_remaining: Some(vseq_counter.dec()),
                active_output_elements: cycle.instr.active_output_elements,
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

        let is_valid: Vec<u64> = (0..num_outputs)
            .map(|i| {
                AssertValidSignedRemainderInstruction::<WORD_SIZE>(r[i], y[i]).to_lookup_output()
            })
            .collect_vec();
        is_valid.iter().for_each(|&valid| {
            assert_eq!(valid, 1, "Invalid signed remainder detected");
        });
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualAssertValidSignedRemainder,
                ts1: v_r,
                ts2: v_c,
                ts3: None,
                td: None,
                imm: None,
                virtual_sequence_remaining: Some(vseq_counter.dec()),
                active_output_elements: cycle.instr.active_output_elements,
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

        let is_valid: Vec<u64> = (0..num_outputs)
            .map(|i| AssertValidDiv0Instruction::<WORD_SIZE>(y[i], q[i]).to_lookup_output())
            .collect_vec();
        is_valid.iter().for_each(|&valid| {
            assert_eq!(valid, 1, "Invalid division by zero detected");
        });
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualAssertValidDiv0,
                ts1: v_c,
                ts2: v_q,
                ts3: None,
                td: None,
                imm: None,
                virtual_sequence_remaining: Some(vseq_counter.dec()),
                active_output_elements: cycle.instr.active_output_elements,
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

        let q_y = (0..num_outputs)
            .map(|i| MulInstruction::<WORD_SIZE>(q[i], y[i]).to_lookup_output())
            .collect_vec();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Mul,
                ts1: v_q,
                ts2: v_c,
                ts3: None,
                td: v_qy,
                imm: None,
                virtual_sequence_remaining: Some(vseq_counter.dec()),
                active_output_elements: cycle.instr.active_output_elements,
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

        let add_0 = (0..num_outputs)
            .map(|i| AddInstruction::<WORD_SIZE>(q_y[i], r[i]).to_lookup_output())
            .collect_vec();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Add,
                ts1: v_qy,
                ts2: v_r,
                ts3: None,
                td: v_0,
                imm: None,
                virtual_sequence_remaining: Some(vseq_counter.dec()),
                active_output_elements: cycle.instr.active_output_elements,
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

        let _assert_eq = (0..num_outputs)
            .map(|i| BeqInstruction::<WORD_SIZE>(add_0[i], x[i]).to_lookup_output())
            .collect_vec();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualAssertEq,
                ts1: v_0,
                ts2: r_x,
                ts3: None,
                td: None,
                imm: None,
                virtual_sequence_remaining: Some(vseq_counter.dec()),
                active_output_elements: cycle.instr.active_output_elements,
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

        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualMove,
                ts1: v_q,
                ts2: None,
                ts3: None,
                td: cycle.instr.td,
                imm: None,
                virtual_sequence_remaining: Some(vseq_counter.dec()),
                active_output_elements: cycle.instr.active_output_elements,
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

    fn sequence_output(x: Vec<u64>, y: Vec<u64>, _: Option<ONNXOpcode>) -> Vec<u64> {
        let num_outputs = x.len();
        let mut output = vec![0; num_outputs];
        for i in 0..num_outputs {
            let x = x[i];
            let y = y[i];
            let x = x as i32;
            let y = y as i32;
            if y == 0 {
                output[i] = (1 << WORD_SIZE) - 1;
                continue;
            }
            let mut quotient = x / y;
            let remainder = x % y;
            if (remainder < 0 && y > 0) || (remainder > 0 && y < 0) {
                quotient -= 1;
            }
            output[i] = quotient as u32 as u64
        }
        output
    }
}

#[cfg(test)]
mod test {
    use crate::jolt::executor::instructions::test::jolt_virtual_sequence_test;

    use super::*;

    #[test]
    fn div_virtual_sequence_32() {
        jolt_virtual_sequence_test::<DivInstruction<32>>(ONNXOpcode::Div, 16);
    }
}
