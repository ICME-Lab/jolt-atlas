use onnx_tracer::{
    constants::virtual_tensor_index,
    tensor::{Tensor, ops::nonlinearities},
    trace_types::{MemoryState, ONNXCycle, ONNXInstr, ONNXOpcode},
};

use crate::{
    jolt::executor::instructions::{VirtualInstructionSequence, virtual_pow2::Pow2Instruction},
    utils::{VirtualSequenceCounter, u64_vec_to_i32_iter},
};
use jolt_core::zkvm::instruction::LookupQuery;

/// Quantized Softmax virtual instruction.
/// Implements a stable base-2 softmax with manual quantization factor `Q = 128`.
pub struct SoftmaxInstruction;

impl VirtualInstructionSequence for SoftmaxInstruction {
    const SEQUENCE_LENGTH: usize = 14;

    fn virtual_trace(cycle: ONNXCycle, k: usize) -> Vec<ONNXCycle> {
        const WORD_SIZE: usize = 32;
        const Q: u64 = 128;

        assert_eq!(cycle.instr.opcode, ONNXOpcode::Softmax);

        let num_outputs = cycle.instr.num_output_elements();
        if num_outputs == 0 {
            return Vec::new();
        }

        let remaining = cycle
            .instr
            .virtual_sequence_remaining
            .unwrap_or(Self::SEQUENCE_LENGTH);
        assert!(
            remaining >= Self::SEQUENCE_LENGTH,
            "Not enough remaining virtual sequence steps"
        );
        let mut counter = VirtualSequenceCounter::new(remaining);

        let td = cycle.instr.td.unwrap_or(0);
        let output_dims = cycle.instr.output_dims.clone();

        let z_tensor = cycle.memory_state.ts1_val.clone().unwrap_or_else(|| {
            Tensor::from(u64_vec_to_i32_iter(
                &cycle.ts1_vals().unwrap_or_else(|| vec![0u64; num_outputs]),
            ))
        });
        let mut z_vals = vec![0i32; num_outputs];
        for (i, value) in z_tensor.inner.iter().take(num_outputs).enumerate() {
            z_vals[i] = *value;
        }

        // Virtual tensor register mapping
        let v_zero = Some(virtual_tensor_index(0, k, td));
        let v_ge0 = Some(virtual_tensor_index(1, k, td));
        let v_neg = Some(virtual_tensor_index(2, k, td));
        let v_abs = Some(virtual_tensor_index(3, k, td));
        let v_pow2 = Some(virtual_tensor_index(4, k, td));
        let v_q_const = Some(virtual_tensor_index(5, k, td));
        let v_q_div_c = Some(virtual_tensor_index(6, k, td));
        let v_q_mul_c = Some(virtual_tensor_index(7, k, td));
        let v_d = Some(virtual_tensor_index(8, k, td));
        let v_sum = Some(virtual_tensor_index(9, k, td));
        let v_broadcast_sum = Some(virtual_tensor_index(10, k, td));
        let v_q_mul_d = Some(virtual_tensor_index(11, k, td));
        let v_output = Some(virtual_tensor_index(12, k, td));

        let mut vt = Vec::with_capacity(Self::SEQUENCE_LENGTH);

        // (1) zero constant
        let zero_vals = vec![0u64; num_outputs];
        let zero_tensor = Tensor::from(u64_vec_to_i32_iter(&zero_vals));
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualConst,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_zero,
                imm: Some(zero_tensor.clone()),
                virtual_sequence_remaining: Some(counter.dec()),
                output_dims: output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: None,
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(zero_tensor.clone()),
            },
            advice_value: None,
        });

        // (2) ge0 = (z >= 0)
        let ge0_vals: Vec<u64> = z_vals.iter().map(|&v| u64::from(v >= 0)).collect();
        let ge0_tensor = Tensor::from(u64_vec_to_i32_iter(&ge0_vals));
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Gte,
                ts1: cycle.instr.ts1,
                ts2: None,
                ts3: None,
                td: v_ge0,
                imm: None,
                virtual_sequence_remaining: Some(counter.dec()),
                output_dims: output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: Some(z_tensor.clone()),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(ge0_tensor.clone()),
            },
            advice_value: None,
        });

        // (3) neg = -z
        let neg_vals: Vec<i32> = z_vals.iter().map(|&v| -v).collect();
        let neg_tensor = Tensor::from(neg_vals.clone().into_iter());
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Sub,
                ts1: v_zero,
                ts2: cycle.instr.ts1,
                ts3: None,
                td: v_neg,
                imm: None,
                virtual_sequence_remaining: Some(counter.dec()),
                output_dims: output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: Some(zero_tensor.clone()),
                ts2_val: Some(z_tensor.clone()),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(neg_tensor.clone()),
            },
            advice_value: None,
        });

        // (4) abs = select(ge0, z, -z)
        let abs_vals: Vec<i32> = z_vals
            .iter()
            .zip(ge0_vals.iter())
            .map(|(&z, &cond)| if cond != 0 { z } else { -z })
            .collect();
        let abs_tensor = Tensor::from(abs_vals.clone().into_iter());
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Select,
                ts1: v_ge0,
                ts2: cycle.instr.ts1,
                ts3: v_neg,
                td: v_abs,
                imm: None,
                virtual_sequence_remaining: Some(counter.dec()),
                output_dims: output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: Some(ge0_tensor.clone()),
                ts2_val: Some(z_tensor.clone()),
                ts3_val: Some(neg_tensor.clone()),
                td_pre_val: None,
                td_post_val: Some(abs_tensor.clone()),
            },
            advice_value: None,
        });

        // (5) c = 2^{|z|}
        let c_vals: Vec<u64> = abs_vals
            .iter()
            .map(|&abs| Pow2Instruction::<WORD_SIZE>(abs as u32 as u64).to_lookup_output())
            .collect();
        let c_tensor = Tensor::from(u64_vec_to_i32_iter(&c_vals));
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualPow2,
                ts1: v_abs,
                ts2: None,
                ts3: None,
                td: v_pow2,
                imm: None,
                virtual_sequence_remaining: Some(counter.dec()),
                output_dims: output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: Some(abs_tensor.clone()),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(c_tensor.clone()),
            },
            advice_value: None,
        });

        // (6) Q constant
        let q_vals = vec![Q; num_outputs];
        let q_tensor = Tensor::from(u64_vec_to_i32_iter(&q_vals));
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualConst,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_q_const,
                imm: Some(q_tensor.clone()),
                virtual_sequence_remaining: Some(counter.dec()),
                output_dims: output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: None,
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(q_tensor.clone()),
            },
            advice_value: None,
        });

        // (7) d_q_over_c = Q / c
        let d_q_over_c_vals: Vec<u64> = c_vals
            .iter()
            .map(|&c| if c == 0 { 0 } else { Q.saturating_div(c) })
            .collect();
        let d_q_over_c_tensor = Tensor::from(u64_vec_to_i32_iter(&d_q_over_c_vals));
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Div,
                ts1: v_q_const,
                ts2: None,
                ts3: None,
                td: v_q_div_c,
                imm: Some(c_tensor.clone()),
                virtual_sequence_remaining: Some(counter.dec()),
                output_dims: output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: Some(q_tensor.clone()),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(d_q_over_c_tensor.clone()),
            },
            advice_value: None,
        });

        // (8) d_q_mul_c = Q * c
        let d_q_mul_c_vals: Vec<u64> = c_vals.iter().map(|&c| Q.saturating_mul(c)).collect();
        let d_q_mul_c_tensor = Tensor::from(u64_vec_to_i32_iter(&d_q_mul_c_vals));
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Mul,
                ts1: v_q_const,
                ts2: v_pow2,
                ts3: None,
                td: v_q_mul_c,
                imm: None,
                virtual_sequence_remaining: Some(counter.dec()),
                output_dims: output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: Some(q_tensor.clone()),
                ts2_val: Some(c_tensor.clone()),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(d_q_mul_c_tensor.clone()),
            },
            advice_value: None,
        });

        // (9) d = select(ge0, Q*c, Q/c)
        let d_vals: Vec<u64> = ge0_vals
            .iter()
            .enumerate()
            .map(|(i, cond)| {
                if *cond != 0 {
                    d_q_mul_c_vals[i]
                } else {
                    d_q_over_c_vals[i]
                }
            })
            .collect();
        let d_tensor = Tensor::from(u64_vec_to_i32_iter(&d_vals));
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Select,
                ts1: v_ge0,
                ts2: v_q_mul_c,
                ts3: v_q_div_c,
                td: v_d,
                imm: None,
                virtual_sequence_remaining: Some(counter.dec()),
                output_dims: output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: Some(ge0_tensor.clone()),
                ts2_val: Some(d_q_mul_c_tensor.clone()),
                ts3_val: Some(d_q_over_c_tensor.clone()),
                td_pre_val: None,
                td_post_val: Some(d_tensor.clone()),
            },
            advice_value: None,
        });

        // (10) sum = Σ d (store as scalar tensor with broadcast-friendly shape)
        let d_sum = d_vals
            .iter()
            .fold(0u64, |acc, value| acc.saturating_add(*value));
        let mut sum_tensor = Tensor::from(u64_vec_to_i32_iter(&[d_sum]));
        let sum_tensor_dims = if output_dims.is_empty() {
            vec![1]
        } else {
            vec![1; output_dims.len()]
        };
        sum_tensor.reshape(&sum_tensor_dims).unwrap();
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualSaturatingSum,
                ts1: v_d,
                ts2: None,
                ts3: None,
                td: v_sum,
                imm: None,
                virtual_sequence_remaining: Some(counter.dec()),
                output_dims: sum_tensor_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: Some(d_tensor.clone()),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(sum_tensor.clone()),
            },
            advice_value: None,
        });

        // (11) broadcast sum
        let broadcast_vals = vec![d_sum; num_outputs];
        let mut broadcast_tensor = Tensor::from(u64_vec_to_i32_iter(&broadcast_vals));
        if !output_dims.is_empty() {
            broadcast_tensor.reshape(&output_dims).unwrap();
        }
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Broadcast,
                ts1: v_sum,
                ts2: None,
                ts3: None,
                td: v_broadcast_sum,
                imm: None,
                virtual_sequence_remaining: Some(counter.dec()),
                output_dims: output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: Some(sum_tensor.clone()),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(broadcast_tensor.clone()),
            },
            advice_value: None,
        });

        // (12) f = Q * d
        let f_vals: Vec<u64> = d_vals.iter().map(|&d| Q.saturating_mul(d)).collect();
        let f_tensor = Tensor::from(u64_vec_to_i32_iter(&f_vals));
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Mul,
                ts1: v_d,
                ts2: v_q_const,
                ts3: None,
                td: v_q_mul_d,
                imm: None,
                virtual_sequence_remaining: Some(counter.dec()),
                output_dims: output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: Some(d_tensor.clone()),
                ts2_val: Some(q_tensor.clone()),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(f_tensor.clone()),
            },
            advice_value: None,
        });

        // (13) g = f / sum
        let g_vals: Vec<u64> = f_vals
            .iter()
            .map(|&f| if d_sum == 0 { 0 } else { f / d_sum })
            .collect();
        let g_tensor = Tensor::from(u64_vec_to_i32_iter(&g_vals));
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Div,
                ts1: v_q_mul_d,
                ts2: None,
                ts3: None,
                td: v_output,
                imm: Some(broadcast_tensor.clone()),
                virtual_sequence_remaining: Some(counter.dec()),
                output_dims: output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: Some(f_tensor.clone()),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(g_tensor.clone()),
            },
            advice_value: None,
        });

        // (14) move final result to td
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualMove,
                ts1: v_output,
                ts2: None,
                ts3: None,
                td: cycle.instr.td,
                imm: None,
                virtual_sequence_remaining: Some(counter.dec()),
                output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(g_tensor.clone()),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: cycle.memory_state.td_pre_val.clone(),
                td_post_val: Some(g_tensor),
            },
            advice_value: None,
        });

        debug_assert_eq!(vt.len(), Self::SEQUENCE_LENGTH, "sequence length mismatch");
        vt
    }

    fn sequence_output(x: Vec<u64>, _y: Vec<u64>) -> Vec<u64> {
        if x.is_empty() {
            return Vec::new();
        }

        let input_tensor = Tensor::from(u64_vec_to_i32_iter(&x));
        let (softmax_tensor, _) = nonlinearities::softmax(&input_tensor, 128.0);

        softmax_tensor
            .into_iter()
            .map(|value| value as u64)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jolt::executor::instructions::test::jolt_virtual_sequence_test;

    #[test]
    fn virtual_trace_matches_sequence_output() {
        let input = vec![0u64, 1, 2, 3];
        let cycle = ONNXCycle {
            instr: ONNXInstr {
                address: 0,
                opcode: ONNXOpcode::Softmax,
                ts1: Some(1),
                ts2: None,
                ts3: None,
                td: Some(2),
                imm: None,
                virtual_sequence_remaining: None,
                output_dims: vec![1, input.len()],
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&input))),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: None,
            },
            advice_value: None,
        };

        let trace = SoftmaxInstruction::virtual_trace(cycle, 32);
        assert_eq!(trace.len(), SoftmaxInstruction::SEQUENCE_LENGTH);

        let expected = SoftmaxInstruction::sequence_output(input.clone(), vec![]);
        let actual = trace
            .last()
            .and_then(|cycle| cycle.td_post_vals())
            .expect("last cycle must write td");

        assert_eq!(expected, actual);
    }

    #[test]
    fn fuzz_virtual_sequence() {
        jolt_virtual_sequence_test::<SoftmaxInstruction>(ONNXOpcode::Softmax, 8);
    }
}
