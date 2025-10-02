use jolt_core::jolt::instruction::LookupQuery;
use onnx_tracer::{
    constants::{MAX_TENSOR_SIZE, virtual_tensor_index},
    tensor::Tensor,
    trace_types::{MemoryState, ONNXCycle, ONNXInstr, ONNXOpcode},
};
use serde::{Deserialize, Serialize};

use crate::{
    jolt::instruction::{VirtualInstructionSequence, virtual_pow2::VirtualPow2},
    utils::u64_vec_to_i32_iter,
};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct SigmoidInstruction<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> VirtualInstructionSequence for SigmoidInstruction<WORD_SIZE> {
    const SEQUENCE_LENGTH: usize = 14;

    fn virtual_trace(cycle: ONNXCycle) -> Vec<ONNXCycle> {
        assert_eq!(cycle.instr.opcode, ONNXOpcode::Sigmoid);

        let mut vt = Vec::with_capacity(Self::SEQUENCE_LENGTH);
        let remain = |vt_len: usize| Some(Self::SEQUENCE_LENGTH - (vt_len + 1));

        // ---- Virtual registers (all distinct) ----
        let v_clamped = Some(virtual_tensor_index(0));
        let v_neg_z_clamped = Some(virtual_tensor_index(1));
        let v_abs = Some(virtual_tensor_index(2));
        let v_pow2 = Some(virtual_tensor_index(3));
        let v_q_const = Some(virtual_tensor_index(4));
        let v_q_const_squared = Some(virtual_tensor_index(5));
        let v_mul_Q_pow = Some(virtual_tensor_index(6));
        let v_div_Q_pow = Some(virtual_tensor_index(7));
        let v_zero = Some(virtual_tensor_index(8));
        let v_ge0 = Some(virtual_tensor_index(9));
        let v_a = Some(virtual_tensor_index(10));
        let v_b = Some(virtual_tensor_index(11));
        let v_c = Some(virtual_tensor_index(12));

        // ------------------------------------------------------------------
        // Step 1. Clamp input into [-8, 8] to avoid overflow in pow2
        // ------------------------------------------------------------------
        let z_u64 = cycle.ts1_vals();
        let z_i64: Vec<i64> = z_u64.iter().map(|&v| v as u32 as i32 as i64).collect();
        let z_clamped: Vec<i64> = z_i64.iter().map(|&v| v.clamp(-8, 8)).collect();
        let z_clamped_u64: Vec<u64> = z_clamped.iter().map(|&x| x as u64).collect();

        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualAdvice,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_clamped,
                imm: None,
                virtual_sequence_remaining: remain(vt.len()),
                active_output_elements: MAX_TENSOR_SIZE,
                output_dims: [1, MAX_TENSOR_SIZE],
            },
            memory_state: MemoryState {
                ts1_val: None,
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&z_clamped_u64))),
            },
            advice_value: Some(Tensor::from(u64_vec_to_i32_iter(&z_clamped_u64))),
        });

        // ------------------------------------------------------------------
        // Step 2. Materialize constant zero for comparisons
        // ------------------------------------------------------------------
        let zero_tensor = vec![0u64; MAX_TENSOR_SIZE];
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualConst,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_zero,
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&zero_tensor))),
                virtual_sequence_remaining: remain(vt.len()),
                active_output_elements: MAX_TENSOR_SIZE,
                output_dims: [1, MAX_TENSOR_SIZE],
            },
            memory_state: MemoryState {
                ts1_val: None,
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&zero_tensor))),
            },
            advice_value: None,
        });

        // ------------------------------------------------------------------
        // Step 3. Compute ge0 = (z >= 0), needed for select later
        // ------------------------------------------------------------------
        let ge0_vals: Vec<u64> = z_clamped
            .iter()
            .map(|&v| if v >= 0 { 1 } else { 0 })
            .collect();
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Gte,
                ts1: v_clamped,
                ts2: v_zero,
                ts3: None,
                td: v_ge0,
                imm: None,
                virtual_sequence_remaining: remain(vt.len()),
                active_output_elements: MAX_TENSOR_SIZE,
                output_dims: [1, MAX_TENSOR_SIZE],
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&z_clamped_u64))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&zero_tensor))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&ge0_vals))),
            },
            advice_value: None,
        });

        // ------------------------------------------------------------------
        // Step 4. Compute -z (for abs)
        // ------------------------------------------------------------------
        let neg_vals: Vec<u64> = z_clamped
            .iter()
            .map(|&x| {
                let xi = x as i32;
                (-xi) as u32 as u64
            })
            .collect();

        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Sub,
                ts1: v_zero, // 0 - x
                ts2: v_clamped,
                ts3: None,
                td: v_neg_z_clamped,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 2),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&vec![0; MAX_TENSOR_SIZE]))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&z_clamped_u64))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&neg_vals))),
            },
            advice_value: None,
        });

        // ------------------------------------------------------------------
        // Step 5. abs(z) = select(ge0, z, -z)
        // ------------------------------------------------------------------
        let abs_vals: Vec<u64> = z_clamped_u64
            .iter()
            .zip(&ge0_vals)
            .zip(&neg_vals)
            .map(|((&x, &cond), &nx)| if cond != 0 { x } else { nx })
            .collect();

        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Select,
                ts1: v_ge0,
                ts2: v_clamped,
                ts3: v_neg_z_clamped,
                td: v_abs,
                imm: None,
                virtual_sequence_remaining: Some(0),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&ge0_vals))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&z_clamped_u64))),
                ts3_val: Some(Tensor::from(u64_vec_to_i32_iter(&neg_vals))),
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&abs_vals))),
            },
            advice_value: None,
        });

        // ------------------------------------------------------------------
        // Step 6. Compute 2^{|z|}
        // ------------------------------------------------------------------
        let pow2_vals: Vec<u64> = abs_vals
            .iter()
            .map(|&ai| VirtualPow2::<WORD_SIZE>(ai).to_lookup_output())
            .collect();

        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualPow2,
                ts1: v_abs,
                ts2: None,
                ts3: None,
                td: v_pow2,
                imm: None,
                virtual_sequence_remaining: remain(vt.len()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&abs_vals))),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&pow2_vals))),
            },
            advice_value: None,
        });

        // ------------------------------------------------------------------
        // Step 7. Load Q and compute Q^2
        // ------------------------------------------------------------------
        const Q: u64 = 1 << 8;
        let q_tensor = vec![Q; MAX_TENSOR_SIZE];

        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualConst,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_q_const,
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&q_tensor))),
                virtual_sequence_remaining: remain(vt.len()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: None,
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&q_tensor))),
            },
            advice_value: None,
        });

        let q2_vals: Vec<u64> = q_tensor.iter().map(|&t| t * t).collect();
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Mul,
                ts1: v_q_const,
                ts2: v_q_const,
                ts3: None,
                td: v_q_const_squared,
                imm: None,
                virtual_sequence_remaining: remain(vt.len()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&q_tensor))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&q_tensor))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&q2_vals))),
            },
            advice_value: None,
        });

        // ------------------------------------------------------------------
        // Step 8. Compute Q/2^{|z|} and Q*2^{|z|}
        // ------------------------------------------------------------------

        // div_Q_pow = Q / 2^{|z|}
        let div_Q_pow_vals: Vec<u64> = pow2_vals.iter().map(|&p| Q / p).collect();
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Div,
                ts1: v_q_const,
                ts2: None,
                ts3: None,
                td: v_div_Q_pow,
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&pow2_vals))),
                virtual_sequence_remaining: None, // inner trace manages its own counters
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&q_tensor))),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&div_Q_pow_vals))),
            },
            advice_value: None,
        });

        // mul_Q_pow = Q * 2^{|z|}
        let mul_Q_pow_vals: Vec<u64> = pow2_vals.iter().map(|&p| p * Q).collect();
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Mul,
                ts1: v_pow2,
                ts2: v_q_const,
                ts3: None,
                td: v_mul_Q_pow,
                imm: None,
                virtual_sequence_remaining: remain(vt.len()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&pow2_vals))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&q_tensor))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&mul_Q_pow_vals))),
            },
            advice_value: None,
        });

        // ------------------------------------------------------------------
        // Step 9. Select the branch:
        // if z >= 0 then a = Q/2^{|z|}
        // else          a = Q*2^{|z|}
        // ------------------------------------------------------------------
        let a_vals: Vec<u64> = (0..MAX_TENSOR_SIZE)
            .map(|i| {
                if ge0_vals[i] != 0 {
                    div_Q_pow_vals[i]
                } else {
                    mul_Q_pow_vals[i]
                }
            })
            .collect();
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Select,
                ts1: v_ge0,
                ts2: v_div_Q_pow,
                ts3: v_mul_Q_pow,
                td: v_a,
                imm: None,
                virtual_sequence_remaining: remain(vt.len()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&ge0_vals))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&div_Q_pow_vals))),
                ts3_val: Some(Tensor::from(u64_vec_to_i32_iter(&mul_Q_pow_vals))),
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&a_vals))),
            },
            advice_value: None,
        });

        // ------------------------------------------------------------------
        // Step 10. b = Q + a
        // ------------------------------------------------------------------
        let b_vals: Vec<u64> = a_vals.iter().map(|&a| a + Q).collect();
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Add,
                ts1: v_q_const,
                ts2: v_a,
                ts3: None,
                td: v_b,
                imm: None,
                virtual_sequence_remaining: remain(vt.len()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&q_tensor))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&a_vals))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&b_vals))),
            },
            advice_value: None,
        });

        // ------------------------------------------------------------------
        // Step 11. c = Q^2 / b   => final σ_Q(z)
        // ------------------------------------------------------------------
        let c_vals: Vec<u64> = b_vals.iter().map(|&b| (Q * Q) / b).collect();
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Div,
                ts1: v_q_const_squared,
                ts2: None,
                ts3: None,
                td: v_c,
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&b_vals))),
                virtual_sequence_remaining: None,
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&q2_vals))),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&c_vals))),
            },
            advice_value: None,
        });

        // ------------------------------------------------------------------
        // Step 12. Move result to final td
        // ------------------------------------------------------------------
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualMove,
                ts1: v_c,
                ts2: None,
                ts3: None,
                td: cycle.instr.td,
                imm: None,
                virtual_sequence_remaining: remain(vt.len()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&c_vals))),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: cycle.memory_state.td_pre_val.clone(),
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&c_vals))),
            },
            advice_value: None,
        });

        debug_assert_eq!(vt.len(), Self::SEQUENCE_LENGTH, "sequence length mismatch");
        vt
    }

    fn sequence_output(x: Vec<u64>, _: Vec<u64>, _: Option<ONNXOpcode>) -> Vec<u64> {
        // Reference (host-side) quantized σ_Q(z) with base 2, clamped to [-8,8]
        let mut out = vec![0u64; MAX_TENSOR_SIZE];
        const Q: u128 = 256;

        for i in 0..MAX_TENSOR_SIZE {
            let xi = x[i] as i32;
            let e = xi.clamp(-8, 8);
            let (num, den): (u128, u128) = if e >= 0 {
                let a = 1u128 << (e as u32); // 2^e
                (a, 1 + a)
            } else {
                let b = 1u128 << ((-e) as u32); // 2^{|e|}
                (1, 1 + b)
            };
            let q = (Q * num) / den;
            out[i] = (q as u64).min(255);
        }
        out
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::jolt::instruction::test::jolt_virtual_sequence_test;

    #[test]
    fn sigmoid_virtual_sequence_32() {
        jolt_virtual_sequence_test::<SigmoidInstruction<32>>(ONNXOpcode::Sigmoid);
    }
}
