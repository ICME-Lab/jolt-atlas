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

/// Quantized Softmax producing `Q * softmax(z)`
///
/// ### Overview
/// This implementation computes a **quantized approximation of the Softmax function**
/// that supports signed inputs (positive and negative) without stabilization
///
/// The algorithm behaves as:
/// ```text
/// if z_i >= 0 → d_i = Q * 2^{z_i}
/// if z_i <  0 → d_i = Q / 2^{|z_i|}
/// g_i = (Q * d_i) / Σ_j d_j
/// ```
///
/// ### Virtual trace sequence
/// The sequence performs 14 virtual steps:
///
/// | Step | Operation | Description |
/// |------|------------|--------------|
/// | 1 | `VirtualConst(0)` | Initialize zero tensor |
/// | 2 | `Gte` | Compute `ge0 = (z >= 0)` |
/// | 3 | `Sub` | Compute `neg_z = -z` |
/// | 4 | `Select` | Compute `abs_z = select(ge0, z, -z)` |
/// | 5 | `VirtualPow2` | Compute `c = 2^{|z|}` |
/// | 6 | `VirtualConst(Q)` | Constant quantization scalar |
/// | 7 | `Div` | Compute `d_q_over_c = Q / c` |
/// | 8 | `Mul` | Compute `d_q_times_c = Q * c` |
/// | 9 | `Select` | Select `d = (z >= 0 ? Q * c : Q / c)` |
/// | 10 | `Sum` | ReduceSum(d) to get total |
/// | 11 | `Broadcast` | Broadcast the sum |
/// | 12 | `Mul` | Compute `f = Q * d` |
/// | 13 | `Div` | Normalize `g = f / e_sum` |
/// | 14 | `VirtualMove` | Write final result to output tensor |
///
/// Each `ONNXCycle` represents one of these steps in the virtualized trace.
///
/// ### Notes
/// - This implementation preserves Softmax monotonicity: larger inputs → larger outputs.
/// - It avoids overflow with capped exponentiation (`2^|z|` up to 2^63).
/// - Designed for quantized circuits or proof backends where fractional values are approximated in integer space.
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct SoftmaxInstruction<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> VirtualInstructionSequence for SoftmaxInstruction<WORD_SIZE> {
    const SEQUENCE_LENGTH: usize = 14;

    fn virtual_trace(cycle: ONNXCycle) -> Vec<ONNXCycle> {
        assert_eq!(cycle.instr.opcode, ONNXOpcode::Softmax);

        let mut vt = Vec::with_capacity(Self::SEQUENCE_LENGTH);
        let remain = |vt_len: usize| Some(Self::SEQUENCE_LENGTH - (vt_len + 1));

        // ---- Virtual tensor register mapping ----
        //
        // v0  -> zero constant (0)
        // v1  -> ge0 = (z >= 0)
        // v2  -> neg_z = -z
        // v3  -> abs_z = |z|
        // v4  -> c = 2^{|z|}
        // v5  -> q_const = Q
        // v6  -> d_q_over_c = Q / c
        // v7  -> d_q_times_c = Q * c
        // v8  -> d = select(ge0, Q*c, Q/c)
        // v9  -> e_sum = Σ d
        // v10 -> broadcast_e_sum
        // v11 -> f = Q * d
        // v12 -> g = f / e_sum
        // v13 -> final output (VirtualMove)
        let v_zero = Some(virtual_tensor_index(0));
        let v_ge0 = Some(virtual_tensor_index(1));
        let v_neg_b = Some(virtual_tensor_index(2));
        let v_abs_b = Some(virtual_tensor_index(3));
        let v_c_pow2 = Some(virtual_tensor_index(4));
        let v_q_const = Some(virtual_tensor_index(5));
        let v_d_q_over_c = Some(virtual_tensor_index(6));
        let v_d_q_times_c = Some(virtual_tensor_index(7));
        let v_d = Some(virtual_tensor_index(8));
        let v_e_sum = Some(virtual_tensor_index(9));
        let v_broadcast_e_sum = Some(virtual_tensor_index(10));
        let v_f_q_times_d = Some(virtual_tensor_index(11));
        let v_g_out = Some(virtual_tensor_index(12));

        // ---- Input tensor ----
        let z_u64 = cycle.ts1_vals();
        let z_tensor = Tensor::from(u64_vec_to_i32_iter(&z_u64));

        // (1) Constant zero tensor
        let zero_tensor = Tensor::from(u64_vec_to_i32_iter(&vec![0u64; MAX_TENSOR_SIZE]));
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualConst,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_zero,
                imm: Some(zero_tensor.clone()),
                virtual_sequence_remaining: remain(vt.len()),
                active_output_elements: MAX_TENSOR_SIZE,
                output_dims: [1, MAX_TENSOR_SIZE],
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
        let ge0_vals: Vec<u64> = z_tensor
            .inner
            .iter()
            .map(|&v| if v >= 0 { 1 } else { 0 })
            .collect();
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
                virtual_sequence_remaining: remain(vt.len()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
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

        // (3) neg_z = -z
        let neg_b_vals: Vec<i32> = z_tensor.inner.iter().map(|&bi| -bi).collect();
        let neg_b_tensor = Tensor::from(neg_b_vals.into_iter());
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Sub,
                ts1: v_zero,
                ts2: cycle.instr.ts1,
                ts3: None,
                td: v_neg_b,
                imm: None,
                virtual_sequence_remaining: remain(vt.len()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(zero_tensor.clone()),
                ts2_val: Some(z_tensor.clone()),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(neg_b_tensor.clone()),
            },
            advice_value: None,
        });

        // (4) abs_z = select(ge0, z, -z)
        let abs_b_vals: Vec<i32> = z_tensor
            .inner
            .iter()
            .zip(&ge0_vals)
            .map(|(&b, &cond)| if cond != 0 { b } else { -b })
            .collect();
        let abs_b_tensor = Tensor::from(abs_b_vals.clone().into_iter());
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Select,
                ts1: v_ge0,
                ts2: cycle.instr.ts1,
                ts3: v_neg_b,
                td: v_abs_b,
                imm: None,
                virtual_sequence_remaining: remain(vt.len()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(ge0_tensor.clone()),
                ts2_val: Some(z_tensor.clone()),
                ts3_val: Some(neg_b_tensor.clone()),
                td_pre_val: None,
                td_post_val: Some(abs_b_tensor.clone()),
            },
            advice_value: None,
        });

        // (5) c = 2^{|z|}
        let c_vals: Vec<u64> = abs_b_vals
            .iter()
            .map(|&ai| VirtualPow2::<WORD_SIZE>(ai as u64).to_lookup_output())
            .collect();
        let c_tensor = Tensor::from(u64_vec_to_i32_iter(&c_vals));
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualPow2,
                ts1: v_abs_b,
                ts2: None,
                ts3: None,
                td: v_c_pow2,
                imm: None,
                virtual_sequence_remaining: remain(vt.len()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(abs_b_tensor.clone()),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(c_tensor.clone()),
            },
            advice_value: None,
        });

        // (6) Q constant
        const Q: u64 = 128;
        let q_tensor = Tensor::from(u64_vec_to_i32_iter(&vec![Q; MAX_TENSOR_SIZE]));
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualConst,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_q_const,
                imm: Some(q_tensor.clone()),
                virtual_sequence_remaining: remain(vt.len()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
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
        let d_q_over_c_vals: Vec<u64> = c_vals.iter().map(|&ci| Q / ci).collect();
        let d_q_over_c_tensor = Tensor::from(u64_vec_to_i32_iter(&d_q_over_c_vals));
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Div,
                ts1: v_q_const,
                ts2: None,
                ts3: None,
                td: v_d_q_over_c,
                imm: Some(c_tensor.clone()),
                virtual_sequence_remaining: remain(vt.len()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
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

        // (8) d_q_times_c = Q * c
        let d_q_times_c_vals: Vec<u64> = c_vals.iter().map(|&ci| Q * ci).collect();
        let d_q_times_c_tensor = Tensor::from(u64_vec_to_i32_iter(&d_q_times_c_vals));
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Mul,
                ts1: v_q_const,
                ts2: v_c_pow2,
                ts3: None,
                td: v_d_q_times_c,
                imm: None,
                virtual_sequence_remaining: remain(vt.len()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(q_tensor.clone()),
                ts2_val: Some(c_tensor.clone()),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(d_q_times_c_tensor.clone()),
            },
            advice_value: None,
        });

        // (9) d = select(ge0, Q * c, Q / c)
        let d_vals: Vec<u64> = ge0_vals
            .iter()
            .enumerate()
            .map(|(i, cond)| {
                if *cond != 0 {
                    Q * c_vals[i]
                } else {
                    Q / c_vals[i]
                }
            })
            .collect();
        let d_tensor = Tensor::from(u64_vec_to_i32_iter(&d_vals));
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Select,
                ts1: v_ge0,
                ts2: v_d_q_times_c,
                ts3: v_d_q_over_c,
                td: v_d,
                imm: None,
                virtual_sequence_remaining: remain(vt.len()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(ge0_tensor.clone()),
                ts2_val: Some(d_q_times_c_tensor.clone()),
                ts3_val: Some(d_q_over_c_tensor.clone()),
                td_pre_val: None,
                td_post_val: Some(d_tensor.clone()),
            },
            advice_value: None,
        });

        // (10) e_sum = Σ d
        let e_sum: u64 = d_vals.iter().copied().sum();
        let mut e_tensor = Tensor::from(u64_vec_to_i32_iter(&vec![0; MAX_TENSOR_SIZE]));
        e_tensor[0] = e_sum as u32 as i32;
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Sum,
                ts1: v_d,
                ts2: None,
                ts3: None,
                td: v_e_sum,
                imm: None,
                virtual_sequence_remaining: remain(vt.len()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: [1, 1],
            },
            memory_state: MemoryState {
                ts1_val: Some(d_tensor.clone()),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(e_tensor.clone()),
            },
            advice_value: None,
        });

        // (11) Broadcast e_sum
        let broadcast_e_sum_vals: Vec<u64> = vec![e_sum; MAX_TENSOR_SIZE];
        let broadcast_e_sum_tensor = Tensor::from(u64_vec_to_i32_iter(&broadcast_e_sum_vals));
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Broadcast,
                ts1: v_e_sum,
                ts2: None,
                ts3: None,
                td: v_broadcast_e_sum,
                imm: None,
                virtual_sequence_remaining: remain(vt.len()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(e_tensor.clone()),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(broadcast_e_sum_tensor.clone()),
            },
            advice_value: None,
        });

        // (12) f = Q * d
        let f_vals: Vec<u64> = d_vals.iter().map(|&di| Q.saturating_mul(di)).collect();
        let f_tensor = Tensor::from(u64_vec_to_i32_iter(&f_vals));
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Mul,
                ts1: v_d,
                ts2: v_q_const,
                ts3: None,
                td: v_f_q_times_d,
                imm: None,
                virtual_sequence_remaining: remain(vt.len()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
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

        // (13) g = f / e
        let g_vals: Vec<u64> = f_vals
            .iter()
            .map(|&fi| if e_sum == 0 { 0 } else { fi / e_sum })
            .collect();
        let g_tensor = Tensor::from(u64_vec_to_i32_iter(&g_vals));
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Div,
                ts1: v_f_q_times_d,
                ts2: None,
                ts3: None,
                td: v_g_out,
                imm: Some(broadcast_e_sum_tensor.clone()),
                virtual_sequence_remaining: remain(vt.len()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
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

        // (14) Move final result
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualMove,
                ts1: v_g_out,
                ts2: None,
                ts3: None,
                td: cycle.instr.td,
                imm: None,
                virtual_sequence_remaining: remain(vt.len()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(g_tensor.clone()),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: cycle.memory_state.td_pre_val.clone(),
                td_post_val: Some(g_tensor.clone()),
            },
            advice_value: None,
        });

        debug_assert_eq!(vt.len(), Self::SEQUENCE_LENGTH, "sequence length mismatch");
        vt
    }

    // Reference CPU-side quantized implementation for verification.
    fn sequence_output(x: Vec<u64>, _y: Vec<u64>, _op: Option<ONNXOpcode>) -> Vec<u64> {
        let mut out = vec![0u64; MAX_TENSOR_SIZE];
        const Q: u64 = 128;

        let mut d_sum: u64 = 0;
        let mut d_vec: [u64; MAX_TENSOR_SIZE] = [0; MAX_TENSOR_SIZE];

        for i in 0..MAX_TENSOR_SIZE {
            let b = x[i] as i32;

            // Compute 2^|b| safely:
            let abs_b = b.unsigned_abs();
            let abs_b = abs_b.min(63); // cap to avoid overflow
            let pow2 = 1u64.checked_shl(abs_b).unwrap_or(u64::MAX);

            // Standard softmax semantics
            let d = if b >= 0 {
                Q.saturating_mul(pow2)
            } else if pow2 == 0 {
                Q
            } else {
                Q.saturating_div(pow2)
            };

            d_vec[i] = d;
            d_sum = d_sum.saturating_add(d);
        }

        // Normalize
        for i in 0..MAX_TENSOR_SIZE {
            let f = Q.saturating_mul(d_vec[i]);
            let g = if d_sum == 0 { 0 } else { f / d_sum };
            out[i] = g;
        }

        out
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::jolt::instruction::test::jolt_virtual_sequence_test;

    #[test]
    fn softmax_virtual_sequence_32() {
        jolt_virtual_sequence_test::<SoftmaxInstruction<32>>(ONNXOpcode::Softmax);
    }
}
