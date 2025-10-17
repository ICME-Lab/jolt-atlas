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

/// Quantized Softmax producing Q * softmax(z)
///
/// Steps:
/// 1. a <- z_max via ArgMax + Gather (replicate z_max to a full vector)
/// 2. b <- Sub(a, z)                      // b_i = z_max - z_i  (>= 0)
/// 3. c <- Pow2(b)                        // c_i = 2^{b_i}
/// 4. d <- Div(Q, c)                      // d_i = Q / 2^{b_i}
/// 5. e <- ReduceSum(d)                   // e = sum_j d_j  (replicated as a vector)
/// 6. f <- Mul(Q, d)                      // f_i = Q * d_i
/// 7. g <- Div(f, e)                      // g_i = f_i / e    == Q * softmax(z_i)
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct SoftmaxInstruction<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> VirtualInstructionSequence for SoftmaxInstruction<WORD_SIZE> {
    // ArgMax, Gather, Broadcast, Sub, VirtualPow2, VirtualConst(Q), Div(Q,c), ReduceSum(d),
    // Broadcast, Mul(Q,d), Div(f,e), VirtualMove  => 12 steps
    const SEQUENCE_LENGTH: usize = 12;

    fn virtual_trace(cycle: ONNXCycle) -> Vec<ONNXCycle> {
        assert_eq!(cycle.instr.opcode, ONNXOpcode::Softmax);

        let mut vt = Vec::with_capacity(Self::SEQUENCE_LENGTH);
        let remain = |vt_len: usize| Some(Self::SEQUENCE_LENGTH - (vt_len + 1));

        // ---- Virtual registers (distinct) ----
        let v_argmax_idx = Some(virtual_tensor_index(0));
        let v_zmax_vec = Some(virtual_tensor_index(1));
        let v_broadcast_zmax = Some(virtual_tensor_index(2));
        let v_b = Some(virtual_tensor_index(3));
        let v_c_pow2 = Some(virtual_tensor_index(4));
        let v_q_const = Some(virtual_tensor_index(5));
        let v_d_q_over_c = Some(virtual_tensor_index(6));
        let v_e_sum = Some(virtual_tensor_index(7));
        let v_broadcast_e_sum = Some(virtual_tensor_index(8));
        let v_f_q_times_d = Some(virtual_tensor_index(9));
        let v_g_out = Some(virtual_tensor_index(10));

        // Input tensor z
        let z_u64 = cycle.ts1_vals();
        let z_tensor = Tensor::from(u64_vec_to_i32_iter(&z_u64));

        // 1a) ArgMax(z)
        let (argmax_idx, _zmax_val) = {
            let mut idx = 0usize;
            let mut best = z_u64[0];
            for (i, &z) in z_u64.iter().enumerate().skip(1) {
                if z >= best {
                    best = z;
                    idx = i;
                }
            }
            (idx, best)
        };
        let mut argmax_tensor = Tensor::from(u64_vec_to_i32_iter(&vec![0; MAX_TENSOR_SIZE]));
        argmax_tensor[0] = argmax_idx as u32 as i32;
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::ArgMax,
                ts1: cycle.instr.ts1,
                ts2: None,
                ts3: None,
                td: v_argmax_idx,
                imm: None,
                virtual_sequence_remaining: remain(vt.len()),
                active_output_elements: 1,
                output_dims: [1, 1],
            },
            memory_state: MemoryState {
                ts1_val: Some(z_tensor.clone()),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(argmax_tensor.clone()),
            },
            advice_value: None,
        });

        // 1b) Gather -> replicate z_max as a full vector
        let zmax_val_u64 = z_u64[argmax_idx];
        let mut zmax_vec: Vec<u64> = vec![0; MAX_TENSOR_SIZE];
        zmax_vec[0] = zmax_val_u64;
        let zmax_vec_tensor = Tensor::from(u64_vec_to_i32_iter(&zmax_vec));
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Gather,
                ts1: cycle.instr.ts1,
                ts2: v_argmax_idx,
                ts3: None,
                td: v_zmax_vec,
                imm: None,
                virtual_sequence_remaining: remain(vt.len()),
                active_output_elements: 1,
                output_dims: [1, 1],
            },
            memory_state: MemoryState {
                ts1_val: Some(z_tensor.clone()),
                ts2_val: Some(argmax_tensor.clone()),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(zmax_vec_tensor.clone()),
            },
            advice_value: None,
        });

        let broadcast_zmax_vals: Vec<u64> = vec![zmax_val_u64; MAX_TENSOR_SIZE];
        let broadcast_zmax_tensor = Tensor::from(u64_vec_to_i32_iter(&broadcast_zmax_vals));
        // TODO: Broadcast z_max to all elements in tensor
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Broadcast,
                ts1: v_zmax_vec,
                ts2: None,
                ts3: None,
                td: v_broadcast_zmax,
                imm: None,
                virtual_sequence_remaining: remain(vt.len()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(zmax_vec_tensor.clone()),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(broadcast_zmax_tensor.clone()),
            },
            advice_value: None,
        });

        // 2) b = z_max - z
        let b_vals: Vec<u64> = broadcast_zmax_vals
            .iter()
            .zip(z_u64.iter())
            .map(|(&a, &z)| a.wrapping_sub(z))
            .collect();
        let b_tensor = Tensor::from(u64_vec_to_i32_iter(&b_vals));
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Sub,
                ts1: v_broadcast_zmax,
                ts2: cycle.instr.ts1,
                ts3: None,
                td: v_b,
                imm: None,
                virtual_sequence_remaining: remain(vt.len()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(broadcast_zmax_tensor.clone()),
                ts2_val: Some(z_tensor.clone()),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(b_tensor.clone()),
            },
            advice_value: None,
        });

        // 3) c = 2^{b}
        let c_vals: Vec<u64> = b_vals
            .iter()
            .map(|&bi| VirtualPow2::<WORD_SIZE>(bi).to_lookup_output())
            .collect();
        let c_tensor = Tensor::from(u64_vec_to_i32_iter(&c_vals));
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualPow2,
                ts1: v_b,
                ts2: None,
                ts3: None,
                td: v_c_pow2,
                imm: None,
                virtual_sequence_remaining: remain(vt.len()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(b_tensor.clone()),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(c_tensor.clone()),
            },
            advice_value: None,
        });

        // 4) d = Q / c
        const Q: u64 = 128;
        let q_tensor = Tensor::from(u64_vec_to_i32_iter(&vec![Q; MAX_TENSOR_SIZE]));
        let d_vals: Vec<u64> = c_vals.iter().map(|&ci| Q / ci).collect();
        let d_tensor = Tensor::from(u64_vec_to_i32_iter(&d_vals));

        // Q const
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

        // Div(Q, c) with c as imm (mirrors your Sigmoid Div pattern)
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
                td_post_val: Some(d_tensor.clone()),
            },
            advice_value: None,
        });

        // 5) e = ReduceSum(d)  (store replicated)
        let e_sum: u64 = d_vals.iter().copied().sum();
        let mut e_tensor = Tensor::from(u64_vec_to_i32_iter(&vec![0; MAX_TENSOR_SIZE]));
        e_tensor[0] = e_sum as u32 as i32;
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Sum,
                ts1: v_d_q_over_c,
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

        // broadcast e_sum to all elements in tensor
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

        // 6) f = Q * d
        let f_vals: Vec<u64> = d_vals.iter().map(|&di| Q.saturating_mul(di)).collect();
        let f_tensor = Tensor::from(u64_vec_to_i32_iter(&f_vals));
        vt.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Mul,
                ts1: v_q_const,
                ts2: v_d_q_over_c,
                ts3: None,
                td: v_f_q_times_d,
                imm: None,
                virtual_sequence_remaining: remain(vt.len()),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(q_tensor.clone()),
                ts2_val: Some(d_tensor.clone()),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(f_tensor.clone()),
            },
            advice_value: None,
        });

        // 7) g = f / e
        // Use Div with e (replicated scalar) as ts2
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

        // Move to final td
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

    fn sequence_output(x: Vec<u64>, _y: Vec<u64>, _op: Option<ONNXOpcode>) -> Vec<u64> {
        let mut out = vec![0u64; MAX_TENSOR_SIZE];
        const Q: u64 = 128;

        // z_max
        let mut zmax = x[0];
        for &v in x.iter().skip(1) {
            if v > zmax {
                zmax = v;
            }
        }

        // c_i = 2^{zmax - z_i}, d_i = Q / c_i
        let mut d_sum: u64 = 0;
        let mut d_vec: [u64; MAX_TENSOR_SIZE] = [0; MAX_TENSOR_SIZE];
        for i in 0..MAX_TENSOR_SIZE {
            let b = zmax.saturating_sub(x[i]);
            let c = 1u64 << (b as u32); // 2^b
            let d = Q / c; // integer division
            d_vec[i] = d;
            d_sum = d_sum.saturating_add(d);
        }

        // g_i = (Q * d_i) / d_sum
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
