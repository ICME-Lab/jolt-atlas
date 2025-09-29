use itertools::Itertools;
use jolt_core::jolt::instruction::LookupQuery;
use onnx_tracer::{
    constants::{MAX_TENSOR_SIZE, virtual_tensor_index},
    tensor::Tensor,
    trace_types::{MemoryState, ONNXCycle, ONNXInstr, ONNXOpcode},
};

use crate::{
    jolt::instruction::{
        VirtualInstructionSequence, add::ADD, argmax::ArgMaxInstruction, beq::BEQInstruction,
        mul::MUL, virtual_advice::ADVICEInstruction,
        virtual_assert_valid_div0::AssertValidDiv0Instruction,
        virtual_assert_valid_signed_remainder::AssertValidSignedRemainderInstruction,
        virtual_pow2::VirtualPow2,
    },
    utils::u64_vec_to_i32_iter,
};

/// Perform softmax and return the result
pub struct SoftmaxInstruction<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> VirtualInstructionSequence for SoftmaxInstruction<WORD_SIZE> {
    const SEQUENCE_LENGTH: usize = ArgMaxInstruction::<WORD_SIZE>::SEQUENCE_LENGTH + MAX_TENSOR_SIZE + 8;
        // ArgMaxInstruction::<WORD_SIZE>::SEQUENCE_LENGTH + (5 * MAX_TENSOR_SIZE) + 3;

    fn virtual_trace(cycle: ONNXCycle) -> Vec<ONNXCycle> {
        assert_eq!(cycle.instr.opcode, ONNXOpcode::Softmax);

        let z = cycle.ts1_vals();
        let mut virtual_trace = vec![];

        // Virtual registers
        let v_idx = Some(virtual_tensor_index(0)); // argmax index
        let v_zmax = Some(virtual_tensor_index(1)); // z_max value
        let v_num = Some(virtual_tensor_index(2)); // per-elem numerator = z_i * 63
        let v_scaled = Some(virtual_tensor_index(3)); // per-elem z'_i after division
        let v_a = Some(virtual_tensor_index(4)); // per-elem a_i = 2^(z'_i)
        let v_sum = Some(virtual_tensor_index(5)); // running sum for N
        let v_broadcast_sum = Some(virtual_tensor_index(6)); // running sum for N
        let v_probs = Some(virtual_tensor_index(7)); // per-elem p_i
        let v_const_63 = Some(virtual_tensor_index(8)); // const 63
        // let v_diff = Some(virtual_tensor_index(9)); // per-elem z_i - z_max

        // ----------------------------------------------------------------
        // Find z_max
        // 1. Use ArgMaxInstruction to get the index of the max.
        // 2. Use Gather to fetch the max value.
        // Let this be z_max.
        // ----------------------------------------------------------------
        let argmax_cycles = ArgMaxInstruction::<WORD_SIZE>::virtual_trace(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::ArgMax,
                ts1: cycle.instr.ts1,
                ts2: None,
                ts3: None,
                td: v_idx,
                imm: None,
                virtual_sequence_remaining: None,
                active_output_elements: 1,
                output_dims: [1, 1],
            },
            memory_state: cycle.memory_state.clone(),
            advice_value: None,
        });

        virtual_trace.extend(argmax_cycles);

        // 2) Gather z[idx] into v_zmax
        let x = z
            .iter()
            .map(|&v| v as u32 as i32 as i64)
            .collect::<Vec<_>>();
        let mut z_max_idx = 0;
        let mut z_max_val = x[0];
        for (i, &xi) in x.iter().enumerate().skip(1) {
            if xi >= z_max_val {
                z_max_val = xi;
                z_max_idx = i;
            }
        }
        let mut zmax_tensor = vec![0; MAX_TENSOR_SIZE];
        zmax_tensor[0] = z[z_max_idx];

        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Gather,
                ts1: cycle.instr.ts1,
                ts2: v_idx,
                ts3: None,
                td: v_zmax,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: 1,
                output_dims: [1, 1],
            },
            memory_state: MemoryState {
                ts1_val: cycle.memory_state.ts1_val.clone(),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&vec![z_max_idx as u64]))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&zmax_tensor))),
            },
            advice_value: None,
        });

        // TODO: Use VirtualAdvice instead of VirtualConst
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualConst,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_const_63,
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&[63]))),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: 1,
                output_dims: [1, 1],
            },
            memory_state: MemoryState {
                ts1_val: None,
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&[63]))),
            },
            advice_value: None,
        });

        let const_63_tensor = vec![63u64; MAX_TENSOR_SIZE];
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualConst,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_const_63,
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&const_63_tensor))),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: MAX_TENSOR_SIZE,
                output_dims: [1, MAX_TENSOR_SIZE],
            },
            memory_state: MemoryState {
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&const_63_tensor))),
                ts1_val: None,
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
            },
            advice_value: None,
        });

        let z_times_63: Vec<u64> = z.iter().map(|&zi| zi * 63).collect();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Mul,
                ts1: cycle.instr.ts1,
                ts2: v_const_63,
                ts3: None,
                td: v_num,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: MAX_TENSOR_SIZE,
                output_dims: [1, MAX_TENSOR_SIZE],
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&z))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&const_63_tensor))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&z_times_63))),
            },
            advice_value: None,
        });

        println!("virtual_trace: {:#?}", virtual_trace);

        let z_max = z[z_max_idx];
        let zmax_tensor = vec![z_max; MAX_TENSOR_SIZE];
        let scaled: Vec<u64> = z_times_63.iter().map(|&ni| ni / z_max).collect();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Div,
                ts1: v_num,
                ts2: v_zmax, // broadcast z_max
                ts3: None,
                td: v_scaled,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: MAX_TENSOR_SIZE,
                output_dims: [1, MAX_TENSOR_SIZE],
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&z_times_63))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&zmax_tensor))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&scaled))),
            },
            advice_value: None,
        });

        let a_vals: Vec<u64> = scaled
            .iter()
            .map(|&e| VirtualPow2::<WORD_SIZE>(e).to_lookup_output())
            .collect();

        // One vectorized VirtualPow2 cycle
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualPow2,
                ts1: v_scaled, // exponent tensor (z'_i)
                ts2: None,
                ts3: None,
                td: v_a, // weights a
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: MAX_TENSOR_SIZE,
                output_dims: [1, MAX_TENSOR_SIZE],
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&scaled))),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&a_vals))),
            },
            advice_value: None,
        });

        // ----------------------------------------------------------
        // (4) N = sum_i a_i  via ADD reductions (linear chain)
        // ----------------------------------------------------------
        let mut running_sum = a_vals.get(0).copied().unwrap_or(0);

        for i in 1..MAX_TENSOR_SIZE {
            let prev = running_sum;
            let next = running_sum + a_vals[i];
            running_sum = running_sum.saturating_add(a_vals[i]);

            // ADD: v_sum = v_sum + a_i
            virtual_trace.push(ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::Add,
                    ts1: v_sum,
                    ts2: v_a, // we materialize a_i for this step
                    ts3: None,
                    td: v_sum,
                    imm: None,
                    virtual_sequence_remaining: Some(
                        Self::SEQUENCE_LENGTH - virtual_trace.len() - 1,
                    ),
                    active_output_elements: 1,
                    output_dims: [1, 1],
                },
                memory_state: MemoryState {
                    ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&[prev]))),
                    ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&[a_vals[i]]))),
                    ts3_val: None,
                    td_pre_val: Some(Tensor::from(u64_vec_to_i32_iter(&[prev]))),
                    td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&[next]))),
                },
                advice_value: None,
            });
        }

        let n_tensor = vec![running_sum; MAX_TENSOR_SIZE];
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualConst,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_broadcast_sum, // virtual register holding N
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&n_tensor))),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: MAX_TENSOR_SIZE,
                output_dims: [1, MAX_TENSOR_SIZE],
            },
            memory_state: MemoryState {
                ts1_val: None,
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&n_tensor))),
            },
            advice_value: None,
        });

        let probs: Vec<u64> = a_vals.iter().map(|&ai| ai / running_sum).collect();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Div,
                ts1: v_a, // the tensor [a_1, ..., a_n]
                ts2: v_broadcast_sum, // the broadcasted N
                ts3: None,
                td: v_probs,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: MAX_TENSOR_SIZE,
                output_dims: [1, MAX_TENSOR_SIZE],
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&a_vals))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&n_tensor))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&probs))),
            },
            advice_value: None,
        });

        println!("virtual_trace.len(): {}", virtual_trace.len());

        println!("Self::SEQUENCE_LENGTH: {}", Self::SEQUENCE_LENGTH);
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualMove,
                ts1: v_probs,
                ts2: None,
                ts3: None,
                td: cycle.instr.td,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&probs))),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: cycle.memory_state.td_pre_val.clone(),
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&probs))),
            },
            advice_value: None,
        });

        virtual_trace
    }

    fn sequence_output(z: Vec<u64>, _imm: Vec<u64>, _: Option<ONNXOpcode>) -> Vec<u64> {
        // Host-side oracle (integer version)
        if MAX_TENSOR_SIZE == 0 {
            return vec![];
        }
        let mut z_max = i64::MIN;
        for i in 0..MAX_TENSOR_SIZE {
            z_max = z_max.max(z[i] as i64);
        }
        let zmax_u64 = (z_max as u64).max(1);

        let mut a: Vec<u128> = vec![0; MAX_TENSOR_SIZE];
        for i in 0..MAX_TENSOR_SIZE {
            let num = (z[i] as u128).saturating_mul(63);
            let zpi = (num / zmax_u64 as u128) as u64;
            let zpi = zpi.min(63);
            a[i] = 1u128 << zpi;
        }
        let mut N: u128 = 0;
        for i in 0..MAX_TENSOR_SIZE {
            N = N.saturating_add(a[i]);
        }
        if N == 0 {
            return vec![0; MAX_TENSOR_SIZE];
        }

        let mut out = vec![0u64; MAX_TENSOR_SIZE];
        for i in 0..MAX_TENSOR_SIZE {
            out[i] = (a[i] / N) as u64; // integer division (0/1 mostly)
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
