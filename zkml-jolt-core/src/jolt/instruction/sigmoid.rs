use itertools::Itertools;
use jolt_core::jolt::instruction::LookupQuery;
use onnx_tracer::{
    constants::{MAX_TENSOR_SIZE, virtual_tensor_index},
    tensor::Tensor,
    trace_types::{MemoryState, ONNXCycle, ONNXInstr, ONNXOpcode},
};
use serde::{Deserialize, Serialize};

use crate::{
    jolt::instruction::{
        add::ADD, beq::BEQInstruction, mul::MUL, virtual_advice::ADVICEInstruction, virtual_assert_valid_div0::AssertValidDiv0Instruction, virtual_assert_valid_signed_remainder::AssertValidSignedRemainderInstruction, virtual_pow2::VirtualPow2, VirtualInstructionSequence
    },
    utils::u64_vec_to_i32_iter,
};
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct SigmoidInstruction<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> VirtualInstructionSequence for SigmoidInstruction<WORD_SIZE> {
    // Pow2 + Const(Q) + Mul(a,Q) + Const(1) + Add(a,1) + Div(num,den via imm) + Move
    const SEQUENCE_LENGTH: usize = 7;

    fn virtual_trace(cycle: ONNXCycle) -> Vec<ONNXCycle> {
        assert_eq!(cycle.instr.opcode, ONNXOpcode::Sigmoid);

        let z = cycle.ts1_vals();
        let mut virtual_trace = vec![];

        // Virtual registers
        let v_pow2      = Some(virtual_tensor_index(0)); // a = 2^z
        let v_q_const   = Some(virtual_tensor_index(1)); // broadcast Q
        let v_num       = Some(virtual_tensor_index(2)); // num = a * Q
        let v_one_const = Some(virtual_tensor_index(3)); // broadcast 1
        let v_den       = Some(virtual_tensor_index(4)); // den = a + 1
        let v_probs     = Some(virtual_tensor_index(5)); // result σ_Q(z)

        // ----------------------------
        // 1) a = 2^z  (vectorized)
        // ----------------------------
        let a_vals: Vec<u64> = z.iter()
            .map(|&zi| {
                VirtualPow2::<WORD_SIZE>(zi).to_lookup_output()
            })
            .collect();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualPow2,
                ts1: cycle.instr.ts1,
                ts2: None,
                ts3: None,
                td: v_pow2,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 1),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&z))),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&a_vals))),
            },
            advice_value: None,
        });

        // ----------------------------
        // 2) Q = 256 (broadcast)
        // ----------------------------
        const Q: u64 = 1 << 8;
        let q_tensor = vec![Q; MAX_TENSOR_SIZE];
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualConst,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_q_const,
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&q_tensor))),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 2),
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

        // --------------------------------------
        // 3) num = a * Q (vectorized multiply)
        // --------------------------------------
        let num_vals: Vec<u64> = a_vals.iter().map(|&a| a * Q).collect();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Mul,
                ts1: v_pow2,
                ts2: v_q_const,
                ts3: None,
                td: v_num,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 3),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&a_vals))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&q_tensor))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&num_vals))),
            },
            advice_value: None,
        });

        // ----------------------------
        // 4) 1 (broadcast)
        // ----------------------------
        let one_tensor = vec![1u64; MAX_TENSOR_SIZE];
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualConst,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_one_const,
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&one_tensor))),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 4),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: None,
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&one_tensor))),
            },
            advice_value: None,
        });

        // ----------------------------
        // 5) den = a + 1 (vectorized)
        // ----------------------------
        let den_vals: Vec<u64> = a_vals.iter().map(|&ai| ai + 1).collect();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Add,
                ts1: v_pow2,
                ts2: v_one_const,
                ts3: None,
                td: v_den,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 5),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&a_vals))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&one_tensor))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&den_vals))),
            },
            advice_value: None,
        });

        // ------------------------------------------------------------
        // 6) probs = num / den  (Div reads denominator from imm !)
        // ------------------------------------------------------------
        // 1) Materialize exactly what will go into the registers:
        let num_td: Vec<i32> = u64_vec_to_i32_iter(&num_vals).collect();
        let den_td: Vec<i32> = u64_vec_to_i32_iter(&den_vals).collect();

        // 2) Read them back as u64s the same way the harness does:
        let num_seen: Vec<u64> = num_td.iter().map(|&n| n as u64).collect();
        let den_seen: Vec<u64> = den_td.iter().map(|&d| d as u64).collect();

        // 3) Compute probs from *these* (the machine-view values):
        let probs_vals: Vec<u64> = num_seen.iter().zip(&den_seen)
            .map(|(&n, &d)| if d == 0 { 0 } else { n / d })
            .collect();
        // let probs_vals: Vec<u64> = num_vals
        //     .iter()
        //     .zip(&den_vals)
        //     .map(|(&n, &d)| if d == 0 { 0 } else { n / d })
        //     .collect();

        // assert_eq!(probs_vals, probs_vals_1);
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Div,
                ts1: v_num,                            
                ts2: None,                              
                ts3: None,
                td: v_probs,
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&den_vals))), 
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 6),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&num_vals))),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&probs_vals))),
            },
            advice_value: None,
        });

        // ----------------------------
        // 7) Move to destination
        // ----------------------------
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualMove,
                ts1: v_probs,
                ts2: None,
                ts3: None,
                td: cycle.instr.td,
                imm: None,
                virtual_sequence_remaining: Some(0),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&probs_vals))),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: cycle.memory_state.td_pre_val.clone(),
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&probs_vals))),
            },
            advice_value: None,
        });

        // println!("virtual_trace: {:#?}", virtual_trace);

        virtual_trace
    }

    fn sequence_output(x: Vec<u64>, _: Vec<u64>, _: Option<ONNXOpcode>) -> Vec<u64> {
        const Q: u64 = 1 << 8;
        let mut out = vec![0u64; MAX_TENSOR_SIZE];
        for i in 0..MAX_TENSOR_SIZE {
            let a = VirtualPow2::<WORD_SIZE>(x[i]).to_lookup_output(); // 2^z
            let den = a.saturating_add(1);
            let num = a.saturating_mul(Q);
            out[i] = if den == 0 { 0 } else { num / den };
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
