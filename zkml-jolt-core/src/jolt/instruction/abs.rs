use jolt_core::jolt::instruction::LookupQuery;
use onnx_tracer::{
    constants::{MAX_TENSOR_SIZE, virtual_tensor_index},
    tensor::Tensor,
    trace_types::{MemoryState, ONNXCycle, ONNXInstr, ONNXOpcode},
};
use serde::{Deserialize, Serialize};

use crate::{
    jolt::instruction::{VirtualInstructionSequence, ge::GEInstruction},
    utils::u64_vec_to_i32_iter,
};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct AbsInstruction<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> VirtualInstructionSequence for AbsInstruction<WORD_SIZE> {
    const SEQUENCE_LENGTH: usize = 3; // GE + Sub + Select

    fn virtual_trace(cycle: ONNXCycle) -> Vec<ONNXCycle> {
        assert_eq!(cycle.instr.opcode, ONNXOpcode::Abs);

        let x_vals = cycle.ts1_vals();
        let mut virtual_trace = vec![];

        // Virtual registers
        let v_cond = Some(virtual_tensor_index(0)); // mask for x >= 0
        let v_neg  = Some(virtual_tensor_index(1)); // -x
        let v_abs  = cycle.instr.td;                // final result

        // 1. Condition: x >= 0
        let cond_vals: Vec<u64> = x_vals
            .iter()
            .map(|&x| GEInstruction::<WORD_SIZE>(x, 0).to_lookup_output())
            .collect();

        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Gte,
                ts1: cycle.instr.ts1,
                ts2: None, // comparing to 0
                ts3: None,
                td: v_cond,
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&vec![0; MAX_TENSOR_SIZE]))),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 1),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&x_vals))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&vec![0; MAX_TENSOR_SIZE]))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&cond_vals))),
            },
            advice_value: None,
        });

        // 2. Negation: -x
        let neg_vals: Vec<u64> = x_vals.iter().map(|&x| {
            let xi = x as i32;
            (-xi) as u32 as u64
        }).collect();

        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Sub,
                ts1: None, // 0 - x
                ts2: cycle.instr.ts1,
                ts3: None,
                td: v_neg,
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&vec![0; MAX_TENSOR_SIZE]))),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 2),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&vec![0; MAX_TENSOR_SIZE]))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&x_vals))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&neg_vals))),
            },
            advice_value: None,
        });

        // 3. Select(cond, x, -x)
        let abs_vals: Vec<u64> = x_vals.iter().zip(&cond_vals).zip(&neg_vals)
            .map(|((&x, &cond), &nx)| if cond != 0 { x } else { nx })
            .collect();

        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Select,
                ts1: v_cond,
                ts2: cycle.instr.ts1,
                ts3: v_neg,
                td: v_abs,
                imm: None,
                virtual_sequence_remaining: Some(0),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&cond_vals))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&x_vals))),
                ts3_val: Some(Tensor::from(u64_vec_to_i32_iter(&neg_vals))),
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&abs_vals))),
            },
            advice_value: None,
        });

        virtual_trace
    }

    fn sequence_output(x: Vec<u64>, _y: Vec<u64>, _: Option<ONNXOpcode>) -> Vec<u64> {
        x.into_iter()
            .map(|xi| {
                let val = xi as i32;
                val.abs() as u32 as u64
            })
            .collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::jolt::instruction::test::jolt_virtual_sequence_test;

    #[test]
    fn abs_virtual_sequence_32() {
        jolt_virtual_sequence_test::<AbsInstruction<32>>(ONNXOpcode::Abs);
    }
}
