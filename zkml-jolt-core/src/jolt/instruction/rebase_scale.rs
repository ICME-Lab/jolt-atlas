use onnx_tracer::{
    constants::{MAX_TENSOR_SIZE, virtual_tensor_index},
    tensor::Tensor,
    trace_types::{MemoryState, ONNXCycle, ONNXInstr, ONNXOpcode},
};

use crate::{
    jolt::{
        instruction::{VirtualInstructionSequence, div::DIVInstruction},
        precompiles::matmult::{MatMultPrecompile, MatMultPrecompileDims},
    },
    utils::u64_vec_to_i32_iter,
};

macro_rules! expect_rebase_scale {
    ($cycle:expr) => {
        match $cycle.instr.opcode {
            ONNXOpcode::RebaseScale(_) => {}
            _ => panic!("Expected ONNXOpcode::RebaseScale"),
        }
    };
}

/// Perform signed division and return the result
pub struct REBASEInstruction<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> VirtualInstructionSequence for REBASEInstruction<WORD_SIZE> {
    const SEQUENCE_LENGTH: usize = 1 + DIVInstruction::<WORD_SIZE>::SEQUENCE_LENGTH;

    fn virtual_trace(cycle: ONNXCycle) -> Vec<ONNXCycle> {
        expect_rebase_scale!(cycle);

        let inner_opcode = match &cycle.instr.opcode {
            ONNXOpcode::RebaseScale(inner) => inner,
            _ => unreachable!(),
        };

        // Virtual registers used in sequence
        let v_0 = Some(virtual_tensor_index(0)); // Inner operator output to be rebased 

        let mut virtual_trace = vec![];

        // Apply inner operator
        let inner_opcode = (**inner_opcode).clone();
        let mut instr = cycle.instr.clone();
        instr.opcode = inner_opcode;
        let inner_res_0: Vec<u64> = {
            let x = cycle.ts1_vals();
            let y = cycle.ts2_vals();
            match instr.opcode {
                ONNXOpcode::MatMult => {
                    let k = cycle
                        .memory_state
                        .ts1_val
                        .as_ref()
                        .map(|tensor| tensor.dims()[1])
                        .unwrap_or(1);
                    let m = instr.output_dims[0];
                    let n = instr.output_dims[1];
                    let dims: MatMultPrecompileDims = (m, n, k);
                    let (result, _shape) =
                        MatMultPrecompile::new(x, y).matmult_rhs_transposed(dims);
                    let mut output = result
                        .iter()
                        .map(|&x| x as u32 as u64)
                        .collect::<Vec<u64>>();
                    output.resize(MAX_TENSOR_SIZE, 0);
                    output
                }
                ONNXOpcode::Mul => match WORD_SIZE {
                    8 => x
                        .iter()
                        .zip(y.iter())
                        .map(|(&a, &b)| (a as u8).wrapping_mul(b as u8) as u64)
                        .collect::<Vec<u64>>(),
                    32 => x
                        .iter()
                        .zip(y.iter())
                        .map(|(&a, &b)| (a as u32).wrapping_mul(b as u32) as u64)
                        .collect::<Vec<u64>>(),
                    64 => x
                        .iter()
                        .zip(y.iter())
                        .map(|(&a, &b)| a.wrapping_mul(b))
                        .collect::<Vec<u64>>(),
                    _ => panic!("Unsupported WORD_SIZE: {WORD_SIZE}"),
                },
                _ => panic!("Unimplemented inner opcode: {:?}", instr.opcode),
            }
        };

        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: instr.opcode,
                ts1: cycle.instr.ts1,
                ts2: cycle.instr.ts2,
                ts3: cycle.instr.ts3,
                td: v_0,
                imm: cycle.instr.imm.clone(),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: cycle.memory_state.ts1_val.clone(),
                ts2_val: cycle.memory_state.ts2_val.clone(),
                ts3_val: cycle.memory_state.ts3_val.clone(),
                td_pre_val: cycle.memory_state.td_pre_val.clone(),
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&inner_res_0))), // Needs to set to the inner operator's expected output
            },
            advice_value: None,
        });
        // Apply div operator by 2^scale
        let res = DIVInstruction::<WORD_SIZE>::sequence_output(
            inner_res_0.clone(),
            vec![128; MAX_TENSOR_SIZE],
            None,
        );
        assert_eq!(res, cycle.td_post_vals());

        let div_cycle = ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Div,
                ts1: v_0,
                ts2: None,
                ts3: None,
                td: cycle.instr.td,
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&vec![
                    128;
                    MAX_TENSOR_SIZE
                ]))),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&inner_res_0))),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&res))),
            },
            advice_value: None,
        };

        let div_virtual_trace = DIVInstruction::<WORD_SIZE>::virtual_trace(div_cycle);
        virtual_trace.extend(div_virtual_trace);

        virtual_trace
    }

    fn sequence_output(x: Vec<u64>, y: Vec<u64>, inner: Option<ONNXOpcode>) -> Vec<u64> {
        if inner.is_none() {
            panic!("Inner opcode must be provided for RebaseScale");
        }
        let Some(inner_opcode) = inner else {
            unreachable!()
        };

        match inner_opcode {
            ONNXOpcode::Mul => {
                let mul_res = match WORD_SIZE {
                    8 => x
                        .iter()
                        .zip(y.iter())
                        .map(|(&a, &b)| (a as u8).wrapping_mul(b as u8) as u64)
                        .collect::<Vec<u64>>(),
                    32 => x
                        .iter()
                        .zip(y.iter())
                        .map(|(&a, &b)| (a as u32).wrapping_mul(b as u32) as u64)
                        .collect::<Vec<u64>>(),
                    64 => x
                        .iter()
                        .zip(y.iter())
                        .map(|(&a, &b)| a.wrapping_mul(b))
                        .collect::<Vec<u64>>(),
                    _ => panic!("Unsupported WORD_SIZE: {WORD_SIZE}"),
                };
                DIVInstruction::<WORD_SIZE>::sequence_output(
                    mul_res,
                    vec![128; MAX_TENSOR_SIZE],
                    None,
                )
            }
            // TODO: Need to check other possible inner opcodes
            _ => panic!("Unimplemented inner opcode: {inner_opcode:?}"),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::jolt::instruction::test::jolt_virtual_sequence_test;

    #[test]
    fn rebasescale_virtual_sequence_32() {
        jolt_virtual_sequence_test::<REBASEInstruction<32>>(ONNXOpcode::RebaseScale(Box::new(
            ONNXOpcode::Mul,
        )));
    }
}
