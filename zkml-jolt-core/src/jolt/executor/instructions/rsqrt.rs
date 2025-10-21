use itertools::Itertools;
use jolt_core::zkvm::instruction::LookupQuery;
use onnx_tracer::{
    constants::virtual_tensor_index,
    tensor::Tensor,
    trace_types::{MemoryState, ONNXCycle, ONNXInstr, ONNXOpcode},
};

use crate::{
    jolt::executor::instructions::{
        VirtualInstructionSequence, add::AddInstruction, mul::MulInstruction, sub::SubInstruction,
    },
    utils::u64_vec_to_i32_iter,
};

// Scale factor
const SF: u64 = 128;
const SF_LOG: u64 = 7;

/// Performs 1 / sqrt(x) and return the result
pub struct RsqrtInstruction<const WORD_SIZE: usize>;
// TODO(AntoineF4C5): Handle x < 0 case
impl<const WORD_SIZE: usize> VirtualInstructionSequence for RsqrtInstruction<WORD_SIZE> {
    const SEQUENCE_LENGTH: usize = 14;

    // TODO(AntoineF4C5): Think about how to implement rescaling.
    // Could create new instruction (Rescale) that consists on a left or right shift by the SF_LOG amount.
    // Probably Less costly than a div by SF

    // Follows https://rusteddreams.bitbucket.io/2017/03/05/sqrt.html, here E corresponds to -E in the article
    fn virtual_trace(cycle: ONNXCycle, K: usize) -> Vec<ONNXCycle> {
        const DIV_VIRTUAL_ADDRESSES: usize = 5;
        assert_eq!(cycle.instr.opcode, ONNXOpcode::Rsqrt);
        let num_outputs = cycle.instr.active_output_elements;
        // DIV source registers
        let r_x = cycle.instr.ts1;

        // Virtual registers used in sequence
        let v_one = Some(virtual_tensor_index(
            DIV_VIRTUAL_ADDRESSES,
            K,
            cycle.instr.td.unwrap(),
        ));
        let v_a = Some(virtual_tensor_index(
            1 + DIV_VIRTUAL_ADDRESSES,
            K,
            cycle.instr.td.unwrap(),
        ));
        let v_d = Some(virtual_tensor_index(
            2 + DIV_VIRTUAL_ADDRESSES,
            K,
            cycle.instr.td.unwrap(),
        ));
        let v_d_sq_ns = Some(virtual_tensor_index(
            3 + DIV_VIRTUAL_ADDRESSES,
            K,
            cycle.instr.td.unwrap(),
        ));
        let v_d_sq = Some(virtual_tensor_index(
            4 + DIV_VIRTUAL_ADDRESSES,
            K,
            cycle.instr.td.unwrap(),
        ));
        let v_xd_sq_ns = Some(virtual_tensor_index(
            5 + DIV_VIRTUAL_ADDRESSES,
            K,
            cycle.instr.td.unwrap(),
        ));
        let v_xd_sq = Some(virtual_tensor_index(
            6 + DIV_VIRTUAL_ADDRESSES,
            K,
            cycle.instr.td.unwrap(),
        ));
        let v_xd_sq_minus_1 = Some(virtual_tensor_index(
            7 + DIV_VIRTUAL_ADDRESSES,
            K,
            cycle.instr.td.unwrap(),
        ));
        let v_xd_cub_minus_d_ns = Some(virtual_tensor_index(
            8 + DIV_VIRTUAL_ADDRESSES,
            K,
            cycle.instr.td.unwrap(),
        ));
        let v_xd_cub_minus_d = Some(virtual_tensor_index(
            9 + DIV_VIRTUAL_ADDRESSES,
            K,
            cycle.instr.td.unwrap(),
        ));
        let v_axd_cub_minus_d_ns = Some(virtual_tensor_index(
            10 + DIV_VIRTUAL_ADDRESSES,
            K,
            cycle.instr.td.unwrap(),
        ));
        let v_axd_cub_minus_d = Some(virtual_tensor_index(
            11 + DIV_VIRTUAL_ADDRESSES,
            K,
            cycle.instr.td.unwrap(),
        ));
        let v_res = Some(virtual_tensor_index(
            12 + DIV_VIRTUAL_ADDRESSES,
            K,
            cycle.instr.td.unwrap(),
        ));

        // RSQRT operand
        let x = cycle.ts1_vals().unwrap_or(vec![0; num_outputs]);
        println!("x:    {}", x[0]);
        let mut virtual_trace = vec![];

        // const one (scaled)
        let one = vec![SF; num_outputs];
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualConst,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_one,
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&vec![SF; num_outputs]))),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: num_outputs,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: None,
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&vec![SF; num_outputs]))),
            },
            advice_value: None,
        });

        let sqrt_2 = (2f32.sqrt() * SF as f32).round() as u64;
        let a = (sqrt_2 as i32 / 2 - SF as i32) as u32 as u64;
        let a = vec![a; num_outputs];
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualConst,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_one,
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&a))),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: num_outputs,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: None,
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&a))),
            },
            advice_value: None,
        });

        // Will later use new log instruction
        let d = (0..num_outputs)
            .map(|i| 2_u64.pow((3 * SF_LOG as u32 - x[i].ilog2()) / 2))
            .collect_vec();
        println!("d:    {}", d[0]);

        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualConst,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_d,
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&d))),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: num_outputs,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: None,
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&d))),
            },
            advice_value: None,
        });

        let d_sq_ns = (0..num_outputs)
            .map(|i| MulInstruction::<WORD_SIZE>(d[i], d[i]).to_lookup_output())
            .collect_vec();
        println!("d_sq_ns:  {}", d_sq_ns[0]);
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Mul,
                ts1: v_d,
                ts2: v_d,
                ts3: None,
                td: v_d_sq_ns,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&d))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&d))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&d_sq_ns))),
            },
            advice_value: None,
        });

        let d_sq = (0..num_outputs).map(|i| d_sq_ns[i] / 128).collect_vec();
        println!("d_sq: {}", d_sq[0]);
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Div,
                ts1: v_d_sq_ns,
                ts2: None,
                ts3: None,
                td: v_d_sq,
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&vec![128; num_outputs]))),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&d_sq_ns))),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&d_sq))),
            },
            advice_value: None,
        });

        let xd_sq_ns = (0..num_outputs)
            .map(|i| MulInstruction::<WORD_SIZE>(x[i], d_sq[i]).to_lookup_output())
            .collect_vec();
        println!("xd_sq_ns: {}", xd_sq_ns[0]);
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Mul,
                ts1: r_x,
                ts2: v_d_sq,
                ts3: None,
                td: v_xd_sq_ns,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&x))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&d_sq))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd_sq_ns))),
            },
            advice_value: None,
        });

        let xd_sq = (0..num_outputs).map(|i| xd_sq_ns[i] / 128).collect_vec();
        println!("xd_sq:    {}", xd_sq[0]);
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Div,
                ts1: v_xd_sq_ns,
                ts2: None,
                ts3: None,
                td: v_xd_sq,
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&vec![128; num_outputs]))),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd_sq_ns))),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd_sq))),
            },
            advice_value: None,
        });

        let xd_sq_minus_1 = (0..num_outputs)
            .map(|i| SubInstruction::<WORD_SIZE>(xd_sq[i], one[i]).to_lookup_output())
            .collect_vec();
        println!("xd_sq_minus_1:    {}", xd_sq_minus_1[0]);
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Sub,
                ts1: v_xd_sq,
                ts2: v_one,
                ts3: None,
                td: v_xd_sq_minus_1,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd_sq))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&one))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd_sq_minus_1))),
            },
            advice_value: None,
        });

        let xd_cub_minus_d_ns = (0..num_outputs)
            .map(|i| MulInstruction::<WORD_SIZE>(xd_sq_minus_1[i], d[i]).to_lookup_output())
            .collect_vec();
        println!("xd_cub_minus_d_ns:    {}", xd_cub_minus_d_ns[0]);
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Mul,
                ts1: v_xd_sq_minus_1,
                ts2: v_d,
                ts3: None,
                td: v_xd_cub_minus_d_ns,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd_sq_minus_1))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&d))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd_cub_minus_d_ns))),
            },
            advice_value: None,
        });

        let xd_cub_minus_d = (0..num_outputs)
            .map(|i| xd_cub_minus_d_ns[i] / SF)
            .collect_vec();
        println!("xd_cub_minus_d:   {}", xd_cub_minus_d[0]);
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Div,
                ts1: v_xd_cub_minus_d_ns,
                ts2: None,
                ts3: None,
                td: v_xd_cub_minus_d,
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&vec![SF; num_outputs]))),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd_cub_minus_d_ns))),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd_cub_minus_d))),
            },
            advice_value: None,
        });

        let axd_cub_minus_d_ns = (0..num_outputs)
            .map(|i| MulInstruction::<WORD_SIZE>(xd_cub_minus_d[i], a[i]).to_lookup_output())
            .collect_vec();
        println!("axd_cub_minus_d_ns:   {}", axd_cub_minus_d_ns[0]);
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Mul,
                ts1: v_xd_cub_minus_d,
                ts2: v_a,
                ts3: None,
                td: v_axd_cub_minus_d_ns,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd_cub_minus_d))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&a))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&axd_cub_minus_d_ns))),
            },
            advice_value: None,
        });

        let axd_cub_minus_d = (0..num_outputs)
            .map(|i| axd_cub_minus_d_ns[i] / SF)
            .collect_vec();
        println!("axd_cub_minus_d:  {}", axd_cub_minus_d[0]);
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Div,
                ts1: v_axd_cub_minus_d_ns,
                ts2: None,
                ts3: None,
                td: v_axd_cub_minus_d,
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&vec![SF; num_outputs]))),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&axd_cub_minus_d_ns))),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&axd_cub_minus_d))),
            },
            advice_value: None,
        });

        let res: Vec<u64> = (0..num_outputs)
            .map(|i| AddInstruction::<WORD_SIZE>(d[i], axd_cub_minus_d[i]).to_lookup_output())
            .collect_vec();
        println!("res: {}", res[0]);
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Mul,
                ts1: v_xd_cub_minus_d,
                ts2: v_a,
                ts3: None,
                td: v_axd_cub_minus_d_ns,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&xd_cub_minus_d))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&a))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&axd_cub_minus_d_ns))),
            },
            advice_value: None,
        });

        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualMove,
                ts1: v_res,
                ts2: None,
                ts3: None,
                td: cycle.instr.td,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&res))),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: cycle.memory_state.td_pre_val.clone(),
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&res))),
            },
            advice_value: None,
        });

        virtual_trace
    }

    fn sequence_output(x: Vec<u64>, _: Vec<u64>, _: Option<ONNXOpcode>) -> Vec<u64> {
        let sf_log = SF.ilog2();
        let sf = SF as i32;
        let num_outputs = x.len();
        let mut output = vec![0; num_outputs];
        for i in 0..num_outputs {
            let q_x = x[i] as u32 as i32;
            let d = 2_i32.pow((3 * sf_log - q_x.ilog2()) / 2);

            let sqrt_2 = (2f32.sqrt() * SF as f32).round() as i32;

            let a = sqrt_2 / 2 - sf;
            let d_sq = d * d / sf;
            let xd_sq = q_x * d_sq / sf;
            let xd_sq_minus_1 = xd_sq - sf;
            let xd_cub_minus_d = d * xd_sq_minus_1 / sf;
            let axd_cub_minus_d = a * xd_cub_minus_d / sf;
            let approximation = d + axd_cub_minus_d;

            // Newton's method
            // let approximation = (approximation
            //     * (3 * sf - ((approximation * q_x) / sf * approximation) / sf))
            //     / (2 * sf);

            // output[i] = ((approximation
            //     * (3 * sf - ((approximation * q_x) / sf * approximation) / sf))
            //     / (2 * sf)) as u32 as u64;
            output[i] = approximation as u32 as u64;
        }
        output
    }
}

#[cfg(test)]
mod test {
    use std::collections::BTreeMap;

    use crate::jolt::bytecode::JoltONNXBytecode;
    use crate::jolt::executor::instructions::test::jolt_virtual_sequence_test;
    use crate::jolt::trace::inline_tensor_cycle;

    use super::*;
    use rand::prelude::*;

    const SF_LOG: usize = 7;

    #[test]
    fn div_virtual_sequence_32() {
        jolt_virtual_sequence_test::<RsqrtInstruction<32>>(ONNXOpcode::Rsqrt, 16);
    }

    fn to_lookup_output(cycle: &ONNXCycle) -> Vec<u64> {
        let output_els = cycle.instr.active_output_elements;
        let mut bytecode_line = JoltONNXBytecode::no_op();
        bytecode_line.opcode = cycle.instr.opcode.clone();
        let mut bytecode = vec![bytecode_line.clone(); output_els];
        for i in 0..output_els {
            bytecode[i].imm = cycle.instr.imm.as_ref().map_or(0, |t| t[i] as u32 as u64);
        }
        let cycles = inline_tensor_cycle(cycle, &bytecode);
        (0..output_els)
            .map(|i| cycles[i].to_lookup_output())
            .collect()
    }

    #[test]
    fn check_virtual_steps() {
        let mut rng = StdRng::seed_from_u64(123456);
        let output_size = 1;

        let next_input = |rng: &mut StdRng| rng.next_u32() >> (1 + SF_LOG + 12); // shift left by 1 (so sign_bit == 0) + SF_LOG (those msb's value is lost in some calculations)

        for _ in 0..1000 {
            // Randomly select tensor register's indices for t_x, t_y, and td (destination tensor register).
            // t_x and t_y are source tensor_registers, td is the destination tensor register.
            let t_x = rng.next_u64() % 32;
            let t_y = rng.next_u64() % 32;

            // Ensure td is not zero
            let mut td = rng.next_u64() % 32;
            while td == 0 {
                td = rng.next_u64() % 32;
            }

            // Assign a random value to x, but if t_x is zero, force x to be zero.
            // This simulates the behavior of register zero.
            let x = if t_x == 0 {
                vec![0u64; output_size]
            } else {
                (0..output_size)
                    .map(|_| next_input(&mut rng) as u64)
                    .collect::<Vec<u64>>()
            };

            // Assign a value to y:
            // - If t_y == t_x, y is set to x (ensures both source (tensor) tensor_registers have the same value).
            // - If t_y is zero, y is forced to zero (simulating zero (tensor) register).
            // - Otherwise, y is assigned a random value.
            let y = if t_y == t_x {
                x.clone()
            } else if t_y == 0 {
                vec![0u64; output_size]
            } else {
                (0..output_size)
                    .map(|_| next_input(&mut rng) as u64)
                    .collect::<Vec<u64>>()
            };

            let result = RsqrtInstruction::<32>::sequence_output(x.clone(), y.clone(), None);

            let mut tensor_registers = vec![vec![0u64; output_size]; 64];
            tensor_registers[t_x as usize] = x.clone();
            tensor_registers[t_y as usize] = y.clone();

            let cycle = ONNXCycle {
                instr: ONNXInstr {
                    address: rng.next_u64() as usize,
                    opcode: ONNXOpcode::Rsqrt,
                    ts1: Some(t_x as usize),
                    ts2: Some(t_y as usize),
                    ts3: None,
                    td: Some(td as usize),
                    imm: Some(Tensor::from(u64_vec_to_i32_iter(&y))),
                    virtual_sequence_remaining: None,
                    active_output_elements: output_size,
                    output_dims: [1, output_size],
                },
                memory_state: MemoryState {
                    ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&x))),
                    ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&y))),
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&result))),
                },
                advice_value: None,
            };

            let virtual_sequence = RsqrtInstruction::<32>::virtual_trace(cycle, 32);
            assert_eq!(
                virtual_sequence.len(),
                RsqrtInstruction::<32>::SEQUENCE_LENGTH
            );

            // Create a mapping for virtual registers (>= 32) to available register slots
            let mut virtual_register_map = BTreeMap::new();
            let mut next_virtual_slot = 33;

            for (i, cycle) in virtual_sequence.iter().enumerate() {
                if let Some(ts1_addr) = cycle.instr.ts1 {
                    let mapped_addr = if ts1_addr >= 32 {
                        *virtual_register_map.entry(ts1_addr).or_insert_with(|| {
                            let slot = next_virtual_slot;
                            next_virtual_slot += 1;
                            slot
                        })
                    } else {
                        ts1_addr
                    };
                    assert_eq!(
                        tensor_registers[mapped_addr],
                        cycle.ts1_vals().unwrap(),
                        "{i}: {cycle:#?}"
                    );
                }

                if let Some(ts2_addr) = cycle.instr.ts2 {
                    let mapped_addr = if ts2_addr >= 32 {
                        *virtual_register_map.entry(ts2_addr).or_insert_with(|| {
                            let slot = next_virtual_slot;
                            next_virtual_slot += 1;
                            slot
                        })
                    } else {
                        ts2_addr
                    };
                    assert_eq!(
                        tensor_registers[mapped_addr],
                        cycle.ts2_vals().unwrap(),
                        "{i}: {cycle:#?}"
                    );
                }

                let output = to_lookup_output(cycle);

                if let Some(td_addr) = cycle.instr.td {
                    let mapped_addr = if td_addr >= 32 {
                        *virtual_register_map.entry(td_addr).or_insert_with(|| {
                            let slot = next_virtual_slot;
                            next_virtual_slot += 1;
                            slot
                        })
                    } else {
                        td_addr
                    };
                    // Only write active output elements, rest should be zero
                    let mut td_output = vec![0u64; output_size];
                    td_output[..cycle.instr.active_output_elements.min(output.len())]
                        .copy_from_slice(
                            &output[..cycle.instr.active_output_elements.min(output.len())],
                        );
                    tensor_registers[mapped_addr] = td_output;
                    assert_eq!(
                        tensor_registers[mapped_addr],
                        cycle.td_post_vals().unwrap(),
                        "{i}: {cycle:#?}"
                    );
                } else {
                    assert!(output == vec![1; output_size], "{cycle:#?}");
                }
            }

            // Find the mapped address for td if it was used in virtual instructions
            let mapped_td = virtual_register_map.values().find(|&&mapped_addr| {
                // Check if this mapped address contains the result
                if mapped_addr < tensor_registers.len() {
                    tensor_registers[mapped_addr] == result
                } else {
                    false
                }
            });

            for (index, val) in tensor_registers.iter().enumerate() {
                let is_mapped_td = matches!(mapped_td, Some(&mapped) if index == mapped);

                if index as u64 == t_x {
                    if t_x != td && !is_mapped_td {
                        // Check that t_x hasn't been clobbered
                        assert_eq!(*val, x);
                    }
                } else if index as u64 == t_y {
                    if t_y != td && !is_mapped_td {
                        // Check that t_y hasn't been clobbered
                        assert_eq!(*val, y);
                    }
                } else if index as u64 == td || is_mapped_td {
                    // Check that result was written to td (or its mapped virtual register)
                    assert_eq!(
                        *val, result,
                        "Lookup mismatch for x {x:?} y {y:?} td {td:?}"
                    );
                } else if index < 32 {
                    // None of the other "real" registers were touched
                    assert_eq!(
                        *val,
                        vec![0u64; output_size],
                        "Other 'real' registers should not be touched"
                    );
                }
            }
        }
    }
    const SF_FLOAT: f32 = SF as f32;
    const INFORMATION_BITS: usize = 32 - SF_LOG - 1; // WORD_SIZE - SF_LOG - sign_bit 

    fn next_pos_i32(rng: &mut impl Rng) -> i32 {
        // by doing right shift we ensure sign_bit == 0
        (rng.next_u32() >> 1) as i32
    }

    #[test]
    #[ignore]
    fn sequence_output() {
        let mut rng = StdRng::seed_from_u64(123456);

        for _ in 0..10_000 {
            // Input to the zkvm, this is a float with a precision of 1/SF, and lower than 2^(32 - 2*SF_LOG - 1) = 2^17 for SF_LOG = 7
            let input = {
                let input = next_pos_i32(&mut rng) % (1 << (INFORMATION_BITS));
                input as f32 / SF_FLOAT
            };
            let x = (input * SF_FLOAT) as i32 as u32 as u64; // cast to vm compatible type
            let expected = todo!(); // Will use onnx-tracer's operator to ensure compatibility
            let result = RsqrtInstruction::<32>::sequence_output(vec![x], vec![], None)[0];
            assert_eq!(
                result, expected as u64,
                "Mismatch: x = {x}, result =! expected: {result} != {expected}"
            );
        }
    }

    const E: u32 = 7;
    // Follows https://rusteddreams.bitbucket.io/2017/03/05/sqrt.html, here E corresponds to -E in the article
    #[test]
    fn test_approximation_error() {
        let mut rng = StdRng::seed_from_u64(123456);
        let sf_f32 = (1 << E) as f32;
        let sf = sf_f32 as i32;

        let mut total_error = 0.0;
        for _ in 0..1
        /* 0_000 */
        {
            let x: f32 = 13.171875; //rng.r#gen::<f32>() * num_traits::Pow::pow(2f32, 16) + 0.1f32; // get numbers between 0.1 and 100
            let expected = sf_f32 / x.sqrt();

            let q_x = {
                // quantize
                let q_x = (x * sf_f32).round() as i32;
                if q_x == 0 { 1 } else { q_x }
            };
            if q_x < 0 {
                println!("marker");
            }

            let approximation = if q_x.ilog2() > 3 * E {
                0 // really high 
            } else {
                let d = 2_i32.pow((3 * E - q_x.ilog2()) / 2);

                let sqrt_2 = (2f32.sqrt() * sf_f32).round() as i32;

                // f2
                // d + d * (d * d * x - 1) * (sqrt_2 / 2 - 1) // Scales are omitted in this expression
                let a = sqrt_2 / 2 - sf;
                let d_sq = d * d / sf;
                let xd_sq = q_x * d_sq / sf;
                let xd_sq_minus_1 = xd_sq - sf;
                let xd_cub_minus_d = d * xd_sq_minus_1 / sf;
                let axd_cub_minus_d = a * xd_cub_minus_d / sf;
                let approximation = d + axd_cub_minus_d;

                println!(
                    "d:    {d}
d^2:    {d_sq}
xd^2:   {xd_sq}
xd^2-1: {xd_sq_minus_1}
d(xd^2*1):   {xd_cub_minus_d}
ad(xd^2-1):  {axd_cub_minus_d}
d+ad(xd^2-1):{approximation}"
                );

                // Newton's method
                // y_{i+1} = y_i * (3 - x * y_i^2) / 2
                let approximation = (approximation
                    * (3 * sf - ((approximation * q_x) / sf * approximation) / sf))
                    / (2 * sf);

                (approximation * (3 * sf - ((approximation * q_x) / sf * approximation) / sf))
                    / (2 * sf)
            };

            // When expected < 1 and approximation is quantized to 0, this would incur an error of 100%, we however count it as valid as can't be more precise.
            let error = if approximation == 0 && expected < 1.0 {
                0.0
            } else {
                (100.0 - (approximation as f32 / expected * 100.0).abs()).abs()
            };

            total_error += error;

            // Never exceed 3% error, unless this is partly due to quantization (rsqrt output is relatively close to 0)
            assert!(
                error < 3.0 || expected < 0.5 * sf_f32,
                "Error exceeds 2 ULP and expected result >> ULP\n{approximation} != {expected}"
            )
        }

        let mean_error = total_error / 10_000.0;
        println!("mean error: {mean_error}%");
    }
}
