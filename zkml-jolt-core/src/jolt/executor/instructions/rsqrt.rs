use itertools::Itertools;
use jolt_core::zkvm::instruction::LookupQuery;
use num::integer::Roots;
use onnx_tracer::{
    constants::virtual_tensor_index,
    tensor::Tensor,
    trace_types::{MemoryState, ONNXCycle, ONNXInstr, ONNXOpcode},
};

use crate::{
    jolt::executor::instructions::{
        VirtualInstructionSequence, beq::BeqInstruction, div::DivInstruction, mul::MulInstruction,
        virtual_advice::AdviceInstruction,
    },
    utils::u64_vec_to_i32_iter,
};

// Scale factor
const SF_LOG: usize = 7;
const SF: u64 = 128;

/// Performs 1 / sqrt(x) and return the result
pub struct RsqrtInstruction<const WORD_SIZE: usize>;
// TODO(AntoineF4C5): Handle x < 0 case
impl<const WORD_SIZE: usize> VirtualInstructionSequence for RsqrtInstruction<WORD_SIZE> {
    const SEQUENCE_LENGTH: usize = 6 + 2 * DivInstruction::<WORD_SIZE>::SEQUENCE_LENGTH;

    // TODO(AntoineF4C5): Think about how to implement rescaling.
    // Could create new instruction (Rescale) that consists on a left or right shift by the SF_LOG amount.
    // Probably Less costly than a div by SF

    /// Uses a virtual advice with following trace, using the result as advice:
    ///
    /// - advice = 1 / sqrt(x)                      | scale_log = 7
    /// - a_sq_unscaled = advice * advice (= 1 / x) | scale_log = 14
    /// - a_sq = a_sq_unscaled / SF                 | scale_log = 7
    /// - xa_sq_unscaled = x * a_sq (= x * (1 / x)) | scale_log = 14
    /// - xa_sq = xa_sq_unscaled / SF               | scale_log = 7
    /// - assert_eq(xa_sq, 1)
    /// - write advice to td
    ///
    fn virtual_trace(cycle: ONNXCycle, K: usize) -> Vec<ONNXCycle> {
        assert_eq!(cycle.instr.opcode, ONNXOpcode::Rsqrt);
        let num_outputs = cycle.instr.active_output_elements;
        // DIV source registers
        let r_x = cycle.instr.ts1;

        // Virtual registers used in sequence
        let v_res = Some(virtual_tensor_index(0, K, cycle.instr.td.unwrap()));
        let v_res_sq_unscaled = Some(virtual_tensor_index(1, K, cycle.instr.td.unwrap()));
        let v_res_sq = Some(virtual_tensor_index(2, K, cycle.instr.td.unwrap()));
        let v_xres_sq_unscaled = Some(virtual_tensor_index(3, K, cycle.instr.td.unwrap()));
        let v_xres_sq = Some(virtual_tensor_index(4, K, cycle.instr.td.unwrap()));
        let v_one = Some(virtual_tensor_index(5, K, cycle.instr.td.unwrap()));

        // RSQRT operand
        let x = cycle.ts1_vals().unwrap_or(vec![0; num_outputs]);
        println!("x:    {}", x[0]);
        let mut virtual_trace = vec![];

        let rsqrt = {
            let mut rsqrt_tensor = vec![0; num_outputs];
            for i in 0..num_outputs {
                let x = x[i];
                let rsqrt = match WORD_SIZE {
                    32 => {
                        if x == 0 {
                            u32::MAX as u64
                        } else {
                            // x is input represented in the vm, i.e. scaled by SF (x = input * SF)
                            let sqrt = ((x * SF) as i32).sqrt(); // sqrt  = sqrt(x * SF)                 = sqrt(input * SF * SF)  = sqrt(input) * SF
                            let rsqrt = (SF * SF) as i32 / sqrt; // rsqrt = SF * SF / (sqrt(input) * SF) = (1 / sqrt(input)) * SF = rsqrt(input) * SF
                            rsqrt as u32 as u64
                        }
                    }
                    64 => {
                        if x == 0 {
                            u64::MAX
                        } else {
                            let sqrt = ((x * SF) as i64).sqrt();
                            let rsqrt = (SF * SF) as i64 / sqrt;
                            rsqrt as u64
                        }
                    }
                    _ => panic!("Unsupported WORD_SIZE: {WORD_SIZE}",),
                };
                rsqrt_tensor[i] = rsqrt;
            }
            rsqrt_tensor
        };

        let res = (0..num_outputs)
            .map(|i| AdviceInstruction::<WORD_SIZE>(rsqrt[i]).to_lookup_output())
            .collect_vec();
        println!("res:  {}", res[0]);
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualAdvice,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_res,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: None,
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&res))),
            },
            advice_value: Some(Tensor::from(u64_vec_to_i32_iter(&rsqrt))),
        });

        let res_sq_unscaled = (0..num_outputs)
            .map(|i| MulInstruction::<WORD_SIZE>(res[i], res[i]).to_lookup_output())
            .collect_vec();
        println!("res_sq_unscaled:  {}", res_sq_unscaled[0]);
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Mul,
                ts1: v_res,
                ts2: v_res,
                ts3: None,
                td: v_res_sq_unscaled,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&res))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&res))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&res_sq_unscaled))),
            },
            advice_value: None,
        });

        let res_sq = (0..num_outputs)
            .map(|i| res_sq_unscaled[i] / 128)
            .collect_vec();
        println!("res_sq:   {}", res_sq[0]);
        let div_trace = DivInstruction::<WORD_SIZE>::virtual_trace(
            ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::Div,
                    ts1: v_res_sq_unscaled,
                    ts2: None,
                    ts3: None,
                    td: v_res_sq,
                    imm: Some(Tensor::from(u64_vec_to_i32_iter(&vec![128; num_outputs]))),
                    virtual_sequence_remaining: Some(
                        Self::SEQUENCE_LENGTH - virtual_trace.len() - 1,
                    ),
                    active_output_elements: cycle.instr.active_output_elements,
                    output_dims: cycle.instr.output_dims,
                },
                memory_state: MemoryState {
                    ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&res_sq_unscaled))),
                    ts2_val: None,
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&res_sq))),
                },
                advice_value: None,
            },
            K + 6,
        );

        virtual_trace.extend(div_trace);

        let xres_sq_unscaled = (0..num_outputs)
            .map(|i| MulInstruction::<WORD_SIZE>(x[i], res_sq[i]).to_lookup_output())
            .collect_vec();
        println!("xres_sq_unscaled: {}", xres_sq_unscaled[0]);
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Mul,
                ts1: r_x,
                ts2: v_res_sq,
                ts3: None,
                td: v_xres_sq_unscaled,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&x))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&res_sq))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&xres_sq_unscaled))),
            },
            advice_value: None,
        });

        let xres_sq = (0..num_outputs)
            .map(|i| xres_sq_unscaled[i] / 128)
            .collect_vec();
        println!("xres_sq:   {}", xres_sq[0]);
        let div_trace = DivInstruction::<WORD_SIZE>::virtual_trace(
            ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::Div,
                    ts1: v_xres_sq_unscaled,
                    ts2: None,
                    ts3: None,
                    td: v_xres_sq,
                    imm: Some(Tensor::from(u64_vec_to_i32_iter(&vec![128; num_outputs]))),
                    virtual_sequence_remaining: Some(
                        Self::SEQUENCE_LENGTH - virtual_trace.len() - 1,
                    ),
                    active_output_elements: cycle.instr.active_output_elements,
                    output_dims: cycle.instr.output_dims,
                },
                memory_state: MemoryState {
                    ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&xres_sq_unscaled))),
                    ts2_val: None,
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&xres_sq))),
                },
                advice_value: None,
            },
            K + 12,
        );

        virtual_trace.extend(div_trace);

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

        let _assert_eq = (0..num_outputs)
            .map(|i| BeqInstruction::<WORD_SIZE>(xres_sq[i], 128).to_lookup_output())
            .collect_vec();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualAssertEq,
                ts1: v_xres_sq,
                ts2: v_one,
                ts3: None,
                td: None,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&xres_sq))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&one))),
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
        let num_outputs = x.len();
        let mut output = vec![0; num_outputs];
        for i in 0..num_outputs {
            let x = x[i];
            let x = x as i32;
            if x == 0 {
                output[i] = (1 << WORD_SIZE) - 1; // TODO(AntoineF4C5): current implementation might be -1 rather than intended i32::max
                continue;
            }
            let sqrt = (x * SF as i32).sqrt();
            let rsqrt = (SF * SF) as i32 / sqrt;
            output[i] = rsqrt as u32 as u64
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

                let output = to_lookup_output(&cycle);

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
    fn sequence_output() {
        let mut rng = StdRng::seed_from_u64(123456);

        for _ in 0..10000 {
            // Input to the zkvm, this is a float with a precision of 1/SF, and lower than 2^(32 - 2*SF_LOG - 1) = 2^17 for SF_LOG = 7
            let input = {
                let input = next_pos_i32(&mut rng) % (1 << (INFORMATION_BITS));
                input as f32 / SF_FLOAT
            };
            let x = (input * SF_FLOAT) as i32 as u32 as u64; // cast to vm compatible type
            let expected = (1.0 / input.sqrt()) * SF_FLOAT;
            let result = RsqrtInstruction::<32>::sequence_output(vec![x], vec![], None)[0];
            assert_eq!(
                result, expected as u64,
                "Mismatch: x = {x}, result =! expected: {result} != {expected}"
            );
        }
    }
}
