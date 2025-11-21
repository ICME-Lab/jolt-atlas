use std::collections::BTreeMap;

use crate::{
    jolt::{
        bytecode::JoltONNXBytecode,
        executor::instructions::{
            InstructionLookup, VirtualInstructionSequence, div::DivInstruction,
        },
        trace::{JoltONNXCycle, inline_tensor_cycle},
    },
    utils::u64_vec_to_i32_iter,
};
use ark_std::test_rng;
use jolt_core::zkvm::instruction::LookupQuery;
use onnx_tracer::{
    tensor::Tensor,
    trace_types::{MemoryState, ONNXCycle, ONNXInstr, ONNXOpcode},
};
use rand::prelude::*;

pub fn materialize_entry_test(opcode: ONNXOpcode) {
    let mut rng = StdRng::seed_from_u64(12345);
    for _ in 0..10000 {
        let cycle_lookup = JoltONNXCycle::random(opcode.clone(), &mut rng)
            .lookup
            .unwrap();
        let table = cycle_lookup.lookup_table().unwrap();
        assert_eq!(
            cycle_lookup.to_lookup_output(),
            table.materialize_entry(cycle_lookup.to_lookup_index())
        );
    }
}

/// Tests the consistency and correctness of a virtual instruction sequence.
/// In detail:
/// 1. Sets the tensor_registers to given values for `x` and `y`.
/// 2. Constructs an `RVTraceRow` with the provided opcode and register values.
/// 3. Generates the virtual instruction sequence using the specified instruction type.
/// 4. Iterates over each row in the virtual sequence and validates the state changes.
/// 5. Verifies that the tensor_registers `t_x` and `t_y` have not been modified (not clobbered).
/// 6. Ensures that the result of the instruction sequence is correctly written to the `td` register.
/// 7. Checks that no unintended modifications have been made to other tensor_registers.
pub fn jolt_virtual_sequence_test<I: VirtualInstructionSequence>(
    opcode: ONNXOpcode,
    output_size: usize,
) {
    let mut rng = test_rng();

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
                .map(|_| rng.gen_range(-8..=8) as u32 as u64)
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
                .map(|_| rng.gen_range(-8..=8) as u32 as u64)
                .collect::<Vec<u64>>()
        };

        let result = I::sequence_output(x.clone(), y.clone());

        let mut tensor_registers = vec![vec![0u64; output_size]; 128];
        tensor_registers[t_x as usize] = x.clone();
        tensor_registers[t_y as usize] = y.clone();

        let cycle = ONNXCycle {
            instr: ONNXInstr {
                address: rng.next_u64() as usize,
                opcode: opcode.clone(),
                ts1: Some(t_x as usize),
                ts2: Some(t_y as usize),
                ts3: None,
                td: Some(td as usize),
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&y))),
                virtual_sequence_remaining: None,
                output_dims: vec![1, output_size],
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

        let virtual_sequence = I::virtual_trace(cycle, 32);
        assert_eq!(virtual_sequence.len(), I::SEQUENCE_LENGTH);

        // Create a mapping for virtual registers (>= 32) to available register slots
        let mut virtual_register_map = BTreeMap::new();
        let mut next_virtual_slot = 33;

        for cycle in virtual_sequence {
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
                let actual = cycle.ts1_vals().unwrap();
                let expected_prefix = &tensor_registers[mapped_addr][..actual.len()];
                assert_eq!(expected_prefix, actual, "{cycle:#?}");
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
                let actual = cycle.ts2_vals().unwrap();
                let expected_prefix = &tensor_registers[mapped_addr][..actual.len()];
                assert_eq!(expected_prefix, actual, "{cycle:#?}");
            }

            let output = to_instruction_output(&cycle);

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
                td_output[..cycle.instr.num_output_elements().min(output.len())].copy_from_slice(
                    &output[..cycle.instr.num_output_elements().min(output.len())],
                );
                tensor_registers[mapped_addr] = td_output;
                let actual_output = cycle.td_post_vals().unwrap();
                let expected_prefix = &tensor_registers[mapped_addr][..actual_output.len()];
                assert_eq!(expected_prefix, actual_output, "{cycle:#?}");
                assert!(
                    tensor_registers[mapped_addr][actual_output.len()..]
                        .iter()
                        .all(|&v| v == 0),
                    "{cycle:#?}"
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

/// Special helper function to compute Broadcast operation output
fn compute_broadcast_output(cycle: &ONNXCycle) -> Vec<u64> {
    let output_els = cycle.instr.num_output_elements();
    let input = cycle.ts1_vals().unwrap_or(vec![0; output_els]);

    // Broadcast operation: replicate the first element to all positions
    let broadcast_value = if input.is_empty() { 0 } else { input[0] };
    vec![broadcast_value; output_els]
}

/// Special helper function to compute Sum operation output
fn compute_saturating_sum(cycle: &ONNXCycle) -> Vec<u64> {
    let output_els = cycle.instr.num_output_elements();
    let input = cycle.ts1_vals().unwrap_or(vec![0; output_els]);

    // Saturating Sum operation: sum all elements and put result in first position
    let mut total_sum: u64 = 0;
    let mut result = vec![0; output_els];
    for &val in &input {
        total_sum = total_sum.saturating_add(val);
    }
    if output_els > 0 {
        result[0] = total_sum;
    }
    result
}

fn to_instruction_output(cycle: &ONNXCycle) -> Vec<u64> {
    match cycle.instr.opcode {
        ONNXOpcode::Broadcast => compute_broadcast_output(cycle),
        ONNXOpcode::VirtualSaturatingSum => compute_saturating_sum(cycle),
        ONNXOpcode::Div => {
            DivInstruction::<32>::sequence_output(cycle.ts1_vals().unwrap(), cycle.imm().unwrap())
        }
        ONNXOpcode::Select => {
            let cond = cycle.ts1_vals().unwrap();
            let x_true = cycle.ts2_vals().unwrap();
            let x_false = cycle.ts3_vals().unwrap();

            cond.iter()
                .zip(x_true)
                .zip(x_false)
                .map(
                    |((&cond, value_true), value_false)| {
                        if cond != 0 { value_true } else { value_false }
                    },
                )
                .collect()
        }
        _ => {
            let output_els = cycle.instr.num_output_elements();
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
    }
}
