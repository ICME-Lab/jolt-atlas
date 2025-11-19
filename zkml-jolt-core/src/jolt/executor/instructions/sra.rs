use jolt_core::zkvm::instruction::LookupQuery;
use onnx_tracer::{
    constants::virtual_tensor_index,
    tensor::Tensor,
    trace_types::{MemoryState, ONNXCycle, ONNXInstr, ONNXOpcode},
};

use crate::{
    jolt::executor::instructions::{
        VirtualInstructionSequence,
        virtual_shift_right_bitmask::VirtualShiftRightBitmaskInstruction,
        virtual_sra::VirtualSraInstruction,
    },
    utils::{VirtualSequenceCounter, u64_vec_to_i32_iter},
};

/// Perform shift right arithmetical and return the result
pub struct SraInstruction<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> VirtualInstructionSequence for SraInstruction<WORD_SIZE> {
    const SEQUENCE_LENGTH: usize = 2;

    fn virtual_trace(cycle: ONNXCycle, K: usize) -> Vec<ONNXCycle> {
        assert_eq!(cycle.instr.opcode, ONNXOpcode::Sra);
        let num_outputs = cycle.instr.num_output_elements();

        // If Sra is part of a longer virtual sequence, recover the counter to continue decrementing it
        let virtual_sequence_remaining =
            if let Some(remaining) = cycle.instr.virtual_sequence_remaining {
                assert!(
                    remaining >= Self::SEQUENCE_LENGTH,
                    "Not enough remaining virtual sequence steps"
                );
                remaining
            } else {
                Self::SEQUENCE_LENGTH
            };

        let mut vseq_counter = VirtualSequenceCounter::new(virtual_sequence_remaining);

        // SRA source and destination registers
        let ts1 = cycle.instr.ts1;
        let ts2 = cycle.instr.ts2;
        let td = cycle.instr.td;

        // Virtual registers used in sequence
        let v_bitmask = Some(virtual_tensor_index(0, K, cycle.instr.td.unwrap()));

        // SRA operands
        let x = cycle.ts1_vals().unwrap_or(vec![0; num_outputs]);
        let y = cycle.ts2_vals().unwrap_or(vec![0; num_outputs]);
        let mut virtual_trace = vec![];

        let bitmask = (0..num_outputs)
            .map(|i| VirtualShiftRightBitmaskInstruction::<WORD_SIZE>(y[i]).to_lookup_output())
            .collect::<Vec<_>>();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualShiftRightBitmask,
                ts1: ts2,
                ts2: None,
                ts3: None,
                td: v_bitmask,
                imm: None,
                virtual_sequence_remaining: Some(vseq_counter.dec()),
                output_dims: cycle.instr.output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&y))),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&bitmask))),
            },
            advice_value: None,
        });

        let result = (0..num_outputs)
            .map(|i| VirtualSraInstruction::<WORD_SIZE>(x[i], bitmask[i]).to_lookup_output())
            .collect::<Vec<_>>();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualSra,
                ts1,
                ts2: v_bitmask,
                ts3: None,
                td,
                imm: None,
                virtual_sequence_remaining: Some(vseq_counter.dec()),
                output_dims: cycle.instr.output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&x))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&bitmask))),
                ts3_val: None,
                td_pre_val: cycle.memory_state.td_pre_val.clone(),
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&result))),
            },
            advice_value: None,
        });

        assert_eq!(
            virtual_trace.len(),
            Self::SEQUENCE_LENGTH,
            "Incorrect virtual trace length"
        );

        virtual_trace
    }

    fn sequence_output(x: Vec<u64>, y: Vec<u64>) -> Vec<u64> {
        let mask = match WORD_SIZE {
            32 => 0x1f,
            64 => 0x3f,
            _ => unimplemented!("Unsupported word size."),
        };
        let num_outputs = x.len();
        let mut output = vec![0; num_outputs];
        for i in 0..num_outputs {
            let x = x[i] as i32;
            let y = y[i] as i32;
            output[i] = (x >> (y & mask)) as u32 as u64;
        }
        output
    }
}

#[cfg(test)]
mod test {
    use crate::jolt::executor::instructions::test::jolt_virtual_sequence_test;

    use super::*;

    #[test]
    fn sra_virtual_sequence_32() {
        jolt_virtual_sequence_test::<SraInstruction<32>>(ONNXOpcode::Sra, 16);
    }

    #[test]
    fn test_sra() {
        // (x, y, output)
        let mappings = vec![
            (1, 0, 1),
            (0, 1, 0),
            (1, 1, 0),
            (8, 2, 2),
            (-4, 0, -4),
            (-4, 1, -2),
            (-4, 3, -1), // -1 >> 1 = -1
            (16, 32, 16),
        ];

        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut exp = Vec::new();

        for map in mappings {
            x.push(map.0 as u32 as u64);
            y.push(map.1 as u32 as u64);
            exp.push(map.2 as u32 as u64);
        }

        let output = SraInstruction::<32>::sequence_output(x, y);
        assert_eq!(output, exp);
    }
}
