use crate::{
    instructions::{declare_onnx_instr, div::Div, ElementWise, VirtualInstructionSequence},
    tensor::Tensor,
    trace_types::{
        AtlasCycle, AtlasInstr, AtlasOpcode, MemoryState, ONNXCycle, ONNXInstr, ONNXOpcode,
    },
    utils::{u64_vec_to_i32_iter, VirtualSequenceCounter, VirtualSlotCounter},
};

// Expandable
declare_onnx_instr!(name = DivI);

impl ElementWise for DivI {
    fn exec(x: u64, y: u64) -> u64 {
        Div::exec(x, y)
    }
}

impl VirtualInstructionSequence for DivI {
    const SEQUENCE_LENGTH: usize = 1 + Div::SEQUENCE_LENGTH;

    fn virtual_trace(cycle: ONNXCycle, virtual_slot: &mut VirtualSlotCounter) -> Vec<AtlasCycle> {
        assert_eq!(cycle.instr.opcode, ONNXOpcode::DivI);
        let num_outputs = cycle.instr.num_output_elements();

        // If DIVI is part of a longer virtual sequence, recover the counter to continue decrementing it
        let remaining = cycle
            .instr
            .virtual_sequence_remaining
            .unwrap_or(Self::SEQUENCE_LENGTH);
        assert!(
            remaining >= Self::SEQUENCE_LENGTH,
            "Not enough remaining virtual sequence steps"
        );
        let mut counter = VirtualSequenceCounter::new(remaining);

        // DIVI source registers
        let r_x = cycle.instr.ts1;
        let td = cycle.instr.td.unwrap_or(0);
        let output_dims = cycle.instr.output_dims.clone();

        // Virtual registers used in sequence
        let v_c = Some(virtual_slot.inc());

        // DIVI operands
        let x = cycle.ts1_vals().unwrap_or(vec![0; num_outputs]);
        let y = cycle.imm().unwrap_or(vec![0; num_outputs]);
        let mut virtual_trace = vec![];

        // const
        virtual_trace.push(AtlasCycle {
            instr: AtlasInstr {
                address: cycle.instr.address,
                opcode: AtlasOpcode::Constant,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_c,
                imm: cycle.instr.imm.clone(),
                virtual_sequence_remaining: Some(counter.dec()),
                output_dims: cycle.instr.output_dims.clone(),
            },
            memory_state: MemoryState {
                ts1_val: None,
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: cycle.instr.imm.clone(),
            },
            advice_value: None,
        });

        // Div
        let quotient = Div::sequence_output(&x, &y);
        virtual_trace.extend(Div::virtual_trace(
            ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::Div,
                    ts1: r_x,
                    ts2: v_c,
                    ts3: None,
                    td: Some(td),
                    imm: None,
                    virtual_sequence_remaining: Some(counter.get()),
                    output_dims: output_dims.clone(),
                },
                memory_state: MemoryState {
                    ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&x))),
                    ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&y))),
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&quotient))),
                },
                advice_value: None,
            },
            virtual_slot,
        ));
        counter.subtract(Div::SEQUENCE_LENGTH);

        assert_eq!(
            virtual_trace.len(),
            Self::SEQUENCE_LENGTH,
            "Incorrect virtual trace length"
        );

        virtual_trace
    }

    fn sequence_output(x: &[u64], y: &[u64]) -> Vec<u64> {
        x.iter()
            .zip(y.iter())
            .map(|(&x, &y)| Self::exec(x, y))
            .collect()
    }
}

#[cfg(test)]
mod test {
    use crate::instructions::test::jolt_virtual_sequence_test;

    use super::*;

    #[test]
    fn virtual_sequence_32() {
        jolt_virtual_sequence_test::<DivI>(ONNXOpcode::DivI, 16);
    }
}
