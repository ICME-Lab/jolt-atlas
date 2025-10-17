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
        VirtualInstructionSequence, add::AddInstruction, beq::BeqInstruction, mul::MulInstruction,
        virtual_advice::AdviceInstruction, virtual_assert_valid_div0::AssertValidDiv0Instruction,
        virtual_assert_valid_signed_remainder::AssertValidSignedRemainderInstruction,
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
    const SEQUENCE_LENGTH: usize = 9;

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
        assert_eq!(cycle.instr.opcode, ONNXOpcode::Div);
        let num_outputs = cycle.instr.active_output_elements;
        // DIV source registers
        let r_x = cycle.instr.ts1;

        // Virtual registers used in sequence
        let v_c = Some(virtual_tensor_index(0, K, cycle.instr.td.unwrap()));
        // let v_q = Some(virtual_tensor_index(1, K, cycle.instr.td.unwrap()));
        // let v_r = Some(virtual_tensor_index(2, K, cycle.instr.td.unwrap()));
        // let v_qy = Some(virtual_tensor_index(3, K, cycle.instr.td.unwrap()));
        // let v_0 = Some(virtual_tensor_index(4, K, cycle.instr.td.unwrap()));

        // DIV operands
        let x = cycle.ts1_vals().unwrap_or(vec![0; num_outputs]);
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

        // // const
        // virtual_trace.push(ONNXCycle {
        //     instr: ONNXInstr {
        //         address: cycle.instr.address,
        //         opcode: ONNXOpcode::VirtualConst,
        //         ts1: None,
        //         ts2: None,
        //         ts3: None,
        //         td: v_c,
        //         imm: cycle.instr.imm.clone(),
        //         virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
        //         active_output_elements: cycle.instr.active_output_elements,
        //         output_dims: cycle.instr.output_dims,
        //     },
        //     memory_state: MemoryState {
        //         ts1_val: None,
        //         ts2_val: None,
        //         ts3_val: None,
        //         td_pre_val: None,
        //         td_post_val: cycle.instr.imm.clone(),
        //     },
        //     advice_value: None,
        // });

        // let q = (0..num_outputs)
        //     .map(|i| AdviceInstruction::<WORD_SIZE>(rsqrt[i]).to_lookup_output())
        //     .collect_vec();
        // virtual_trace.push(ONNXCycle {
        //     instr: ONNXInstr {
        //         address: cycle.instr.address,
        //         opcode: ONNXOpcode::VirtualAdvice,
        //         ts1: None,
        //         ts2: None,
        //         ts3: None,
        //         td: v_q,
        //         imm: None,
        //         virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
        //         active_output_elements: cycle.instr.active_output_elements,
        //         output_dims: cycle.instr.output_dims,
        //     },
        //     memory_state: MemoryState {
        //         ts1_val: None,
        //         ts2_val: None,
        //         ts3_val: None,
        //         td_pre_val: None,
        //         td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&q))),
        //     },
        //     advice_value: Some(Tensor::from(u64_vec_to_i32_iter(&rsqrt))),
        // });

        // let r = (0..num_outputs)
        //     .map(|i| AdviceInstruction::<WORD_SIZE>(remainder[i]).to_lookup_output())
        //     .collect_vec();
        // virtual_trace.push(ONNXCycle {
        //     instr: ONNXInstr {
        //         address: cycle.instr.address,
        //         opcode: ONNXOpcode::VirtualAdvice,
        //         ts1: None,
        //         ts2: None,
        //         ts3: None,
        //         td: v_r,
        //         imm: None,
        //         virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
        //         active_output_elements: cycle.instr.active_output_elements,
        //         output_dims: cycle.instr.output_dims,
        //     },
        //     memory_state: MemoryState {
        //         ts1_val: None,
        //         ts2_val: None,
        //         ts3_val: None,
        //         td_pre_val: None,
        //         td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&r))),
        //     },
        //     advice_value: Some(Tensor::from(u64_vec_to_i32_iter(&remainder))),
        // });

        // let is_valid: Vec<u64> = (0..num_outputs)
        //     .map(|i| {
        //         AssertValidSignedRemainderInstruction::<WORD_SIZE>(r[i], y[i]).to_lookup_output()
        //     })
        //     .collect_vec();
        // is_valid.iter().for_each(|&valid| {
        //     assert_eq!(valid, 1, "Invalid signed remainder detected");
        // });
        // virtual_trace.push(ONNXCycle {
        //     instr: ONNXInstr {
        //         address: cycle.instr.address,
        //         opcode: ONNXOpcode::VirtualAssertValidSignedRemainder,
        //         ts1: v_r,
        //         ts2: v_c,
        //         ts3: None,
        //         td: None,
        //         imm: None,
        //         virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
        //         active_output_elements: cycle.instr.active_output_elements,
        //         output_dims: cycle.instr.output_dims,
        //     },
        //     memory_state: MemoryState {
        //         ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&r))),
        //         ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&y))),
        //         ts3_val: None,
        //         td_pre_val: None,
        //         td_post_val: None,
        //     },
        //     advice_value: None,
        // });

        // let is_valid: Vec<u64> = (0..num_outputs)
        //     .map(|i| AssertValidDiv0Instruction::<WORD_SIZE>(y[i], q[i]).to_lookup_output())
        //     .collect_vec();
        // is_valid.iter().for_each(|&valid| {
        //     assert_eq!(valid, 1, "Invalid division by zero detected");
        // });
        // virtual_trace.push(ONNXCycle {
        //     instr: ONNXInstr {
        //         address: cycle.instr.address,
        //         opcode: ONNXOpcode::VirtualAssertValidDiv0,
        //         ts1: v_c,
        //         ts2: v_q,
        //         ts3: None,
        //         td: None,
        //         imm: None,
        //         virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
        //         active_output_elements: cycle.instr.active_output_elements,
        //         output_dims: cycle.instr.output_dims,
        //     },
        //     memory_state: MemoryState {
        //         ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&y))),
        //         ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&q))),
        //         ts3_val: None,
        //         td_pre_val: None,
        //         td_post_val: None,
        //     },
        //     advice_value: None,
        // });

        // let q_y = (0..num_outputs)
        //     .map(|i| MulInstruction::<WORD_SIZE>(q[i], y[i]).to_lookup_output())
        //     .collect_vec();
        // virtual_trace.push(ONNXCycle {
        //     instr: ONNXInstr {
        //         address: cycle.instr.address,
        //         opcode: ONNXOpcode::Mul,
        //         ts1: v_q,
        //         ts2: v_c,
        //         ts3: None,
        //         td: v_qy,
        //         imm: None,
        //         virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
        //         active_output_elements: cycle.instr.active_output_elements,
        //         output_dims: cycle.instr.output_dims,
        //     },
        //     memory_state: MemoryState {
        //         ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&q))),
        //         ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&y))),
        //         ts3_val: None,
        //         td_pre_val: None,
        //         td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&q_y))),
        //     },
        //     advice_value: None,
        // });

        // let add_0 = (0..num_outputs)
        //     .map(|i| AddInstruction::<WORD_SIZE>(q_y[i], r[i]).to_lookup_output())
        //     .collect_vec();
        // virtual_trace.push(ONNXCycle {
        //     instr: ONNXInstr {
        //         address: cycle.instr.address,
        //         opcode: ONNXOpcode::Add,
        //         ts1: v_qy,
        //         ts2: v_r,
        //         ts3: None,
        //         td: v_0,
        //         imm: None,
        //         virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
        //         active_output_elements: cycle.instr.active_output_elements,
        //         output_dims: cycle.instr.output_dims,
        //     },
        //     memory_state: MemoryState {
        //         ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&q_y))),
        //         ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&r))),
        //         ts3_val: None,
        //         td_pre_val: None,
        //         td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&add_0))),
        //     },
        //     advice_value: None,
        // });

        // let _assert_eq = (0..num_outputs)
        //     .map(|i| BeqInstruction::<WORD_SIZE>(add_0[i], x[i]).to_lookup_output())
        //     .collect_vec();
        // virtual_trace.push(ONNXCycle {
        //     instr: ONNXInstr {
        //         address: cycle.instr.address,
        //         opcode: ONNXOpcode::VirtualAssertEq,
        //         ts1: v_0,
        //         ts2: r_x,
        //         ts3: None,
        //         td: None,
        //         imm: None,
        //         virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
        //         active_output_elements: cycle.instr.active_output_elements,
        //         output_dims: cycle.instr.output_dims,
        //     },
        //     memory_state: MemoryState {
        //         ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&add_0))),
        //         ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&x))),
        //         ts3_val: None,
        //         td_pre_val: None,
        //         td_post_val: None,
        //     },
        //     advice_value: None,
        // });

        // virtual_trace.push(ONNXCycle {
        //     instr: ONNXInstr {
        //         address: cycle.instr.address,
        //         opcode: ONNXOpcode::VirtualMove,
        //         ts1: v_q,
        //         ts2: None,
        //         ts3: None,
        //         td: cycle.instr.td,
        //         imm: None,
        //         virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
        //         active_output_elements: cycle.instr.active_output_elements,
        //         output_dims: cycle.instr.output_dims,
        //     },
        //     memory_state: MemoryState {
        //         ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&q))),
        //         ts2_val: None,
        //         ts3_val: None,
        //         td_pre_val: cycle.memory_state.td_pre_val.clone(),
        //         td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&q))),
        //     },
        //     advice_value: None,
        // });

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
    use crate::jolt::executor::instructions::test::jolt_virtual_sequence_test;

    use super::*;
    use rand::prelude::*;

    #[test]
    #[ignore]
    fn div_virtual_sequence_32() {
        jolt_virtual_sequence_test::<RsqrtInstruction<32>>(ONNXOpcode::Rsqrt, 16);
    }

    const SF_FLOAT: f32 = SF as f32;
    const INFORMATION_BITS: usize = 32 - SF_LOG - 1; // WORD_SIZE - SF_LOG - sign_bit 

    fn next_pos_i32(rng: &mut StdRng) -> i32 {
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
