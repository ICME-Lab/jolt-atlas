use crate::{
    jolt::instruction::{VirtualInstructionSequence, ge::GEInstruction},
    utils::u64_vec_to_i32_iter,
};
use jolt_core::jolt::instruction::LookupQuery;
use onnx_tracer::{
    constants::{MAX_TENSOR_SIZE, virtual_tensor_index},
    tensor::Tensor,
    trace_types::{MemoryState, ONNXCycle, ONNXInstr, ONNXOpcode},
};

pub struct ReduceMaxInstruction<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> VirtualInstructionSequence for ReduceMaxInstruction<WORD_SIZE> {
    const SEQUENCE_LENGTH: usize = (MAX_TENSOR_SIZE - 1) * 6 + 3;

    fn virtual_trace(cycle: ONNXCycle) -> Vec<ONNXCycle> {
        assert_eq!(cycle.instr.opcode, ONNXOpcode::ReduceMax);
        let zero_tensor = || Tensor::from((0..MAX_TENSOR_SIZE).map(|_| 0i32));
        let scalar_tensor = |scalar: u64| {
            let mut value = vec![0; MAX_TENSOR_SIZE];
            value[0] = scalar;
            Some(Tensor::from(u64_vec_to_i32_iter(&value)))
        };

        // Get the active output elements from the input tensor (ts1)
        let active_elements = cycle
            .memory_state
            .ts1_val
            .as_ref()
            .map(|t| t.dims().iter().product::<usize>().min(MAX_TENSOR_SIZE))
            .unwrap_or(cycle.instr.active_output_elements);

        // Create tensors for each value in the input array
        let gathered_ts1 = (0..MAX_TENSOR_SIZE)
            .map(|i| {
                let mut tensor = zero_tensor();
                tensor[0] = cycle.ts1_vals()[i] as u32 as i32;
                tensor
            })
            .collect::<Vec<_>>();

        // Create tensors representing possible indices for gathering
        let indices = (0..MAX_TENSOR_SIZE)
            .map(|i| {
                let mut tensor = zero_tensor();
                tensor[0] = i as u32 as i32;
                tensor
            })
            .collect::<Vec<_>>();

        // Create validity mask tensors: 1 for valid indices (< active_elements), 0 for padded
        let validity_masks = (0..MAX_TENSOR_SIZE)
            .map(|i| {
                let mut tensor = zero_tensor();
                tensor[0] = if i < active_elements { 1 } else { 0 };
                tensor
            })
            .collect::<Vec<_>>();

        // Virtual registers used in sequence (no need for index tracking)
        let vmax_val = Some(virtual_tensor_index(0));
        let vxi_val = Some(virtual_tensor_index(1));
        let vxi_idx = Some(virtual_tensor_index(2));
        let vcond = Some(virtual_tensor_index(3));
        let vmask = Some(virtual_tensor_index(4));
        let vmasked_cond = Some(virtual_tensor_index(5));

        // ReduceMax operands
        let x = cycle.ts1_vals();

        let mut virtual_trace = vec![];

        // Initialize max_val = x[0]
        // Use constant for index 0
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualConst,
                ts1: None,
                ts2: None,
                ts3: None,
                td: vxi_idx,
                imm: Some(indices[0].clone()),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: 1,
                output_dims: [1, 1],
            },
            memory_state: MemoryState {
                ts1_val: None,
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(indices[0].clone()),
            },
            advice_value: None,
        });

        // max_val = Gather(data=x, indices=0, axis=1) → [1]
        let mut max_val = x[0];
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Gather,
                ts1: cycle.instr.ts1,
                ts2: vxi_idx,
                ts3: None,
                td: vmax_val,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                output_dims: [1, 1],
                active_output_elements: 1,
            },
            memory_state: MemoryState {
                ts1_val: cycle.memory_state.ts1_val.clone(),
                ts2_val: Some(indices[0].clone()),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(gathered_ts1[0].clone()),
            },
            advice_value: None,
        });

        for i in 1..MAX_TENSOR_SIZE {
            // idx_i = Constant(value=[i]) → [1]
            virtual_trace.push(ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::VirtualConst,
                    ts1: None,
                    ts2: None,
                    ts3: None,
                    td: vxi_idx,
                    imm: Some(indices[i].clone()),
                    virtual_sequence_remaining: Some(
                        Self::SEQUENCE_LENGTH - virtual_trace.len() - 1,
                    ),
                    output_dims: [1, 1],
                    active_output_elements: 1,
                },
                memory_state: MemoryState {
                    ts1_val: None,
                    ts2_val: None,
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: Some(indices[i].clone()),
                },
                advice_value: None,
            });

            // x[i] = Gather(data=x, indices=idx_i, axis=1)
            let xi = x[i];
            virtual_trace.push(ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::Gather,
                    ts1: cycle.instr.ts1,
                    ts2: vxi_idx,
                    ts3: None,
                    td: vxi_val,
                    imm: None,
                    virtual_sequence_remaining: Some(
                        Self::SEQUENCE_LENGTH - virtual_trace.len() - 1,
                    ),
                    output_dims: [1, 1],
                    active_output_elements: 1,
                },
                memory_state: MemoryState {
                    ts1_val: cycle.memory_state.ts1_val.clone(),
                    ts2_val: Some(indices[i].clone()),
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: Some(gathered_ts1[i].clone()),
                },
                advice_value: None,
            });

            // Create validity mask constant for this index
            virtual_trace.push(ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::VirtualConst,
                    ts1: None,
                    ts2: None,
                    ts3: None,
                    td: vmask,
                    imm: Some(validity_masks[i].clone()),
                    virtual_sequence_remaining: Some(
                        Self::SEQUENCE_LENGTH - virtual_trace.len() - 1,
                    ),
                    output_dims: [1, 1],
                    active_output_elements: 1,
                },
                memory_state: MemoryState {
                    ts1_val: None,
                    ts2_val: None,
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: Some(validity_masks[i].clone()),
                },
                advice_value: None,
            });

            // ge = (xi >= max_val)
            let ge = { GEInstruction::<WORD_SIZE>(xi, max_val).to_lookup_output() };
            virtual_trace.push(ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::Gte,
                    ts1: vxi_val,
                    ts2: vmax_val,
                    ts3: None,
                    td: vcond,
                    imm: None,
                    virtual_sequence_remaining: Some(
                        Self::SEQUENCE_LENGTH - virtual_trace.len() - 1,
                    ),
                    output_dims: [1, 1],
                    active_output_elements: 1,
                },
                memory_state: MemoryState {
                    ts1_val: Some(gathered_ts1[i].clone()),
                    ts2_val: scalar_tensor(max_val),
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: scalar_tensor(ge),
                },
                advice_value: None,
            });

            // masked_cond = ge * validity_mask (constrains padded elements to have ge=0)
            let masked_ge = if i < active_elements { ge } else { 0 };
            virtual_trace.push(ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::Mul,
                    ts1: vcond,
                    ts2: vmask,
                    ts3: None,
                    td: vmasked_cond,
                    imm: None,
                    virtual_sequence_remaining: Some(
                        Self::SEQUENCE_LENGTH - virtual_trace.len() - 1,
                    ),
                    output_dims: [1, 1],
                    active_output_elements: 1,
                },
                memory_state: MemoryState {
                    ts1_val: scalar_tensor(ge),
                    ts2_val: Some(validity_masks[i].clone()),
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: scalar_tensor(masked_ge),
                },
                advice_value: None,
            });

            // max_val = select(masked_cond, xi, max_val) - only update the value, no index tracking
            virtual_trace.push(ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::Select,
                    ts1: vmasked_cond,
                    ts2: vxi_val,
                    ts3: vmax_val,
                    td: vmax_val,
                    imm: None,
                    virtual_sequence_remaining: Some(
                        Self::SEQUENCE_LENGTH - virtual_trace.len() - 1,
                    ),
                    output_dims: [1, 1],
                    active_output_elements: 1,
                },
                memory_state: MemoryState {
                    ts1_val: scalar_tensor(masked_ge),
                    ts2_val: Some(gathered_ts1[i].clone()),
                    ts3_val: scalar_tensor(max_val),
                    td_pre_val: None,
                    td_post_val: scalar_tensor(if masked_ge == 1 { xi } else { max_val }),
                },
                advice_value: None,
            });
            max_val = if masked_ge == 1 { xi } else { max_val };
        }

        // Move final result to output
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualMove,
                ts1: vmax_val,
                ts2: None,
                ts3: None,
                td: cycle.instr.td,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                output_dims: [1, 1],
                active_output_elements: 1,
            },
            memory_state: MemoryState {
                ts1_val: scalar_tensor(max_val),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: cycle.memory_state.td_pre_val.clone(),
                td_post_val: scalar_tensor(max_val),
            },
            advice_value: None,
        });

        virtual_trace
    }

    /// This function implements the reduce max operation, which finds the maximum
    /// value in an array. The algorithm works as follows:
    ///
    /// 1. Initialize max_val to the first element v[0]
    ///
    /// 2. For each subsequent element i from 1 to N-1:
    ///    a. Compare current element v[i] with max_val using >= operator
    ///
    ///    b. If v[i] >= max_val (condition c is true):
    ///       - Update max_val to v[i] (select new maximum value)
    ///
    ///    c. Otherwise, keep current max_val unchanged
    ///
    /// 3. Return the maximum value found
    ///
    /// Time Complexity: O(N) where N is the array length
    /// Space Complexity: O(1) as only constant extra space is used
    ///
    /// Example:
    /// Input array: [3, 7, 2, 9, 1]
    /// Step 1: max_val = 3
    /// Step 2: i=1, v[1]=7 >= 3 → max_val = 7
    /// Step 3: i=2, v[2]=2 < 7 → no change
    /// Step 4: i=3, v[3]=9 >= 7 → max_val = 9
    /// Step 5: i=4, v[4]=1 < 9 → no change
    /// Result: max_val = 9
    fn sequence_output(x: Vec<u64>, _: Vec<u64>) -> Vec<u64> {
        let x = x
            .iter()
            .map(|&v| v as u32 as i32 as i64)
            .collect::<Vec<_>>();
        let mut max_val = x[0];
        for &xi in x.iter().skip(1) {
            let c = xi >= max_val;
            max_val = if c { xi } else { max_val };
        }
        // Pad output tensor to MAX_TENSOR_SIZE
        // The first element is the maximum value
        // The rest are zeros
        let mut output = vec![0; MAX_TENSOR_SIZE];
        output[0] = max_val as i32 as u32 as u64;
        output
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::jolt::instruction::test::jolt_virtual_sequence_test;
    use rand::Rng;

    #[test]
    fn reduce_max_virtual_sequence_32() {
        jolt_virtual_sequence_test::<ReduceMaxInstruction<32>>(ONNXOpcode::ReduceMax);
    }

    #[test]
    fn test_reduce_max() {
        // Helper function to test a single case
        let test_case = |input: Vec<u64>, expected_max: u64, description: &str| {
            let mut expected_output = vec![0; MAX_TENSOR_SIZE];
            expected_output[0] = expected_max;

            let result = ReduceMaxInstruction::<32>::sequence_output(input.clone(), vec![]);
            assert_eq!(result, expected_output, "Failed for case: {description}");

            let cycle = ONNXCycle {
                instr: ONNXInstr {
                    address: 1,
                    opcode: ONNXOpcode::ReduceMax,
                    ts1: Some(0),
                    ts2: None,
                    ts3: None,
                    td: Some(1),
                    imm: None,
                    virtual_sequence_remaining: None,
                    active_output_elements: 1,
                    output_dims: [1, 1],
                },
                memory_state: MemoryState {
                    ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&input))),
                    ts2_val: None,
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&expected_output))),
                },
                advice_value: None,
            };

            let reduce_max_trace = ReduceMaxInstruction::<32>::virtual_trace(cycle);
            assert_eq!(
                reduce_max_trace.len(),
                ReduceMaxInstruction::<32>::SEQUENCE_LENGTH
            );
            let output_cycle = reduce_max_trace.last().unwrap();
            assert_eq!(
                output_cycle.td_post_vals(),
                expected_output,
                "Virtual trace failed for case: {description}",
            );
        };

        // Edge cases
        test_case(vec![5], 5, "single element");
        test_case(vec![1, 2], 2, "two elements, max at end");
        test_case(vec![2, 1], 2, "two elements, max at start");
        test_case(vec![5, 5, 5], 5, "all elements equal");
        test_case(vec![1, 5, 3, 5, 2], 5, "duplicate max values");
        test_case(vec![10, 1, 2, 3, 4], 10, "max at beginning");
        test_case(vec![1, 2, 3, 4, 10], 10, "max at end");
        test_case(vec![0, 0, 0, 0], 0, "all zeros");
        test_case(vec![u32::MAX as u64], u32::MAX as u64, "maximum u32 value");
        test_case(
            vec![0, u32::MAX as u64, 0],
            0,
            "maximum u32 value in middle",
        );

        // Random test cases
        let mut rng = rand::thread_rng();
        for i in 0..1000 {
            let size = rng.gen_range(1..=std::cmp::min(50, MAX_TENSOR_SIZE));
            let input: Vec<u64> = (0..size).map(|_| rng.gen_range(0..=1000)).collect();
            let expected_max = *input.iter().max().unwrap();

            test_case(input, expected_max, &format!("random case {i}"));
        }

        println!("All reduce max tests passed!");
    }
}
